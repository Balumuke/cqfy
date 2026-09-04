import os
import joblib
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from patsy import build_design_matrices, dmatrix
from scipy.special import logit
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
#%%
DATA_FILE = "Dec2023_Dec2026_data.xlsx"
TEMPORAL_DATA_FILE = "Jan2026_Jun2026_data.xlsx"

PREFERRED_SHEET = "Eng_revise"
TEMPORAL_SHEET = "Eng_revise"
TARGET_COLUMN = "十天随访结果"
OUTPUT_DIR = "model_results"

TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 577
FEATURE_METHOD = 0
TOP_N = 10
CORR_THRESHOLD = 0.80
CLASSIFICATION_THRESHOLD = 0.50

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CI = 0.95
CALIBRATION_BINS = 10
DCA_THRESHOLDS = np.arange(0.05, 0.51, 0.01)
SHAP_SAMPLE_SIZE = 1000
RCS_DF = 4
RCS_GRID_POINTS = 200

RCS_KEY_CONTINUOUS = [
    "gestational days",
    "neutrophil percentage",
    "maximum amniotic fluid depth",
    "BMI",
    "daily temperature",
]

RCS_BASE_COVARIATES = [
    "maternal age",
    "maternal occupation",
    "maternal educational level",
    "maternal ethnicity",
    "paternal age",
    "paternal occupation",
    "paternal educational level",
    "paternal ethnicity",
    "gravidity (G)",
    "parity (P)",
    "history of spontaneous abortion",
    "history of induced abortion",
    "history of cesarean section",
    "history of vaginal delivery",
    "systolic blood pressure",
    "diastolic blood pressure",
    "white blood cell count (WBC)",
    "placental location",
    "indication for amniocentesis",
]


feature_columns = [
    "maternal age",
    "maternal occupation",
    "maternal educational level",
    "maternal ethnicity",
    "paternal age",
    "paternal occupation",
    "paternal educational level",
    "paternal ethnicity",
    "gestational days",
    "gravidity (G)",
    "parity (P)",
    "history of spontaneous abortion",
    "history of induced abortion",
    "history of cesarean section",
    "history of vaginal delivery",
    "systolic blood pressure",
    "diastolic blood pressure",
    "BMI",
    "biparietal diameter (BPD)",
    "femur length (FL)",
    "maximum amniotic fluid depth",
    "placental location",
    "white blood cell count (WBC)",
    "neutrophil percentage",
    "indication for amniocentesis",
    "daily temperature",
]

categorical_columns = [
    "maternal occupation",
    "maternal educational level",
    "maternal ethnicity",
    "paternal occupation",
    "paternal educational level",
    "paternal ethnicity",
    "gravidity (G)",
    "parity (P)",
    "history of spontaneous abortion",
    "history of induced abortion",
    "history of cesarean section",
    "history of vaginal delivery",
    "placental location",
    "indication for amniocentesis",
]

continuous_columns = [
    "maternal age",
    "paternal age",
    "gestational days",
    "systolic blood pressure",
    "diastolic blood pressure",
    "BMI",
    "biparietal diameter (BPD)",
    "femur length (FL)",
    "maximum amniotic fluid depth",
    "white blood cell count (WBC)",
    "neutrophil percentage",
    "daily temperature",
]

feature_method_names = {
    0: "full",
    1: "random_forest",
    2: "lasso",
    3: "univariable",
}
#%%
def read_data(file_path, preferred_sheet):
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)

    workbook = pd.ExcelFile(file_path)
    sheet_name = (
        preferred_sheet
        if preferred_sheet in workbook.sheet_names
        else workbook.sheet_names[0]
    )
    return pd.read_excel(file_path, sheet_name=sheet_name)


def prepare_data(data):
    required_columns = feature_columns + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.loc[data[TARGET_COLUMN].notna(), required_columns].copy()
    X = data[feature_columns]
    y = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)

    if set(y.unique()) != {0, 1}:
        raise ValueError("The outcome must contain both classes coded as 0 and 1")

    if X.isna().any().any():
        raise ValueError(
            "Predictor data contain missing values. "
            "Complete the prespecified imputation before model training."
        )

    return X, y


def preprocess_data(X_train, X_test):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                categorical_columns,
            ),
        ]
    )

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    categorical_feature_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
    )
    processed_feature_names = continuous_columns + categorical_feature_names

    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=processed_feature_names,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=processed_feature_names,
        index=X_test.index,
    )

    return preprocessor, X_train_processed, X_test_processed, categorical_feature_names

#%%
def select_features(X_train, y_train, method, categorical_feature_names):
    if method == 0:
        return list(X_train.columns)

    if method == 1:
        selector = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)
        order = np.argsort(selector.feature_importances_)[::-1]
        return list(X_train.columns[order[: min(TOP_N, X_train.shape[1])]])

    if method == 2:
        selector = LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        selector.fit(X_train, y_train)
        coefficients = selector.coef_.ravel()
        selected_index = np.flatnonzero(coefficients != 0)

        if selected_index.size == 0:
            selected_index = np.argsort(np.abs(coefficients))[::-1][: min(TOP_N, X_train.shape[1])]

        return list(X_train.columns[selected_index])

    if method == 3:
        numerical_names = [column for column in continuous_columns if column in X_train.columns]
        categorical_names = [
            column
            for column in categorical_feature_names
            if column in X_train.columns
        ]

        numerical_p_values = f_classif(X_train[numerical_names], y_train)[1]
        categorical_p_values = chi2(X_train[categorical_names], y_train)[1]

        p_values = pd.Series(
            np.concatenate([numerical_p_values, categorical_p_values]),
            index=numerical_names + categorical_names,
        )
        return list(p_values.sort_values().index[: min(TOP_N, len(p_values))])

    raise ValueError("FEATURE_METHOD must be 0, 1, 2, or 3")


def remove_redundant_features(X_train, selected_features):
    X_selected = X_train[selected_features]

    selected_features = X_selected.columns[
        X_selected.nunique(dropna=False) > 1
    ].tolist()
    X_selected = X_selected[selected_features]

    if X_selected.shape[1] <= 1:
        return selected_features

    correlation = X_selected.corr().abs()
    upper_triangle = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    correlated_features = [
        column
        for column in upper_triangle.columns
        if (upper_triangle[column] > CORR_THRESHOLD).any()
    ]

    return [column for column in selected_features if column not in correlated_features]

#%%
def build_models():
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "GaussianNB": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def predict_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    decision_score = model.decision_function(X)
    return 1 / (1 + np.exp(-decision_score))


def calculate_metrics(y_true, y_probability):
    y_prediction = (y_probability >= CLASSIFICATION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prediction, labels=[0, 1]).ravel()

    return {
        "n": len(y_true),
        "events": int(np.sum(y_true)),
        "auc": roc_auc_score(y_true, y_probability),
        "accuracy": accuracy_score(y_true, y_prediction),
        "f1": f1_score(y_true, y_prediction, zero_division=0),
        "precision": precision_score(y_true, y_prediction, zero_division=0),
        "sensitivity": recall_score(y_true, y_prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

#%%
def run_cross_validation(X_development, y_development):
    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    models = build_models()
    results = []

    for fold, (train_index, validation_index) in enumerate(
        cross_validation.split(X_development, y_development),
        start=1,
    ):
        X_train = X_development.iloc[train_index]
        y_train = y_development.iloc[train_index]
        X_validation = X_development.iloc[validation_index]
        y_validation = y_development.iloc[validation_index]

        _, X_train_processed, X_validation_processed, categorical_names = preprocess_data(
            X_train,
            X_validation,
        )

        selected_features = select_features(
            X_train_processed,
            y_train,
            FEATURE_METHOD,
            categorical_names,
        )
        selected_features = remove_redundant_features(
            X_train_processed,
            selected_features,
        )

        for model_name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(X_train_processed[selected_features], y_train)

            y_probability = predict_probability(
                fitted_model,
                X_validation_processed[selected_features],
            )
            metrics = calculate_metrics(y_validation, y_probability)

            results.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "feature_method": feature_method_names[FEATURE_METHOD],
                    "n_features": len(selected_features),
                    **metrics,
                }
            )

    results = pd.DataFrame(results)
    summary = results.groupby("model", as_index=False).agg(
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        sensitivity_mean=("sensitivity", "mean"),
        sensitivity_std=("sensitivity", "std"),
        specificity_mean=("specificity", "mean"),
        specificity_std=("specificity", "std"),
    )
    summary = summary.sort_values(
        ["auc_mean", "f1_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return results, summary

#%%
def fit_final_model(X_development, y_development, model_name):
    preprocessor, X_processed, _, categorical_names = preprocess_data(
        X_development,
        X_development,
    )

    selected_features = select_features(
        X_processed,
        y_development,
        FEATURE_METHOD,
        categorical_names,
    )
    selected_features = remove_redundant_features(
        X_processed,
        selected_features,
    )

    model = clone(build_models()[model_name])
    model.fit(X_processed[selected_features], y_development)

    return {
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "model_name": model_name,
        "feature_method": feature_method_names[FEATURE_METHOD],
        "model": model,
    }


def evaluate_model(model_bundle, X_test, y_test, cohort_name):
    preprocessor = model_bundle["preprocessor"]
    X_test_array = preprocessor.transform(X_test)

    categorical_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
    )
    processed_feature_names = continuous_columns + categorical_names
    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=processed_feature_names,
        index=X_test.index,
    )

    y_probability = predict_probability(
        model_bundle["model"],
        X_test_processed[model_bundle["selected_features"]],
    )
    y_prediction = (y_probability >= CLASSIFICATION_THRESHOLD).astype(int)
    metrics = calculate_metrics(y_test, y_probability)
    metrics.update(
        {
            "cohort": cohort_name,
            "model": model_bundle["model_name"],
            "feature_method": model_bundle["feature_method"],
            "n_features": len(model_bundle["selected_features"]),
        }
    )

    predictions = pd.DataFrame(
        {
            "observed": np.asarray(y_test),
            "predicted_probability": y_probability,
            "predicted_class": y_prediction,
        },
        index=X_test.index,
    )

    return metrics, predictions

#%%

def bootstrap_confidence_intervals(
    y_true,
    y_probability,
    n_resamples=BOOTSTRAP_RESAMPLES,
    confidence_level=BOOTSTRAP_CI,
    random_state=RANDOM_STATE,
):
    y_true = np.asarray(y_true).astype(int)
    y_probability = np.asarray(y_probability, dtype=float)
    rng = np.random.default_rng(random_state)

    metric_names = [
        "auc",
        "accuracy",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
    ]
    bootstrap_values = {name: [] for name in metric_names}

    for _ in range(n_resamples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        y_boot = y_true[indices]
        p_boot = y_probability[indices]

        if np.unique(y_boot).size < 2:
            continue

        metrics = calculate_metrics(y_boot, p_boot)
        for name in metric_names:
            bootstrap_values[name].append(metrics[name])

    alpha = (1.0 - confidence_level) / 2.0
    point_estimates = calculate_metrics(y_true, y_probability)

    rows = []
    for name in metric_names:
        values = np.asarray(bootstrap_values[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": point_estimates[name],
                "ci_lower": np.quantile(values, alpha),
                "ci_upper": np.quantile(values, 1.0 - alpha),
                "n_bootstrap_valid": len(values),
            }
        )

    return pd.DataFrame(rows)


def calibration_analysis(
    y_true,
    y_probability,
    n_bins=CALIBRATION_BINS,
):
    y_true = np.asarray(y_true).astype(int)
    y_probability = np.asarray(y_probability, dtype=float)
    clipped_probability = np.clip(y_probability, 1e-6, 1.0 - 1e-6)

    brier = brier_score_loss(y_true, y_probability)

    calibration_predictor = logit(clipped_probability)
    calibration_design = sm.add_constant(calibration_predictor, has_constant="add")
    calibration_model = sm.GLM(
        y_true,
        calibration_design,
        family=sm.families.Binomial(),
    ).fit()

    calibration_intercept = float(calibration_model.params[0])
    calibration_slope = float(calibration_model.params[1])

    calibration_data = pd.DataFrame(
        {
            "observed": y_true,
            "predicted_probability": y_probability,
        }
    )

    n_unique = calibration_data["predicted_probability"].nunique()
    q = min(n_bins, n_unique)

    if q < 2:
        calibration_data["bin"] = 0
    else:
        calibration_data["bin"] = pd.qcut(
            calibration_data["predicted_probability"],
            q=q,
            labels=False,
            duplicates="drop",
        )

    grouped_rows = []
    for bin_id, group in calibration_data.groupby("bin", observed=True):
        n = len(group)
        events = int(group["observed"].sum())
        observed_rate = events / n
        ci_low, ci_high = proportion_confint(
            count=events,
            nobs=n,
            alpha=0.05,
            method="wilson",
        )

        grouped_rows.append(
            {
                "bin": int(bin_id),
                "n": n,
                "events": events,
                "mean_predicted_probability": group["predicted_probability"].mean(),
                "observed_event_proportion": observed_rate,
                "observed_ci_lower": ci_low,
                "observed_ci_upper": ci_high,
            }
        )

    calibration_groups = pd.DataFrame(grouped_rows)

    calibration_metrics = {
        "brier_score": float(brier),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }

    return calibration_metrics, calibration_groups


def decision_curve_analysis(
    y_true,
    y_probability,
    thresholds=DCA_THRESHOLDS,
):
    y_true = np.asarray(y_true).astype(int)
    y_probability = np.asarray(y_probability, dtype=float)

    n = len(y_true)
    prevalence = y_true.mean()
    rows = []

    for threshold in thresholds:
        predicted_high_risk = y_probability >= threshold

        tp = np.sum((predicted_high_risk == 1) & (y_true == 1))
        fp = np.sum((predicted_high_risk == 1) & (y_true == 0))
        odds_at_threshold = threshold / (1.0 - threshold)

        model_net_benefit = (
            tp / n
            - fp / n * odds_at_threshold
        )
        treat_all_net_benefit = (
            prevalence
            - (1.0 - prevalence) * odds_at_threshold
        )

        rows.append(
            {
                "threshold_probability": float(threshold),
                "model_net_benefit": float(model_net_benefit),
                "treat_all_net_benefit": float(treat_all_net_benefit),
                "treat_none_net_benefit": 0.0,
                "true_positive": int(tp),
                "false_positive": int(fp),
            }
        )

    return pd.DataFrame(rows)


def _processed_feature_names(preprocessor):
    categorical_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
    )
    return continuous_columns + categorical_names


def _processed_to_original_feature(processed_name):
    if processed_name in continuous_columns:
        return processed_name

    for original_name in sorted(categorical_columns, key=len, reverse=True):
        if processed_name == original_name or processed_name.startswith(f"{original_name}_"):
            return original_name

    return processed_name


def run_shap_analysis(
    model_bundle,
    X_analysis,
    y_analysis,
    output_dir,
):
    if model_bundle["model_name"] not in {"LightGBM", "XGBoost"}:
        raise ValueError(
            "The current SHAP implementation is intended for the tree-based final model "
            "used in the manuscript (LightGBM or XGBoost)."
        )

    n_sample = min(SHAP_SAMPLE_SIZE, len(X_analysis))

    if n_sample < len(X_analysis):
        sample_indices, _ = train_test_split(
            np.arange(len(X_analysis)),
            train_size=n_sample,
            stratify=np.asarray(y_analysis),
            random_state=RANDOM_STATE,
        )
        X_sample = X_analysis.iloc[sample_indices].copy()
        y_sample = y_analysis.iloc[sample_indices].copy()
    else:
        X_sample = X_analysis.copy()
        y_sample = y_analysis.copy()

    preprocessor = model_bundle["preprocessor"]
    X_sample_array = preprocessor.transform(X_sample)
    processed_names = _processed_feature_names(preprocessor)

    X_sample_processed = pd.DataFrame(
        X_sample_array,
        columns=processed_names,
        index=X_sample.index,
    )
    X_model = X_sample_processed[model_bundle["selected_features"]]

    explainer = shap.TreeExplainer(model_bundle["model"])
    shap_result = explainer.shap_values(X_model)

    if isinstance(shap_result, list):
        shap_values = np.asarray(shap_result[-1])
    elif hasattr(shap_result, "values"):
        shap_values = np.asarray(shap_result.values)
    else:
        shap_values = np.asarray(shap_result)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    if shap_values.shape != X_model.shape:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values.shape}; expected {X_model.shape}."
        )

    processed_importance = pd.DataFrame(
        {
            "processed_feature": X_model.columns,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    processed_importance["original_feature"] = processed_importance[
        "processed_feature"
    ].map(_processed_to_original_feature)

    original_importance = (
        processed_importance.groupby("original_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_values_frame = pd.DataFrame(
        shap_values,
        columns=[f"SHAP::{name}" for name in X_model.columns],
        index=X_sample.index,
    )

    dependence_data = pd.DataFrame(
        {
            "observed": np.asarray(y_sample),
            "predicted_probability": predict_probability(model_bundle["model"], X_model),
        },
        index=X_sample.index,
    )

    for feature in RCS_KEY_CONTINUOUS:
        dependence_data[feature] = X_sample[feature]
        if feature in X_model.columns:
            dependence_data[f"SHAP::{feature}"] = shap_values_frame[f"SHAP::{feature}"]

    processed_importance.to_csv(
        os.path.join(output_dir, "shap_feature_importance_processed.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    original_importance.to_csv(
        os.path.join(output_dir, "shap_feature_importance_original.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    dependence_data.to_csv(
        os.path.join(output_dir, "shap_dependence_data.csv"),
        index=True,
        encoding="utf-8-sig",
    )

    return processed_importance, original_importance, dependence_data


def _prepare_rcs_covariates(X, columns):
    covariate_data = X[columns].copy()

    categorical_covariates = [
        column for column in columns if column in categorical_columns
    ]
    continuous_covariates = [
        column for column in columns if column not in categorical_covariates
    ]

    parts = []

    if continuous_covariates:
        continuous_data = covariate_data[continuous_covariates].apply(
            pd.to_numeric,
            errors="raise",
        )
        parts.append(continuous_data.astype(float))

    if categorical_covariates:
        categorical_data = pd.get_dummies(
            covariate_data[categorical_covariates].astype(str),
            drop_first=True,
            dtype=float,
        )
        parts.append(categorical_data)

    if parts:
        covariate_matrix = pd.concat(parts, axis=1)
    else:
        covariate_matrix = pd.DataFrame(index=X.index)

    return covariate_matrix.astype(float)


def run_rcs_analysis(
    X_analysis,
    y_analysis,
    output_dir,
):
    X_analysis = X_analysis.reset_index(drop=True).copy()
    y_analysis = pd.Series(np.asarray(y_analysis).astype(int)).reset_index(drop=True)

    summary_rows = []

    for focal_feature in RCS_KEY_CONTINUOUS:
        adjustment_columns = list(RCS_BASE_COVARIATES)
        adjustment_columns.extend(
            [
                feature
                for feature in RCS_KEY_CONTINUOUS
                if feature != focal_feature
            ]
        )

        adjustment_columns = list(dict.fromkeys(adjustment_columns))
        covariate_matrix = _prepare_rcs_covariates(
            X_analysis,
            adjustment_columns,
        ).reset_index(drop=True)

        focal_values = pd.to_numeric(
            X_analysis[focal_feature],
            errors="raise",
        ).astype(float)

        spline_basis = dmatrix(
            f"cr(x, df={RCS_DF}) - 1",
            {"x": focal_values},
            return_type="dataframe",
        )
        spline_design_info = spline_basis.design_info
        spline_basis = spline_basis.reset_index(drop=True)
        spline_columns = [
            f"spline_{index}"
            for index in range(spline_basis.shape[1])
        ]
        spline_basis.columns = spline_columns

        full_design = pd.concat(
            [spline_basis, covariate_matrix],
            axis=1,
        )
        full_design = sm.add_constant(full_design, has_constant="add").astype(float)

        spline_model = sm.GLM(
            y_analysis,
            full_design,
            family=sm.families.Binomial(),
        ).fit()

        linear_feature = pd.DataFrame(
            {"x_linear": focal_values}
        )
        linear_design = pd.concat(
            [linear_feature, covariate_matrix],
            axis=1,
        )
        linear_design = sm.add_constant(
            linear_design,
            has_constant="add",
        ).astype(float)

        linear_model = sm.GLM(
            y_analysis,
            linear_design,
            family=sm.families.Binomial(),
        ).fit()

        likelihood_ratio = max(
            0.0,
            2.0 * (spline_model.llf - linear_model.llf),
        )
        df_difference = max(1, len(spline_columns) - 1)
        p_nonlinearity = float(
            chi2.sf(likelihood_ratio, df_difference)
        )

        grid = np.linspace(
            focal_values.min(),
            focal_values.max(),
            RCS_GRID_POINTS,
        )

        grid_basis = build_design_matrices(
            [spline_design_info],
            {"x": grid},
        )[0]
        grid_basis = np.asarray(grid_basis, dtype=float)

        beta_spline = spline_model.params[spline_columns].to_numpy(dtype=float)
        covariance_spline = spline_model.cov_params().loc[
            spline_columns,
            spline_columns,
        ].to_numpy(dtype=float)

        linear_predictor = grid_basis @ beta_spline
        reference_index = int(np.argmin(linear_predictor))
        reference_value = float(grid[reference_index])
        reference_basis = grid_basis[reference_index]

        contrast = grid_basis - reference_basis
        log_or = contrast @ beta_spline

        variance = np.einsum(
            "ij,jk,ik->i",
            contrast,
            covariance_spline,
            contrast,
        )
        standard_error = np.sqrt(np.clip(variance, 0.0, None))

        odds_ratio = np.exp(log_or)
        ci_lower = np.exp(log_or - 1.96 * standard_error)
        ci_upper = np.exp(log_or + 1.96 * standard_error)

        curve = pd.DataFrame(
            {
                "feature": focal_feature,
                "value": grid,
                "adjusted_or": odds_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "reference_value": reference_value,
                "p_nonlinearity": p_nonlinearity,
            }
        )

        safe_name = (
            focal_feature.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )

        curve.to_csv(
            os.path.join(output_dir, f"rcs_curve_{safe_name}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        summary_rows.append(
            {
                "feature": focal_feature,
                "reference_value": reference_value,
                "p_nonlinearity": p_nonlinearity,
                "likelihood_ratio": likelihood_ratio,
                "df_difference": df_difference,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        os.path.join(output_dir, "rcs_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def save_evaluation_diagnostics(
    cohort_name,
    y_true,
    predictions,
    output_dir,
    run_dca=False,
):
    y_probability = predictions["predicted_probability"].to_numpy()

    bootstrap_ci = bootstrap_confidence_intervals(
        y_true,
        y_probability,
        random_state=RANDOM_STATE,
    )
    bootstrap_ci.insert(0, "cohort", cohort_name)
    bootstrap_ci.to_csv(
        os.path.join(output_dir, f"{cohort_name}_bootstrap_ci.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    calibration_metrics, calibration_groups = calibration_analysis(
        y_true,
        y_probability,
    )

    calibration_metrics_frame = pd.DataFrame(
        [
            {
                "cohort": cohort_name,
                **calibration_metrics,
            }
        ]
    )
    calibration_metrics_frame.to_csv(
        os.path.join(output_dir, f"{cohort_name}_calibration_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    calibration_groups.insert(0, "cohort", cohort_name)
    calibration_groups.to_csv(
        os.path.join(output_dir, f"{cohort_name}_calibration_groups.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    dca = None
    if run_dca:
        dca = decision_curve_analysis(
            y_true,
            y_probability,
        )
        dca.insert(0, "cohort", cohort_name)
        dca.to_csv(
            os.path.join(output_dir, f"{cohort_name}_decision_curve.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    return bootstrap_ci, calibration_metrics_frame, calibration_groups, dca


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    historical_data = read_data(DATA_FILE, PREFERRED_SHEET)
    X, y = prepare_data(historical_data)

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Development cohort: {len(X_development)}")
    print(f"Internal test cohort: {len(X_test)}")

    cv_results, cv_summary = run_cross_validation(
        X_development,
        y_development,
    )
    best_model_name = cv_summary.loc[0, "model"]

    print("\nCross-validation summary")
    print(cv_summary.to_string(index=False))
    print(f"\nSelected model: {best_model_name}")

    final_model = fit_final_model(
        X_development,
        y_development,
        best_model_name,
    )

    evaluation_results = []

    internal_metrics, internal_predictions = evaluate_model(
        final_model,
        X_test,
        y_test,
        "internal_test",
    )
    evaluation_results.append(internal_metrics)
    internal_predictions.to_csv(
        os.path.join(OUTPUT_DIR, "internal_test_predictions.csv"),
        encoding="utf-8-sig",
    )

    save_evaluation_diagnostics(
        "internal_test",
        y_test,
        internal_predictions,
        OUTPUT_DIR,
        run_dca=False,
    )

    if TEMPORAL_DATA_FILE:
        temporal_data = read_data(TEMPORAL_DATA_FILE, TEMPORAL_SHEET)
        X_temporal, y_temporal = prepare_data(temporal_data)

        temporal_metrics, temporal_predictions = evaluate_model(
            final_model,
            X_temporal,
            y_temporal,
            "temporal_validation",
        )
        evaluation_results.append(temporal_metrics)
        temporal_predictions.to_csv(
            os.path.join(OUTPUT_DIR, "temporal_validation_predictions.csv"),
            encoding="utf-8-sig",
        )

        save_evaluation_diagnostics(
            "temporal_validation",
            y_temporal,
            temporal_predictions,
            OUTPUT_DIR,
            run_dca=True,
        )

    run_shap_analysis(
        final_model,
        X,
        y,
        OUTPUT_DIR,
    )

    run_rcs_analysis(
        X,
        y,
        OUTPUT_DIR,
    )

    evaluation_results = pd.DataFrame(evaluation_results)

    cv_results.to_csv(
        os.path.join(OUTPUT_DIR, "cross_validation_fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    cv_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cross_validation_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    evaluation_results.to_csv(
        os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, "final_model.joblib"))

    print("\nIndependent evaluation")
    print(evaluation_results.to_string(index=False))


if __name__ == "__main__":
    main()
