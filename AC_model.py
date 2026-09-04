import os

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
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
DATA_FILE = "extracted_data.xlsx"
TEMPORAL_DATA_FILE = None

PREFERRED_SHEET = "Eng_revise"
TEMPORAL_SHEET = "Eng_revise"
TARGET_COLUMN = "十天随访结果"
OUTPUT_DIR = "model_results"

TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 577
FEATURE_METHOD = 0
TOP_N = 20
CORR_THRESHOLD = 0.80
CLASSIFICATION_THRESHOLD = 0.50


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
