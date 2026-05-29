import os
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from statsmodels.nonparametric.smoothers_lowess import lowess

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
#%%

# ============================================================
# Global plotting style
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


# ============================================================
# Configuration
# ============================================================
DATA_FILE = "extracted_data.xlsx"
PREFERRED_SHEET = "Eng_revise"
FALLBACK_SHEET = "Sheet1"
TARGET_COL = "十天随访结果"

FEATURE_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
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

CONTINUOUS_COLUMNS = [
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

FEATURE_METHOD = 0  # 0: full feature set; 1: RF importance; 2: Logistic L1; 3: univariable screening
TOP_N = 20
CORR_THRESHOLD = 0.8
CV_SPLITS = 5
CV_RANDOM_STATE = 577

MODELS = [
    ("LogisticRegression", LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced")),
    ("SVM_linear", SVC(kernel="linear", probability=True, random_state=42, class_weight="balanced")),
    ("MLP", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)),
    ("GaussianNB", GaussianNB()),
    (
        "XGBoost",
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
        ),
    ),
    (
        "LightGBM",
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    ),
]

MODEL_COLOR_MAP = {
    "LogisticRegression": "#4F81C7",
    "SVM_linear": "#F2A65A",
    "MLP": "#9E8CC2",
    "GaussianNB": "#6FAE8E",
    "XGBoost": "#D1615D",
    "LightGBM": "#5A6F9A",
}

ROC_DIR = Path("roc_plots")
SHAP_DIR = Path("shap_plots")
ROC_DIR.mkdir(exist_ok=True)
SHAP_DIR.mkdir(exist_ok=True)


# ============================================================
# Data I/O
# ============================================================
def read_data(file_path: str, preferred_sheet: str, fallback_sheet: str) -> pd.DataFrame:
    """Read the preferred sheet if present; otherwise fall back to Sheet1 or the first sheet."""
    xls = pd.ExcelFile(file_path)
    if preferred_sheet in xls.sheet_names:
        return pd.read_excel(file_path, sheet_name=preferred_sheet)
    if fallback_sheet in xls.sheet_names:
        return pd.read_excel(file_path, sheet_name=fallback_sheet)
    return pd.read_excel(file_path, sheet_name=xls.sheet_names[0])


# ============================================================
# Feature selection utilities
# ============================================================
def select_features(
    X_train_proc: pd.DataFrame,
    y_train_fold: pd.Series,
    X_test_proc: pd.DataFrame,
    method: int,
    top_n: int,
    num_feature_names: list[str],
    cat_feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply feature selection within the training fold only."""
    if method == 0:
        return X_train_proc, X_test_proc

    if method == 1:
        rf_fs = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf_fs.fit(X_train_proc, y_train_fold)
        idx_rf = rf_fs.feature_importances_.argsort()[-top_n:][::-1]
        selected_cols = X_train_proc.columns[idx_rf]
        return X_train_proc[selected_cols], X_test_proc[selected_cols]

    if method == 2:
        lasso_logit = LogisticRegressionCV(
            Cs=10,
            cv=5,
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        lasso_logit.fit(X_train_proc, y_train_fold)
        coef = lasso_logit.coef_.ravel()
        selected_idx = np.where(coef != 0)[0]
        if len(selected_idx) == 0:
            selected_idx = np.argsort(np.abs(coef))[-top_n:]
        selected_cols = X_train_proc.columns[selected_idx]
        return X_train_proc[selected_cols], X_test_proc[selected_cols]

    if method == 3:
        f_vals, f_pvals = f_classif(X_train_proc[num_feature_names], y_train_fold)
        f_df = pd.DataFrame(
            {"feature": num_feature_names, "test": "F-test", "score": f_vals, "p_value": f_pvals}
        )

        chi2_vals, chi2_pvals = chi2(X_train_proc[cat_feature_names], y_train_fold)
        chi2_df = pd.DataFrame(
            {"feature": cat_feature_names, "test": "chi-square", "score": chi2_vals, "p_value": chi2_pvals}
        )

        all_results = pd.concat([f_df, chi2_df], ignore_index=True)
        selected_cols = all_results.sort_values("p_value").head(top_n)["feature"].tolist()
        return X_train_proc[selected_cols], X_test_proc[selected_cols]

    raise ValueError(f"Unknown feature selection method: {method}")


# ============================================================
# Collinearity filtering
# ============================================================
def drop_constant_cols(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Drop zero-variance columns identified in the training fold."""
    nunique = X_train.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if not const_cols:
        return X_train, X_test, []
    return X_train.drop(columns=const_cols), X_test.drop(columns=const_cols), const_cols


def drop_high_corr_features(X_train: pd.DataFrame, X_test: pd.DataFrame, corr_threshold: float = 0.8):
    """Drop one feature from each pair of highly correlated training-fold features."""
    if X_train.shape[1] <= 1:
        return X_train, X_test, []

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
    if not to_drop:
        return X_train, X_test, []
    return X_train.drop(columns=to_drop), X_test.drop(columns=to_drop), to_drop


# ============================================================
# Training and evaluation
# ============================================================
def train_and_evaluate(name: str, model, X_train, y_train, X_test, y_test):
    """Fit one model and return metrics plus predicted probabilities."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    return metrics, y_pred, y_proba


def run_cross_validation(X: pd.DataFrame, y: pd.Series):
    """Run stratified K-fold cross-validation and keep the best fold for each model."""
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    results = []
    best_by = "auc"
    best_roc = {}

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== Fold {fold_id} ==========")

        X_train_raw = X.iloc[train_idx].copy()
        y_train_raw = y.iloc[train_idx].copy()
        X_test_raw = X.iloc[test_idx].copy()
        y_test_fold = y.iloc[test_idx].copy()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), CONTINUOUS_COLUMNS),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False), CATEGORICAL_COLUMNS),
            ]
        )

        X_train_arr = preprocessor.fit_transform(X_train_raw)
        X_test_arr = preprocessor.transform(X_test_raw)

        num_feature_names = list(CONTINUOUS_COLUMNS)
        cat_feature_names = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLUMNS)
        )
        feature_names = num_feature_names + cat_feature_names

        X_train_proc = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test_proc = pd.DataFrame(X_test_arr, columns=feature_names)

        X_train_sel, X_test_sel = select_features(
            X_train_proc,
            y_train_raw,
            X_test_proc,
            method=FEATURE_METHOD,
            top_n=TOP_N,
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
        )
        print(f"Selected feature shape: {X_train_sel.shape}")

        X_train_sel, X_test_sel, const_dropped = drop_constant_cols(X_train_sel, X_test_sel)
        X_train_sel, X_test_sel, corr_dropped = drop_high_corr_features(
            X_train_sel, X_test_sel, corr_threshold=CORR_THRESHOLD
        )
        if const_dropped or corr_dropped:
            print(
                f"Removed {len(const_dropped)} constant columns and {len(corr_dropped)} highly correlated columns"
            )
        print(f"Post-filter feature shape: {X_train_sel.shape}")

        for name, model in MODELS:
            metrics_dict, y_pred, y_proba = train_and_evaluate(
                name,
                model,
                X_train_sel,
                y_train_raw,
                X_test_sel,
                y_test_fold,
            )
            metrics_dict["fold"] = fold_id
            results.append(metrics_dict)

            current_score = metrics_dict[best_by]
            if (name not in best_roc) or (current_score > best_roc[name][best_by]):
                best_roc[name] = {
                    best_by: current_score,
                    "fold": fold_id,
                    "y_test": np.asarray(y_test_fold),
                    "y_pred": np.asarray(y_pred),
                    "y_proba": np.asarray(y_proba),
                }

    results_df = pd.DataFrame(results)
    summary = results_df.groupby("model")[["accuracy", "auc", "f1", "precision", "recall"]].agg(["mean", "std"])
    summary.to_csv("cv0_summary_metrics.csv", encoding="utf-8-sig")
    return results_df, summary, best_roc


# ============================================================
# ROC and confusion matrix plotting
# ============================================================
def plot_grouped_roc(best_roc: dict):
    """Save grouped ROC curves with three models per panel."""
    model_names = [name for name, _ in MODELS]
    group_size = 3
    groups = [model_names[i : i + group_size] for i in range(0, len(model_names), group_size)]

    for group_id, group in enumerate(groups, start=1):
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=500)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.5, color="gray")

        for model_name in group:
            info = best_roc.get(model_name)
            if info is None:
                continue
            fpr, tpr, _ = roc_curve(info["y_test"], info["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                linewidth=3.0,
                color=MODEL_COLOR_MAP.get(model_name),
                label=f"{model_name} (AUC={roc_auc:.3f})",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.set_xlabel("FPR", fontsize=25)
        ax.set_ylabel("TPR", fontsize=25)
        ax.legend(frameon=True, fontsize=18, loc="lower right")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"ROC_group_{group_id}.pdf", dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)


def plot_best_model_roc_and_cm(best_roc: dict):
    """Save ROC curve and confusion matrix for the model with the highest best-fold AUC."""
    best_model_name = None
    best_auc = -1
    best_info = None

    for model_name, info in best_roc.items():
        fpr, tpr, _ = roc_curve(info["y_test"], info["y_proba"])
        roc_auc = auc(fpr, tpr)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = model_name
            best_info = info

    if best_info is None:
        raise RuntimeError("No best model found.")

    print(f"Best model by ROC AUC: {best_model_name} (AUC = {best_auc:.4f})")

    fpr, tpr, _ = roc_curve(best_info["y_test"], best_info["y_proba"])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.8, 6.2), dpi=500)
    ax.plot(
        fpr,
        tpr,
        linewidth=3.2,
        color=MODEL_COLOR_MAP.get(best_model_name, "#2F4B7C"),
        label=f"{best_model_name} (AUC = {roc_auc:.3f})",
    )
    ax.fill_between(fpr, tpr, alpha=0.20, color=MODEL_COLOR_MAP.get(best_model_name, "#2F4B7C"))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=24)
    ax.set_ylabel("True Positive Rate", fontsize=24)
    ax.tick_params(axis="both", labelsize=22)
    ax.legend(loc="lower right", fontsize=18, frameon=True)
    ax.grid(True, linewidth=0.6, alpha=0.4)
    plt.tight_layout()
    plt.savefig(ROC_DIR / f"ROC_best_{best_model_name}.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)

    y_pred_best = (best_info["y_proba"] >= 0.5).astype(int)
    cm = confusion_matrix(best_info["y_test"], y_pred_best)
    acc = accuracy_score(best_info["y_test"], y_pred_best)
    print(f"Best model accuracy at threshold 0.5: {acc:.4f}")

    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=500)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, annot_kws={"size": 22}, ax=ax)
    ax.set_xlabel("Predicted label", fontsize=22)
    ax.set_ylabel("True label", fontsize=22)
    ax.set_title(f"Confusion Matrix ({best_model_name})", fontsize=24)
    plt.tight_layout()
    plt.savefig(ROC_DIR / f"ConfusionMatrix_best_{best_model_name}.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)


# ============================================================
# SHAP plotting utilities
# ============================================================
def wrap_labels(cols: list[str], width: int = 22) -> list[str]:
    return ["\n".join(textwrap.wrap(str(c), width=width)) for c in cols]


def prepare_shap_dataset(X: pd.DataFrame, y: pd.Series):
    """Preprocess the full dataset using the same pipeline as cross-validation."""
    preprocessor_all = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False), CATEGORICAL_COLUMNS),
        ]
    )
    X_all_arr = preprocessor_all.fit_transform(X)

    num_names = list(CONTINUOUS_COLUMNS)
    cat_names = list(preprocessor_all.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLUMNS))
    all_feature_names = num_names + cat_names
    X_all_proc = pd.DataFrame(X_all_arr, columns=all_feature_names)

    X_sel, _ = select_features(
        X_all_proc,
        y,
        X_all_proc,
        method=FEATURE_METHOD,
        top_n=TOP_N,
        num_feature_names=num_names,
        cat_feature_names=cat_names,
    )
    X_sel, _, _ = drop_constant_cols(X_sel, X_sel)
    X_sel, _, _ = drop_high_corr_features(X_sel, X_sel, corr_threshold=CORR_THRESHOLD)
    return X_sel


def compute_shap_outputs(X: pd.DataFrame, y: pd.Series):
    """Train LightGBM on the full processed dataset and compute SHAP outputs."""
    X_sel = prepare_shap_dataset(X, y)
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    lgbm.fit(X_sel, y)

    n_sample = min(1000, len(X_sel))
    X_shap_sample = X_sel.sample(n_sample, random_state=42)
    explainer = shap.TreeExplainer(lgbm)
    shap_vals = explainer.shap_values(X_shap_sample)
    shap_vals_to_plot = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) == 2 else shap_vals
    return X_sel, X_shap_sample, shap_vals_to_plot


def plot_shap_summary_basic(X_shap_sample: pd.DataFrame, shap_vals_to_plot: np.ndarray):
    """Save the standard SHAP summary plot."""
    X_shap_plot = X_shap_sample.copy()
    X_shap_plot.columns = wrap_labels(X_shap_plot.columns, width=22)

    plt.figure(figsize=(12, 7), dpi=150)
    shap.summary_plot(shap_vals_to_plot, X_shap_plot, show=False, max_display=15)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.45)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_lightgbm.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_summary_with_topbar(X_shap_sample: pd.DataFrame, shap_vals_to_plot: np.ndarray):
    """Save a combined SHAP beeswarm plot with top mean-|SHAP| bar overlay."""
    custom_cmap = LinearSegmentedColormap.from_list("blue_red_custom", ["#4DA3FF", "#FF0051"])

    X_shap_plot = X_shap_sample.copy()
    X_shap_plot.columns = wrap_labels(X_shap_plot.columns, width=22)
    display_n = 15

    plt.figure(figsize=(12, 7), dpi=500)
    shap.summary_plot(
        shap_vals_to_plot,
        X_shap_plot,
        show=False,
        max_display=display_n,
        cmap=custom_cmap,
    )

    fig = plt.gcf()
    ax = plt.gca()
    mean_abs = np.abs(shap_vals_to_plot).mean(axis=0)
    wrapped_cols = list(X_shap_plot.columns)
    mean_abs_series = pd.Series(mean_abs, index=wrapped_cols)

    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    yticks = ax.get_yticks()
    bar_vals = np.array([mean_abs_series.get(lbl, np.nan) for lbl in yticklabels], dtype=float)
    valid = np.isfinite(bar_vals)
    yticks_v = np.array(yticks)[valid]
    bar_vals_v = bar_vals[valid]

    ax_top = ax.twiny()
    ax_top.barh(yticks_v, bar_vals_v, height=0.78, alpha=0.20, zorder=0)
    ax_top.tick_params(axis="x", labelsize=20)
    for collection in ax.collections:
        collection.set_zorder(3)

    ax_top.set_yticks(ax.get_yticks())
    ax_top.set_yticklabels([])
    ax_top.grid(False)
    fig.subplots_adjust(left=0.45)

    ax.set_xlim(-1.5, 3.0)
    bottom_ticks = [-1.5, -0.75, 0, 0.75, 1.5, 2.0, 2.5, 3.0]
    ax.set_xticks(bottom_ticks)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)

    top_min, top_max = 0.0, 0.30
    ax_top.set_xlim(top_min, top_max)
    bottom_min, bottom_max = ax.get_xlim()
    bottom_ticks_np = np.array(bottom_ticks, dtype=float)
    top_ticks = top_min + (bottom_ticks_np - bottom_min) / (bottom_max - bottom_min) * (top_max - top_min)
    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels([f"{v:.2f}" for v in top_ticks])
    ax_top.set_xlabel("Mean |Shapley Value| (Feature Importance)", fontsize=11)

    ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.50)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.50)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        ax_top.spines[spine].set_visible(False)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", width=0.8, labelsize=20)
    ax_top.tick_params(axis="x", width=0.8)

    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_lgbm_dot_topbar.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.close()


def plot_top_shap_dependence(X_shap_sample: pd.DataFrame, shap_vals_to_plot: np.ndarray):
    """Save one standard dependence plot for the top-ranked feature."""
    top_feat_idx = int(np.argmax(np.abs(shap_vals_to_plot).mean(axis=0)))
    top_feat = X_shap_sample.columns[top_feat_idx]

    plt.figure(figsize=(9, 6), dpi=150)
    shap.dependence_plot(top_feat, shap_vals_to_plot, X_shap_sample, show=False)
    plt.tight_layout()
    safe_name = top_feat.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(SHAP_DIR / f"shap_dependence_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def compute_threshold_by_curvature(x: np.ndarray, y: np.ndarray, frac: float = 0.25):
    """Estimate an empirical turning point based on smoothed curvature."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return np.nan

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    smoothed = lowess(ys, xs, frac=frac, return_sorted=True)

    x_s = smoothed[:, 0]
    y_s = smoothed[:, 1]
    lo, hi = np.quantile(x_s, [0.05, 0.95])
    mid = (x_s >= lo) & (x_s <= hi)

    dy = np.gradient(y_s, x_s)
    d2y = np.gradient(dy, x_s)
    curvature = np.abs(d2y)
    idx = np.where(mid)[0]
    if len(idx) == 0:
        return np.nan
    return float(x_s[idx[np.argmax(curvature[idx])]])


def plot_temperature_grouped_dependence(
    X_all_bal: pd.DataFrame,
    X_shap_sample: pd.DataFrame,
    shap_vals_to_plot: np.ndarray,
):
    """Save 2x2 and single-panel dependence plots stratified by daily temperature."""
    axis_label_size = 20
    tick_label_size = 20
    legend_font_size = 15
    title_size = 14

    main_vars = [
        "gestational days",
        "neutrophil percentage",
        "maximum amniotic fluid depth",
        "BMI",
    ]
    interaction_var = "daily temperature"
    temp_colors = {
        "Low temperature": "#2F6DAE",
        "Medium temperature": "#B0B7C3",
        "High temperature": "#E0455F",
    }

    def get_shap_for_feature(feature_name: str):
        idx = list(X_shap_sample.columns).index(feature_name)
        return shap_vals_to_plot[:, idx].astype(float)

    def get_raw_x(feature_name: str):
        return X_all_bal.loc[X_shap_sample.index, feature_name].to_numpy(dtype=float)

    temp_raw = get_raw_x(interaction_var)
    temp_group = pd.qcut(temp_raw, q=3, labels=["Low temperature", "Medium temperature", "High temperature"])

    def plot_dependence(ax, feature_name: str):
        x = get_raw_x(feature_name)
        y = get_shap_for_feature(feature_name)
        med = np.nanmedian(x)
        thr = compute_threshold_by_curvature(x, y)

        for grp in ["Low temperature", "Medium temperature", "High temperature"]:
            mask = (temp_group == grp) & np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue

            ax.scatter(x[mask], y[mask], s=18, alpha=0.7, color=temp_colors[grp], edgecolors="none", label=grp)
            smoothed = lowess(y[mask], x[mask], frac=0.8, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=temp_colors[grp], linewidth=1.8)

        ax.axvline(med, color="black", linestyle="--", linewidth=1.5, label=f"Median: {med:.2f}")
        if np.isfinite(thr):
            ax.axvline(thr, color="#666666", linestyle=":", linewidth=1.8, label=f"Threshold: {thr:.2f}")

        ax.set_title(feature_name, fontsize=title_size)
        ax.set_xlabel(f"{feature_name} (original units)", fontsize=axis_label_size)
        ax.set_ylabel("SHAP value", fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        ax.grid(False)
        ax.legend(fontsize=legend_font_size, frameon=True, loc="upper left")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5), dpi=350)
    axes = axes.ravel()
    for i, var_name in enumerate(main_vars):
        plot_dependence(axes[i], var_name)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_dependence_4vars_tempGrouped.png", dpi=350, bbox_inches="tight")
    plt.savefig(SHAP_DIR / "shap_dependence_4vars_tempGrouped.pdf", dpi=350, bbox_inches="tight")
    plt.close(fig)

    for var_name in main_vars:
        fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=350)
        plot_dependence(ax, var_name)
        plt.tight_layout()
        safe_name = var_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(SHAP_DIR / f"shap_dependence_{safe_name}_tempGrouped.png", dpi=350, bbox_inches="tight")
        plt.savefig(SHAP_DIR / f"shap_dependence_{safe_name}_tempGrouped.pdf", dpi=350, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Main execution
# ============================================================
def main():
    df = read_data(DATA_FILE, PREFERRED_SHEET, FALLBACK_SHEET)
    missing_cols = [c for c in (FEATURE_COLUMNS + [TARGET_COL]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in the dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].copy()

    _, summary, best_roc = run_cross_validation(X, y)
    print("\nCross-validation summary:")
    print(summary)

    plot_grouped_roc(best_roc)
    plot_best_model_roc_and_cm(best_roc)

    if not SHAP_AVAILABLE:
        print("\nSHAP is not available in the current environment. Skipping SHAP-related analyses.")
        return

    try:
        print("\n=========== Running LightGBM SHAP analysis ===========")
        X_all_bal = X.copy()
        y_all_bal = y.copy()
        X_sel, X_shap_sample, shap_vals_to_plot = compute_shap_outputs(X_all_bal, y_all_bal)
        print(f"Prepared SHAP dataset with shape: {X_sel.shape}")

        plot_shap_summary_basic(X_shap_sample, shap_vals_to_plot)
        plot_top_shap_dependence(X_shap_sample, shap_vals_to_plot)
        plot_shap_summary_with_topbar(X_shap_sample, shap_vals_to_plot)
        plot_temperature_grouped_dependence(X_all_bal, X_shap_sample, shap_vals_to_plot)
        print("Saved SHAP plots to the 'shap_plots' directory.")
    except Exception as exc:
        print(f"SHAP analysis failed, but the main modeling results are unaffected: {exc!r}")


if __name__ == "__main__":
    main()
