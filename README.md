# Amniocentesis Short-Term Adverse Outcome Prediction

This repository contains the analysis code for the study:

**Machine learning–based risk prediction of short-term adverse outcomes following amniocentesis: a retrospective cohort study**

## Analysis

Main script:

`AC_model.py`

The analysis includes:

- 80/20 stratified development and internal test split
- Five-fold stratified cross-validation
- Comparison of Logistic Regression, SVM, MLP, GaussianNB, XGBoost, and LightGBM
- Feature selection using Random Forest, LASSO, univariable screening, or the full feature set
- Internal held-out validation
- Temporal validation
- Bootstrap 95% confidence intervals
- Calibration analysis and Brier score
- Decision curve analysis
- SHAP model interpretation
- Restricted cubic spline analysis

## Data

Expected input files:

```text
Dec2023_Dec2025_data.xlsx
Jan2026_Jun2026_data.xlsx
```

Worksheet:

Eng_revise

Outcome variable:

outcomes

## Requirements

Python 3.9 with:

```
numpy
pandas
scikit-learn
lightgbm
xgboost
shap
statsmodels
scipy
patsy
joblib
```

## Run

Place the input data files in the same directory as the script and run:

```Python
python AC_model.py
```

Results are saved to:

```
model_results/
```

## Key settings

```
Random seed: 577
Five-fold cross-validation
Classification threshold: 0.50
Bootstrap resamples: 2000
Calibration groups: 10
DCA threshold range: 0.05–0.50
```