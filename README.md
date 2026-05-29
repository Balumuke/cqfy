# Machine Learning–Based Risk Prediction of Post-Amniocentesis Complications

This repository contains a Python script for developing and evaluating machine learning models to predict post-amniocentesis complications or discomforts based on clinical, obstetric, laboratory, ultrasound, and environmental variables.

The workflow includes data preprocessing, feature selection, model comparison, cross-validation, ROC curve visualization, confusion matrix plotting, and SHAP-based model interpretation.

## Project Overview

Amniocentesis is an invasive prenatal diagnostic procedure. Although it is generally considered safe, some patients may experience post-procedural discomforts or complications. This project aims to build machine learning models for individualized risk prediction using real-world clinical data.

The target outcome is the 10-day follow-up result after amniocentesis.

## Main Features

The script performs the following analyses:

* Reads clinical data from an Excel file
* Uses predefined clinical, obstetric, laboratory, ultrasound, and environmental predictors
* Applies numerical standardization and categorical one-hot encoding
* Supports multiple feature-selection strategies
* Compares several machine learning classifiers
* Uses stratified 5-fold cross-validation
* Reports model performance using:

  * Accuracy
  * AUC
  * F1 score
  * Precision
  * Recall
* Generates grouped ROC curves
* Generates the ROC curve and confusion matrix for the best-performing model
* Performs SHAP analysis for LightGBM model interpretation
* Produces SHAP summary plots and SHAP dependence plots

## Models Included

The following machine learning models are implemented:

* Logistic Regression
* Linear Support Vector Machine
* Multilayer Perceptron
* Gaussian Naive Bayes
* XGBoost
* LightGBM

## Data Requirements

The script expects an Excel file named:

```text
extracted_data.xlsx
```

By default, the script first tries to read the sheet:

```text
Eng_revise
```

If this sheet is not found, it falls back to:

```text
Sheet1
```

The target column should be:

```text
十天随访结果
```

The input features include maternal and paternal demographic variables, obstetric history, blood pressure, BMI, ultrasound measurements, laboratory test results, amniocentesis indication, and daily temperature.

## Required Python Packages

Install the required packages before running the script:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels lightgbm xgboost shap openpyxl
```

## How to Run

Place the following files in the same directory:

```text
AC_model.py
extracted_data.xlsx
```

Then run:

```bash
python AC_model.py
```

If your script filename is different, for example:

```text
AC_model(2).py
```

you can run:

```bash
python "AC_model(2).py"
```

## Configuration

Key parameters can be modified at the beginning of the script:

```python
DATA_FILE = "extracted_data.xlsx"
PREFERRED_SHEET = "Eng_revise"
FALLBACK_SHEET = "Sheet1"
TARGET_COL = "十天随访结果"
FEATURE_METHOD = 0
TOP_N = 20
CORR_THRESHOLD = 0.8
CV_SPLITS = 5
CV_RANDOM_STATE = 577
```

The feature selection method is controlled by:

```python
FEATURE_METHOD
```

Available options are:

```text
0 = full feature set
1 = Random Forest importance
2 = Logistic L1 selection
3 = univariable screening
```

## Output Files

After running the script, the following outputs will be generated.

### Cross-validation summary

```text
cv0_summary_metrics.csv
```

This file contains the mean and standard deviation of model performance metrics across the 5-fold cross-validation.

### ROC plots

ROC curves are saved in:

```text
roc_plots/
```

Main outputs include:

```text
ROC_group_1.pdf
ROC_group_2.pdf
ROC_best_<model_name>.pdf
ConfusionMatrix_best_<model_name>.pdf
```

### SHAP plots

If the `shap` package is available, SHAP-related figures are saved in:

```text
shap_plots/
```

Main outputs include:

```text
shap_summary_lightgbm.png
shap_summary_lgbm_dot_topbar.pdf
shap_dependence_<feature_name>.png
shap_dependence_4vars_tempGrouped.png
shap_dependence_4vars_tempGrouped.pdf
```

## SHAP Interpretation

The script trains a LightGBM model on the processed dataset and calculates SHAP values to evaluate feature contributions. It generates:

* Standard SHAP summary plot
* SHAP summary plot with mean absolute SHAP value bar overlay
* SHAP dependence plot for the top-ranked feature
* Temperature-stratified SHAP dependence plots for selected clinical predictors

These plots are designed to support model interpretability and help identify important predictors and potential nonlinear risk patterns.

## Notes

This repository only contains the modeling script. The clinical dataset is not included due to privacy and ethical considerations.

Please do not upload patient-level clinical data, identifiable information, or raw hospital records to a public GitHub repository.

For compatibility with newer versions of scikit-learn, the following line may need to be updated:

```python
OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)
```

to:

```python
OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
```

## Suggested `.gitignore`

It is recommended to add a `.gitignore` file to avoid uploading sensitive or unnecessary files:

```text
# Data files
*.xlsx
*.xls
*.csv

# Output folders
roc_plots/
shap_plots/

# Python cache
__pycache__/
*.pyc

# Jupyter checkpoints
.ipynb_checkpoints/

# System files
.DS_Store
Thumbs.db
```

## Citation

If this code is used in a manuscript, please cite the corresponding study or project:

```text
Machine learning–based risk prediction of post-amniocentesis complications: a retrospective cohort study
```
