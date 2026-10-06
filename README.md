# Ridge and Lasso on the Wisconsin Breast Cancer Dataset

An academic experiment comparing regularized linear regression models on scikit-learn's binary breast-cancer dataset, with thresholded predictions and coefficient visualizations.

## Method

[breast-cancer-prediction.py](breast-cancer-prediction.py) loads the bundled scikit-learn dataset, creates an 80/20 split with seed 42, and standardizes features using the training subset.

It searches regularization strengths with 5-fold GridSearchCV, using negative mean squared error as the selection criterion. Ridge and Lasso produce continuous outputs; values at or above 0.5 are converted to class 1.

The script prints accuracy, precision, recall, F1, MSE, and the features retained by Lasso. Here class 1 means **benign**, so the printed binary precision/recall/F1 describe that class.

## Run

With Python 3.12:

```bash
git clone https://github.com/minhiungan2608/breast_cancer_prdmodel.git
cd breast_cancer_prdmodel
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python breast-cancer-prediction.py
```

On Windows, activate with `.venv\Scripts\activate`. No external CSV or API key is required.

The script writes `ridge_lasso_weights.png` and `lasso_predictions.png` to the working directory. It has been smoke-tested with the bundled dataset; no fixed performance claim is made here.

## Evaluation limits

These are regression models applied to binary labels, not dedicated classification estimators. Standardization occurs before cross-validation, so validation folds influence the scaler within the training subset. Moving preprocessing into a Pipeline and adding a logistic-regression baseline would provide a cleaner comparison.

**Stack:** Python · NumPy · scikit-learn · Matplotlib.

This repository is coursework demonstrating model comparison and feature analysis. No repository-wide software license has been specified.
