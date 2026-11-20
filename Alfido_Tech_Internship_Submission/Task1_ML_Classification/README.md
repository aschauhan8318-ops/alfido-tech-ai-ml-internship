# Task 1 – ML Classification: SMS Spam Detection

## Goal
Train and compare Logistic Regression and Random Forest classifiers to identify spam messages.

## Dataset
UCI SMS Spam Collection. The script downloads the dataset automatically from the UCI Machine Learning Repository on first run.

## Setup (Windows PowerShell)
```powershell
cd Task1_ML_Classification
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Notebook
Open `notebooks/ML_Classification.ipynb` in Jupyter or VS Code and run the cells. The notebook calls the reproducible training script.

## Outputs
- `model/spam_classifier.joblib`: complete TF-IDF + selected classifier pipeline
- `outputs/model_metrics.csv` and `.json`
- confusion matrix and ROC curve for both models
- `outputs/model_comparison.png`

## Metrics
Accuracy, precision, recall, F1-score and ROC-AUC are calculated on a stratified 20% held-out test set. Five-fold stratified cross-validation is performed on the training split.

## Reproducibility
Random seed is 42. The selected model is chosen by held-out test F1; for a strict final scientific comparison, select the model using cross-validation only and reserve the test set for one final evaluation.
