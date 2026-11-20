# Task 4 – Responsible AI and Model Interpretability

## Goal
Explain the Task 1 spam classifier and perform a limited fairness/error analysis.

## Prerequisite
Run Task 1 first. Task 4 loads its saved model and recreates the same stratified 80/20 split (random_state=42).

## Setup
```powershell
cd Task4_Responsible_AI
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python analyze.py
```

## Outputs
- `outputs/feature_importance.png` – Logistic Regression signed coefficients
- `outputs/shap_summary.png` – SHAP summary plot
- `outputs/shap_local_example.png` – local SHAP waterfall explanation
- `outputs/message_length_error_rates.csv` – subgroup error rates on held-out test data
- `outputs/fairness_proxy_analysis.png` – false positive / false negative rates by message length
- `outputs/heldout_confusion_matrix.png`

## Fairness limitation
The SMS Spam Collection dataset does not provide sensitive demographic attributes such as age, gender, race or religion. Therefore, this project cannot claim to have measured demographic fairness. Message length is used only as a non-sensitive proxy subgroup for exploratory error analysis. A proper demographic audit requires appropriate, consented and representative group labels, privacy safeguards and legal/ethical review.

## Mitigation recommendations
- Expand and rebalance training data where evidence shows under-representation.
- Inspect false positives and false negatives for each subgroup.
- Review vocabulary and data collection practices for unwanted proxy effects.
- Tune thresholds only with documented precision/recall and group-impact analysis.
- Monitor drift and subgroup performance after deployment.
- Minimize retention of message content and remove personal data where possible.
