import io, zipfile, urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = PROJECT_ROOT / "Task1_ML_Classification" / "dataset"
MODEL_PATH = PROJECT_ROOT / "Task1_ML_Classification" / "model" / "spam_classifier.joblib"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def load_data():
    path = DATA_DIR / "SMSSpamCollection"
    if not path.exists():
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member = next(n for n in zf.namelist() if n.endswith("SMSSpamCollection"))
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(zf.read(member))
    df = pd.read_csv(path, sep="\t", names=["label", "message"]).dropna().drop_duplicates()
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    return df

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run Task 1 training first to create the saved spam model.")
    model = joblib.load(MODEL_PATH)
    df = load_data()
    _, X_test, _, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    predictions = model.predict(X_test)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    # Global feature interpretation using signed Logistic Regression coefficients.
    if classifier.__class__.__name__ == "LogisticRegression":
        names = vectorizer.get_feature_names_out()
        coef = classifier.coef_[0]
        feature_df = pd.DataFrame({"feature": names, "coefficient": coef})
        selected = pd.concat([
            feature_df.nlargest(15, "coefficient"),
            feature_df.nsmallest(15, "coefficient")
        ]).sort_values("coefficient")
        plt.figure(figsize=(10, 9))
        sns.barplot(data=selected, x="coefficient", y="feature")
        plt.title("Words most associated with each class")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "feature_importance.png", dpi=160)
        plt.close()
    else:
        print("The selected model is not Logistic Regression; coefficient feature plot skipped.")

    # Local/global SHAP explanation for the Logistic Regression pipeline's linear classifier.
    if classifier.__class__.__name__ == "LogisticRegression":
        sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
        transformed = vectorizer.transform(sample)
        explainer = shap.LinearExplainer(classifier, transformed)
        shap_values = explainer(transformed)
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()

        # One local explanation waterfall plot.
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_local_example.png", dpi=160, bbox_inches="tight")
        plt.close()

    # Fairness proxy audit: compare error rates by message-length buckets on held-out test data.
    audit = pd.DataFrame({"message": X_test, "actual": y_test, "predicted": predictions})
    audit["message_length"] = audit["message"].str.len()
    audit["length_group"] = pd.cut(
        audit["message_length"], bins=[0, 50, 150, np.inf],
        labels=["Short", "Medium", "Long"], include_lowest=True
    )
    audit["fp"] = ((audit["actual"] == 0) & (audit["predicted"] == 1)).astype(int)
    audit["fn"] = ((audit["actual"] == 1) & (audit["predicted"] == 0)).astype(int)

    rows = []
    for group, subset in audit.groupby("length_group", observed=True):
        negatives = int((subset["actual"] == 0).sum())
        positives = int((subset["actual"] == 1).sum())
        rows.append({
            "message_length_group": str(group),
            "samples": int(len(subset)),
            "false_positive_rate": float(subset["fp"].sum() / negatives) if negatives else None,
            "false_negative_rate": float(subset["fn"].sum() / positives) if positives else None
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "message_length_error_rates.csv", index=False)

    if not results.empty:
        results.set_index("message_length_group")[
            ["false_positive_rate", "false_negative_rate"]
        ].plot(kind="bar", figsize=(9,5))
        plt.title("Error rates by message-length group (held-out test set)")
        plt.ylabel("Rate")
        plt.xlabel("Message length group")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "fairness_proxy_analysis.png", dpi=160)
        plt.close()

    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm, display_labels=["Not spam", "Spam"]).plot(cmap="Blues")
    plt.title("Held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heldout_confusion_matrix.png", dpi=160)
    plt.close()

    print("Message-length error rate analysis:")
    print(results.to_string(index=False))
    print("\nDataset does not contain demographic attributes such as age or gender.")
    print("Message length is a proxy subgroup, not a demographic fairness assessment.")
    print("Outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
