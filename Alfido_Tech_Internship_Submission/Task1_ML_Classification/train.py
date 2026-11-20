import io, zipfile, urllib.request, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "model"
for folder in (DATA_DIR, OUT_DIR, MODEL_DIR):
    folder.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_data():
    if not DATA_FILE.exists():
        print("Downloading UCI SMS Spam Collection...")
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            member = next((n for n in zf.namelist() if n.endswith("SMSSpamCollection")), None)
            if member is None:
                raise FileNotFoundError("SMSSpamCollection was not found inside the downloaded ZIP.")
            DATA_FILE.write_bytes(zf.read(member))
    df = pd.read_csv(DATA_FILE, sep="\t", names=["label", "message"], encoding="utf-8")
    df = df.dropna().drop_duplicates().copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

def main():
    df = load_data()
    print("Dataset shape after cleaning:", df.shape)
    print(df["label"].value_counts().rename(index={0:"ham", 1:"spam"}))

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=2, max_features=30000)),
            ("classifier", RandomForestClassifier(
                n_estimators=250, class_weight="balanced_subsample",
                random_state=42, n_jobs=-1
            ))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall", "f1":"f1", "roc_auc":"roc_auc"}
    rows, fitted = [], {}
    for name, estimator in models.items():
        print(f"\nCross-validating {name}...")
        cv_result = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        prob = estimator.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
            "cv_f1_mean": float(np.mean(cv_result["test_f1"])),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob))
        }
        rows.append(metrics)
        fitted[name] = estimator
        print(metrics)

        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not spam", "Spam"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close()

        RocCurveDisplay.from_predictions(y_test, prob)
        plt.title(f"ROC Curve – {name}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"roc_curve_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close()

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    (OUT_DIR / "model_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = results.set_index("model")[metric_cols].plot(kind="bar", figsize=(11,6), ylim=(0,1))
    ax.set_title("Test-set model comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Select best model by held-out F1 score; save complete TF-IDF + classifier pipeline.
    best_row = results.sort_values("f1", ascending=False).iloc[0]
    best_name = best_row["model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "spam_classifier.joblib")
    (MODEL_DIR / "selected_model.txt").write_text(best_name, encoding="utf-8")

    print("\nSelected model:", best_name)
    print("Saved model:", MODEL_DIR / "spam_classifier.joblib")
    print("Metrics and plots saved in outputs/")

if __name__ == "__main__":
    main()
