from pathlib import Path
import shutil

source = Path(__file__).resolve().parents[1] / "Task1_ML_Classification" / "model" / "spam_classifier.joblib"
destination_dir = Path(__file__).resolve().parent / "model"
destination_dir.mkdir(exist_ok=True)
destination = destination_dir / "spam_classifier.joblib"

if not source.exists():
    raise FileNotFoundError(
        f"Trained Task 1 model not found: {source}\n"
        "Run Task1_ML_Classification/train.py first."
    )
shutil.copy2(source, destination)
print(f"Copied trained pipeline to: {destination}")
