import json
import re
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent
model = tf.keras.models.load_model(ROOT / "models" / "sentiment_model.keras")
word_index = json.loads((ROOT / "models" / "imdb_word_index.json").read_text(encoding="utf-8"))

VOCAB_SIZE = 10000
MAX_LENGTH = 250

def encode_review(text):
    tokens = re.findall(r"[a-z']+", text.lower())
    # IMDB's integer encoding reserves 0 for padding, 1 for start, 2 for OOV,
    # and shifts ordinary word indices by 3.
    sequence = [1]
    for token in tokens:
        idx = word_index.get(token)
        sequence.append((idx + 3) if idx is not None and idx + 3 < VOCAB_SIZE else 2)
    return pad_sequences([sequence], maxlen=MAX_LENGTH, padding="post", truncating="post")

def predict(text):
    probability = float(model.predict(encode_review(text), verbose=0)[0][0])
    label = "Positive" if probability >= 0.5 else "Negative"
    return {"sentiment": label, "positive_probability": round(probability, 4)}

if __name__ == "__main__":
    sample = input("Enter a movie review: ")
    print(predict(sample))
