import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def main():
    tf.keras.utils.set_random_seed(42)
    vocab_size = 10000
    max_length = 250

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    x_train = pad_sequences(x_train, maxlen=max_length, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=max_length, padding="post", truncating="post")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(vocab_size, 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )],
        verbose=1
    )

    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0))
    }
    print("\nTest metrics:", metrics)
    print(classification_report(y_test, predictions, target_names=["Negative", "Positive"]))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(history.history["accuracy"], label="Training accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation accuracy")
    ax.set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_loss.png", dpi=160)
    plt.close(fig)

    model.save(MODEL_DIR / "sentiment_model.keras")
    word_index = tf.keras.datasets.imdb.get_word_index()
    (MODEL_DIR / "imdb_word_index.json").write_text(json.dumps(word_index), encoding="utf-8")
    (OUT_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved model, word index, training curves and metrics.")

if __name__ == "__main__":
    main()
