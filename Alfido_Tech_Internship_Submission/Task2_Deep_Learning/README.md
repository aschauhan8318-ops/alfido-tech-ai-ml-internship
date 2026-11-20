# Task 2 – Deep Learning: IMDB Movie Review Sentiment Analysis

## Goal
Classify movie reviews as positive or negative using a Bidirectional LSTM neural network.

## Dataset
TensorFlow/Keras IMDB review dataset (25,000 training and 25,000 test reviews). It downloads automatically the first time it is loaded.

## Setup (Windows PowerShell)
```powershell
cd Task2_Deep_Learning
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

## Inference
After training has created the model files:
```powershell
python predict.py
```
Type a movie review when prompted.

## Outputs
- `models/sentiment_model.keras`
- `models/imdb_word_index.json` (required tokenizer vocabulary mapping)
- `outputs/training_accuracy.png`
- `outputs/training_loss.png`
- `outputs/evaluation_metrics.json`

## Notes
This is a trained-from-scratch LSTM baseline with text token IDs and an embedding layer. Transfer learning is not applicable to this baseline; the task brief says to use pretrained models / transfer learning where applicable. The IMDB dataset is pre-tokenized by Keras. The saved word-index file enables inference from raw review text.
