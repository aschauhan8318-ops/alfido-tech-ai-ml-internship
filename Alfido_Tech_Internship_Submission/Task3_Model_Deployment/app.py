from pathlib import Path
import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "model" / "spam_classifier.joblib"
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "service": "Alfido Spam Classification API",
        "status": "running",
        "endpoints": ["/health", "/predict"]
    })

@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("message"), str):
        return jsonify({"error": "Send JSON with a string field named 'message'."}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Message must not be empty."}), 400
    if len(message) > 10000:
        return jsonify({"error": "Message is too long (maximum 10000 characters)."}), 413

    prediction = int(model.predict([message])[0])
    result = "Spam" if prediction == 1 else "Not Spam"
    response = {
        "input_message": message,
        "prediction": result,
        "prediction_label": prediction,
        "status": "success"
    }
    if hasattr(model, "predict_proba"):
        response["spam_probability"] = round(float(model.predict_proba([message])[0][1]), 4)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
