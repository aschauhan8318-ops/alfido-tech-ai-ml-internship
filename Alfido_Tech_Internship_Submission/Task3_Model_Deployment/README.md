# Task 3 – Flask API and Docker Deployment

## Prerequisite
Run Task 1 first to generate `Task1_ML_Classification/model/spam_classifier.joblib`.

## Local setup (Windows PowerShell)
```powershell
cd Task3_Model_Deployment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python copy_model.py
python app.py
```
Open `http://127.0.0.1:5000/health`.

## Test prediction in PowerShell
```powershell
$body = @{ message = "Congratulations! You won a free prize. Click now!" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body
```

## Docker
After copying the trained model:
```powershell
docker build -t alfido-spam-api .
docker run --rm -p 5000:5000 --name alfido-spam-api alfido-spam-api
```
Test:
```powershell
curl.exe http://localhost:5000/health
```
Prediction:
```powershell
curl.exe -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"message\":\"You have won a prize\"}"
```

## Endpoints
- `GET /` – service information
- `GET /health` – health check
- `POST /predict` – JSON model inference

## Example request
```json
{"message": "Congratulations! You won a free prize. Click now!"}
```
Example response shape (prediction is model-dependent):
```json
{
  "input_message": "Congratulations! You won a free prize. Click now!",
  "prediction": "Spam",
  "prediction_label": 1,
  "spam_probability": 0.91,
  "status": "success"
}
```
The probability shown above is illustrative, not a measured result.
