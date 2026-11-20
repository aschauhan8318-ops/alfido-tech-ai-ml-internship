# How to Run and Submit All Four Alfido Tech Tasks

## Before starting
Use Python 3.11 if possible. Install Git. Docker Desktop is needed only for Task 3. Make sure your computer has an internet connection for first-time dataset downloads.

## Step 1 – Run Task 1
Open PowerShell in the project root:
```powershell
cd Task1_ML_Classification
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```
Wait for training to finish. Check:
- `model/spam_classifier.joblib`
- `outputs/model_metrics.csv`
- `outputs/model_comparison.png`
- confusion matrix and ROC plots

## Step 2 – Run Task 2
Open a NEW terminal:
```powershell
cd Task2_Deep_Learning
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```
TensorFlow downloads the IMDB dataset on first run. Training can take time. Check:
- `models/sentiment_model.keras`
- `models/imdb_word_index.json`
- `outputs/training_accuracy.png`
- `outputs/training_loss.png`
- `outputs/evaluation_metrics.json`

Test inference:
```powershell
python predict.py
```

## Step 3 – Run Task 3
First ensure Task 1 has generated its model.
```powershell
cd Task3_Model_Deployment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python copy_model.py
python app.py
```
Open `http://127.0.0.1:5000/health` in your browser. In a second PowerShell terminal, test:
```powershell
$body = @{ message = "Congratulations! You won a free prize. Click now!" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body
```
For Docker, start Docker Desktop, then:
```powershell
docker build -t alfido-spam-api .
docker run --rm -p 5000:5000 --name alfido-spam-api alfido-spam-api
```
Take screenshots of `/health` and the prediction request/response.

## Step 4 – Run Task 4
Ensure Task 1 has been run.
```powershell
cd Task4_Responsible_AI
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python analyze.py
```
Check the generated SHAP plots, feature importance, CSV and fairness proxy plot.

## Step 5 – Open notebooks
Each task has a notebook in its `notebooks` folder. Open it in VS Code or Jupyter and run all cells. The notebooks invoke the task scripts so the same reproducible code is used.

## Step 6 – Prepare final GitHub repository
1. Create a GitHub repository named `alfido-tech-ai-ml-internship`.
2. Extract this ZIP.
3. Run all tasks first so generated models, metrics and plots exist.
4. Copy the generated outputs into the same task folders.
5. From the extracted project root, run:
```powershell
git init
git add .
git commit -m "Complete Alfido Tech AI ML internship tasks"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/alfido-tech-ai-ml-internship.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username.

## Step 7 – What links to submit
Open the official Alfido Tech submission page:
https://www.alfidotech.com/submission.html

Submit the same GitHub repository link under each task, but clearly label each task and include its folder path:
- Task 1 – ML Classification: `Task1_ML_Classification/`
- Task 2 – Deep Learning: `Task2_Deep_Learning/`
- Task 3 – Model Deployment: `Task3_Model_Deployment/`
- Task 4 – Responsible AI: `Task4_Responsible_AI/`

If the form asks for separate document/report links, upload each task's DOCX report to Google Drive, set sharing to "Anyone with the link – Viewer", and paste the corresponding link. Do not share private data or credentials.

## Screenshots to capture
- Task 1: terminal showing metrics and model comparison plot.
- Task 2: terminal showing test metrics, training accuracy and loss plots, inference prediction.
- Task 3: browser `/health` response and PowerShell POST prediction response; Docker `docker ps`.
- Task 4: SHAP summary, local explanation, feature importance and fairness proxy chart.

## Final checks
- Do not submit model metrics until the code has run successfully.
- Keep model and dataset files in the repo only if their licenses and file sizes allow; otherwise provide dataset download instructions.
- Never upload `.venv`, passwords, API keys or private credentials.
- Test all README commands before final submission.
