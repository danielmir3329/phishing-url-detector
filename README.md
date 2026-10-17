# Phishing URL Detector

An ETL and MLOps project that trains a URL phishing classifier,
serves predictions through an API, and monitors application performance.

## Tools

Python, pandas, Dask, SQLite, SQLAlchemy, scikit-learn, MLflow,
FastAPI, Airflow, GitHub Actions, Prometheus, and Grafana.

## Setup

Use Python 3.14 on Ubuntu or Windows with WSL2.
Run these commands from the project folder:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    mkdir -p data/raw data/processed models

Download and extract the PhiUSIIL dataset from:
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Place PhiUSIIL_Phishing_URL_Dataset.csv in data/raw/.

Dataset credit: Prasad and Chandra (2024), UCI Machine Learning
Repository, CC BY 4.0. Original labels: 0 = phishing, 1 = legitimate.

## Run the Pipeline

    python src/clean_data.py
    python src/build_features.py
    python src/load_data.py
    python src/train_model.py

Cleaning removes invalid records and duplicate URLs. Dask extracts
13 URL features. SQLAlchemy loads them into SQLite in 10,000-row
batches. Training reads SQLite and fits a StandardScaler and
logistic regression pipeline. The model target uses 1 = phishing.

## Predictions

    python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

Open http://localhost:8000/docs.

- GET /health checks service status.
- POST /predict accepts a JSON object containing a url.
- POST /predict-csv accepts a CSV with a URL column.
- GET /metrics exposes Prometheus metrics.

CSV output includes URL, prediction, phishing_score, and error.
The application analyzes URL text without visiting websites.

## Experiments and Results

MLflow stores parameters, metrics, model artifacts, and reports
in the phishing-url-detection experiment. Its tracking database
is data/mlflow_tracking.db.

The initial random-split baseline achieved phishing precision
0.9981, recall 0.9806, and F1 0.9892. These results apply to the
dataset split and do not establish real-world accuracy.
A legitimate prediction does not guarantee a safe website.

## Automation and Monitoring

Airflow runs dags/phishing_pipeline.py daily:
API extraction → title-text extraction → cleaning → features → database loading → training.
It uses a separate .airflow-venv environment.

GitHub Actions CI checks dependencies, syntax, and feature tests.
The manual deployment workflow validates code, updates the local
project, retrains, restarts the API service, and checks health.
Deployment requires the configured self-hosted Linux runner.
Service files in deployment/ contain machine-specific paths.

Prometheus uses monitoring/prometheus.yml. Grafana displays
request counts, response duration, and rejected requests.
Python and Airflow logs record processing and errors.
Monitoring measures service operation, not prediction accuracy.

## Tests

    .venv/bin/python -m unittest discover -s tests -v


## Additional Extraction and Tuning

src/extract_text.py processes webpage-title text in 10,000-row batches.
src/extract_api.py retrieves domain registration JSON through RDAP.
Both store their results in SQLite and are included in the Airflow DAG.
These supplementary records are separate from the URL-only classifier.

Training compares C values 0.1, 1.0, and 10.0 using three-fold
stratified cross-validation on the training partition. MLflow records
candidate scores and the selected setting. The held-out test partition
is used after selection.

The observed tuned run selected C=10.0, with cross-validation F1
0.991876. Test precision was 0.9992, recall 0.9857, F1 0.9924,
ROC AUC 0.9966, and average precision 0.9973.

Models are retained in folders named by MLflow run ID.
models/model_version.json identifies the selected model.
The API reads this identifier at startup.

A manual deployment workflow has completed successfully.
The final updated six-task Airflow run should also be checked.
Generated data, models, environments, and credentials are excluded
from Git. The written assignment report accompanies this repository.
