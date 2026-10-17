# Phishing URL Detector

This project demonstrates a Python ETL and MLOps pipeline for
classifying URLs as likely phishing or legitimate.

## Tools

pandas, Dask, SQLite, SQLAlchemy, scikit-learn, MLflow,
FastAPI, Airflow, GitHub Actions, Prometheus, and Grafana.

## Setup

Use Python 3.14 in Ubuntu or Windows WSL2.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    mkdir -p data/raw data/processed models

Download the PhiUSIIL dataset and place its extracted CSV in data/raw/:
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Dataset credit: Prasad and Chandra (2024), UCI, CC BY 4.0.

## Run the Pipeline

    python src/extract_api.py
    python src/extract_text.py
    python src/clean_data.py
    python src/build_features.py
    python src/load_data.py
    python src/train_model.py

The scripts extract RDAP API records and webpage-title text,
clean URL data, calculate 13 URL features with Dask, load SQLite,
and train a classifier. Text and registration data are supplementary
ETL demonstrations; the model uses URL features only.

Training compares C values 0.1, 1.0, and 10.0 with three-fold
cross-validation. MLflow tracks settings, scores, and model artifacts.
The selected model is identified by its MLflow run ID.

## Use the API

    python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

Open http://localhost:8000/docs to predict one URL or upload a
CSV containing a URL column.

The API returns predictions and scores without visiting websites.
A legitimate prediction does not guarantee a safe URL.

## Results

After removing 425 duplicates, 235,370 URLs remained.
The tuned model selected C=10.0 and achieved:

| Test metric | Score |
| --- | ---: |
| Phishing precision | 0.9992 |
| Phishing recall | 0.9857 |
| Phishing F1 | 0.9924 |
| ROC AUC | 0.9966 |

Results are from a stratified random split and may not generalize
to new threats.

## Automation and Monitoring

- dags/phishing_pipeline.py defines the daily six-task Airflow workflow.
- .github/workflows/ci.yml runs automated checks.
- .github/workflows/deploy.yml tests, retrains, and deploys locally.
- deployment/ contains the systemd service configuration.
- monitoring/prometheus.yml configures API metric collection.
- Grafana displays request counts, response times, and rejected requests.
- Python and Airflow logs record processing and errors.

Airflow uses a separate .airflow-venv environment.
Deployment requires the configured self-hosted Linux runner and
machine-specific service paths. It requires a clean Git checkout.

## Tests

    .venv/bin/python -m unittest discover -s tests -v

## Verification

An initial deployment and four-task Airflow run succeeded.
The latest deployment stopped because local changes were present.
The updated six-task Airflow run still requires confirmation.

Generated data, models, environments, and credentials are excluded
from Git.
