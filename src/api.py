from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.build_features import extract_features, FEATURE_COLUMNS

PROJECT = Path(__file__).resolve().parents[1]
model = joblib.load(PROJECT / "models/baseline_model.joblib")

app = FastAPI(title="Phishing URL Detector", version="0.1.0")


class PredictionRequest(BaseModel):
    url: str = Field(min_length=1, max_length=8192)


@app.get("/health")
def health():
    return {"status": "ok", "model": "logistic-regression-baseline"}


@app.post("/predict")
def predict(request: PredictionRequest):
    url = request.url.strip()

    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.hostname)
            and not any(char.isspace() for char in url)
        )
    except ValueError:
        valid = False

    if not valid:
        raise HTTPException(
            status_code=422,
            detail="Enter a valid URL starting with http:// or https://"
        )

    features = pd.DataFrame(
        [extract_features(url)],
        columns=FEATURE_COLUMNS
    )

    prediction = int(model.predict(features)[0])
    phishing_index = list(model.classes_).index(1)
    score = float(model.predict_proba(features)[0, phishing_index])

    return {
        "url": url,
        "prediction": (
            "Likely phishing" if prediction == 1 else "Likely legitimate"
        ),
        "phishing_score": score,
        "model_version": "logistic-regression-baseline"
    }

from io import BytesIO
from fastapi import File, UploadFile
from fastapi.responses import Response


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read(5_000_001)
    if len(content) > 5_000_000:
        raise HTTPException(413, "Upload a CSV smaller than 5 MB")

    try:
        data = pd.read_csv(
            BytesIO(content),
            dtype=str,
            keep_default_na=False,
            nrows=10001
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
        raise HTTPException(422, "Upload a readable UTF-8 CSV file")

    if "URL" not in data.columns:
        raise HTTPException(422, "CSV must contain a column named URL")
    if data.empty or len(data) > 10000:
        raise HTTPException(422, "CSV must contain 1–10,000 URLs")

    results = pd.DataFrame({
        "URL": data["URL"].str.strip(),
        "prediction": "",
        "phishing_score": float("nan"),
        "error": ""
    })

    valid_indices = []
    records = []

    for index, url in results["URL"].items():
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme.lower() in {"http", "https"}
                and bool(parsed.hostname)
                and not any(char.isspace() for char in url)
            )
        except ValueError:
            valid = False

        if valid:
            valid_indices.append(index)
            records.append(extract_features(url))
        else:
            results.at[index, "error"] = "Invalid HTTP/HTTPS URL"

    if records:
        features = pd.DataFrame(records, columns=FEATURE_COLUMNS)
        predictions = model.predict(features)
        phishing_index = list(model.classes_).index(1)
        scores = model.predict_proba(features)[:, phishing_index]

        results.loc[valid_indices, "prediction"] = [
            "Likely phishing" if value == 1 else "Likely legitimate"
            for value in predictions
        ]
        results.loc[valid_indices, "phishing_score"] = scores

    # Prevent invalid input text being treated as a spreadsheet formula.
    results["URL"] = results["URL"].map(
        lambda value: "'" + value
        if value.startswith(("=", "+", "-", "@", "\t", "\r", "\n"))
        else value
    )

    return Response(
        content=results.to_csv(index=False),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="predictions.csv"'
        }
    )

import logging
import time
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

logger = logging.getLogger("uvicorn.error")

REQUESTS = Counter(
    "phishing_api_requests_total",
    "Number of API requests",
    ["endpoint", "method", "status"]
)

LATENCY = Histogram(
    "phishing_api_request_duration_seconds",
    "Time spent handling API requests",
    ["endpoint", "method"]
)


@app.middleware("http")
async def monitor_requests(request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    started = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        logger.exception("Unhandled API error")
        raise
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        elapsed = time.perf_counter() - started

        REQUESTS.labels(
            endpoint, request.method, str(status)
        ).inc()

        LATENCY.labels(
            endpoint, request.method
        ).observe(elapsed)

        logger.info(
            "endpoint=%s method=%s status=%s duration=%.4fs",
            endpoint, request.method, status, elapsed
        )


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST}
    )
