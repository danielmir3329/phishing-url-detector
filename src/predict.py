from pathlib import Path
import sys
import joblib
import pandas as pd
from build_features import extract_features, FEATURE_COLUMNS

project = Path(__file__).resolve().parents[1]
model = joblib.load(project / "models/baseline_model.joblib")

urls = sys.argv[1:]
if not urls:
    raise SystemExit('Usage: python src/predict.py "https://example.com"')

features = pd.DataFrame(
    [extract_features(url) for url in urls],
    columns=FEATURE_COLUMNS
)

predictions = model.predict(features)
phishing_index = list(model.classes_).index(1)
scores = model.predict_proba(features)[:, phishing_index]

for url, prediction, score in zip(urls, predictions, scores):
    label = "Likely phishing" if prediction == 1 else "Likely legitimate"
    print(f"\nURL: {url}")
    print(f"Prediction: {label}")
    print(f"Model phishing score: {score:.4f}")
