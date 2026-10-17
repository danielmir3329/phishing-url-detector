from pathlib import Path
import logging

import joblib
import mlflow
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
PROJECT = Path(__file__).resolve().parents[1]


def main():
    engine = create_engine(
        f"sqlite:///{PROJECT / 'data/phishing.db'}"
    )
    with engine.connect() as connection:
        df = pd.read_sql_table("url_features", connection)
    engine.dispose()

    X = df.drop(columns="label")

    # Original dataset: 0 = phishing.
    # Model target: 1 = phishing, so phishing is the positive class.
    y = 1 - df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT / 'data/mlflow_tracking.db'}"
    )
    mlflow.set_experiment("phishing-url-detection")

    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_params({
            "model": "logistic_regression",
            "C": 1.0,
            "features": X.shape[1],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "split": "stratified_random_baseline",
            "random_state": 42,
            "positive_class": "phishing"
        })

        logging.info("Training model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "phishing_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "phishing_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "phishing_f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(
                y_test, probabilities
            )
        }
        mlflow.log_metrics(metrics)

        output = PROJECT / "models"
        output.mkdir(exist_ok=True)

        model_path = output / "baseline_model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0
        )
        report_path = output / "baseline_report.txt"
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        print("\n", report)
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print(f"\nSaved model: {model_path}")


if __name__ == "__main__":
    main()
