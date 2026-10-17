from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess

from airflow.sdk import DAG, task

PROJECT = Path.home() / "phishing-detector"
PYTHON = PROJECT / ".venv/bin/python"

with DAG(
    dag_id="phishing_pipeline",
    description="Prepare URL data and train the phishing detector",
    start_date=datetime(2026, 10, 16, tzinfo=timezone.utc),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
        "execution_timeout": timedelta(minutes=15),
    },
    tags=["phishing", "etl", "mlops"],
) as dag:

    @task
    def run_script(script_name: str):
        subprocess.run(
            [str(PYTHON), str(PROJECT / "src" / script_name)],
            cwd=str(PROJECT),
            check=True,
        )

    clean = run_script.override(
        task_id="clean_data"
    )("clean_data.py")

    features = run_script.override(
        task_id="build_features"
    )("build_features.py")

    load = run_script.override(
        task_id="load_database"
    )("load_data.py")

    train = run_script.override(
        task_id="train_model"
    )("train_model.py")

    clean >> features >> load >> train
