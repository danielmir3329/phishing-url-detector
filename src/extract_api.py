from pathlib import Path
from datetime import datetime, timezone
import json
import logging

import pandas as pd
import requests
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
PROJECT = Path(__file__).resolve().parents[1]


def main():
    output = PROJECT / "data/raw/rdap"
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for domain in ["wikipedia.org", "python.org"]:
        response = requests.get(
            f"https://rdap.org/domain/{domain}",
            headers={"Accept": "application/rdap+json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        (output / f"{domain}.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

        registration = next(
            (
                event.get("eventDate")
                for event in data.get("events", [])
                if event.get("eventAction") == "registration"
            ),
            None
        )

        records.append({
            "domain": domain,
            "registration_date": registration,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "source_url": response.url
        })
        logging.info("Extracted API record for %s", domain)

    engine = create_engine(
        f"sqlite:///{PROJECT / 'data/phishing.db'}"
    )
    with engine.begin() as connection:
        pd.DataFrame(records).to_sql(
            "domain_registration",
            connection,
            if_exists="replace",
            index=False
        )
    engine.dispose()
    logging.info("Saved %s API records to SQLite", len(records))


if __name__ == "__main__":
    main()
