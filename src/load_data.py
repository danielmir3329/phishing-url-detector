from pathlib import Path
import logging

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/processed/url_features.csv"
DATABASE = PROJECT / "data/phishing.db"


def main():
    engine = create_engine(f"sqlite:///{DATABASE}")
    total = 0

    with engine.begin() as connection:
        for batch_number, chunk in enumerate(
            pd.read_csv(SOURCE, chunksize=10000),
            start=1
        ):
            chunk.to_sql(
                "url_features",
                con=connection,
                if_exists="replace" if batch_number == 1 else "append",
                index=False
            )
            total += len(chunk)

        stored = connection.execute(
            text("SELECT COUNT(*) FROM url_features")
        ).scalar_one()

        if stored != total:
            raise ValueError(
                f"Row count mismatch: expected {total}, found {stored}"
            )

    logging.info("Loaded %s rows into SQLite", stored)
    logging.info("Database: %s", DATABASE)
    engine.dispose()


if __name__ == "__main__":
    main()
