from pathlib import Path
import logging

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PROJECT = Path(__file__).resolve().parents[1]


def main():
    source = PROJECT / "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
    engine = create_engine(
        f"sqlite:///{PROJECT / 'data/phishing.db'}"
    )
    total = 0

    with engine.begin() as connection:
        batches = pd.read_csv(
            source,
            usecols=["URL", "Title"],
            dtype=str,
            keep_default_na=False,
            chunksize=10000
        )

        for number, batch in enumerate(batches, start=1):
            batch = batch.rename(columns={"Title": "title"})
            batch["clean_title"] = (
                batch["title"]
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            batch["title_length"] = batch["clean_title"].str.len()
            batch["title_word_count"] = (
                batch["clean_title"]
                .str.findall(r"\b\w+\b")
                .str.len()
            )

            batch.to_sql(
                "webpage_text",
                connection,
                if_exists="replace" if number == 1 else "append",
                index=False
            )
            total += len(batch)
            logging.info("Extracted and processed batch %s", number)

        stored = connection.execute(
            text("SELECT COUNT(*) FROM webpage_text")
        ).scalar_one()
        if stored != total:
            raise ValueError("Stored row count does not match extraction")

    engine.dispose()
    logging.info("Saved %s title records to SQLite", total)


if __name__ == "__main__":
    main()
