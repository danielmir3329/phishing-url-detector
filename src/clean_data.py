from pathlib import Path
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"
OUTPUT = PROJECT / "data/processed/clean_urls.csv"


def main():
    df = pd.read_csv(SOURCE, usecols=["URL", "label"])
    original_rows = len(df)
    logging.info("Loaded %s rows", original_rows)

    # Remove invalid records and surrounding whitespace.
    df = df.dropna(subset=["URL", "label"]).copy()
    df["URL"] = df["URL"].str.strip()
    df = df.loc[
        df["URL"].ne("") & df["label"].isin([0, 1])
    ].copy()

    # Stop if a URL has contradictory training labels.
    conflicts = df.groupby("URL")["label"].nunique().gt(1)
    if conflicts.any():
        raise ValueError(
            f"Found {conflicts.sum()} URLs with conflicting labels"
        )

    df = df.drop_duplicates(subset=["URL"])
    df["label"] = df["label"].astype(int)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    logging.info("Removed %s rows", original_rows - len(df))
    logging.info("Saved %s rows to %s", len(df), OUTPUT)


if __name__ == "__main__":
    main()
