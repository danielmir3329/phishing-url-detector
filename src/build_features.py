from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import logging

import dask.dataframe as dd
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/processed/clean_urls.csv"
OUTPUT = PROJECT / "data/processed/url_features.csv"

FEATURE_COLUMNS = [
    "url_length", "digit_count", "dot_count", "hyphen_count",
    "at_count", "question_count", "equals_count",
    "percent_count", "uses_https", "hostname_length",
    "hostname_dot_count", "hostname_is_ip", "path_length"
]


def extract_features(url):
    """Calculate features from text without visiting the website."""
    url = str(url).strip()

    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        hostname = parsed.hostname or ""
        path = parsed.path
        uses_https = int(parsed.scheme.lower() == "https")

        try:
            ipaddress.ip_address(hostname)
            hostname_is_ip = 1
        except ValueError:
            hostname_is_ip = 0

    except ValueError:
        hostname = ""
        path = ""
        uses_https = 0
        hostname_is_ip = 0

    return {
        "url_length": len(url),
        "digit_count": sum(char.isdigit() for char in url),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equals_count": url.count("="),
        "percent_count": url.count("%"),
        "uses_https": uses_https,
        "hostname_length": len(hostname),
        "hostname_dot_count": hostname.count("."),
        "hostname_is_ip": hostname_is_ip,
        "path_length": len(path),
    }


def transform_partition(partition):
    records = [
        extract_features(url) for url in partition["URL"]
    ]
    features = pd.DataFrame(
        records,
        columns=FEATURE_COLUMNS,
        index=partition.index
    ).astype("int64")

    features["label"] = partition["label"].astype("int64")
    return features


def main():
    data = dd.read_csv(
        SOURCE,
        dtype={"URL": "string", "label": "int64"},
        blocksize="1MB"
    )
    logging.info("Processing %s partitions", data.npartitions)

    meta = pd.DataFrame({
        column: pd.Series(dtype="int64")
        for column in FEATURE_COLUMNS + ["label"]
    })

    features = data.map_partitions(
        transform_partition,
        meta=meta
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, single_file=True, index=False)
    logging.info("Saved features to %s", OUTPUT)


if __name__ == "__main__":
    main()
