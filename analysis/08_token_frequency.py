from pathlib import Path
from collections import Counter
import pandas as pd
import re
import unicodedata


# Project root:
# D:\amzn_ml_challenge
ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"

SOURCES = {
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def get_tokens(value):
    normalized = normalize_text(value)

    if not normalized:
        return set()

    return set(normalized.split())


def analyze_source(source_name, path):

    print("\n" + "=" * 90)
    print(f"TOKEN FREQUENCY ANALYSIS — {source_name}")
    print("=" * 90)

    name_counter = Counter()
    address_counter = Counter()

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["business_name", "business_address"],
        dtype=str,
        chunksize=100_000,
    ):
        total_rows += len(chunk)

        for value in chunk["business_name"]:
            name_counter.update(get_tokens(value))

        for value in chunk["business_address"]:
            address_counter.update(get_tokens(value))

        print(f"Processed {total_rows:,} rows...", end="\r")

    print(f"\nFinished {total_rows:,} rows.")

    def print_stats(counter, label):

        print("\n" + "-" * 80)
        print(label)
        print("-" * 80)

        print(f"Unique tokens: {len(counter):,}")

        if not counter:
            return

        frequencies = sorted(counter.values(), reverse=True)

        series = pd.Series(frequencies)

        print(f"Total token occurrences: {sum(frequencies):,}")
        print(f"Median frequency: {series.median():,.0f}")
        print(f"Mean frequency:   {series.mean():,.2f}")
        print(f"Max frequency:    {frequencies[0]:,}")

        print("\nTop 30 most frequent tokens:")

        for token, count in counter.most_common(30):
            print(f"{token:30s} {count:>12,}")

        print("\nFrequency buckets:")

        buckets = [
            ("1", 1, 1),
            ("2-5", 2, 5),
            ("6-10", 6, 10),
            ("11-50", 11, 50),
            ("51-100", 51, 100),
            ("101-500", 101, 500),
            ("501-1K", 501, 1_000),
            ("1K-5K", 1_001, 5_000),
            ("5K-10K", 5_001, 10_000),
            ("10K-50K", 10_001, 50_000),
            ("50K+", 50_001, float("inf")),
        ]

        for bucket_name, low, high in buckets:

            count = sum(
                1
                for frequency in frequencies
                if low <= frequency <= high
            )

            print(f"{bucket_name:10s}: {count:>10,} tokens")

    print_stats(name_counter, "BUSINESS NAME TOKENS")
    print_stats(address_counter, "BUSINESS ADDRESS TOKENS")


for source_name, path in SOURCES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{path}"
        )

    analyze_source(source_name, path)


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)