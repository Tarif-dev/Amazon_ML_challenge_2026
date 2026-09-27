from pathlib import Path
from collections import Counter
import pandas as pd
import re
import unicodedata


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"
PAIR_FILE = ROOT / "analysis" / "output" / "positive_pairs_sample.tsv"

SOURCES = {
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}

THRESHOLDS = [
    5,
    10,
    25,
    50,
    100,
    500,
    1000,
    5000,
    10000,
    50000,
]


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


def build_token_counter(path, column):

    counter = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[column],
        dtype=str,
        chunksize=100_000,
    ):
        for value in chunk[column]:
            counter.update(get_tokens(value))

    return counter


def main():

    print("=" * 90)
    print("TOKEN FREQUENCY → POSITIVE PAIR RECALL")
    print("=" * 90)

    pairs = pd.read_csv(
        PAIR_FILE,
        sep="\t",
        dtype=str,
    )

    print(f"\nPositive pairs: {len(pairs):,}")

    for source_name, source_path in SOURCES.items():

        print("\n" + "=" * 90)
        print(f"SOURCE: {source_name}")
        print("=" * 90)

        print("\nBuilding name token frequencies...")

        name_counter = build_token_counter(
            source_path,
            "business_name",
        )

        print("Building address token frequencies...")

        address_counter = build_token_counter(
            source_path,
            "business_address",
        )

        source_pairs = pairs[
            pairs["target_source"] == source_name
        ].copy()

        print(f"\nPositive pairs for {source_name}: {len(source_pairs):,}")

        results = []

        for threshold in THRESHOLDS:

            name_hits = 0
            address_hits = 0
            combined_hits = 0

            for _, row in source_pairs.iterrows():

                s1_name_tokens = get_tokens(
                    row["s1_business_name"]
                )

                target_name_tokens = get_tokens(
                    row["target_business_name"]
                )

                s1_address_tokens = get_tokens(
                    row["s1_business_address"]
                )

                target_address_tokens = get_tokens(
                    row["target_business_address"]
                )

                informative_name_s1 = {
                    token
                    for token in s1_name_tokens
                    if name_counter[token] <= threshold
                }

                informative_name_target = {
                    token
                    for token in target_name_tokens
                    if name_counter[token] <= threshold
                }

                informative_address_s1 = {
                    token
                    for token in s1_address_tokens
                    if address_counter[token] <= threshold
                }

                informative_address_target = {
                    token
                    for token in target_address_tokens
                    if address_counter[token] <= threshold
                }

                name_match = bool(
                    informative_name_s1
                    & informative_name_target
                )

                address_match = bool(
                    informative_address_s1
                    & informative_address_target
                )

                if name_match:
                    name_hits += 1

                if address_match:
                    address_hits += 1

                if name_match or address_match:
                    combined_hits += 1

            total = len(source_pairs)

            results.append({
                "threshold": threshold,
                "name_recall": name_hits / total,
                "address_recall": address_hits / total,
                "combined_recall": combined_hits / total,
            })

        result_df = pd.DataFrame(results)

        print("\n")
        print(
            result_df.to_string(
                index=False,
                formatters={
                    "name_recall": "{:.4%}".format,
                    "address_recall": "{:.4%}".format,
                    "combined_recall": "{:.4%}".format,
                },
            )
        )


if __name__ == "__main__":
    main()