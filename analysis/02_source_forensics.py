from pathlib import Path
from collections import Counter
import pandas as pd


DATASET = Path("student_resource/dataset")


FILES = {
    "train_source1": DATASET / "train/train_source1.tsv",
    "train_source2": DATASET / "train/train_source2.tsv",
    "train_source3": DATASET / "train/train_source3.tsv",
    "test_source1": DATASET / "test/test_source1.tsv",
    "test_source2": DATASET / "test/test_source2.tsv",
    "test_source3": DATASET / "test/test_source3.tsv",
}


def profile_file(name, path):
    print("\n" + "=" * 100)
    print(name)
    print(path)
    print("=" * 100)

    # Read only the columns we need.
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print(f"Rows: {len(df):,}")

    # ---------------------------------------------------------
    # ID uniqueness
    # ---------------------------------------------------------
    duplicate_ids = df["entity_id"].duplicated().sum()

    print("\n[1] ENTITY IDs")
    print(f"Unique IDs: {df['entity_id'].nunique():,}")
    print(f"Duplicate ID rows: {duplicate_ids:,}")

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    print("\n[2] MISSING VALUES")

    for column in [
        "business_name",
        "business_address",
        "country",
    ]:
        missing = (df[column].str.strip() == "").sum()
        percentage = missing / len(df) * 100

        print(
            f"{column:20s}: "
            f"{missing:10,} "
            f"({percentage:7.3f}%)"
        )

    # ---------------------------------------------------------
    # Country distribution
    # ---------------------------------------------------------
    print("\n[3] TOP COUNTRIES")

    country_counts = (
        df["country"]
        .replace("", "<MISSING>")
        .value_counts()
        .head(20)
    )

    for country, count in country_counts.items():
        print(
            f"{str(country):20s} "
            f"{count:10,} "
            f"({count / len(df) * 100:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Name length
    # ---------------------------------------------------------
    name_lengths = df["business_name"].str.len()

    print("\n[4] BUSINESS NAME LENGTH")

    print(
        name_lengths.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )

    # ---------------------------------------------------------
    # Address length
    # ---------------------------------------------------------
    address_lengths = df["business_address"].str.len()

    print("\n[5] BUSINESS ADDRESS LENGTH")

    print(
        address_lengths.describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
    )

    # ---------------------------------------------------------
    # Exact duplicate records
    # ---------------------------------------------------------
    print("\n[6] EXACT DUPLICATE RECORDS")

    duplicate_rows = df.duplicated(
        subset=[
            "business_name",
            "business_address",
            "country",
        ],
        keep=False,
    ).sum()

    duplicate_groups = (
        df.groupby(
            [
                "business_name",
                "business_address",
                "country",
            ],
            dropna=False,
        )
        .size()
    )

    repeated_groups = (duplicate_groups > 1).sum()

    print(f"Rows belonging to duplicate groups: {duplicate_rows:,}")
    print(f"Repeated (name,address,country) groups: {repeated_groups:,}")

    # ---------------------------------------------------------
    # Name-only duplicates
    # ---------------------------------------------------------
    name_group_sizes = (
        df.groupby(
            ["business_name"],
            dropna=False,
        )
        .size()
    )

    repeated_names = (name_group_sizes > 1).sum()

    print("\n[7] BUSINESS NAME DUPLICATION")
    print(f"Names appearing more than once: {repeated_names:,}")

    # ---------------------------------------------------------
    # Sample multilingual / unusual records
    # ---------------------------------------------------------
    print("\n[8] SAMPLE NON-ASCII NAMES")

    non_ascii = df[
        df["business_name"].map(
            lambda x: any(ord(ch) > 127 for ch in x)
        )
    ]

    print(
        non_ascii[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return df


def main():
    print("=" * 100)
    print("AMAZON ML CHALLENGE — SOURCE FORENSICS")
    print("=" * 100)

    # IMPORTANT:
    #
    # The files are large (~200–500 MB each).
    # We therefore process them ONE AT A TIME.
    #
    # We do NOT load all six source files simultaneously.

    for name, path in FILES.items():
        profile_file(name, path)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()