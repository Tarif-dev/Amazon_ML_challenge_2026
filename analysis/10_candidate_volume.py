from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import re
import unicodedata
import math
import time


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"


# We do NOT process all 2.2M S1 rows yet.
# This is only an experiment to estimate candidate volume.
S1_SAMPLE_SIZE = 10_000

# Random seed makes the experiment reproducible.
RANDOM_SEED = 42

# Frequency thresholds to test.
#
# A token occurring more than the threshold in the target
# source is ignored as a blocking token.
THRESHOLDS = [
    500,
    1_000,
    5_000,
    10_000,
    50_000,
]

CHUNK_SIZE = 100_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text while preserving Unicode characters.

    Steps:
    1. Handle missing values.
    2. Unicode NFKC normalization.
    3. Case folding.
    4. Replace punctuation with spaces.
    5. Collapse repeated whitespace.
    """

    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()

    # Keep Unicode word characters.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def get_tokens(value):
    """
    Return unique normalized tokens.

    Using a set is important because if a token appears twice
    inside the same address/name, it should not create duplicate
    candidate IDs.
    """

    normalized = normalize_text(value)

    if not normalized:
        return set()

    return set(normalized.split())


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

def load_s1_sample():

    print("=" * 90)
    print("LOADING SOURCE 1 SAMPLE")
    print("=" * 90)

    print(f"File: {S1_FILE}")
    print(f"Sample size: {S1_SAMPLE_SIZE:,}")

    start = time.time()

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        dtype=str,
    )

    print(f"Full S1 rows: {len(s1):,}")

    if S1_SAMPLE_SIZE < len(s1):
        s1 = s1.sample(
            n=S1_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )

    s1 = s1.reset_index(drop=True)

    print(f"Sample rows: {len(s1):,}")
    print(f"Loading time: {time.time() - start:.2f}s")

    return s1


# ============================================================
# BUILD TOKEN FREQUENCIES
# ============================================================

def build_token_frequencies(path):

    print("\n" + "=" * 90)
    print("BUILDING TOKEN FREQUENCIES")
    print("=" * 90)

    print(f"File: {path}")

    name_counter = Counter()
    address_counter = Counter()

    total_rows = 0

    start = time.time()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "business_address",
        ],
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):

        total_rows += len(chunk)

        for value in chunk["business_name"]:
            name_counter.update(get_tokens(value))

        for value in chunk["business_address"]:
            address_counter.update(get_tokens(value))

        print(
            f"Processed {total_rows:,} rows...",
            end="\r",
        )

    elapsed = time.time() - start

    print()
    print(f"Finished: {total_rows:,} rows")
    print(f"Unique name tokens: {len(name_counter):,}")
    print(f"Unique address tokens: {len(address_counter):,}")
    print(f"Time: {elapsed:.2f}s")

    return name_counter, address_counter


# ============================================================
# BUILD TARGET TOKEN → ROW-ID INDEX
# ============================================================

def build_indexes(
    path,
    name_counter,
    address_counter,
    threshold,
):
    """
    Build:

        name_token -> set(target row IDs)
        address_token -> set(target row IDs)

    ONLY tokens whose global frequency <= threshold are indexed.

    This deliberately uses integer row IDs instead of entity_id
    strings to reduce memory consumption during the experiment.
    """

    print("\n" + "-" * 90)
    print(f"BUILDING INDEX — MAX TOKEN FREQUENCY = {threshold:,}")
    print("-" * 90)

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    row_id = 0
    total_rows = 0

    start = time.time()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "business_name",
            "business_address",
        ],
        dtype=str,
        chunksize=CHUNK_SIZE,
    ):

        for name, address in zip(
            chunk["business_name"],
            chunk["business_address"],
        ):

            name_tokens = get_tokens(name)
            address_tokens = get_tokens(address)

            # ------------------------------------------------
            # NAME INDEX
            # ------------------------------------------------

            for token in name_tokens:

                if name_counter[token] <= threshold:
                    name_index[token].append(row_id)

            # ------------------------------------------------
            # ADDRESS INDEX
            # ------------------------------------------------

            for token in address_tokens:

                if address_counter[token] <= threshold:
                    address_index[token].append(row_id)

            row_id += 1

        total_rows += len(chunk)

        print(
            f"Indexed {total_rows:,} rows...",
            end="\r",
        )

    print()

    elapsed = time.time() - start

    print(f"Rows indexed: {total_rows:,}")
    print(f"Indexed name tokens: {len(name_index):,}")
    print(f"Indexed address tokens: {len(address_index):,}")
    print(f"Index build time: {elapsed:.2f}s")

    return name_index, address_index


# ============================================================
# CANDIDATE COUNT CALCULATION
# ============================================================

def calculate_candidate_counts(
    s1,
    name_index,
    address_index,
):
    """
    For each S1 record:

        candidates =
            candidates from name tokens
            UNION
            candidates from address tokens

    Returns candidate-count statistics.
    """

    counts = []

    start = time.time()

    for i, row in s1.iterrows():

        candidate_ids = set()

        # ----------------------------------------------------
        # NAME BLOCK
        # ----------------------------------------------------

        name_tokens = get_tokens(
            row["business_name"]
        )

        for token in name_tokens:

            candidate_ids.update(
                name_index.get(token, [])
            )

        # ----------------------------------------------------
        # ADDRESS BLOCK
        # ----------------------------------------------------

        address_tokens = get_tokens(
            row["business_address"]
        )

        for token in address_tokens:

            candidate_ids.update(
                address_index.get(token, [])
            )

        counts.append(len(candidate_ids))

        if (i + 1) % 1000 == 0:
            print(
                f"Processed S1: {i + 1:,}/{len(s1):,}",
                end="\r",
            )

    print()

    elapsed = time.time() - start

    series = pd.Series(counts)

    statistics = {
        "mean": series.mean(),
        "median": series.median(),
        "p90": series.quantile(0.90),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "max": series.max(),
        "zero": (series == 0).sum(),
        "under_10": (series < 10).sum(),
        "under_50": (series < 50).sum(),
        "under_100": (series < 100).sum(),
        "under_500": (series < 500).sum(),
        "under_1000": (series < 1000).sum(),
        "under_5000": (series < 5000).sum(),
        "under_10000": (series < 10000).sum(),
    }

    statistics["time_seconds"] = elapsed

    return statistics


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_statistics(
    source_name,
    threshold,
    stats,
    total_s1,
):

    print("\n")
    print("=" * 90)
    print(
        f"CANDIDATE VOLUME — {source_name} — "
        f"THRESHOLD {threshold:,}"
    )
    print("=" * 90)

    print(
        f"Mean candidates:       {stats['mean']:,.2f}"
    )

    print(
        f"Median candidates:     {stats['median']:,.0f}"
    )

    print(
        f"P90 candidates:        {stats['p90']:,.0f}"
    )

    print(
        f"P95 candidates:        {stats['p95']:,.0f}"
    )

    print(
        f"P99 candidates:        {stats['p99']:,.0f}"
    )

    print(
        f"Maximum candidates:    {stats['max']:,}"
    )

    print()

    print(
        f"Zero candidates:       "
        f"{stats['zero']:,} "
        f"({stats['zero'] / total_s1:.2%})"
    )

    print(
        f"< 10 candidates:       "
        f"{stats['under_10']:,} "
        f"({stats['under_10'] / total_s1:.2%})"
    )

    print(
        f"< 50 candidates:       "
        f"{stats['under_50']:,} "
        f"({stats['under_50'] / total_s1:.2%})"
    )

    print(
        f"< 100 candidates:      "
        f"{stats['under_100']:,} "
        f"({stats['under_100'] / total_s1:.2%})"
    )

    print(
        f"< 500 candidates:      "
        f"{stats['under_500']:,} "
        f"({stats['under_500'] / total_s1:.2%})"
    )

    print(
        f"< 1,000 candidates:    "
        f"{stats['under_1000']:,} "
        f"({stats['under_1000'] / total_s1:.2%})"
    )

    print(
        f"< 5,000 candidates:    "
        f"{stats['under_5000']:,} "
        f"({stats['under_5000'] / total_s1:.2%})"
    )

    print(
        f"< 10,000 candidates:   "
        f"{stats['under_10000']:,} "
        f"({stats['under_10000'] / total_s1:.2%})"
    )

    print(
        f"\nCandidate calculation time: "
        f"{stats['time_seconds']:.2f}s"
    )


# ============================================================
# RUN EXPERIMENT FOR ONE SOURCE
# ============================================================

def analyze_source(
    source_name,
    source_path,
    s1,
):

    print("\n\n")
    print("#" * 100)
    print(f"# ANALYZING {source_name}")
    print("#" * 100)

    # --------------------------------------------------------
    # STEP 1 — TOKEN FREQUENCIES
    # --------------------------------------------------------

    name_counter, address_counter = build_token_frequencies(
        source_path
    )

    all_results = []

    # --------------------------------------------------------
    # STEP 2 — TEST EACH THRESHOLD
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        name_index, address_index = build_indexes(
            source_path,
            name_counter,
            address_counter,
            threshold,
        )

        stats = calculate_candidate_counts(
            s1,
            name_index,
            address_index,
        )

        print_statistics(
            source_name,
            threshold,
            stats,
            len(s1),
        )

        all_results.append({
            "source": source_name,
            "threshold": threshold,
            **stats,
        })

        # Explicitly delete indexes before next threshold.
        del name_index
        del address_index

    return all_results


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("\n")
    print("#" * 100)
    print("# AMAZON ML CHALLENGE — CANDIDATE VOLUME EXPERIMENT")
    print("#" * 100)

    print()
    print(f"Project root: {ROOT}")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"S1 sample size: {S1_SAMPLE_SIZE:,}")
    print(f"Thresholds: {THRESHOLDS}")
    print()

    # --------------------------------------------------------
    # LOAD S1
    # --------------------------------------------------------

    s1 = load_s1_sample()

    # --------------------------------------------------------
    # ANALYZE S2
    # --------------------------------------------------------

    s2_results = analyze_source(
        "S2",
        S2_FILE,
        s1,
    )

    # --------------------------------------------------------
    # ANALYZE S3
    # --------------------------------------------------------

    s3_results = analyze_source(
        "S3",
        S3_FILE,
        s1,
    )

    # --------------------------------------------------------
    # COMBINED SUMMARY
    # --------------------------------------------------------

    results = pd.DataFrame(
        s2_results + s3_results
    )

    print("\n\n")
    print("#" * 100)
    print("# FINAL SUMMARY")
    print("#" * 100)

    summary_columns = [
        "source",
        "threshold",
        "mean",
        "median",
        "p90",
        "p95",
        "p99",
        "max",
        "zero",
    ]

    print(
        results[summary_columns].to_string(
            index=False,
            formatters={
                "mean": "{:,.2f}".format,
                "median": "{:,.0f}".format,
                "p90": "{:,.0f}".format,
                "p95": "{:,.0f}".format,
                "p99": "{:,.0f}".format,
                "max": "{:,.0f}".format,
                "zero": "{:,.0f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_path = (
        ROOT
        / "analysis"
        / "08_candidate_volume_results.tsv"
    )

    results.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print()
    print(f"Results saved to:")
    print(output_path)

    print()
    print(
        f"Total experiment time: "
        f"{time.time() - total_start:.2f}s"
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()