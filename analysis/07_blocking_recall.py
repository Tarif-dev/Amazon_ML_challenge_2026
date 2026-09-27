import re
import unicodedata
import pandas as pd

FEATURES = "analysis/output/positive_pair_features.tsv"
PAIRS = "analysis/output/positive_pairs_sample.tsv"

print("=" * 100)
print("BLOCKING KEY ANALYSIS")
print("=" * 100)

# =========================================================
# LOAD FILES
# =========================================================

features = pd.read_csv(FEATURES, sep="\t")
pairs = pd.read_csv(PAIRS, sep="\t")

print(f"Feature rows: {len(features):,}")
print(f"Pair rows:    {len(pairs):,}")

print("\nPair columns:")
print(list(pairs.columns))

print("\nFeature columns:")
print(list(features.columns))


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()

    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def tokens(value):
    text = normalize_text(value)

    if not text:
        return set()

    return set(text.split())


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_pair_columns = [
    "source1_entity_id",
    "target_entity_id",
    "target_source",
]

missing = [
    col
    for col in required_pair_columns
    if col not in pairs.columns
]

if missing:
    raise ValueError(
        f"Missing required columns in positive_pairs_sample.tsv: {missing}"
    )


# =========================================================
# MERGE FEATURES + ORIGINAL PAIR DATA
# =========================================================

key_cols = [
    "source1_entity_id",
    "target_entity_id",
    "target_source",
]

df = features.merge(
    pairs,
    on=key_cols,
    how="left",
    suffixes=("", "_pair")
)

print(f"\nMerged rows: {len(df):,}")


# =========================================================
# IMPORTANT:
# Inspect what original text columns are actually available.
# =========================================================

print("\nColumns after merge:")
print(list(df.columns))


# =========================================================
# We need the actual text fields.
# =========================================================

required_text_columns = [
    "s1_business_name",
    "s1_business_address",
    "target_business_name",
    "target_business_address",
]

missing_text = [
    col
    for col in required_text_columns
    if col not in df.columns
]

if missing_text:
    print("\n" + "=" * 100)
    print("TEXT COLUMNS ARE MISSING")
    print("=" * 100)

    print("Missing:")
    for col in missing_text:
        print(f"  - {col}")

    print("\nAvailable columns:")
    for col in df.columns:
        print(f"  - {col}")

    raise ValueError(
        "\nThe positive-pair file does not contain all original "
        "business text fields. We need to retrieve them from the "
        "original source files before measuring blocking recall."
    )


# =========================================================
# CREATE NORMALIZED REPRESENTATIONS
# =========================================================

df["s1_name_norm"] = (
    df["s1_business_name"]
    .map(normalize_text)
)

df["target_name_norm"] = (
    df["target_business_name"]
    .map(normalize_text)
)

df["s1_address_norm"] = (
    df["s1_business_address"]
    .map(normalize_text)
)

df["target_address_norm"] = (
    df["target_business_address"]
    .map(normalize_text)
)


# =========================================================
# BLOCK A — NORMALIZED NAME EXACT
# =========================================================

df["block_name_exact"] = (
    df["s1_name_norm"] == df["target_name_norm"]
)

name_exact_recall = df["block_name_exact"].mean()


# =========================================================
# BLOCK B — SHARED NAME TOKEN
# =========================================================

df["s1_name_tokens"] = (
    df["s1_business_name"]
    .map(tokens)
)

df["target_name_tokens"] = (
    df["target_business_name"]
    .map(tokens)
)

df["block_name_token"] = df.apply(
    lambda row:
        bool(
            row["s1_name_tokens"]
            &
            row["target_name_tokens"]
        ),
    axis=1
)

name_token_recall = df["block_name_token"].mean()


# =========================================================
# BLOCK C — NORMALIZED ADDRESS EXACT
# =========================================================

df["block_address_exact"] = (
    (df["s1_address_norm"] != "")
    &
    (df["target_address_norm"] != "")
    &
    (df["s1_address_norm"] == df["target_address_norm"])
)

address_exact_recall = (
    df["block_address_exact"].mean()
)


# =========================================================
# BLOCK D — SHARED ADDRESS TOKEN
# =========================================================

df["s1_address_tokens"] = (
    df["s1_business_address"]
    .map(tokens)
)

df["target_address_tokens"] = (
    df["target_business_address"]
    .map(tokens)
)

df["block_address_token"] = df.apply(
    lambda row:
        bool(
            row["s1_address_tokens"]
            &
            row["target_address_tokens"]
        ),
    axis=1
)

address_token_recall = (
    df["block_address_token"].mean()
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 100)
print("BLOCKING RECALL")
print("=" * 100)

print(
    f"Normalized name exact:        "
    f"{name_exact_recall:.2%}"
)

print(
    f"Shared name token:             "
    f"{name_token_recall:.2%}"
)

print(
    f"Normalized address exact:      "
    f"{address_exact_recall:.2%}"
)

print(
    f"Shared address token:          "
    f"{address_token_recall:.2%}"
)


# =========================================================
# SOURCE-SPECIFIC
# =========================================================

print("\n" + "=" * 100)
print("BLOCKING RECALL BY TARGET SOURCE")
print("=" * 100)

for source in ["S2", "S3"]:

    subset = df[
        df["target_source"] == source
    ]

    print(f"\n--- {source} ---")
    print(f"Pairs: {len(subset):,}")

    print(
        f"Normalized name exact:   "
        f"{subset['block_name_exact'].mean():.2%}"
    )

    print(
        f"Shared name token:        "
        f"{subset['block_name_token'].mean():.2%}"
    )

    print(
        f"Normalized address exact: "
        f"{subset['block_address_exact'].mean():.2%}"
    )

    print(
        f"Shared address token:      "
        f"{subset['block_address_token'].mean():.2%}"
    )


# =========================================================
# COMBINATIONS
# =========================================================

print("\n" + "=" * 100)
print("COMBINED BLOCKING RECALL")
print("=" * 100)

strategies = {

    "name_exact OR address_exact":
        (
            df["block_name_exact"]
            |
            df["block_address_exact"]
        ),

    "name_token OR address_exact":
        (
            df["block_name_token"]
            |
            df["block_address_exact"]
        ),

    "name_exact OR address_token":
        (
            df["block_name_exact"]
            |
            df["block_address_token"]
        ),

    "name_token OR address_token":
        (
            df["block_name_token"]
            |
            df["block_address_token"]
        ),
}

for name, mask in strategies.items():

    print(
        f"{name:<40}"
        f"{mask.mean():.2%}"
    )


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)