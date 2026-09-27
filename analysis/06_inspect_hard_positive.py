import pandas as pd

FEATURES = "analysis/output/positive_pair_features.tsv"
SAMPLE = "analysis/output/positive_pairs_sample.tsv"

features = pd.read_csv(FEATURES, sep="\t")
pairs = pd.read_csv(SAMPLE, sep="\t")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 150)

# ---------------------------------------------------------
# Find the hardest pairs from the feature table
# ---------------------------------------------------------

hard_name = (
    features
    .sort_values("name_edit_similarity")
    .head(15)
)

hard_address = (
    features
    .sort_values("address_edit_similarity")
    .head(15)
)

features["combined_similarity"] = (
    0.5 * features["name_edit_similarity"]
    + 0.5 * features["address_edit_similarity"]
)

hard_combined = (
    features
    .sort_values("combined_similarity")
    .head(15)
)

# ---------------------------------------------------------
# Retrieve original text
# ---------------------------------------------------------

key_cols = [
    "source1_entity_id",
    "target_entity_id",
    "target_source",
]

# Only keep columns that exist
pair_cols = [
    "source1_entity_id",
    "target_entity_id",
    "target_source",
    "source1_business_name",
    "source1_business_address",
    "source1_country",
    "target_business_name",
    "target_business_address",
    "target_country",
]

available = [c for c in pair_cols if c in pairs.columns]

pairs_lookup = pairs[available].drop_duplicates(
    ["source1_entity_id", "target_entity_id", "target_source"]
)

def inspect(feature_subset, title):
    merged = feature_subset.merge(
        pairs_lookup,
        on=key_cols,
        how="left"
    )

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    columns = [
        "source1_entity_id",
        "target_entity_id",
        "target_source",
        "source1_business_name",
        "target_business_name",
        "source1_business_address",
        "target_business_address",
        "source1_country",
        "target_country",
        "name_edit_similarity",
        "name_token_jaccard",
        "name_token_containment",
        "address_edit_similarity",
        "address_token_jaccard",
        "address_token_containment",
        "combined_similarity",
    ]

    columns = [c for c in columns if c in merged.columns]

    print(
        merged[columns].to_string(index=False)
    )


inspect(
    hard_name,
    "15 LOWEST NAME-SIMILARITY TRUE MATCHES"
)

inspect(
    hard_address,
    "15 LOWEST ADDRESS-SIMILARITY TRUE MATCHES"
)

inspect(
    hard_combined,
    "15 LOWEST COMBINED-SIMILARITY TRUE MATCHES"
)

print("\n" + "=" * 120)
print("DONE")
print("=" * 120)