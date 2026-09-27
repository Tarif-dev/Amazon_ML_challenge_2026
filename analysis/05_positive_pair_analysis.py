from pathlib import Path
import re
import unicodedata

import pandas as pd


INPUT_PATH = Path(
    "analysis/output/positive_pairs_sample.tsv"
)

OUTPUT_PATH = Path(
    "analysis/output/positive_pair_features.tsv"
)


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text):
    """
    Unicode canonicalization.

    NFKC converts visually/structurally equivalent Unicode
    representations into a more consistent representation.
    """

    if not text:
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_case(text):
    """
    Unicode-aware lowercasing.
    """

    return text.casefold()


def normalize_whitespace(text):
    """
    Collapse repeated whitespace.
    """

    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text):
    """
    Replace punctuation with spaces.

    IMPORTANT:
    We are NOT removing characters aggressively yet.
    This is an experiment.
    """

    return re.sub(r"[^\w\s]", " ", text)


def basic_normalize(text):
    """
    First experimental normalization pipeline.
    """

    text = normalize_unicode(text)
    text = normalize_case(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    return text


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def tokens(text):
    return basic_normalize(text).split()


def sorted_tokens(text):
    return " ".join(sorted(tokens(text)))


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def levenshtein_distance(a, b):
    """
    Standard Levenshtein edit distance.

    Memory optimized to O(min(len(a), len(b))).
    """

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    if len(a) > len(b):
        a, b = b, a

    previous = list(range(len(a) + 1))

    for i, char_b in enumerate(b, start=1):

        current = [i]

        for j, char_a in enumerate(a, start=1):

            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (
                char_a != char_b
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost,
                )
            )

        previous = current

    return previous[-1]


def normalized_edit_similarity(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    distance = levenshtein_distance(a, b)

    max_len = max(len(a), len(b))

    return 1.0 - distance / max_len


# ============================================================
# TOKEN SIMILARITY
# ============================================================

def jaccard_similarity(a, b):

    a_tokens = set(tokens(a))
    b_tokens = set(tokens(b))

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    return intersection / union


def token_containment(a, b):

    a_tokens = set(tokens(a))
    b_tokens = set(tokens(b))

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)

    return intersection / min(
        len(a_tokens),
        len(b_tokens),
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_features(row):

    s1_name = row["s1_business_name"]
    s2_name = row["target_business_name"]

    s1_address = row["s1_business_address"]
    s2_address = row["target_business_address"]

    s1_name_norm = basic_normalize(s1_name)
    s2_name_norm = basic_normalize(s2_name)

    s1_address_norm = basic_normalize(s1_address)
    s2_address_norm = basic_normalize(s2_address)

    return {

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "source1_entity_id":
            row["source1_entity_id"],

        "target_entity_id":
            row["target_entity_id"],

        "target_source":
            row["target_source"],

        "match_count_for_s1":
            row["match_count_for_s1"],

        # ----------------------------------------------------
        # Country
        # ----------------------------------------------------

        "country_same":
            int(
                row["s1_country"].casefold()
                == row["target_country"].casefold()
            ),

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        "name_raw_exact":
            int(s1_name == s2_name),

        "name_casefold_exact":
            int(
                s1_name.casefold()
                == s2_name.casefold()
            ),

        "name_normalized_exact":
            int(
                s1_name_norm
                == s2_name_norm
            ),

        "name_sorted_token_exact":
            int(
                sorted_tokens(s1_name)
                == sorted_tokens(s2_name)
            ),

        "name_edit_similarity":
            normalized_edit_similarity(
                s1_name_norm,
                s2_name_norm,
            ),

        "name_token_jaccard":
            jaccard_similarity(
                s1_name,
                s2_name,
            ),

        "name_token_containment":
            token_containment(
                s1_name,
                s2_name,
            ),

        "name_len_s1":
            len(s1_name),

        "name_len_target":
            len(s2_name),

        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        "address_raw_exact":
            int(s1_address == s2_address),

        "address_casefold_exact":
            int(
                s1_address.casefold()
                == s2_address.casefold()
            ),

        "address_normalized_exact":
            int(
                s1_address_norm
                == s2_address_norm
            ),

        "address_edit_similarity":
            normalized_edit_similarity(
                s1_address_norm,
                s2_address_norm,
            ),

        "address_token_jaccard":
            jaccard_similarity(
                s1_address,
                s2_address,
            ),

        "address_token_containment":
            token_containment(
                s1_address,
                s2_address,
            ),

        "address_len_s1":
            len(s1_address),

        "address_len_target":
            len(s2_address),
    }


def main():

    print("=" * 90)
    print("POSITIVE PAIR ANALYSIS")
    print("=" * 90)

    print(f"\nLoading: {INPUT_PATH}")

    pairs = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Positive pairs: {len(pairs):,}")

    print("\nComputing features...")

    features = [
        compute_features(row)
        for _, row in pairs.iterrows()
    ]

    result = pd.DataFrame(features)

    result.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved features to:")
    print(OUTPUT_PATH)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 90)
    print("FEATURE SUMMARY")
    print("=" * 90)

    binary_features = [
        "country_same",
        "name_raw_exact",
        "name_casefold_exact",
        "name_normalized_exact",
        "name_sorted_token_exact",
        "address_raw_exact",
        "address_casefold_exact",
        "address_normalized_exact",
    ]

    print("\nExact-match rates:")

    for column in binary_features:

        rate = result[column].mean()

        print(
            f"{column:30s}"
            f"{rate:8.3%}"
        )

    similarity_features = [
        "name_edit_similarity",
        "name_token_jaccard",
        "name_token_containment",
        "address_edit_similarity",
        "address_token_jaccard",
        "address_token_containment",
    ]

    print("\nSimilarity distributions:")

    print(
        result[similarity_features]
        .describe(
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
        .T
        .to_string()
    )

    # ========================================================
    # COUNTRY
    # ========================================================

    print("\n" + "=" * 90)
    print("COUNTRY CONSISTENCY")
    print("=" * 90)

    print(
        result["country_same"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # BY SOURCE
    # ========================================================

    print("\n" + "=" * 90)
    print("S2 VS S3")
    print("=" * 90)

    for source, group in result.groupby(
        "target_source"
    ):

        print(f"\n--- {source} ---")

        print(
            f"Pairs: {len(group):,}"
        )

        print(
            f"Name normalized exact: "
            f"{group['name_normalized_exact'].mean():.3%}"
        )

        print(
            f"Address normalized exact: "
            f"{group['address_normalized_exact'].mean():.3%}"
        )

        print(
            f"Name edit similarity median: "
            f"{group['name_edit_similarity'].median():.3f}"
        )

        print(
            f"Address edit similarity median: "
            f"{group['address_edit_similarity'].median():.3f}"
        )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()