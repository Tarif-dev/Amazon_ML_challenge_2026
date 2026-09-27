from pathlib import Path

import pandas as pd


DATASET = Path("student_resource/dataset")

GT_PATH = DATASET / "train/train_ground_truth.tsv"
S1_PATH = DATASET / "train/train_source1.tsv"
S2_PATH = DATASET / "train/train_source2.tsv"
S3_PATH = DATASET / "train/train_source3.tsv"

OUTPUT_DIR = Path("analysis/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_SEED = 42

# Number of S1 entities sampled from each match-count bucket.
PER_BUCKET = 200

# Chunk size chosen conservatively for a 16 GB RAM machine.
CHUNK_SIZE = 250_000


def load_ground_truth_sample():
    print("=" * 90)
    print("STEP 1 — SAMPLING GROUND TRUTH")
    print("=" * 90)

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt["matches"] = gt["matched_entity_ids"].map(
        lambda x: [
            value.strip()
            for value in x.split(",")
            if value.strip()
        ]
        if x.strip()
        else []
    )

    gt["match_count"] = gt["matches"].str.len()

    # ---------------------------------------------------------
    # Stratified sampling by number of true matches.
    # ---------------------------------------------------------
    samples = []

    for match_count, group in gt.groupby("match_count"):
        n = min(PER_BUCKET, len(group))

        sampled = group.sample(
            n=n,
            random_state=RANDOM_SEED + int(match_count),
        )

        samples.append(sampled)

        print(
            f"match_count={match_count:2d} "
            f"available={len(group):8,} "
            f"sampled={n:4,}"
        )

    sample = pd.concat(samples, ignore_index=True)

    print(f"\nTotal S1 entities sampled: {len(sample):,}")

    return sample


def load_source1_records(sample):
    print("\n" + "=" * 90)
    print("STEP 2 — LOADING S1 SAMPLE")
    print("=" * 90)

    s1_ids = set(sample["source1_entity_id"])

    result_parts = []

    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        mask = chunk["entity_id"].isin(s1_ids)

        if mask.any():
            result_parts.append(chunk.loc[mask])

    s1 = pd.concat(result_parts, ignore_index=True)

    print(f"S1 records retrieved: {len(s1):,}")

    return s1


def extract_target_ids(sample):
    s2_ids = set()
    s3_ids = set()

    for matches in sample["matches"]:
        for entity_id in matches:
            if entity_id.startswith("S2-"):
                s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                s3_ids.add(entity_id)

    print("\n" + "=" * 90)
    print("STEP 3 — TARGET IDS TO RETRIEVE")
    print("=" * 90)

    print(f"S2 IDs requested: {len(s2_ids):,}")
    print(f"S3 IDs requested: {len(s3_ids):,}")

    return s2_ids, s3_ids


def retrieve_records(path, target_ids, source_name):
    print("\n" + "=" * 90)
    print(f"STEP 4 — RETRIEVING {source_name}")
    print("=" * 90)

    result_parts = []

    processed = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        processed += len(chunk)

        mask = chunk["entity_id"].isin(target_ids)

        if mask.any():
            result_parts.append(chunk.loc[mask])

        if processed % 1_000_000 < CHUNK_SIZE:
            print(
                f"Processed {processed:,} rows...",
                flush=True,
            )

    if result_parts:
        result = pd.concat(result_parts, ignore_index=True)
    else:
        result = pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        )

    print(f"{source_name} records retrieved: {len(result):,}")

    missing = target_ids - set(result["entity_id"])

    if missing:
        print(
            f"WARNING: {len(missing):,} requested IDs "
            f"were not found."
        )
    else:
        print("All requested IDs were found.")

    return result


def build_pair_table(sample, s1, s2, s3):
    print("\n" + "=" * 90)
    print("STEP 5 — BUILDING POSITIVE PAIR TABLE")
    print("=" * 90)

    s1_lookup = s1.set_index("entity_id").to_dict("index")
    s2_lookup = s2.set_index("entity_id").to_dict("index")
    s3_lookup = s3.set_index("entity_id").to_dict("index")

    rows = []

    for _, gt_row in sample.iterrows():

        s1_id = gt_row["source1_entity_id"]

        s1_record = s1_lookup[s1_id]

        for target_id in gt_row["matches"]:

            if target_id.startswith("S2-"):
                target_record = s2_lookup[target_id]
                source = "S2"

            else:
                target_record = s3_lookup[target_id]
                source = "S3"

            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "target_entity_id": target_id,
                    "target_source": source,

                    "s1_business_name":
                        s1_record["business_name"],

                    "s1_business_address":
                        s1_record["business_address"],

                    "s1_country":
                        s1_record["country"],

                    "target_business_name":
                        target_record["business_name"],

                    "target_business_address":
                        target_record["business_address"],

                    "target_country":
                        target_record["country"],

                    "match_count_for_s1":
                        gt_row["match_count"],
                }
            )

    pairs = pd.DataFrame(rows)

    print(f"Positive pairs created: {len(pairs):,}")

    return pairs


def main():

    # ---------------------------------------------------------
    # 1. Sample S1 entities
    # ---------------------------------------------------------
    sample = load_ground_truth_sample()

    # ---------------------------------------------------------
    # 2. Retrieve S1 records
    # ---------------------------------------------------------
    s1 = load_source1_records(sample)

    # ---------------------------------------------------------
    # 3. Determine target IDs
    # ---------------------------------------------------------
    s2_ids, s3_ids = extract_target_ids(sample)

    # ---------------------------------------------------------
    # 4. Retrieve only those target records
    # ---------------------------------------------------------
    s2 = retrieve_records(
        S2_PATH,
        s2_ids,
        "Source 2",
    )

    s3 = retrieve_records(
        S3_PATH,
        s3_ids,
        "Source 3",
    )

    # ---------------------------------------------------------
    # 5. Build pair table
    # ---------------------------------------------------------
    pairs = build_pair_table(
        sample,
        s1,
        s2,
        s3,
    )

    # ---------------------------------------------------------
    # 6. Save
    # ---------------------------------------------------------
    output_path = (
        OUTPUT_DIR / "positive_pairs_sample.tsv"
    )

    pairs.to_csv(
        output_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    print(f"\nSaved to:")
    print(output_path)

    print("\nPair distribution by source:")
    print(
        pairs["target_source"]
        .value_counts()
        .to_string()
    )

    print("\nPair distribution by S1 match count:")
    print(
        pairs["match_count_for_s1"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()