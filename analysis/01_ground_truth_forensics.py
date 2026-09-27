from pathlib import Path
from collections import Counter
import pandas as pd


GT_PATH = Path("student_resource/dataset/train/train_ground_truth.tsv")


def main():
    print("=" * 80)
    print("GROUND TRUTH FORENSICS")
    print("=" * 80)

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"\nRows: {len(gt):,}")
    print(f"Columns: {list(gt.columns)}")

    # ---------------------------------------------------------
    # 1. Check Source 1 uniqueness
    # ---------------------------------------------------------
    duplicate_s1 = gt["source1_entity_id"].duplicated().sum()

    print("\n[1] SOURCE 1 DUPLICATES")
    print(f"Duplicate S1 rows: {duplicate_s1:,}")

    # ---------------------------------------------------------
    # 2. Number of matched IDs per S1
    # ---------------------------------------------------------
    def parse_matches(value):
        if not value.strip():
            return []

        return [x.strip() for x in value.split(",") if x.strip()]

    gt["matches"] = gt["matched_entity_ids"].map(parse_matches)

    gt["match_count"] = gt["matches"].map(len)

    print("\n[2] MATCH COUNT DISTRIBUTION")
    print(gt["match_count"].describe())

    print("\nExact distribution:")
    print(
        gt["match_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # 3. Singleton rate
    # ---------------------------------------------------------
    singleton_count = (gt["match_count"] == 0).sum()

    singleton_rate = singleton_count / len(gt)

    print("\n[3] SINGLETONS")
    print(f"Singletons: {singleton_count:,}")
    print(f"Singleton rate: {singleton_rate:.4%}")

    # ---------------------------------------------------------
    # 4. Count S2 vs S3 matches
    # ---------------------------------------------------------
    s2_counts = []
    s3_counts = []

    for matches in gt["matches"]:
        s2 = sum(x.startswith("S2-") for x in matches)
        s3 = sum(x.startswith("S3-") for x in matches)

        s2_counts.append(s2)
        s3_counts.append(s3)

    gt["s2_count"] = s2_counts
    gt["s3_count"] = s3_counts

    print("\n[4] S2 / S3 MATCH DISTRIBUTION")

    print("\nS2 matches per S1:")
    print(
        gt["s2_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nS3 matches per S1:")
    print(
        gt["s3_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # 5. Multiple matches from same source
    # ---------------------------------------------------------
    multiple_s2 = (gt["s2_count"] > 1).sum()
    multiple_s3 = (gt["s3_count"] > 1).sum()

    print("\n[5] MULTIPLE MATCHES FROM SAME SOURCE")

    print(f"S1 entities with >1 S2 match: {multiple_s2:,}")
    print(f"S1 entities with >1 S3 match: {multiple_s3:,}")

    # ---------------------------------------------------------
    # 6. Global ID reuse
    # ---------------------------------------------------------
    print("\n[6] TARGET ID REUSE")

    s2_counter = Counter()
    s3_counter = Counter()

    for matches in gt["matches"]:
        for match_id in matches:
            if match_id.startswith("S2-"):
                s2_counter[match_id] += 1
            elif match_id.startswith("S3-"):
                s3_counter[match_id] += 1

    reused_s2 = sum(count > 1 for count in s2_counter.values())
    reused_s3 = sum(count > 1 for count in s3_counter.values())

    print(f"Unique S2 IDs appearing in GT: {len(s2_counter):,}")
    print(f"S2 IDs reused across S1 rows: {reused_s2:,}")

    print(f"Unique S3 IDs appearing in GT: {len(s3_counter):,}")
    print(f"S3 IDs reused across S1 rows: {reused_s3:,}")

    # ---------------------------------------------------------
    # 7. Maximum match counts
    # ---------------------------------------------------------
    print("\n[7] MAXIMUM MATCH COUNTS")

    print(f"Maximum total matches for one S1: {gt['match_count'].max()}")
    print(f"Maximum S2 matches for one S1: {gt['s2_count'].max()}")
    print(f"Maximum S3 matches for one S1: {gt['s3_count'].max()}")

    # ---------------------------------------------------------
    # 8. Show examples of complex entities
    # ---------------------------------------------------------
    print("\n[8] EXAMPLES")

    complex_rows = (
        gt[gt["match_count"] >= 4]
        .sort_values("match_count", ascending=False)
        .head(10)
    )

    for _, row in complex_rows.iterrows():
        print(
            f"\n{row['source1_entity_id']}"
            f"\n  total matches: {row['match_count']}"
            f"\n  S2 matches: {row['s2_count']}"
            f"\n  S3 matches: {row['s3_count']}"
            f"\n  IDs: {row['matched_entity_ids']}"
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()