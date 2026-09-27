from pathlib import Path
import pandas as pd


GT_PATH = Path("student_resource/dataset/train/train_ground_truth.tsv")


def main():
    print("=" * 90)
    print("GROUND TRUTH TARGET ANALYSIS")
    print("=" * 90)

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2_ids = set()
    s3_ids = set()

    for value in gt["matched_entity_ids"]:
        if not value:
            continue

        for entity_id in value.split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                s3_ids.add(entity_id)

    print(f"\nUnique S2 IDs referenced by ground truth: {len(s2_ids):,}")
    print(f"Unique S3 IDs referenced by ground truth: {len(s3_ids):,}")

    print("\nExpected target coverage:")

    print(
        f"S2 GT coverage: "
        f"{len(s2_ids):,} unique IDs"
    )

    print(
        f"S3 GT coverage: "
        f"{len(s3_ids):,} unique IDs"
    )

    print("\nNow checking overlap between S2 and S3 IDs...")

    overlap = s2_ids & s3_ids

    print(f"S2/S3 ID overlap: {len(overlap):,}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()