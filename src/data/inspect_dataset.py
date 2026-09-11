import csv
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_PATH = Path("data/raw/twcs.csv")


# ---------------------------------------------------------
# Main dataset inspection
# ---------------------------------------------------------

def inspect_dataset():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return

    print("=" * 70)
    print("HIVER SDE ASSIGNMENT - DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset: {DATASET_PATH}")
    print(f"File size: {DATASET_PATH.stat().st_size / (1024 * 1024):.2f} MB")

    row_count = 0
    inbound_counts = Counter()
    author_counts = Counter()

    missing_values = Counter()

    min_date = None
    max_date = None

    sample_rows = []

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        print("\nColumns:")
        print("-" * 70)

        for column in reader.fieldnames:
            print(f"  - {column}")

        print("\nReading dataset...")

        for row in reader:
            row_count += 1

            # Store first few rows for inspection
            if len(sample_rows) < 5:
                sample_rows.append(row)

            # Count inbound/outbound tweets
            inbound_counts[row.get("inbound", "")] += 1

            # Count tweets per author
            author_id = row.get("author_id", "")
            if author_id:
                author_counts[author_id] += 1

            # Missing values
            for key, value in row.items():
                if value is None or value.strip() == "":
                    missing_values[key] += 1

            # Date range
            created_at = row.get("created_at", "")

            if created_at:
                if min_date is None or created_at < min_date:
                    min_date = created_at

                if max_date is None or created_at > max_date:
                    max_date = created_at

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"\nTotal tweets: {row_count:,}")

    print("\nDate range:")
    print(f"  Earliest: {min_date}")
    print(f"  Latest:   {max_date}")

    print("\nInbound / Outbound:")
    for value, count in inbound_counts.items():
        label = "Customer / inbound" if value == "True" else "Brand / outbound"
        print(f"  {label}: {count:,}")

    print("\nUnique authors:")
    print(f"  {len(author_counts):,}")

    print("\nTop authors by number of tweets:")
    print("-" * 70)

    for author_id, count in author_counts.most_common(20):
        print(f"  {author_id}: {count:,} tweets")

    print("\nMissing values:")
    print("-" * 70)

    for column, count in missing_values.items():
        percentage = (count / row_count * 100) if row_count else 0
        print(f"  {column}: {count:,} ({percentage:.2f}%)")

    print("\nSample records:")
    print("=" * 70)

    for index, row in enumerate(sample_rows, start=1):
        print(f"\n--- Sample {index} ---")

        for key, value in row.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    inspect_dataset()