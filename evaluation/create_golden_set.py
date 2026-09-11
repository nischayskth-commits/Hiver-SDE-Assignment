import csv
import random
import os


INPUT_FILE = "data/processed/virgintrains_support_pairs.csv"
OUTPUT_FILE = "data/golden/golden_set.csv"

SAMPLE_SIZE = 200

random.seed(42)


def main():

    print("=" * 80)
    print("CREATING GOLDEN EVALUATION SET")
    print("=" * 80)

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["customer_message"].strip():
                rows.append(row)

    print(f"\nAvailable conversations: {len(rows):,}")

    # Random sample with fixed seed for reproducibility
    sample = random.sample(
        rows,
        min(SAMPLE_SIZE, len(rows))
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        fieldnames = [
            "id",
            "customer_message",
            "historical_response",
            "intent",
            "expected_resolution",
            "should_escalate",
            "escalation_reason"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index, row in enumerate(sample, start=1):

            writer.writerow({
                "id": index,
                "customer_message": row["customer_message"],
                "historical_response": row["brand_response"],
                "intent": "",
                "expected_resolution": "",
                "should_escalate": "",
                "escalation_reason": ""
            })

    print(f"\nGolden set created: {len(sample)} examples")

    print(f"Output:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("NEXT STEP")
    print("=" * 80)

    print("""
We will manually label:
    - intent
    - expected resolution
    - should_escalate
    - escalation reason
""")


if __name__ == "__main__":
    main()