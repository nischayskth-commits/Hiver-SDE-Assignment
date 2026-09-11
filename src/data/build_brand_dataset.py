import csv
import os

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/virgintrains_support_pairs.csv"

TARGET_BRAND = "VirginTrains"


def main():
    print("=" * 80)
    print("BUILDING VIRGINTRAINS SUPPORT DATASET")
    print("=" * 80)

    # ---------------------------------------------------------
    # PASS 1: Store customer tweets
    # ---------------------------------------------------------

    customer_tweets = {}

    print("\nPass 1: Reading customer tweets...")

    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["inbound"].lower() == "true":
                customer_tweets[row["tweet_id"]] = row

    print(f"Customer tweets stored: {len(customer_tweets):,}")

    # ---------------------------------------------------------
    # PASS 2: Find VirginTrains responses
    # ---------------------------------------------------------

    pairs = []

    print("\nPass 2: Finding VirginTrains responses...")

    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:

            # We only want VirginTrains outbound messages
            if row["author_id"] != TARGET_BRAND:
                continue

            parent_id = row["in_response_to_tweet_id"]

            if not parent_id:
                continue

            # Find the customer tweet being answered
            customer = customer_tweets.get(parent_id)

            if customer is None:
                continue

            pairs.append({
                "customer_tweet_id": customer["tweet_id"],
                "brand_tweet_id": row["tweet_id"],
                "created_at": customer["created_at"],
                "customer_message": customer["text"],
                "brand_response": row["text"],
            })

    print(f"Support pairs found: {len(pairs):,}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        fieldnames = [
            "customer_tweet_id",
            "brand_tweet_id",
            "created_at",
            "customer_message",
            "brand_response",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(pairs)

    print("\nDataset created successfully.")
    print(f"Output: {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()