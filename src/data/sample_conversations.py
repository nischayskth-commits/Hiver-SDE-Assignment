import csv
from collections import defaultdict
from pathlib import Path


DATASET_PATH = Path("data/raw/twcs.csv")


# ---------------------------------------------------------
# Candidate brands
# ---------------------------------------------------------

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "VirginTrains",
    "XboxSupport",
]


# Number of examples to show per brand
SAMPLES_PER_BRAND = 5


def load_brand_tweets():

    brand_tweets = defaultdict(list)

    print("=" * 80)
    print("HIVER - SAMPLING REAL CUSTOMER SUPPORT CONVERSATIONS")
    print("=" * 80)

    print("\nPass 1: Finding brand support tweets...")

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            author = row.get("author_id", "").strip()

            if author in BRANDS:

                brand_tweets[author].append({
                    "tweet_id": row.get("tweet_id", "").strip(),
                    "text": row.get("text", "").strip(),
                    "response_tweet_id": row.get(
                        "response_tweet_id", ""
                    ).strip(),
                    "in_response_to_tweet_id": row.get(
                        "in_response_to_tweet_id", ""
                    ).strip(),
                })

    return brand_tweets


def collect_related_tweets(brand_tweets):

    # IDs we want to retrieve
    wanted_ids = set()

    for tweets in brand_tweets.values():

        for tweet in tweets:

            parent_id = tweet["in_response_to_tweet_id"]

            if parent_id:
                wanted_ids.add(parent_id)

    print(f"\nCustomer tweets to retrieve: {len(wanted_ids):,}")

    related = {}

    print("Pass 2: Retrieving linked customer tweets...")

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            tweet_id = row.get("tweet_id", "").strip()

            if tweet_id in wanted_ids:

                related[tweet_id] = {
                    "tweet_id": tweet_id,
                    "author_id": row.get(
                        "author_id", ""
                    ).strip(),
                    "inbound": row.get(
                        "inbound", ""
                    ).strip(),
                    "created_at": row.get(
                        "created_at", ""
                    ).strip(),
                    "text": row.get(
                        "text", ""
                    ).strip(),
                    "response_tweet_id": row.get(
                        "response_tweet_id", ""
                    ).strip(),
                    "in_response_to_tweet_id": row.get(
                        "in_response_to_tweet_id", ""
                    ).strip(),
                }

    return related


def display_examples(brand_tweets, related):

    print("\n" + "=" * 80)
    print("REAL CONVERSATION EXAMPLES")
    print("=" * 80)

    for brand in BRANDS:

        tweets = brand_tweets.get(brand, [])

        print("\n")
        print("#" * 80)
        print(f"BRAND: {brand}")
        print("#" * 80)

        shown = 0

        for brand_tweet in tweets:

            parent_id = brand_tweet["in_response_to_tweet_id"]

            if not parent_id:
                continue

            customer = related.get(parent_id)

            if not customer:
                continue

            print("\n--- Conversation Example ---")

            print("\nCUSTOMER:")
            print(customer["text"])

            print("\nBRAND:")
            print(brand_tweet["text"])

            print("\nCustomer tweet ID:")
            print(customer["tweet_id"])

            print("\nBrand tweet ID:")
            print(brand_tweet["tweet_id"])

            shown += 1

            if shown >= SAMPLES_PER_BRAND:
                break

        if shown == 0:
            print("\nNo linked examples found.")


def main():

    if not DATASET_PATH.exists():

        print(
            f"ERROR: Dataset not found at {DATASET_PATH}"
        )

        return

    brand_tweets = load_brand_tweets()

    related = collect_related_tweets(
        brand_tweets
    )

    display_examples(
        brand_tweets,
        related
    )


if __name__ == "__main__":
    main()