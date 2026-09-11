import csv
from collections import defaultdict, Counter
from pathlib import Path


DATASET_PATH = Path("data/raw/twcs.csv")


# Number of top brands we want to inspect
TOP_BRANDS = 30


def analyze_brands():

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return

    print("=" * 80)
    print("HIVER SDE ASSIGNMENT - BRAND ANALYSIS")
    print("=" * 80)

    # ---------------------------------------------------------
    # Statistics for each brand
    # ---------------------------------------------------------

    brand_stats = defaultdict(
        lambda: {
            "total": 0,
            "inbound": 0,
            "outbound": 0,
            "tweets_with_response": 0,
            "tweets_with_parent": 0,
        }
    )

    # ---------------------------------------------------------
    # Read dataset
    # ---------------------------------------------------------

    print("\nReading dataset...")

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

            if not author:
                continue

            inbound = row.get("inbound", "").strip().lower()

            stats = brand_stats[author]

            stats["total"] += 1

            if inbound == "true":
                stats["inbound"] += 1
            else:
                stats["outbound"] += 1

            if row.get("response_tweet_id", "").strip():
                stats["tweets_with_response"] += 1

            if row.get("in_response_to_tweet_id", "").strip():
                stats["tweets_with_parent"] += 1

    # ---------------------------------------------------------
    # Identify likely support brands
    #
    # We focus on authors with substantial outbound activity.
    # ---------------------------------------------------------

    brands = []

    for author, stats in brand_stats.items():

        if stats["outbound"] >= 5000:

            brands.append(
                (
                    author,
                    stats
                )
            )

    # Sort primarily by outbound support activity
    brands.sort(
        key=lambda item: item[1]["outbound"],
        reverse=True
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print(f"TOP {TOP_BRANDS} POTENTIAL SUPPORT BRANDS")
    print("=" * 80)

    print(
        f"\n{'Brand':<22}"
        f"{'Total':>12}"
        f"{'Inbound':>12}"
        f"{'Outbound':>12}"
        f"{'Replies':>12}"
        f"{'Parents':>12}"
    )

    print("-" * 80)

    for author, stats in brands[:TOP_BRANDS]:

        print(
            f"{author:<22}"
            f"{stats['total']:>12,}"
            f"{stats['inbound']:>12,}"
            f"{stats['outbound']:>12,}"
            f"{stats['tweets_with_response']:>12,}"
            f"{stats['tweets_with_parent']:>12,}"
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("WHAT THESE NUMBERS MEAN")
    print("=" * 80)

    print("""
Total:
    All tweets associated with the author.

Inbound:
    Customer messages directed toward the brand.

Outbound:
    Brand/support-agent messages.

Replies:
    Tweets that point toward one or more response tweets.

Parents:
    Tweets that are replies to another tweet.

We will use these statistics to select a brand
for our AI support agent.
""")

    print("=" * 80)


if __name__ == "__main__":
    analyze_brands()