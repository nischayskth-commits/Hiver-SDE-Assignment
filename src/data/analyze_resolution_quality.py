import csv
import re
from collections import defaultdict
from pathlib import Path


DATASET_PATH = Path("data/raw/twcs.csv")


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


# Words/phrases commonly associated with generic escalation.
# This is exploratory analysis, NOT an AI classifier.
GENERIC_PATTERNS = [
    r"\bdm\b",
    r"direct message",
    r"send us a",
    r"send me a",
    r"contact us",
    r"get in touch",
    r"reach out",
    r"private message",
    r"customer service",
    r"we('re| are) here to help",
    r"we('d| would) be happy to help",
    r"we('ll| will) be in touch",
]


def is_generic_response(text):
    """
    Rough heuristic for identifying responses that mainly
    redirect the customer instead of providing a concrete
    resolution.

    This is only exploratory and should NOT be presented
    as a ground-truth metric.
    """

    text = text.lower().strip()

    if not text:
        return True

    matches = 0

    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, text):
            matches += 1

    # Very short responses are often not useful resolution evidence.
    if len(text.split()) < 8:
        return True

    # If response contains escalation language and is short,
    # classify it as likely generic.
    if matches >= 1 and len(text.split()) < 30:
        return True

    return False


def analyze():

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return

    # ---------------------------------------------------------
    # First pass: collect all tweets
    # ---------------------------------------------------------

    tweets = {}

    print("=" * 80)
    print("HIVER - HISTORICAL RESOLUTION QUALITY ANALYSIS")
    print("=" * 80)

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

            tweet_id = row.get("tweet_id", "").strip()

            if tweet_id:
                tweets[tweet_id] = {
                    "author_id": row.get("author_id", "").strip(),
                    "inbound": row.get("inbound", "").strip().lower(),
                    "text": row.get("text", "").strip(),
                    "in_response_to": row.get(
                        "in_response_to_tweet_id", ""
                    ).strip(),
                }

    print(f"Loaded {len(tweets):,} tweets.")

    # ---------------------------------------------------------
    # Analyze brand -> customer -> response relationships
    # ---------------------------------------------------------

    stats = defaultdict(
        lambda: {
            "support_responses": 0,
            "linked_customer_messages": 0,
            "generic_responses": 0,
            "substantive_responses": 0,
            "response_words": 0,
        }
    )

    for tweet in tweets.values():

        brand = tweet["author_id"]

        if brand not in BRANDS:
            continue

        if tweet["inbound"] == "true":
            continue

        parent_id = tweet["in_response_to"]

        if not parent_id:
            continue

        customer = tweets.get(parent_id)

        if not customer:
            continue

        # We only want brand -> customer interactions.
        if customer["inbound"] != "true":
            continue

        stats[brand]["support_responses"] += 1
        stats[brand]["linked_customer_messages"] += 1

        response_text = tweet["text"]

        stats[brand]["response_words"] += len(
            response_text.split()
        )

        if is_generic_response(response_text):
            stats[brand]["generic_responses"] += 1
        else:
            stats[brand]["substantive_responses"] += 1

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"\n{'Brand':<20}"
        f"{'Linked':>12}"
        f"{'Substantive':>15}"
        f"{'Generic':>12}"
        f"{'Substantive %':>16}"
    )

    print("-" * 80)

    for brand in BRANDS:

        s = stats[brand]

        total = s["support_responses"]

        substantive_pct = (
            s["substantive_responses"] / total * 100
            if total
            else 0
        )

        print(
            f"{brand:<20}"
            f"{total:>12,}"
            f"{s['substantive_responses']:>15,}"
            f"{s['generic_responses']:>12,}"
            f"{substantive_pct:>15.2f}%"
        )

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("IMPORTANT INTERPRETATION")
    print("=" * 80)

    print("""
The 'substantive %' value is an exploratory heuristic.

It does NOT mean that every response classified as
'substantive' is actually correct or useful.

It is being used only to help us compare candidate brands
before selecting one for the project.

We will manually inspect the selected brand and later create
a proper hand-labelled evaluation set.
""")

    print("=" * 80)


if __name__ == "__main__":
    analyze()