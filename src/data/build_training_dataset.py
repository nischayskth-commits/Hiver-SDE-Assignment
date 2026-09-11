import csv
from pathlib import Path
from collections import Counter


PROCESSED_FILE = Path("data/processed/virgintrains_support_pairs.csv")
GOLDEN_FILE = Path("data/golden/golden_set_final.csv")
OUTPUT_FILE = Path("data/processed/training_pairs.csv")


# These are candidate labels for silver training data.
# They are NOT treated as human ground truth.
INTENT_RULES = {
    "train_status": [
        "delay", "delayed", "late", "cancelled", "cancelled train",
        "cancellation", "running", "platform", "disruption",
        "disrupted", "service", "train running", "train status"
    ],

    "ticket_change": [
        "change ticket", "change my ticket", "change date",
        "change the date", "amend", "amendment", "wrong date",
        "move my ticket", "move the ticket", "different date",
        "transfer my ticket"
    ],

    "refund_compensation": [
        "refund", "refunds", "compensation", "delay repay",
        "delayrepay", "repay", "money back", "reimburse",
        "compensate"
    ],

    "booking_ticket": [
        "ticket", "tickets", "book ticket", "booking",
        "booked", "valid ticket", "ticket valid", "ticket accepted",
        "collect ticket", "collect my ticket", "ticket collection"
    ],

    "fare_price": [
        "price", "prices", "fare", "fares", "cost", "cheap",
        "cheapest", "expensive", "advance ticket", "advance tickets",
        "off peak", "off-peak", "peak ticket", "ticket sale"
    ],

    "seat_reservation": [
        "seat", "seats", "reserved seat", "seat reservation",
        "reservation", "reserved", "unreserved", "coach",
        "carriage", "table seat", "plug seat"
    ],

    "wifi_connectivity": [
        "wifi", "wi-fi", "wi fi", "internet", "connection",
        "connect", "connected", "wifi code", "wifi password"
    ],

    "station_facilities": [
        "station", "lounge", "toilet", "toilets", "platform",
        "station staff", "car park", "parking", "facilities"
    ],

    "lost_property": [
        "lost", "lost property", "lost item", "left my",
        "left behind", "missing item", "found item", "lost phone",
        "lost bag", "lost coat", "lost kindle"
    ],

    "complaint_feedback": [
        "complaint", "complain", "feedback", "poor service",
        "terrible service", "bad service", "rude", "unhelpful",
        "customer service", "staff complaint", "report"
    ],
}


def normalize(text):
    return " ".join(text.lower().split())


def candidate_labels(message):
    """
    Return all matching candidate intents.

    We deliberately allow multiple matches because words such as
    'train', 'ticket', and 'station' occur across many intents.
    """
    text = normalize(message)

    matches = []

    for intent, keywords in INTENT_RULES.items():
        if any(keyword in text for keyword in keywords):
            matches.append(intent)

    return matches


def load_golden_pairs():
    """
    Build a set of (customer_message, historical_response) pairs
    belonging to the golden evaluation set.

    Matching the actual text pair is safer than assuming that the
    golden-set ID corresponds to a row number in the processed file.
    """
    golden_pairs = set()

    with GOLDEN_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            customer = normalize(row["customer_message"])
            response = normalize(row["historical_response"])

            golden_pairs.add((customer, response))

    return golden_pairs


def main():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_FILE}"
        )

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    golden_pairs = load_golden_pairs()

    print(f"Golden evaluation pairs protected: {len(golden_pairs)}")

    total = 0
    excluded = 0
    kept = 0
    skipped_ambiguous = 0
    skipped_unmatched = 0

    label_counts = Counter()

    with PROCESSED_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as source, OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as target:

        reader = csv.DictReader(source)

        fieldnames = [
            "customer_tweet_id",
            "brand_tweet_id",
            "created_at",
            "customer_message",
            "brand_response",
            "silver_intent",
        ]

        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            total += 1

            customer = normalize(row["customer_message"])
            response = normalize(row["brand_response"])

            # -------------------------------------------------
            # Leakage protection
            # -------------------------------------------------
            if (customer, response) in golden_pairs:
                excluded += 1
                continue

            matches = candidate_labels(row["customer_message"])

            # Only keep examples where our weak labeling rules
            # produce exactly one intent.
            if len(matches) == 0:
                skipped_unmatched += 1
                continue

            if len(matches) > 1:
                skipped_ambiguous += 1
                continue

            intent = matches[0]

            writer.writerow({
                "customer_tweet_id": row["customer_tweet_id"],
                "brand_tweet_id": row["brand_tweet_id"],
                "created_at": row["created_at"],
                "customer_message": row["customer_message"],
                "brand_response": row["brand_response"],
                "silver_intent": intent,
            })

            kept += 1
            label_counts[intent] += 1

    print()
    print("=" * 60)
    print("TRAINING DATASET CREATED")
    print("=" * 60)
    print(f"Total historical pairs:       {total:,}")
    print(f"Golden pairs excluded:        {excluded:,}")
    print(f"Ambiguous examples skipped:   {skipped_ambiguous:,}")
    print(f"Unmatched examples skipped:   {skipped_unmatched:,}")
    print(f"Training examples kept:       {kept:,}")
    print()
    print("Silver-label distribution:")

    for intent, count in label_counts.most_common():
        print(f"  {intent:<25} {count:,}")

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()