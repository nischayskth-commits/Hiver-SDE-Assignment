import csv
import random
from pathlib import Path

INPUT_FILE = Path("data/processed/virgintrains_support_pairs.csv")
OUTPUT_FILE = Path("data/golden/golden_set_v2.csv")

TARGET_PER_CATEGORY = 18
RANDOM_SEED = 42


KEYWORD_CATEGORIES = {
    "train_status": [
        "delay",
        "delayed",
        "late",
        "running",
        "cancelled",
        "canceled",
        "disruption",
        "service disruption",
        "train status",
        "train cancelled",
        "train delayed",
        "missed train",
        "next train"
    ],

    "ticket_change": [
        "change my ticket",
        "change ticket",
        "change booking",
        "change date",
        "change journey",
        "amend my ticket",
        "amend ticket",
        "amendment",
        "wrong date",
        "wrong ticket",
        "change reservation"
    ],

    "refund_compensation": [
        "refund",
        "refunded",
        "compensation",
        "delay repay",
        "delayrepay",
        "repay",
        "money back",
        "reimbursement",
        "claim for delay",
        "claim compensation"
    ],

    "booking_ticket": [
        "book a ticket",
        "book ticket",
        "booking",
        "booked",
        "buy a ticket",
        "buy ticket",
        "purchase ticket",
        "ticket",
        "tickets"
    ],

    "fare_price": [
        "price",
        "cost",
        "fare",
        "expensive",
        "cheap",
        "advance fare",
        "advance ticket",
        "charge",
        "fee",
        "£",
        "pound"
    ],

    "seat_reservation": [
        "seat",
        "reserved seat",
        "seat reservation",
        "reserve a seat",
        "reserved seating",
        "window seat",
        "aisle seat",
        "seat number"
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "internet",
        "internet connection",
        "connectivity",
        "online",
        "wireless"
    ],

    "station_facilities": [
        "station",
        "platform",
        "lounge",
        "toilet",
        "bathroom",
        "facilities",
        "waiting room",
        "station staff",
        "station entrance"
    ],

    "lost_property": [
        "lost",
        "lost property",
        "missing",
        "left behind",
        "left my",
        "lost item",
        "lost phone",
        "lost bag",
        "lost luggage",
        "lost coat",
        "lost kindle"
    ],

    "complaint_feedback": [
        "complaint",
        "complain",
        "complaining",
        "poor service",
        "terrible service",
        "awful service",
        "disappointed",
        "disappointing",
        "unhappy",
        "feedback"
    ],
}


def matching_categories(message):
    """
    Return ALL candidate categories whose keywords occur.

    This is candidate sampling only.
    It is NOT the final intent label.
    """
    text = message.lower()

    matches = []

    for category, keywords in KEYWORD_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                matches.append(category)
                break

    return matches


def main():

    random.seed(RANDOM_SEED)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    print("Reading dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total available support pairs: {len(rows)}")

    # ---------------------------------------------------------
    # Put every message into every category it potentially matches
    # ---------------------------------------------------------

    category_rows = {
        category: []
        for category in KEYWORD_CATEGORIES
    }

    unmatched_rows = []

    for row in rows:

        message = row["customer_message"].strip()

        if not message:
            continue

        matches = matching_categories(message)

        if not matches:
            unmatched_rows.append(row)

        for category in matches:
            category_rows[category].append(row)

    # ---------------------------------------------------------
    # Display candidate distribution
    # ---------------------------------------------------------

    print("\nCandidate distribution:")

    for category, items in category_rows.items():
        print(f"{category:25s}: {len(items)}")

    print(f"{'unmatched':25s}: {len(unmatched_rows)}")

    # ---------------------------------------------------------
    # Select examples
    # ---------------------------------------------------------

    selected = {}

    for category, items in category_rows.items():

        random.shuffle(items)

        count = 0

        for row in items:

            tweet_id = row["customer_tweet_id"]

            if tweet_id not in selected:
                selected[tweet_id] = row
                count += 1

            if count >= TARGET_PER_CATEGORY:
                break

    # ---------------------------------------------------------
    # Add unmatched examples
    # These are useful for the "other" class.
    # ---------------------------------------------------------

    random.shuffle(unmatched_rows)

    for row in unmatched_rows[:25]:

        tweet_id = row["customer_tweet_id"]

        if tweet_id not in selected:
            selected[tweet_id] = row

    # ---------------------------------------------------------
    # Add very short messages
    # These are intentionally difficult examples.
    # ---------------------------------------------------------

    short_rows = [
        row
        for row in rows
        if 1 <= len(row["customer_message"].split()) <= 5
    ]

    random.shuffle(short_rows)

    for row in short_rows[:30]:

        tweet_id = row["customer_tweet_id"]

        if tweet_id not in selected:
            selected[tweet_id] = row

    # ---------------------------------------------------------
    # Convert dictionary to list
    # ---------------------------------------------------------

    final_rows = list(selected.values())

    random.shuffle(final_rows)

    # Maximum 200
    final_rows = final_rows[:200]

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "customer_message",
        "historical_response",
        "intent",
        "expected_resolution",
        "should_escalate",
        "escalation_reason"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for i, row in enumerate(final_rows, start=1):

            writer.writerow({
                "id": i,
                "customer_message": row["customer_message"],
                "historical_response": row["brand_response"],
                "intent": "",
                "expected_resolution": "",
                "should_escalate": "",
                "escalation_reason": ""
            })

    print("\n========================================")
    print("SUCCESS!")
    print("========================================")

    print(f"Golden set created:")
    print(f"  {OUTPUT_FILE}")

    print(f"\nNumber of examples:")
    print(f"  {len(final_rows)}")

    print("\nIMPORTANT:")
    print("Keyword categories were used ONLY for sampling.")
    print("They are NOT the final ground-truth labels.")

    print("\nNext step:")
    print("We will manually label the golden set.")


if __name__ == "__main__":
    main()