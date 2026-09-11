import csv
import re
from pathlib import Path
from collections import Counter


PROCESSED_FILE = Path("data/processed/virgintrains_support_pairs.csv")
GOLDEN_FILE = Path("data/golden/golden_set_final.csv")
OUTPUT_FILE = Path("data/processed/training_pairs_v2.csv")


def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


# ------------------------------------------------------------
# Strong intent signals
#
# These rules are deliberately conservative.
# Broad words such as "train", "ticket", and "station" are
# NOT sufficient by themselves.
# ------------------------------------------------------------

RULES = {

    "ticket_change": {
        "customer": [
            "change my ticket",
            "change the ticket",
            "change ticket",
            "change my booking",
            "change booking",
            "change the date",
            "change date",
            "wrong date",
            "wrong day",
            "booked the wrong",
            "amend my ticket",
            "amend ticket",
            "amend my booking",
            "amend booking",
            "move my ticket",
            "move the ticket",
            "move my booking",
            "move booking",
            "transfer my ticket",
            "transfer the ticket",
            "different date",
            "another date",
            "change journey",
            "change my journey",
            "miss my train",
            "missed my train",
            "earlier train",
            "later train",
            "earlier service",
            "later service",
        ],
        "response": [
            "aftersales",
            "amend",
            "amendment",
            "change the date",
            "change your ticket",
            "change ticket",
            "change booking",
        ],
    },


    "refund_compensation": {
        "customer": [
            "refund",
            "refunds",
            "refunded",
            "money back",
            "compensation",
            "compensate",
            "delay repay",
            "delayrepay",
            "delay repay claim",
            "claim compensation",
            "claim for a delay",
            "reimburse",
            "reimbursement",
            "repayment",
        ],
        "response": [
            "delay repay",
            "customer resolutions",
            "refund",
            "compensation",
            "refund form",
        ],
    },


    "wifi_connectivity": {
        "customer": [
            "wifi",
            "wi-fi",
            "wi fi",
            "onboard internet",
            "on board internet",
            "internet on the train",
            "internet connection",
            "connect to the wifi",
            "connect to wifi",
            "wifi code",
            "wifi password",
            "wifi login",
        ],
        "response": [
            "wifi",
            "wi-fi",
            "wi fi",
            "onboard wifi",
            "on board wifi",
        ],
    },


    "seat_reservation": {
        "customer": [
            "reserved seat",
            "seat reservation",
            "seat reservation",
            "reserve a seat",
            "reserve my seat",
            "book a seat",
            "booked a seat",
            "seat allocation",
            "seat allocated",
            "seat wasn't allocated",
            "seat was not allocated",
            "wrong seat",
            "reserved seats",
            "unreserved coach",
            "unreserved coaches",
            "unreserved seat",
            "table seat",
            "table and plug",
            "plug seat",
            "no seat",
            "can't find my seat",
            "cannot find my seat",
        ],
        "response": [
            "seat reservation",
            "reserved seat",
            "seat allocation",
            "seat",
        ],
    },


    "fare_price": {
        "customer": [
            "how much",
            "how much is",
            "how much are",
            "ticket price",
            "ticket prices",
            "ticket cost",
            "ticket costs",
            "fare",
            "fares",
            "cheapest ticket",
            "cheapest tickets",
            "cheapest fare",
            "best price",
            "best priced",
            "expensive ticket",
            "expensive tickets",
            "too expensive",
            "advance tickets available",
            "advance ticket available",
            "advance tickets released",
            "when are advance tickets",
            "off peak",
            "off-peak",
            "peak fare",
            "peak ticket",
        ],
        "response": [
            "advance tickets",
            "fare",
            "fares",
            "off-peak",
            "off peak",
            "price",
        ],
    },


    "booking_ticket": {
        "customer": [
            "book a ticket",
            "book ticket",
            "book tickets",
            "booking a ticket",
            "booking tickets",
            "booked a ticket",
            "bought a ticket",
            "buy a ticket",
            "buy tickets",
            "ticket collection",
            "collect my ticket",
            "collect the ticket",
            "collect tickets",
            "ticket valid",
            "is my ticket valid",
            "ticket accepted",
            "are my tickets valid",
            "valid on",
            "valid ticket",
            "ticket accepted on",
            "accepted on another",
            "use my ticket on",
        ],
        "response": [
            "ticket validity",
            "ticket valid",
            "valid on",
            "accepted on",
            "ticket collection",
            "collect",
        ],
    },


    "lost_property": {
        "customer": [
            "lost property",
            "lost item",
            "lost my",
            "lost phone",
            "lost bag",
            "lost coat",
            "lost kindle",
            "lost case",
            "left my",
            "left behind",
            "forgotten on the train",
            "forgot my",
            "missing item",
            "missing property",
        ],
        "response": [
            "lost property",
            "lost property team",
        ],
    },


    "complaint_feedback": {
        "customer": [
            "make a complaint",
            "made a complaint",
            "complaint",
            "complaints",
            "complain",
            "feedback",
            "poor service",
            "bad service",
            "terrible service",
            "awful service",
            "rude staff",
            "rude member of staff",
            "unhelpful staff",
            "staff complaint",
            "customer service complaint",
            "want to complain",
        ],
        "response": [
            "complaint",
            "customer relations",
            "customer resolutions",
            "feedback",
        ],
    },


    "station_facilities": {
        "customer": [
            "station facilities",
            "station lounge",
            "first class lounge",
            "waiting room",
            "station toilet",
            "station toilets",
            "car park",
            "carpark",
            "parking at the station",
            "station parking",
            "station staff",
        ],
        "response": [
            "station",
            "lounge",
            "facilities",
            "car park",
            "parking",
        ],
    },


    "train_status": {
        "customer": [
            "is the train running",
            "is my train running",
            "where is my train",
            "where is the train",
            "train delayed",
            "train is delayed",
            "train delay",
            "delayed train",
            "how late",
            "running late",
            "still running",
            "cancelled train",
            "train cancelled",
            "train cancellation",
            "service cancelled",
            "service delayed",
            "when will the train",
            "what time will the train",
            "next train",
            "next service",
            "will trains run",
            "will the train run",
            "platform",
            "disruption",
            "disrupted",
        ],
        "response": [
            "delayed",
            "delay",
            "cancelled",
            "cancellation",
            "disruption",
            "service",
            "running",
        ],
    },
}


def get_matches(customer_text, response_text):
    """
    Score each intent using both the customer message and
    historical support response.

    Customer phrases receive more weight because they represent
    the actual request.

    Response phrases provide supporting evidence.
    """

    scores = Counter()

    for intent, rules in RULES.items():

        for phrase in rules["customer"]:
            if phrase in customer_text:
                scores[intent] += 3

        for phrase in rules["response"]:
            if phrase in response_text:
                scores[intent] += 1

    if not scores:
        return []

    highest = max(scores.values())

    # Keep only the strongest intents.
    winners = [
        intent
        for intent, score in scores.items()
        if score == highest
    ]

    return winners


def load_golden_pairs():
    golden_pairs = set()

    with GOLDEN_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            customer = normalize(row["customer_message"])
            response = normalize(row["historical_response"])

            golden_pairs.add((customer, response))

    return golden_pairs


def main():

    print("=" * 60)
    print("BUILDING TRAINING DATASET V2")
    print("=" * 60)

    golden_pairs = load_golden_pairs()

    print(f"Golden pairs protected: {len(golden_pairs)}")

    total = 0
    excluded = 0
    kept = 0
    ambiguous = 0
    unmatched = 0

    label_counts = Counter()

    with PROCESSED_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as source, OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as target:

        reader = csv.DictReader(source)

        fieldnames = [
            "customer_tweet_id",
            "brand_tweet_id",
            "created_at",
            "customer_message",
            "brand_response",
            "silver_intent",
            "label_method",
        ]

        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in reader:

            total += 1

            customer = normalize(
                row["customer_message"]
            )

            response = normalize(
                row["brand_response"]
            )

            # ------------------------------------------------
            # GOLDEN SET LEAKAGE PROTECTION
            # ------------------------------------------------

            if (customer, response) in golden_pairs:
                excluded += 1
                continue

            matches = get_matches(
                customer,
                response
            )

            # ------------------------------------------------
            # Conservative filtering
            # ------------------------------------------------

            if len(matches) == 0:
                unmatched += 1
                continue

            if len(matches) > 1:
                ambiguous += 1
                continue

            intent = matches[0]

            writer.writerow({
                "customer_tweet_id":
                    row["customer_tweet_id"],

                "brand_tweet_id":
                    row["brand_tweet_id"],

                "created_at":
                    row["created_at"],

                "customer_message":
                    row["customer_message"],

                "brand_response":
                    row["brand_response"],

                "silver_intent":
                    intent,

                "label_method":
                    "conservative_customer_response_rules",
            })

            kept += 1
            label_counts[intent] += 1

    print()
    print("=" * 60)
    print("TRAINING DATASET V2 CREATED")
    print("=" * 60)

    print(f"Total historical pairs:     {total:,}")
    print(f"Golden pairs excluded:      {excluded:,}")
    print(f"Ambiguous skipped:          {ambiguous:,}")
    print(f"Unmatched skipped:          {unmatched:,}")
    print(f"Training examples kept:     {kept:,}")

    print()
    print("Silver-label distribution:")

    for intent, count in label_counts.most_common():
        print(
            f"  {intent:<25} {count:,}"
        )

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()