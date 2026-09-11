import csv
import random
import re
from collections import defaultdict


INPUT_FILE = "data/processed/virgintrains_support_pairs.csv"

random.seed(42)


KEYWORDS = {
    "delay": [
        "delay",
        "delayed",
        "late",
        "running late",
        "behind"
    ],

    "cancellation": [
        "cancelled",
        "canceled",
        "cancellation",
        "cancel"
    ],

    "refund": [
        "refund",
        "money back",
        "reimburse",
        "compensation"
    ],

    "ticket_change": [
        "change my ticket",
        "change ticket",
        "amend",
        "amendment",
        "change booking",
        "change booking"
    ],

    "booking": [
        "book",
        "booking",
        "booked",
        "reserve",
        "reservation"
    ],

    "price": [
        "price",
        "cost",
        "£",
        "expensive",
        "charge",
        "charged",
        "fare"
    ],

    "wifi": [
        "wifi",
        "wi-fi",
        "internet"
    ],

    "app": [
        "app",
        "application"
    ],

    "lost_property": [
        "lost property",
        "lost item",
        "lost phone",
        "left my",
        "forgotten"
    ],

    "seat": [
        "seat",
        "seats",
        "reserved seat",
        "reservation"
    ],

    "station": [
        "station",
        "platform",
        "lounge",
        "gate"
    ],

    "bicycle": [
        "bicycle",
        "bike",
        "bikes"
    ],

    "complaint": [
        "complaint",
        "complain",
        "rude",
        "unhelpful",
        "terrible",
        "shocking",
        "disappointed"
    ],

    "journey_information": [
        "what time",
        "when",
        "where",
        "route",
        "journey",
        "train from",
        "train to"
    ]
}


def normalize(text):
    text = text.lower()
    text = text.replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def matches_keyword(text, keyword):
    return keyword.lower() in text


def main():

    print("=" * 80)
    print("VIRGINTRAINS CUSTOMER INTENT EXPLORATION")
    print("=" * 80)

    grouped = defaultdict(list)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            message = normalize(row["customer_message"])

            if not message:
                continue

            for category, keywords in KEYWORDS.items():

                if any(
                    matches_keyword(message, keyword)
                    for keyword in keywords
                ):
                    grouped[category].append(
                        (
                            row["customer_message"],
                            row["brand_response"]
                        )
                    )

    # ---------------------------------------------------------
    # Print samples
    # ---------------------------------------------------------

    for category in KEYWORDS:

        examples = grouped[category]

        print("\n")
        print("=" * 80)
        print(
            f"{category.upper()} "
            f"({len(examples):,} matching messages)"
        )
        print("=" * 80)

        sample_size = min(8, len(examples))

        for i, (customer, response) in enumerate(
            random.sample(examples, sample_size),
            start=1
        ):

            print(f"\nExample {i}")

            print(f"Customer:")
            print(customer)

            print(f"\nVirginTrains:")
            print(response)

            print("-" * 80)

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()