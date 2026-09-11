import csv
import re


FILE = "data/golden/golden_set_final.csv"


INTENT_SIGNALS = {

    "train_status": [
        "delay",
        "delayed",
        "cancel",
        "cancelled",
        "late",
        "service",
        "train",
    ],

    "refund_compensation": [
        "refund",
        "compensation",
        "compensation",
        "delayrepay",
        "delay repay",
        "money back",
    ],

    "ticket_change": [
        "change ticket",
        "change my ticket",
        "amend",
        "amendment",
        "change tickets",
        "earlier train",
        "later train",
    ],

    "booking_ticket": [
        "book",
        "booking",
        "ticket",
        "tickets",
        "buy",
        "purchase",
    ],

    "seat_reservation": [
        "seat",
        "reserved seat",
        "reservation",
        "reserve seats",
        "reserved",
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "internet",
        "connection",
    ],

    "fare_price": [
        "price",
        "fare",
        "cost",
        "charged",
        "charge",
        "expensive",
    ],

    "station_facilities": [
        "station",
        "platform",
        "lounge",
        "socket",
        "toilet",
        "carpark",
        "car park",
    ],

    "lost_property": [
        "lost",
        "lost property",
        "left behind",
        "missing item",
    ],

    "complaint_feedback": [
        "complaint",
        "disappointing",
        "disgraceful",
        "terrible",
        "shocking",
        "unhelpful",
        "poor service",
    ],
}


def detect_signals(message):

    text = message.lower()

    detected = []

    for intent, signals in INTENT_SIGNALS.items():

        for signal in signals:

            if signal in text:

                detected.append(intent)

                break

    return detected


with open(
    FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    rows = list(
        csv.DictReader(f)
    )


multi_intent = []


for row in rows:

    detected = detect_signals(
        row["customer_message"]
    )

    if len(detected) >= 2:

        multi_intent.append(
            (
                row,
                detected
            )
        )


print("=" * 70)
print("MULTI-INTENT ANALYSIS")
print("=" * 70)

print()
print(
    f"Golden examples: {len(rows)}"
)

print(
    f"Potential multi-intent examples: "
    f"{len(multi_intent)}"
)

print()

for row, detected in multi_intent:

    print("-" * 70)

    print(
        f"ID: {row['id']}"
    )

    print(
        f"Gold intent: "
        f"{row['intent']}"
    )

    print(
        f"Detected signals: "
        f"{detected}"
    )

    print(
        f"Message: "
        f"{row['customer_message']}"
    )


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)