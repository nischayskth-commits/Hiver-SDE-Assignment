import csv
from collections import Counter


FILE = "data/golden/golden_set.csv"


def main():

    rows = []

    with open(FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    print("=" * 80)
    print("GOLDEN SET SAMPLE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal examples: {len(rows)}")

    # Message lengths
    lengths = [
        len(row["customer_message"].split())
        for row in rows
    ]

    print("\nMessage length:")
    print(f"Shortest: {min(lengths)} words")
    print(f"Longest:  {max(lengths)} words")
    print(f"Average:  {sum(lengths) / len(lengths):.1f} words")

    # Very short messages
    short = [
        row for row in rows
        if len(row["customer_message"].split()) <= 5
    ]

    print(f"\nVery short messages (<=5 words): {len(short)}")

    # Messages containing useful intent signals
    keywords = {
        "delay": ["delay", "delayed", "late"],
        "cancel": ["cancel", "cancelled", "canceled"],
        "refund": ["refund", "compensation", "repay"],
        "ticket": ["ticket", "tickets"],
        "booking": ["booking", "book", "booked"],
        "seat": ["seat", "seats", "reserved"],
        "wifi": ["wifi", "wi-fi"],
        "station": ["station", "platform", "lounge"],
        "lost": ["lost", "left my", "lost property"],
        "complaint": ["complaint", "complain", "poor", "terrible"]
    }

    print("\nKeyword coverage:")

    for category, words in keywords.items():

        count = 0

        for row in rows:

            text = row["customer_message"].lower()

            if any(word in text for word in words):
                count += 1

        print(f"{category:<15} {count:>4}")


if __name__ == "__main__":
    main()