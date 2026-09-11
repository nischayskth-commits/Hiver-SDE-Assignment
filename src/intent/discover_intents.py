import csv
import re
from collections import Counter


INPUT_FILE = "data/processed/virgintrains_support_pairs.csv"


# Common words that don't tell us much about the intent
STOP_WORDS = {
    "the", "and", "for", "you", "are", "was", "with",
    "this", "that", "have", "has", "had", "but", "not",
    "can", "could", "would", "will", "what", "when",
    "where", "how", "why", "from", "your", "our",
    "about", "please", "just", "been", "they", "them",
    "there", "their", "it's", "its", "into", "than",
    "then", "want", "need", "get", "got", "my", "me",
    "i", "to", "of", "in", "on", "is", "it", "a", "an",
    "do", "did", "be", "we", "at", "or", "if"
}


def tokenize(text):
    text = text.lower()

    words = re.findall(r"[a-z]+", text)

    return [
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


def main():

    print("=" * 80)
    print("VIRGINTRAINS INTENT DISCOVERY")
    print("=" * 80)

    word_counter = Counter()
    examples = {}

    total = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            message = row["customer_message"].strip()

            if not message:
                continue

            total += 1

            words = tokenize(message)

            word_counter.update(words)

            for word in words:
                if word not in examples:
                    examples[word] = message

    print(f"\nTotal customer messages: {total:,}")

    print("\n" + "=" * 80)
    print("MOST COMMON CUSTOMER TERMS")
    print("=" * 80)

    for word, count in word_counter.most_common(100):

        print(
            f"{word:<25} {count:>6}   "
            f"Example: {examples[word][:100]}"
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()