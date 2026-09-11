import csv
from collections import Counter
from pathlib import Path

TRAIN_FILE = Path("data/processed/training_pairs.csv")
GOLDEN_FILE = Path("data/golden/golden_set_final.csv")


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def accuracy(y_true, y_pred):
    correct = sum(
        actual == predicted
        for actual, predicted in zip(y_true, y_pred)
    )
    return correct / len(y_true)


def macro_f1(y_true, y_pred, labels):
    scores = []

    for label in labels:
        tp = sum(
            actual == label and predicted == label
            for actual, predicted in zip(y_true, y_pred)
        )

        fp = sum(
            actual != label and predicted == label
            for actual, predicted in zip(y_true, y_pred)
        )

        fn = sum(
            actual == label and predicted != label
            for actual, predicted in zip(y_true, y_pred)
        )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        scores.append(f1)

    return sum(scores) / len(scores)


def main():
    train_rows = load_csv(TRAIN_FILE)
    golden_rows = load_csv(GOLDEN_FILE)

    label_counts = Counter(
        row["silver_intent"]
        for row in train_rows
    )

    majority_intent, majority_count = label_counts.most_common(1)[0]

    y_true = [
        row["intent"]
        for row in golden_rows
    ]

    y_pred = [
        majority_intent
        for _ in golden_rows
    ]

    labels = sorted(set(y_true))

    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred, labels)

    print("=" * 60)
    print("BASELINE 1: MAJORITY CLASS")
    print("=" * 60)

    print(f"Training examples: {len(train_rows):,}")
    print(f"Golden examples:   {len(golden_rows):,}")
    print(f"Majority intent:   {majority_intent}")
    print(f"Majority count:    {majority_count:,}")

    print()
    print(f"Accuracy:          {acc:.4f}")
    print(f"Macro-F1:          {f1:.4f}")

    print()
    print("Golden-set intent distribution:")

    golden_counts = Counter(y_true)

    for intent, count in golden_counts.most_common():
        print(f"  {intent:<25} {count}")


if __name__ == "__main__":
    main()