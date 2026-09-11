import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


TRAIN_FILE = Path("data/processed/training_pairs_v2.csv")
GOLDEN_FILE = Path("data/golden/golden_set_final.csv")


def load_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():

    print("=" * 60)
    print("BASELINE 2 V2: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 60)

    train_rows = load_csv(TRAIN_FILE)
    golden_rows = load_csv(GOLDEN_FILE)

    X_train = [
        row["customer_message"]
        for row in train_rows
    ]

    y_train = [
        row["silver_intent"]
        for row in train_rows
    ]

    X_test = [
        row["customer_message"]
        for row in golden_rows
    ]

    y_test = [
        row["intent"]
        for row in golden_rows
    ]

    print(f"Training examples: {len(X_train):,}")
    print(f"Golden examples:   {len(X_test):,}")

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    print()
    print("Fitting TF-IDF...")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_tfidf,
        y_train
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test_tfidf
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    # ---------------------------------------------------------
    # Per-intent results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("PER-INTENT RESULTS")
    print("=" * 60)

    labels = sorted(set(y_test))

    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # Sample errors
    # ---------------------------------------------------------

    print("=" * 60)
    print("SAMPLE ERRORS")
    print("=" * 60)

    shown = 0

    for row, actual, predicted in zip(
        golden_rows,
        y_test,
        predictions,
    ):

        if actual != predicted:

            print()
            print(
                f"Message:   "
                f"{row['customer_message']}"
            )

            print(
                f"Expected:  {actual}"
            )

            print(
                f"Predicted: {predicted}"
            )

            shown += 1

            if shown >= 10:
                break


if __name__ == "__main__":
    main()