import csv
from collections import Counter


FILE = "evaluation/trustsupport_predictions.csv"


# =============================================================
# LOAD PREDICTIONS
# =============================================================

with open(
    FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    rows = list(
        csv.DictReader(f)
    )


# =============================================================
# FIND INTENT ERRORS
# =============================================================

wrong = [
    row
    for row in rows
    if row["true_intent"]
    != row["predicted_intent"]
]


print("=" * 70)
print("TRUSTSUPPORT FAILURE ANALYSIS")
print("=" * 70)

print()
print(
    f"Total examples: {len(rows)}"
)

print(
    f"Wrong intent predictions: {len(wrong)}"
)

print(
    f"Correct intent predictions: "
    f"{len(rows) - len(wrong)}"
)


# =============================================================
# TOP INTENT CONFUSIONS
# =============================================================

print()
print("=" * 70)
print("TOP INTENT CONFUSIONS")
print("=" * 70)

confusions = Counter(
    (
        row["true_intent"],
        row["predicted_intent"]
    )
    for row in wrong
)

for (true_intent, predicted_intent), count in (
    confusions.most_common(20)
):

    print(
        f"{true_intent:25s} -> "
        f"{predicted_intent:25s} : "
        f"{count}"
    )


# =============================================================
# WRONG PREDICTION EXAMPLES
# =============================================================

print()
print("=" * 70)
print("WRONG PREDICTION EXAMPLES")
print("=" * 70)

for row in wrong:

    print()
    print(
        f"ID: {row['id']}"
    )

    print(
        f"TRUE: {row['true_intent']}"
    )

    print(
        f"PREDICTED: {row['predicted_intent']}"
    )

    print(
        f"CONFIDENCE: {row['intent_confidence']}"
    )

    print(
        f"MESSAGE: {row['customer_message']}"
    )


# =============================================================
# OTHER CLASS ANALYSIS
# =============================================================

print()
print("=" * 70)
print("OTHER CLASS ANALYSIS")
print("=" * 70)

other_rows = [
    row
    for row in rows
    if row["true_intent"] == "other"
]

other_wrong = [
    row
    for row in other_rows
    if row["predicted_intent"] != "other"
]

print()
print(
    f"True 'other' examples: "
    f"{len(other_rows)}"
)

print(
    f"Correctly detected as 'other': "
    f"{len(other_rows) - len(other_wrong)}"
)

print(
    f"Missed 'other' examples: "
    f"{len(other_wrong)}"
)

print()
print("MISSED 'OTHER' EXAMPLES:")

for row in other_wrong:

    print()

    print(
        f"ID: {row['id']}"
    )

    print(
        f"Predicted: "
        f"{row['predicted_intent']}"
    )

    print(
        f"Confidence: "
        f"{row['intent_confidence']}"
    )

    print(
        f"Message: "
        f"{row['customer_message']}"
    )


# =============================================================
# ESCALATION COLUMN DISCOVERY
# =============================================================

print()
print("=" * 70)
print("AVAILABLE CSV COLUMNS")
print("=" * 70)

if rows:

    for column in rows[0].keys():

        print(
            f"- {column}"
        )


print()
print("=" * 70)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 70)