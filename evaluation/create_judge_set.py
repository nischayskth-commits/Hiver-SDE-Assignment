import csv
import os
import random


INPUT_FILE = "evaluation/reply_quality_metrics.csv"
OUTPUT_FILE = "evaluation/llm_judge_set.csv"

SAMPLE_SIZE = 50
RANDOM_SEED = 42


def load_results():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def main():

    rows = load_results()

    print(f"Loaded evaluation rows: {len(rows)}")

    random.seed(RANDOM_SEED)

    # Stratified sampling by predicted decision where possible.
    auto_rows = [
        row for row in rows
        if row.get("decision") == "AUTO-HANDLE"
    ]

    escalate_rows = [
        row for row in rows
        if row.get("decision") == "ESCALATE"
    ]

    # We want both types represented.
    auto_sample_size = min(
        len(auto_rows),
        SAMPLE_SIZE // 2
    )

    escalate_sample_size = min(
        len(escalate_rows),
        SAMPLE_SIZE - auto_sample_size
    )

    selected = (
        random.sample(auto_rows, auto_sample_size)
        +
        random.sample(escalate_rows, escalate_sample_size)
    )

    # If one category does not have enough examples,
    # fill the remaining slots from all rows.
    if len(selected) < SAMPLE_SIZE:

        selected_ids = {
            row.get("id")
            for row in selected
        }

        remaining = [
            row
            for row in rows
            if row.get("id") not in selected_ids
        ]

        needed = SAMPLE_SIZE - len(selected)

        selected.extend(
            random.sample(
                remaining,
                min(needed, len(remaining))
            )
        )

    # Keep deterministic order for easier review.
    selected.sort(
        key=lambda row: int(row.get("id", 0))
    )

    output_rows = []

    for row in selected:

        output_rows.append({
            "id": row.get("id", ""),
            "customer_message": row.get(
                "customer_message",
                ""
            ),
            "true_intent": row.get(
                "true_intent",
                ""
            ),
            "predicted_intent": row.get(
                "predicted_intent",
                ""
            ),
            "expected_resolution": row.get(
                "expected_resolution",
                ""
            ),
            "historical_evidence": row.get(
                "evidence_actions",
                ""
            ),
            "evidence_case_count": row.get(
                "evidence_case_count",
                ""
            ),
            "retrieval_similarity": row.get(
                "top_similarity",
                ""
            ),
            "evidence_strength": row.get(
                "evidence_strength",
                ""
            ),
            "evidence_agreement": row.get(
                "evidence_agreement",
                ""
            ),
            "predicted_decision": row.get(
                "decision",
                ""
            ),
            "decision_reason": row.get(
                "decision_reason",
                ""
            ),
            "draft_reply": row.get(
                "draft_reply",
                ""
            ),
        })

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    fieldnames = [
        "id",
        "customer_message",
        "true_intent",
        "predicted_intent",
        "expected_resolution",
        "historical_evidence",
        "evidence_case_count",
        "retrieval_similarity",
        "evidence_strength",
        "evidence_agreement",
        "predicted_decision",
        "decision_reason",
        "draft_reply",
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
        writer.writerows(output_rows)

    print()
    print("=" * 60)
    print("LLM JUDGE DATASET CREATED")
    print("=" * 60)
    print(f"Examples selected: {len(output_rows)}")
    print(f"AUTO-HANDLE: {sum(r['predicted_decision'] == 'AUTO-HANDLE' for r in output_rows)}")
    print(f"ESCALATE: {sum(r['predicted_decision'] == 'ESCALATE' for r in output_rows)}")
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()