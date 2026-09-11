from pathlib import Path
import csv
import sys

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)

# Import TrustSupport
from trustsupport import TrustSupport


# ============================================================
# FILE PATHS
# ============================================================

GOLDEN_FILE = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set_final.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "trustsupport_predictions.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_escalation(value):
    """
    Convert different representations of escalation
    into a simple boolean.
    """

    value = str(value).strip().lower()

    return value in {
        "y",
        "yes",
        "true",
        "1",
    }


def safe_float(value):
    """
    Safely convert a value to float.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ============================================================
# LOAD GOLDEN SET
# ============================================================

def load_golden_set():

    if not GOLDEN_FILE.exists():

        raise FileNotFoundError(
            f"Golden set not found:\n"
            f"{GOLDEN_FILE}"
        )

    with GOLDEN_FILE.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:

        raise ValueError(
            "Golden set is empty."
        )

    return rows


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(results):

    fieldnames = [
        "id",
        "customer_message",

        "true_intent",
        "predicted_intent",

        "intent_correct",
        "intent_confidence",

        "true_escalation",
        "predicted_escalation",

        "escalation_correct",

        "decision_reason",

        "evidence_strength",
        "top_similarity",
        "evidence_agreement",

        "risk_signals",

        "draft_reply",
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("TRUSTSUPPORT EVALUATION HARNESS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load golden set
    # --------------------------------------------------------

    golden_rows = load_golden_set()

    print()
    print(
        f"Golden examples loaded: "
        f"{len(golden_rows)}"
    )

    # --------------------------------------------------------
    # Initialize agent ONCE
    # --------------------------------------------------------

    print()
    print(
        "Initializing TrustSupport..."
    )

    agent = TrustSupport()

    # --------------------------------------------------------
    # Storage for predictions
    # --------------------------------------------------------

    results = []

    true_intents = []
    predicted_intents = []

    true_escalations = []
    predicted_escalations = []

    # --------------------------------------------------------
    # Evaluate every golden example
    # --------------------------------------------------------

    print()
    print(
        "Evaluating golden examples..."
    )

    for index, row in enumerate(
        golden_rows,
        start=1
    ):

        message = (
            row["customer_message"]
        )

        true_intent = (
            row["intent"]
        )

        true_escalation = (
            normalize_escalation(
                row["should_escalate"]
            )
        )

        # ----------------------------------------------------
        # Run TrustSupport
        # ----------------------------------------------------

        result = agent.analyze(
            message
        )

        predicted_intent = (
            result["intent"]
        )

        intent_confidence = safe_float(
            result["intent_confidence"]
        )

        decision = result[
            "decision"
        ]

        predicted_escalation = (
            str(
                decision["decision"]
            ).upper()
            == "ESCALATE"
        )

        # ----------------------------------------------------
        # Metrics information
        # ----------------------------------------------------

        intent_correct = (
            predicted_intent
            == true_intent
        )

        escalation_correct = (
            predicted_escalation
            == true_escalation
        )

        # ----------------------------------------------------
        # Evidence information
        # ----------------------------------------------------

        evidence_strength = safe_float(
            decision.get(
                "evidence_strength",
                0
            )
        )

        top_similarity = safe_float(
            decision.get(
                "top_similarity",
                0
            )
        )

        evidence_agreement = safe_float(
            decision.get(
                "evidence_agreement",
                0
            )
        )

        risk_signals = decision.get(
            "risk_signals",
            []
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        results.append(
            {
                "id":
                    row["id"],

                "customer_message":
                    message,

                "true_intent":
                    true_intent,

                "predicted_intent":
                    predicted_intent,

                "intent_correct":
                    intent_correct,

                "intent_confidence":
                    intent_confidence,

                "true_escalation":
                    true_escalation,

                "predicted_escalation":
                    predicted_escalation,

                "escalation_correct":
                    escalation_correct,

                "decision_reason":
                    decision.get(
                        "reason",
                        ""
                    ),

                "evidence_strength":
                    evidence_strength,

                "top_similarity":
                    top_similarity,

                "evidence_agreement":
                    evidence_agreement,

                "risk_signals":
                    ", ".join(
                        risk_signals
                    ),

                "draft_reply":
                    result.get(
                        "reply",
                        ""
                    ),
            }
        )

        true_intents.append(
            true_intent
        )

        predicted_intents.append(
            predicted_intent
        )

        true_escalations.append(
            true_escalation
        )

        predicted_escalations.append(
            predicted_escalation
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index % 25 == 0
            or index == len(golden_rows)
        ):

            print(
                f"Processed "
                f"{index}/"
                f"{len(golden_rows)}"
            )

    # ========================================================
    # INTENT METRICS
    # ========================================================

    intent_accuracy = accuracy_score(
        true_intents,
        predicted_intents
    )

    intent_macro_f1 = f1_score(
        true_intents,
        predicted_intents,
        average="macro",
        zero_division=0
    )

    intent_weighted_f1 = f1_score(
        true_intents,
        predicted_intents,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # ESCALATION METRICS
    # ========================================================

    escalation_accuracy = accuracy_score(
        true_escalations,
        predicted_escalations
    )

    escalation_precision = precision_score(
        true_escalations,
        predicted_escalations,
        average="binary",
        zero_division=0
    )

    escalation_recall = recall_score(
        true_escalations,
        predicted_escalations,
        average="binary",
        zero_division=0
    )

    escalation_f1 = f1_score(
        true_escalations,
        predicted_escalations,
        average="binary",
        zero_division=0
    )

    # ========================================================
    # AUTO-HANDLE PRECISION
    # ========================================================

    auto_handle_indices = [
        i
        for i, prediction in enumerate(
            predicted_escalations
        )
        if not prediction
    ]

    if auto_handle_indices:

        correct_auto_handles = sum(
            1
            for i in auto_handle_indices
            if (
                true_escalations[i]
                is False
            )
        )

        auto_handle_precision = (
            correct_auto_handles
            / len(auto_handle_indices)
        )

    else:

        auto_handle_precision = 0.0

    # ========================================================
    # EVIDENCE METRICS
    # ========================================================

    average_confidence = (
        sum(
            safe_float(
                row["intent_confidence"]
            )
            for row in results
        )
        / len(results)
    )

    average_similarity = (
        sum(
            safe_float(
                row["top_similarity"]
            )
            for row in results
        )
        / len(results)
    )

    average_evidence_strength = (
        sum(
            safe_float(
                row["evidence_strength"]
            )
            for row in results
        )
        / len(results)
    )

    average_evidence_agreement = (
        sum(
            safe_float(
                row["evidence_agreement"]
            )
            for row in results
        )
        / len(results)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_predictions(
        results
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("INTENT METRICS")
    print("=" * 70)

    print(
        f"Accuracy:       "
        f"{intent_accuracy:.4f}"
    )

    print(
        f"Macro-F1:       "
        f"{intent_macro_f1:.4f}"
    )

    print(
        f"Weighted-F1:    "
        f"{intent_weighted_f1:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print()
    print("-" * 70)
    print("PER-INTENT PERFORMANCE")
    print("-" * 70)

    print(
        classification_report(
            true_intents,
            predicted_intents,
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    labels = sorted(
        set(true_intents)
        | set(predicted_intents)
    )

    matrix = confusion_matrix(
        true_intents,
        predicted_intents,
        labels=labels
    )

    print()
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "Labels:"
    )

    print(
        labels
    )

    print()

    for label, values in zip(
        labels,
        matrix
    ):

        print(
            f"{label:25s} "
            f"{values.tolist()}"
        )

    # ========================================================
    # ESCALATION METRICS
    # ========================================================

    print()
    print("=" * 70)
    print("ESCALATION METRICS")
    print("=" * 70)

    print(
        f"Escalation accuracy:   "
        f"{escalation_accuracy:.4f}"
    )

    print(
        f"Escalation precision:  "
        f"{escalation_precision:.4f}"
    )

    print(
        f"Escalation recall:     "
        f"{escalation_recall:.4f}"
    )

    print(
        f"Escalation F1:         "
        f"{escalation_f1:.4f}"
    )

    print(
        f"Auto-handle precision: "
        f"{auto_handle_precision:.4f}"
    )

    # ========================================================
    # EVIDENCE METRICS
    # ========================================================

    print()
    print("=" * 70)
    print("EVIDENCE / CONFIDENCE METRICS")
    print("=" * 70)

    print(
        f"Average intent confidence: "
        f"{average_confidence:.4f}"
    )

    print(
        f"Average retrieval similarity: "
        f"{average_similarity:.4f}"
    )

    print(
        f"Average evidence strength: "
        f"{average_evidence_strength:.4f}"
    )

    print(
        f"Average evidence agreement: "
        f"{average_evidence_agreement:.4f}"
    )

    # ========================================================
    # DATASET DECISION SUMMARY
    # ========================================================

    total_escalations = sum(
        predicted_escalations
    )

    total_auto_handles = (
        len(predicted_escalations)
        - total_escalations
    )

    print()
    print("=" * 70)
    print("DECISION DISTRIBUTION")
    print("=" * 70)

    print(
        f"AUTO-HANDLE: "
        f"{total_auto_handles}"
    )

    print(
        f"ESCALATE:    "
        f"{total_escalations}"
    )

    print(
        f"TOTAL:       "
        f"{len(results)}"
    )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    print()
    print("=" * 70)

    print(
        "Detailed predictions saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)

    print()
    print(
        "Evaluation completed successfully."
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()