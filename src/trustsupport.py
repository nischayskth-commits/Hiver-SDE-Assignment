from pathlib import Path
import sys

# =============================================================
# ALLOW IMPORTS FROM SRC SUBDIRECTORIES
# =============================================================

sys.path.append(str(Path(__file__).parent))


# =============================================================
# PROJECT IMPORTS
# =============================================================

from intent.input_quality import InputQualityDetector

from retrieval.historical_retriever import HistoricalRetriever

from retrieval.resolution_evidence import (
    ResolutionEvidenceExtractor,
)

from decision.evidence_gate import (
    EvidenceConfidenceGate,
)

from generation.reply_generator import (
    GroundedReplyGenerator,
)


# =============================================================
# MACHINE LEARNING IMPORTS
# =============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =============================================================
# TRAINING DATA
# =============================================================

TRAIN_FILE = Path(
    "data/processed/training_pairs_v2.csv"
)


# =============================================================
# TRUSTSUPPORT AGENT
# =============================================================

class TrustSupport:

    def __init__(self):

        print("Initializing TrustSupport...")

        # =====================================================
        # 1. LOAD TRAINING DATA
        # =====================================================

        self.train_rows = self._load_training_data()

        X_train = [
            row["customer_message"]
            for row in self.train_rows
        ]

        y_train = [
            row["silver_intent"]
            for row in self.train_rows
        ]

        # =====================================================
        # 2. INPUT QUALITY DETECTOR
        # =====================================================

        print(
            "Initializing input quality detector..."
        )

        self.input_quality = (
            InputQualityDetector()
        )

        # =====================================================
        # 3. INTENT CLASSIFIER
        # =====================================================

        print(
            "Training intent classifier..."
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        X_train_tfidf = (
            self.vectorizer.fit_transform(
                X_train
            )
        )

        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        self.classifier.fit(
            X_train_tfidf,
            y_train
        )

        print(
            f"Classifier trained on "
            f"{len(self.train_rows)} examples."
        )

        # =====================================================
        # 4. HISTORICAL RETRIEVAL
        # =====================================================

        print(
            "Initializing historical retriever..."
        )

        self.retriever = (
            HistoricalRetriever()
        )

        # =====================================================
        # 5. RESOLUTION EVIDENCE EXTRACTION
        # =====================================================

        self.evidence_extractor = (
            ResolutionEvidenceExtractor()
        )

        # =====================================================
        # 6. EVIDENCE-CONFIDENCE DECISION GATE
        # =====================================================

        self.gate = (
            EvidenceConfidenceGate()
        )

        # =====================================================
        # 7. GROUNDED REPLY GENERATOR
        # =====================================================

        self.reply_generator = (
            GroundedReplyGenerator()
        )

        print(
            "TrustSupport initialized successfully."
        )

    # =========================================================
    # LOAD TRAINING DATA
    # =========================================================

    def _load_training_data(self):

        import csv

        if not TRAIN_FILE.exists():

            raise FileNotFoundError(
                f"Training file not found: "
                f"{TRAIN_FILE}"
            )

        with TRAIN_FILE.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            return list(
                csv.DictReader(f)
            )

    # =========================================================
    # INTENT PREDICTION
    # =========================================================

    def predict_intent(self, message):

        vector = (
            self.vectorizer.transform(
                [message]
            )
        )

        probabilities = (
            self.classifier.predict_proba(
                vector
            )[0]
        )

        best_index = (
            probabilities.argmax()
        )

        intent = (
            self.classifier.classes_[
                best_index
            ]
        )

        confidence = float(
            probabilities[best_index]
        )

        return intent, confidence

    # =========================================================
    # MAIN ANALYSIS PIPELINE
    # =========================================================

    def analyze(
        self,
        message,
        exclude_ids=None
    ):

        # -----------------------------------------------------
        # 0. INPUT QUALITY CHECK
        # -----------------------------------------------------

        quality = (
            self.input_quality.analyze(
                message
            )
        )

        # -----------------------------------------------------
        # If the message is insufficient,
        # do NOT force it through the classifier.
        # -----------------------------------------------------

        if quality["is_insufficient"]:

            return {

                "message": message,

                "intent": "other",

                "intent_confidence": 0.0,

                "retrieved_cases": [],

                "evidence": {
                    "evidence": []
                },

                "decision": {

                    "decision": "ESCALATE",

                    "reason": (
                        "Input quality check: "
                        f"{quality['reason']}"
                    ),

                    "evidence_strength": 0.0,

                    "top_similarity": 0.0,

                    "evidence_agreement": 0.0,

                    "risk_signals": [
                        quality["reason"]
                    ],
                },

                "reply": (
                    "I need a little more "
                    "information to understand "
                    "your request. Please provide "
                    "some details about the issue "
                    "so the support team can assist you."
                ),
            }

        # -----------------------------------------------------
        # 1. INTENT DETECTION
        # -----------------------------------------------------

        intent, intent_confidence = (
            self.predict_intent(
                message
            )
        )

        # -----------------------------------------------------
        # 2. HISTORICAL CASE RETRIEVAL
        # -----------------------------------------------------
        #
        # exclude_ids is used during evaluation to prevent
        # golden-set conversations from being retrieved.
        #
        # During normal interactive use, exclude_ids is None,
        # so the retriever behaves normally.
        # -----------------------------------------------------

        retrieved_cases = (
            self.retriever.search(
                message,
                top_k=5,
                exclude_ids=exclude_ids
            )
        )

        # -----------------------------------------------------
        # 3. RESOLUTION EVIDENCE EXTRACTION
        # -----------------------------------------------------

        evidence = (
            self.evidence_extractor.extract(
                retrieved_cases
            )
        )

        # -----------------------------------------------------
        # 4. EVIDENCE-CONFIDENCE DECISION
        # -----------------------------------------------------

        decision = self.gate.decide(
            message=message,
            intent=intent,
            intent_confidence=intent_confidence,
            retrieval_results=retrieved_cases,
            evidence_result=evidence,
        )

        # -----------------------------------------------------
        # 5. GROUNDED REPLY GENERATION
        # -----------------------------------------------------

        reply = self.reply_generator.generate(
            message=message,
            intent=intent,
            evidence_result=evidence,
            decision=decision,
        )

        # -----------------------------------------------------
        # 6. FINAL RESULT
        # -----------------------------------------------------

        return {

            "message": message,

            "intent": intent,

            "intent_confidence": round(
                intent_confidence,
                3
            ),

            "retrieved_cases": (
                retrieved_cases
            ),

            "evidence": evidence,

            "decision": decision,

            "reply": reply,
        }


# =============================================================
# PRINT RESULT
# =============================================================

def print_result(result):

    print()

    print("=" * 70)
    print("TRUSTSUPPORT ANALYSIS")
    print("=" * 70)

    # =========================================================
    # CUSTOMER MESSAGE
    # =========================================================

    print()
    print("CUSTOMER MESSAGE")
    print("-" * 70)

    print(
        result["message"]
    )

    # =========================================================
    # INTENT
    # =========================================================

    print()
    print("INTENT DETECTION")
    print("-" * 70)

    print(
        f"Intent: "
        f"{result['intent']}"
    )

    print(
        f"Intent confidence: "
        f"{result['intent_confidence']}"
    )

    # =========================================================
    # DECISION
    # =========================================================

    decision = result["decision"]

    print()
    print("AUTOMATION DECISION")
    print("-" * 70)

    print(
        f"Decision: "
        f"{decision['decision']}"
    )

    print(
        f"Reason: "
        f"{decision['reason']}"
    )

    print(
        f"Evidence strength: "
        f"{decision['evidence_strength']}"
    )

    print(
        f"Top similarity: "
        f"{decision['top_similarity']}"
    )

    print(
        f"Evidence agreement: "
        f"{decision['evidence_agreement']}"
    )

    print(
        f"Risk signals: "
        f"{decision['risk_signals']}"
    )

    # =========================================================
    # DRAFT CUSTOMER REPLY
    # =========================================================

    print()
    print("=" * 70)
    print("DRAFT CUSTOMER REPLY")
    print("=" * 70)

    print()

    print(
        result["reply"]
    )

    # =========================================================
    # RESOLUTION EVIDENCE
    # =========================================================

    print()
    print("-" * 70)
    print("RESOLUTION EVIDENCE")
    print("-" * 70)

    evidence_items = (
        result["evidence"].get(
            "evidence",
            []
        )
    )

    if not evidence_items:

        print(
            "No strong resolution evidence "
            "was found in the retrieved cases."
        )

    else:

        for item in evidence_items:

            print(
                f"{item['action']}: "
                f"{item['support_count']} cases "
                f"({item['agreement']:.0%})"
            )

    # =========================================================
    # HISTORICAL CASES
    # =========================================================

    print()
    print("-" * 70)
    print("TOP HISTORICAL CASES")
    print("-" * 70)

    for i, case in enumerate(
        result["retrieved_cases"],
        start=1
    ):

        print()

        print(
            f"{i}. "
            f"Similarity: "
            f"{case['similarity']:.3f}"
        )

        print(
            f"Customer: "
            f"{case['customer_message']}"
        )

        print(
            f"Response: "
            f"{case['brand_response']}"
        )

    print()
    print("=" * 70)


# =============================================================
# INTERACTIVE PROGRAM
# =============================================================

if __name__ == "__main__":

    agent = TrustSupport()

    while True:

        print()

        message = input(
            "Customer message "
            "(type 'quit' to exit): "
        ).strip()

        if message.lower() == "quit":

            print()

            print(
                "TrustSupport stopped."
            )

            break

        if not message:

            print(
                "Please enter a customer message."
            )

            continue

        try:

            result = agent.analyze(
                message
            )

            print_result(
                result
            )

        except Exception as error:

            print()

            print(
                "An error occurred:"
            )

            print(
                error
            )