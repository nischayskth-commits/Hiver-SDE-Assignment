import re


class EvidenceConfidenceGate:

    def __init__(
        self,
        min_intent_confidence=0.55,
        min_retrieval_similarity=0.20,
        min_evidence_agreement=0.40,
    ):

        self.min_intent_confidence = (
            min_intent_confidence
        )

        self.min_retrieval_similarity = (
            min_retrieval_similarity
        )

        self.min_evidence_agreement = (
            min_evidence_agreement
        )

    # =========================================================
    # RISK DETECTION
    # =========================================================

    def detect_risk_signals(self, message):

        text = message.lower()

        risks = []

        serious_patterns = {

            "safety": [
                "unsafe",
                "dangerous",
                "danger",
                "health and safety",
                "fell",
                "fall",
                "injured",
                "injury",
            ],

            "active_disruption": [
                "stranded",
                "stuck",
                "cancelled",
                "cancelled train",
                "train cancelled",
                "massively delayed",
                "hours late",
                "can't get home",
                "cannot get home",
            ],

            "financial_dispute": [
                "charged me",
                "charged twice",
                "wrong charge",
                "£",
                "refund overdue",
                "money missing",
                "paid again",
                "charged",
            ],

            "case_specific": [
                "reference number",
                "case number",
                "complaint reference",
                "already contacted",
                "already called",
                "still waiting",
                "no reply",
                "haven't heard",
                "have not heard",
            ],

            "insufficient_information": [
                "https://",
            ],
        }

        for risk_type, patterns in (
            serious_patterns.items()
        ):

            for pattern in patterns:

                if pattern in text:

                    risks.append(
                        risk_type
                    )

                    break

        # -----------------------------------------------------
        # Very short messages
        # -----------------------------------------------------

        words = text.split()

        if len(words) <= 2:

            if (
                "insufficient_information"
                not in risks
            ):

                risks.append(
                    "insufficient_information"
                )

        return risks

    # =========================================================
    # EVIDENCE STRENGTH
    # =========================================================

    def calculate_evidence_strength(
        self,
        retrieval_results,
        evidence_result,
    ):

        if not retrieval_results:

            return 0.0

        # -----------------------------------------------------
        # Highest retrieval similarity
        # -----------------------------------------------------

        top_similarity = max(
            result["similarity"]
            for result in retrieval_results
        )

        # -----------------------------------------------------
        # Highest resolution agreement
        # -----------------------------------------------------

        agreements = [
            item["agreement"]
            for item in evidence_result.get(
                "evidence",
                []
            )
        ]

        max_agreement = (
            max(agreements)
            if agreements
            else 0.0
        )

        # -----------------------------------------------------
        # Normalize similarity
        # -----------------------------------------------------

        similarity_score = min(
            max(
                (
                    top_similarity - 0.10
                ) / 0.40,
                0.0
            ),
            1.0
        )

        # -----------------------------------------------------
        # Combined evidence score
        # -----------------------------------------------------

        evidence_score = (
            0.5 * similarity_score
            +
            0.5 * max_agreement
        )

        return round(
            evidence_score,
            3
        )

    # =========================================================
    # FINAL DECISION
    # =========================================================

    def decide(
        self,
        message,
        intent,
        intent_confidence,
        retrieval_results,
        evidence_result,
    ):

        risks = self.detect_risk_signals(
            message
        )

        # -----------------------------------------------------
        # Evidence metrics
        # -----------------------------------------------------

        evidence_strength = (
            self.calculate_evidence_strength(
                retrieval_results,
                evidence_result,
            )
        )

        top_similarity = max(
            (
                result["similarity"]
                for result in retrieval_results
            ),
            default=0.0,
        )

        max_agreement = max(
            (
                item["agreement"]
                for item in evidence_result.get(
                    "evidence",
                    []
                )
            ),
            default=0.0,
        )

        # =====================================================
        # HARD ESCALATION CONDITIONS
        # =====================================================

        # -----------------------------------------------------
        # Safety
        # -----------------------------------------------------

        if "safety" in risks:

            return {
                "decision": "ESCALATE",
                "reason": (
                    "Safety-related issue "
                    "requires human review."
                ),
                "intent": intent,
                "intent_confidence":
                    round(
                        intent_confidence,
                        3
                    ),
                "top_similarity":
                    round(
                        top_similarity,
                        3
                    ),
                "evidence_agreement":
                    round(
                        max_agreement,
                        3
                    ),
                "evidence_strength":
                    evidence_strength,
                "risk_signals":
                    risks,
            }

        # -----------------------------------------------------
        # Insufficient information
        # -----------------------------------------------------

        if (
            "insufficient_information"
            in risks
        ):

            return {
                "decision": "ESCALATE",
                "reason": (
                    "The customer message does "
                    "not contain enough information "
                    "for a reliable response."
                ),
                "intent": intent,
                "intent_confidence":
                    round(
                        intent_confidence,
                        3
                    ),
                "top_similarity":
                    round(
                        top_similarity,
                        3
                    ),
                "evidence_agreement":
                    round(
                        max_agreement,
                        3
                    ),
                "evidence_strength":
                    evidence_strength,
                "risk_signals":
                    risks,
            }

        # =====================================================
        # HIGH-RISK BUSINESS CONDITIONS
        # =====================================================

        risk_reasons = []

        if "active_disruption" in risks:

            risk_reasons.append(
                "active disruption detected"
            )

        if "financial_dispute" in risks:

            risk_reasons.append(
                "financial or transaction issue detected"
            )

        if "case_specific" in risks:

            risk_reasons.append(
                "existing case or unresolved contact detected"
            )

        # -----------------------------------------------------
        # High-risk conditions always escalate
        # -----------------------------------------------------

        if risk_reasons:

            return {
                "decision": "ESCALATE",
                "reason": "; ".join(
                    risk_reasons
                ),
                "intent": intent,
                "intent_confidence":
                    round(
                        intent_confidence,
                        3
                    ),
                "top_similarity":
                    round(
                        top_similarity,
                        3
                    ),
                "evidence_agreement":
                    round(
                        max_agreement,
                        3
                    ),
                "evidence_strength":
                    evidence_strength,
                "risk_signals":
                    risks,
            }

        # =====================================================
        # NORMAL SUPPORT REQUEST
        # =====================================================

        # -----------------------------------------------------
        # Strong case
        #
        # High classifier confidence AND reasonable evidence.
        # -----------------------------------------------------

        strong_case = (
            intent_confidence >= 0.55
            and top_similarity >= 0.20
            and max_agreement >= 0.20
        )

        if strong_case:

            return {
                "decision": "AUTO-HANDLE",
                "reason": (
                    "Intent confidence and "
                    "historical evidence provide "
                    "sufficient support for "
                    "automated handling."
                ),
                "intent": intent,
                "intent_confidence":
                    round(
                        intent_confidence,
                        3
                    ),
                "top_similarity":
                    round(
                        top_similarity,
                        3
                    ),
                "evidence_agreement":
                    round(
                        max_agreement,
                        3
                    ),
                "evidence_strength":
                    evidence_strength,
                "risk_signals":
                    risks,
            }

        # -----------------------------------------------------
        # Moderate-confidence case
        #
        # Allow automation when retrieval is useful and
        # classifier confidence is not extremely low.
        # -----------------------------------------------------

        moderate_case = (
            intent_confidence >= 0.35
            and top_similarity >= 0.30
            and max_agreement >= 0.20
        )

        if moderate_case:

            return {
                "decision": "AUTO-HANDLE",
                "reason": (
                    "The request has moderate "
                    "intent confidence and useful "
                    "historical evidence, with no "
                    "high-risk signals detected."
                ),
                "intent": intent,
                "intent_confidence":
                    round(
                        intent_confidence,
                        3
                    ),
                "top_similarity":
                    round(
                        top_similarity,
                        3
                    ),
                "evidence_agreement":
                    round(
                        max_agreement,
                        3
                    ),
                "evidence_strength":
                    evidence_strength,
                "risk_signals":
                    risks,
            }

        # =====================================================
        # VERY WEAK CASE
        # =====================================================

        reasons = []

        if (
            intent_confidence
            < 0.35
        ):

            reasons.append(
                "intent confidence is very low"
            )

        if (
            top_similarity
            < 0.20
        ):

            reasons.append(
                "historical retrieval evidence is weak"
            )

        if (
            max_agreement
            < 0.20
        ):

            reasons.append(
                "historical resolution agreement is weak"
            )

        if not reasons:

            reasons.append(
                "available evidence is insufficient "
                "for reliable automated handling"
            )

        return {
            "decision": "ESCALATE",
            "reason": "; ".join(
                reasons
            ),
            "intent": intent,
            "intent_confidence":
                round(
                    intent_confidence,
                    3
                ),
            "top_similarity":
                round(
                    top_similarity,
                    3
                ),
            "evidence_agreement":
                round(
                    max_agreement,
                    3
                ),
            "evidence_strength":
                evidence_strength,
            "risk_signals":
                risks,
        }


# =============================================================
# SIMPLE TEST
# =============================================================

if __name__ == "__main__":

    gate = EvidenceConfidenceGate()

    # ---------------------------------------------------------
    # Example 1
    # ---------------------------------------------------------

    result = gate.decide(

        message=(
            "My train is delayed by two hours, "
            "can I claim compensation?"
        ),

        intent="refund_compensation",

        intent_confidence=0.82,

        retrieval_results=[
            {"similarity": 0.27},
            {"similarity": 0.25},
            {"similarity": 0.24},
        ],

        evidence_result={
            "evidence": [
                {
                    "action":
                        "delay_repay",
                    "agreement":
                        0.40,
                },
                {
                    "action":
                        "website_form",
                    "agreement":
                        0.40,
                },
            ]
        },
    )

    print("=" * 60)
    print("EVIDENCE-CONFIDENCE GATE")
    print("=" * 60)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )