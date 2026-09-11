import csv
import os
import sys
import re


# =============================================================
# FILE PATHS
# =============================================================

GOLDEN_FILE = "data/golden/golden_set_final.csv"

PROCESSED_FILE = (
    "data/processed/virgintrains_support_pairs.csv"
)

OUTPUT_FILE = (
    "evaluation/reply_quality_metrics.csv"
)


# =============================================================
# PROJECT IMPORT PATH
# =============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from src.trustsupport import TrustSupport


# =============================================================
# LOAD GOLDEN SET
# =============================================================

with open(
    GOLDEN_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    golden_rows = list(reader)


# =============================================================
# TEXT NORMALIZATION
# =============================================================

def normalize(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =============================================================
# BUILD GOLDEN EXCLUSION IDS
# =============================================================
#
# The golden set contains:
#   customer_message
#   historical_response
#
# The processed dataset contains:
#   customer_message
#   brand_response
#   customer_tweet_id
#   brand_tweet_id
#
# We match the golden interaction against the processed
# interaction and collect BOTH tweet IDs.
#
# Those IDs are then excluded from retrieval during evaluation.
#
# This prevents the evaluation example from retrieving its own
# historical conversation.
# =============================================================

print()
print("=" * 70)
print("BUILDING GOLDEN RETRIEVAL EXCLUSION SET")
print("=" * 70)


golden_pairs = set()

for row in golden_rows:

    customer_message = normalize(
        row.get(
            "customer_message",
            ""
        )
    )

    historical_response = normalize(
        row.get(
            "historical_response",
            ""
        )
    )

    if customer_message and historical_response:

        golden_pairs.add(
            (
                customer_message,
                historical_response
            )
        )


print(
    f"Golden interaction pairs: "
    f"{len(golden_pairs)}"
)


golden_exclusion_ids = set()

processed_count = 0
matched_count = 0


with open(
    PROCESSED_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        processed_count += 1

        customer_message = normalize(
            row.get(
                "customer_message",
                ""
            )
        )

        brand_response = normalize(
            row.get(
                "brand_response",
                ""
            )
        )

        pair = (
            customer_message,
            brand_response
        )

        if pair in golden_pairs:

            customer_id = str(
                row.get(
                    "customer_tweet_id",
                    ""
                )
            ).strip()

            brand_id = str(
                row.get(
                    "brand_tweet_id",
                    ""
                )
            ).strip()

            if customer_id:
                golden_exclusion_ids.add(
                    customer_id
                )

            if brand_id:
                golden_exclusion_ids.add(
                    brand_id
                )

            matched_count += 1


print(
    f"Processed interactions scanned: "
    f"{processed_count}"
)

print(
    f"Golden interactions matched: "
    f"{matched_count}"
)

print(
    f"Tweet IDs excluded from retrieval: "
    f"{len(golden_exclusion_ids)}"
)


# =============================================================
# INITIALIZE TRUSTSUPPORT
# =============================================================

print()
print("=" * 70)
print("TRUSTSUPPORT REPLY QUALITY EVALUATION")
print("=" * 70)

print()

print(
    f"Golden examples loaded: "
    f"{len(golden_rows)}"
)

print()

print(
    "Loading TrustSupport pipeline..."
)

agent = TrustSupport()

print(
    "TrustSupport pipeline loaded."
)


# =============================================================
# GROUNDING
# =============================================================

def tokenize(text):

    normalized = normalize(
        text
    )

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


def calculate_grounding(
    reply,
    evidence_responses
):

    reply_tokens = tokenize(
        reply
    )

    if not reply_tokens:
        return 0.0

    evidence_tokens = set()

    for response in evidence_responses:

        evidence_tokens.update(
            tokenize(response)
        )

    if not evidence_tokens:
        return 0.0

    overlap = (
        len(
            reply_tokens
            &
            evidence_tokens
        )
        /
        len(reply_tokens)
    )

    return round(
        overlap,
        3
    )


# =============================================================
# UNSUPPORTED CLAIM DETECTION
# =============================================================

def detect_unsupported_patterns(
    reply
):

    patterns = [

        r"\bguarantee\b",

        r"\bguaranteed\b",

        r"\bdefinitely\b",

        r"\byou will receive\b",

        r"\byou are entitled\b",

        r"\bwe will refund\b",

        r"\bwe have refunded\b",

        r"\bwithin \d+ (?:hours|days)\b",

        r"\bautomatically\b",

        r"\bwill be paid\b",

        r"\bwe promise\b",

    ]

    found = []

    text = reply.lower()

    for pattern in patterns:

        if re.search(
            pattern,
            text
        ):

            found.append(
                pattern
            )

    return found


# =============================================================
# REPLY LENGTH
# =============================================================

def calculate_length_score(
    reply
):

    count = len(
        reply.split()
    )

    if count == 0:
        return 0

    if count < 8:
        return 1

    if count < 15:
        return 2

    if count < 30:
        return 3

    if count < 60:
        return 4

    return 5


# =============================================================
# INTENT RELEVANCE
# =============================================================

INTENT_KEYWORDS = {

    "train_status": [
        "delay",
        "delayed",
        "cancel",
        "cancelled",
        "service",
        "train",
        "running",
    ],

    "ticket_change": [
        "change",
        "amend",
        "modify",
        "ticket",
    ],

    "refund_compensation": [
        "refund",
        "compensation",
        "repay",
        "claim",
        "reimbursement",
    ],

    "booking_ticket": [
        "ticket",
        "booking",
        "book",
        "journey",
        "travel",
    ],

    "fare_price": [
        "fare",
        "price",
        "cost",
        "ticket",
    ],

    "seat_reservation": [
        "seat",
        "reservation",
        "reserved",
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "internet",
        "connection",
        "connectivity",
    ],

    "station_facilities": [
        "station",
        "platform",
        "facility",
        "facilities",
    ],

    "lost_property": [
        "lost",
        "property",
        "item",
        "belongings",
        "found",
    ],

    "complaint_feedback": [
        "complaint",
        "support",
        "customer",
        "complain",
        "feedback",
    ],

    "other": [
        "information",
        "details",
        "support",
        "help",
    ],
}


def check_intent_relevance(
    reply,
    intent
):

    reply_lower = normalize(
        reply
    )

    keywords = INTENT_KEYWORDS.get(
        intent,
        []
    )

    if not keywords:
        return False

    return any(
        keyword in reply_lower
        for keyword in keywords
    )


# =============================================================
# EXTRACT RETRIEVED EVIDENCE
# =============================================================

def extract_evidence_responses(
    result
):

    responses = []

    retrieved_cases = result.get(
        "retrieved_cases",
        []
    )

    for case in retrieved_cases:

        if not isinstance(
            case,
            dict
        ):
            continue

        response = case.get(
            "brand_response",
            ""
        )

        if response:

            responses.append(
                response
            )

    return responses


# =============================================================
# EXTRACT EVIDENCE ACTIONS
# =============================================================

def extract_evidence_actions(
    result
):

    actions = []

    evidence = result.get(
        "evidence",
        {}
    )

    evidence_items = evidence.get(
        "evidence",
        []
    )

    for item in evidence_items:

        if not isinstance(
            item,
            dict
        ):
            continue

        action = item.get(
            "action",
            ""
        )

        if action:

            actions.append(
                action
            )

    return actions


# =============================================================
# EVALUATION
# =============================================================

results = []


for index, row in enumerate(
    golden_rows,
    start=1
):

    customer_message = row.get(
        "customer_message",
        ""
    )

    true_intent = row.get(
        "intent",
        ""
    )

    expected_resolution = row.get(
        "expected_resolution",
        ""
    )

    true_escalation_raw = row.get(
        "should_escalate",
        ""
    )

    true_escalation = (
        str(true_escalation_raw)
        .strip()
        .lower()
        in [
            "true",
            "1",
            "yes",
            "y",
        ]
    )


    # ---------------------------------------------------------
    # RUN PIPELINE
    # ---------------------------------------------------------

    try:

        result = agent.analyze(
            customer_message,
            exclude_ids=golden_exclusion_ids
        )

    except Exception as error:

        print()

        print(
            f"ERROR on example "
            f"{index}: {error}"
        )

        continue


    # ---------------------------------------------------------
    # INTENT
    # ---------------------------------------------------------

    predicted_intent = str(
        result.get(
            "intent",
            ""
        )
    )

    intent_confidence = float(
        result.get(
            "intent_confidence",
            0.0
        )
    )


    # ---------------------------------------------------------
    # DECISION
    # ---------------------------------------------------------

    decision = result.get(
        "decision",
        {}
    )

    decision_name = str(
        decision.get(
            "decision",
            ""
        )
    )

    predicted_escalation = (
        decision_name
        == "ESCALATE"
    )

    decision_reason = str(
        decision.get(
            "reason",
            ""
        )
    )

    top_similarity = float(
        decision.get(
            "top_similarity",
            0.0
        )
    )

    evidence_agreement = float(
        decision.get(
            "evidence_agreement",
            0.0
        )
    )

    evidence_strength = float(
        decision.get(
            "evidence_strength",
            0.0
        )
    )

    risk_signals = decision.get(
        "risk_signals",
        []
    )


    # ---------------------------------------------------------
    # REPLY
    # ---------------------------------------------------------

    draft_reply = str(
        result.get(
            "reply",
            ""
        )
    )


    # ---------------------------------------------------------
    # RETRIEVED EVIDENCE
    # ---------------------------------------------------------

    evidence_responses = (
        extract_evidence_responses(
            result
        )
    )

    evidence_actions = (
        extract_evidence_actions(
            result
        )
    )


    # ---------------------------------------------------------
    # VERIFY LEAKAGE PROTECTION
    # ---------------------------------------------------------

    retrieved_ids = []

    for case in result.get(
        "retrieved_cases",
        []
    ):

        if not isinstance(
            case,
            dict
        ):
            continue

        customer_id = str(
            case.get(
                "customer_tweet_id",
                ""
            )
        ).strip()

        brand_id = str(
            case.get(
                "brand_tweet_id",
                ""
            )
        ).strip()

        if customer_id:
            retrieved_ids.append(
                customer_id
            )

        if brand_id:
            retrieved_ids.append(
                brand_id
            )


    leakage_detected = any(
        tweet_id in golden_exclusion_ids
        for tweet_id in retrieved_ids
    )


    # ---------------------------------------------------------
    # GROUNDING
    # ---------------------------------------------------------

    grounding = calculate_grounding(
        draft_reply,
        evidence_responses
    )


    # ---------------------------------------------------------
    # UNSUPPORTED CLAIMS
    # ---------------------------------------------------------

    unsupported = (
        detect_unsupported_patterns(
            draft_reply
        )
    )


    # ---------------------------------------------------------
    # LENGTH
    # ---------------------------------------------------------

    length_score = (
        calculate_length_score(
            draft_reply
        )
    )


    # ---------------------------------------------------------
    # INTENT RELEVANCE
    # ---------------------------------------------------------

    intent_relevant = (
        check_intent_relevance(
            draft_reply,
            predicted_intent
        )
    )


    # ---------------------------------------------------------
    # CORRECTNESS
    # ---------------------------------------------------------

    intent_correct = (
        predicted_intent
        == true_intent
    )

    escalation_correct = (
        predicted_escalation
        == true_escalation
    )


    # ---------------------------------------------------------
    # EVIDENCE AVAILABILITY
    # ---------------------------------------------------------

    evidence_available = (
        len(evidence_responses)
        > 0
    )


    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    results.append({

        "id":
            row.get(
                "id",
                ""
            ),

        "customer_message":
            customer_message,

        "true_intent":
            true_intent,

        "predicted_intent":
            predicted_intent,

        "intent_correct":
            intent_correct,

        "intent_confidence":
            round(
                intent_confidence,
                4
            ),

        "true_escalation":
            true_escalation,

        "predicted_escalation":
            predicted_escalation,

        "escalation_correct":
            escalation_correct,

        "decision":
            decision_name,

        "decision_reason":
            decision_reason,

        "expected_resolution":
            expected_resolution,

        "draft_reply":
            draft_reply,

        "grounding_overlap":
            grounding,

        "top_similarity":
            round(
                top_similarity,
                4
            ),

        "evidence_strength":
            round(
                evidence_strength,
                4
            ),

        "evidence_agreement":
            round(
                evidence_agreement,
                4
            ),

        "risk_signals":
            "|".join(
                str(x)
                for x in risk_signals
            ),

        "evidence_available":
            evidence_available,

        "evidence_case_count":
            len(evidence_responses),

        "evidence_actions":
            "|".join(
                evidence_actions
            ),

        "unsupported_claim_count":
            len(unsupported),

        "unsupported_patterns":
            "|".join(
                unsupported
            ),

        "length_score":
            length_score,

        "intent_relevant":
            intent_relevant,

        "retrieval_leakage":
            leakage_detected,

    })


    # ---------------------------------------------------------
    # PROGRESS
    # ---------------------------------------------------------

    if index % 25 == 0:

        print(
            f"Processed "
            f"{index}/{len(golden_rows)}"
        )


# =============================================================
# STOP IF NOTHING WAS EVALUATED
# =============================================================

if not results:

    print()

    print(
        "ERROR: No examples were evaluated."
    )

    raise SystemExit(1)


# =============================================================
# AGGREGATE
# =============================================================

count = len(results)


intent_accuracy = (
    sum(
        1
        for r in results
        if r["intent_correct"]
    )
    / count
)


escalation_accuracy = (
    sum(
        1
        for r in results
        if r["escalation_correct"]
    )
    / count
)


avg_grounding = (
    sum(
        r["grounding_overlap"]
        for r in results
    )
    / count
)


avg_similarity = (
    sum(
        r["top_similarity"]
        for r in results
    )
    / count
)


avg_strength = (
    sum(
        r["evidence_strength"]
        for r in results
    )
    / count
)


avg_agreement = (
    sum(
        r["evidence_agreement"]
        for r in results
    )
    / count
)


avg_length = (
    sum(
        r["length_score"]
        for r in results
    )
    / count
)


relevant_count = sum(
    1
    for r in results
    if r["intent_relevant"]
)


unsupported_count = sum(
    r["unsupported_claim_count"]
    for r in results
)


empty_count = sum(
    1
    for r in results
    if not r["draft_reply"].strip()
)


evidence_count = sum(
    1
    for r in results
    if r["evidence_available"]
)


auto_count = sum(
    1
    for r in results
    if not r["predicted_escalation"]
)


escalate_count = sum(
    1
    for r in results
    if r["predicted_escalation"]
)


leakage_count = sum(
    1
    for r in results
    if r["retrieval_leakage"]
)


# =============================================================
# PRINT SUMMARY
# =============================================================

print()

print("=" * 70)
print("LEAKAGE-SAFE AUTOMATED REPLY QUALITY SUMMARY")
print("=" * 70)

print()

print(
    f"Examples evaluated: "
    f"{count}"
)

print(
    f"Golden IDs excluded: "
    f"{len(golden_exclusion_ids)}"
)

print(
    f"Retrieval leakage detected: "
    f"{leakage_count}"
)

print()

print(
    f"Intent accuracy: "
    f"{intent_accuracy:.3f}"
)

print(
    f"Escalation accuracy: "
    f"{escalation_accuracy:.3f}"
)

print()

print(
    f"Average lexical grounding: "
    f"{avg_grounding:.3f}"
)

print(
    f"Average retrieval similarity: "
    f"{avg_similarity:.3f}"
)

print(
    f"Average evidence strength: "
    f"{avg_strength:.3f}"
)

print(
    f"Average evidence agreement: "
    f"{avg_agreement:.3f}"
)

print(
    f"Average reply length score: "
    f"{avg_length:.2f}/5"
)

print()

print(
    f"Replies with historical evidence: "
    f"{evidence_count}/{count}"
)

print(
    f"Intent-relevant replies: "
    f"{relevant_count}/{count}"
)

print(
    f"Unsupported claim patterns: "
    f"{unsupported_count}"
)

print(
    f"Empty replies: "
    f"{empty_count}/{count}"
)

print()

print(
    f"AUTO-HANDLE: "
    f"{auto_count}"
)

print(
    f"ESCALATE: "
    f"{escalate_count}"
)


# =============================================================
# SAMPLE RESULTS
# =============================================================

print()

print("=" * 70)
print("SAMPLE GENERATED REPLIES")
print("=" * 70)


for result in results[:15]:

    print()

    print("-" * 70)

    print(
        f"ID: "
        f"{result['id']}"
    )

    print(
        f"True intent: "
        f"{result['true_intent']}"
    )

    print(
        f"Predicted intent: "
        f"{result['predicted_intent']}"
    )

    print(
        f"Customer: "
        f"{result['customer_message']}"
    )

    print()

    print(
        f"Reply: "
        f"{result['draft_reply']}"
    )

    print()

    print(
        f"Grounding: "
        f"{result['grounding_overlap']}"
    )

    print(
        f"Retrieval similarity: "
        f"{result['top_similarity']}"
    )

    print(
        f"Evidence strength: "
        f"{result['evidence_strength']}"
    )

    print(
        f"Evidence agreement: "
        f"{result['evidence_agreement']}"
    )

    print(
        f"Evidence actions: "
        f"{result['evidence_actions']}"
    )

    print(
        f"Intent relevant: "
        f"{result['intent_relevant']}"
    )

    print(
        f"Decision: "
        f"{result['decision']}"
    )

    print(
        f"Unsupported claims: "
        f"{result['unsupported_patterns']}"
    )

    print(
        f"Retrieval leakage: "
        f"{result['retrieval_leakage']}"
    )


# =============================================================
# SAVE RESULTS
# =============================================================

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

    "decision",

    "decision_reason",

    "expected_resolution",

    "draft_reply",

    "grounding_overlap",

    "top_similarity",

    "evidence_strength",

    "evidence_agreement",

    "risk_signals",

    "evidence_available",

    "evidence_case_count",

    "evidence_actions",

    "unsupported_claim_count",

    "unsupported_patterns",

    "length_score",

    "intent_relevant",

    "retrieval_leakage",

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

    writer.writerows(
        results
    )


# =============================================================
# FINAL
# =============================================================

print()

print("=" * 70)

print(
    "Leakage-safe reply evaluation completed successfully."
)

print()

print(
    f"Detailed results saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)