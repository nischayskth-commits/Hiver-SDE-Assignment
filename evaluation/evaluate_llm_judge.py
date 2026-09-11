import os
import csv
import json
import sys
from pathlib import Path

from openai import OpenAI

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trustsupport import TrustSupport


GOLDEN_PATH = ROOT / "data" / "golden" / "golden_set_final.csv"
OUTPUT_PATH = ROOT / "evaluation" / "llm_judge_results.csv"

MODEL = "gpt-5.6"


def load_golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def judge_case(client, row, result):
    retrieved_cases = result.get("retrieved_cases", [])
    evidence = result.get("evidence", {})
    decision = result.get("decision", {})

    historical_responses = [
        {
            "customer_message": case.get("customer_message", ""),
            "brand_response": case.get("brand_response", ""),
            "similarity": round(float(case.get("similarity", 0)), 3),
        }
        for case in retrieved_cases
    ]

    prompt = f"""
You are evaluating an AI customer-support agent for a software engineering
internship assignment.

The agent is called TrustSupport. It predicts an intent, retrieves historical
support interactions, extracts possible resolution actions, drafts a response,
and decides whether to AUTO-HANDLE or ESCALATE.

IMPORTANT:
- Historical responses are evidence of how the brand responded in the past.
- They are NOT guaranteed to be current policy.
- Do not reward unsupported claims.
- Evaluate the generated reply itself.
- If the predicted intent is wrong, consider whether that caused the reply
  to be inappropriate.
- A concise reply can still receive a high score if it appropriately handles
  the customer message.
- Do not require the reply to solve something that cannot reasonably be solved
  from the available evidence.

CUSTOMER MESSAGE:
{row["customer_message"]}

GOLD INTENT:
{row["intent"]}

GOLD EXPECTED RESOLUTION:
{row["expected_resolution"]}

GOLD ESCALATION:
{row["should_escalate"]}

GOLD ESCALATION REASON:
{row["escalation_reason"]}

PREDICTED INTENT:
{result.get("intent")}

INTENT CONFIDENCE:
{result.get("intent_confidence")}

DECISION:
{decision.get("decision")}

DECISION REASON:
{decision.get("reason")}

EXTRACTED EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}

RETRIEVED HISTORICAL CASES:
{json.dumps(historical_responses, ensure_ascii=False)}

GENERATED REPLY:
{result.get("reply", "")}

Score the response using this rubric.

1. groundedness:
5 = Clearly grounded in retrieved historical evidence and makes no
    unsupported operational/policy claims.
4 = Mostly grounded, with only minor generalization.
3 = Partly grounded, but some claims are weakly supported.
2 = Mostly generic or weakly connected to evidence.
1 = Unsupported or contradictory to the available evidence.

2. resolution_correctness:
5 = Directly addresses the customer's actual request and expected resolution.
4 = Mostly correct, with a minor omission.
3 = Partially correct.
2 = Substantially misses the resolution.
1 = Wrong or harmful resolution.

3. helpfulness:
5 = Gives a useful next step or answer.
4 = Helpful but somewhat incomplete.
3 = Some useful guidance.
2 = Very limited usefulness.
1 = Not useful.

4. completeness:
5 = Covers the important information reasonably possible from the evidence.
4 = Minor omission.
3 = Several omissions.
2 = Major omissions.
1 = Does not meaningfully address the request.

5. tone:
5 = Professional, clear, concise and appropriate for customer support.
4 = Good tone with minor awkwardness.
3 = Acceptable but generic/awkward.
2 = Poor customer-support tone.
1 = Clearly inappropriate.

6. unsupported_claims:
5 = No meaningful unsupported claims.
4 = One minor unsupported/generalized claim.
3 = Some unsupported claims.
2 = Several unsupported claims.
1 = Major unsupported claims or invented policy/facts.

Return ONLY valid JSON with exactly these fields:

{{
  "groundedness": integer,
  "resolution_correctness": integer,
  "helpfulness": integer,
  "completeness": integer,
  "tone": integer,
  "unsupported_claims": integer,
  "overall": number,
  "brief_reason": "one or two sentences"
}}

The overall score should be the arithmetic mean of the six category scores,
rounded to two decimal places.
"""


    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    text = response.output_text.strip()

    # Remove accidental markdown fences
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    rows = load_golden()

    print(f"Golden examples loaded: {len(rows)}")

    agent = TrustSupport()

    results = []

    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] Judging example {row['id']}...")

        result = agent.analyze(row["customer_message"])

        try:
            judgment = judge_case(client, row, result)
        except Exception as e:
            print(f"  Judge error: {e}")
            judgment = {
                "groundedness": "",
                "resolution_correctness": "",
                "helpfulness": "",
                "completeness": "",
                "tone": "",
                "unsupported_claims": "",
                "overall": "",
                "brief_reason": f"Judge error: {e}",
            }

        results.append({
            "id": row["id"],
            "customer_message": row["customer_message"],
            "gold_intent": row["intent"],
            "predicted_intent": result.get("intent"),
            "intent_correct": (
                result.get("intent") == row["intent"]
            ),
            "decision": result.get("decision", {}).get("decision"),
            "gold_escalation": row["should_escalate"],
            "predicted_escalation": (
                "yes"
                if result.get("decision", {}).get("decision") == "ESCALATE"
                else "no"
            ),
            "draft_reply": result.get("reply", ""),
            "groundedness": judgment.get("groundedness"),
            "resolution_correctness": judgment.get(
                "resolution_correctness"
            ),
            "helpfulness": judgment.get("helpfulness"),
            "completeness": judgment.get("completeness"),
            "tone": judgment.get("tone"),
            "unsupported_claims": judgment.get("unsupported_claims"),
            "overall": judgment.get("overall"),
            "brief_reason": judgment.get("brief_reason"),
        })

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    valid = [
        r for r in results
        if r["overall"] != ""
    ]

    if valid:
        metrics = [
            "groundedness",
            "resolution_correctness",
            "helpfulness",
            "completeness",
            "tone",
            "unsupported_claims",
            "overall",
        ]

        print("\n=== LLM JUDGE RESULTS ===")

        for metric in metrics:
            values = [float(r[metric]) for r in valid]
            average = sum(values) / len(values)
            print(f"{metric}: {average:.3f}")

        print(f"\nValid judged examples: {len(valid)}/{len(results)}")

    print(f"\nSaved results to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()