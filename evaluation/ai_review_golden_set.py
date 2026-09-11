from pathlib import Path
import csv

INPUT_FILE = Path("data/golden/golden_set_final.csv")
OUTPUT_FILE = Path("data/golden/golden_set_final.csv")

INTENTS = [
    "train_status",
    "ticket_change",
    "refund_compensation",
    "booking_ticket",
    "fare_price",
    "seat_reservation",
    "wifi_connectivity",
    "station_facilities",
    "lost_property",
    "complaint_feedback",
    "other"
]

def save(rows):
    fields = [
        "id",
        "customer_message",
        "historical_response",
        "intent",
        "expected_resolution",
        "should_escalate",
        "escalation_reason"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def suggest_intent(message):
    text = message.lower()

    if any(x in text for x in ["wifi", "wi-fi", "internet", "connection"]):
        return "wifi_connectivity"

    if any(x in text for x in ["seat", "seating", "reserved", "reservation"]):
        return "seat_reservation"

    if any(x in text for x in ["refund", "compensation", "delay repay", "repay"]):
        return "refund_compensation"

    if any(x in text for x in ["cancelled", "cancelled train", "delayed", "delay", "disruption"]):
        return "train_status"

    if any(x in text for x in ["change ticket", "change my ticket", "wrong date", "amend", "amendment"]):
        return "ticket_change"

    if any(x in text for x in ["price", "fare", "cost", "expensive", "how much"]):
        return "fare_price"

    if any(x in text for x in ["book", "booking", "ticket"]):
        return "booking_ticket"

    if any(x in text for x in ["lost", "lost property", "lost item"]):
        return "lost_property"

    if any(x in text for x in ["complaint", "complain", "poor service"]):
        return "complaint_feedback"

    if any(x in text for x in ["station", "platform", "lounge"]):
        return "station_facilities"

    return "other"

def suggest_resolution(intent):
    resolutions = {
        "train_status": "Provide current train status, disruption information, or appropriate alternative travel guidance.",
        "ticket_change": "Explain the appropriate ticket amendment, cancellation, or ticket-validity process.",
        "refund_compensation": "Explain the applicable refund or compensation process and relevant next steps.",
        "booking_ticket": "Provide guidance about booking, tickets, or ticket validity.",
        "fare_price": "Explain the relevant fare, pricing, or ticket-release information.",
        "seat_reservation": "Provide guidance about seat reservations or investigate the seating issue.",
        "wifi_connectivity": "Provide Wi-Fi information or troubleshooting guidance.",
        "station_facilities": "Provide information about the relevant station facility or service.",
        "lost_property": "Provide guidance for reporting or recovering lost property.",
        "complaint_feedback": "Provide the appropriate complaint or feedback process.",
        "other": "The message does not provide enough information for a specific support intent; request clarification."
    }
    return resolutions[intent]

def suggest_escalation(message, intent):
    text = message.lower()

    serious = [
        "threat",
        "threatened",
        "abuse",
        "abusive",
        "unsafe",
        "security",
        "police",
        "injured",
        "injury"
    ]

    disruption = [
        "cancelled",
        "cancelled train",
        "2 hour",
        "two hour",
        "stranded",
        "missed"
    ]

    if any(x in text for x in serious):
        return "yes", "Potential safety, conduct, or serious customer-service issue requiring human review."

    if intent == "other":
        words = text.split()
        if len(words) <= 3:
            return "yes", "The message is too short or ambiguous to determine the customer's request."

    if any(x in text for x in disruption):
        return "yes", "The issue may require case-specific or current operational investigation."

    return "no", ""

def main():
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("TRUSTSUPPORT AI + HUMAN REVIEW")
    print("========================================")
    print()
    print("Reading:", INPUT_FILE)
    print("Total rows:", len(rows))
    print("Already labeled:", sum(bool(r.get("intent", "").strip()) for r in rows))
    print()

    for row in rows:
        if row.get("intent", "").strip():
            continue

        print("=" * 70)
        print(f"Example {row['id']} / {len(rows)}")
        print("=" * 70)

        print()
        print("CUSTOMER:")
        print("-" * 70)
        print(row["customer_message"])

        print()
        print("HISTORICAL RESPONSE:")
        print("-" * 70)
        print(row["historical_response"])

        intent = suggest_intent(row["customer_message"])
        resolution = suggest_resolution(intent)
        escalate, reason = suggest_escalation(row["customer_message"], intent)

        print()
        print("AI PRE-LABEL SUGGESTION")
        print("-" * 70)
        print("Intent:    ", intent)
        print("Resolution:", resolution)
        print("Escalate:  ", escalate)

        choice = input("\n[A]ccept [C]hange intent [E]scalation [S]kip [Q]uit: ").strip().lower()

        if choice == "q":
            save(rows)
            print("\nSaved. Exiting.")
            return

        if choice == "s":
            continue

        if choice == "a":
            row["intent"] = intent
            row["expected_resolution"] = resolution
            row["should_escalate"] = escalate
            row["escalation_reason"] = reason

        elif choice == "c":
            print("\nINTENTS:")
            for i, item in enumerate(INTENTS, 1):
                print(f"{i}. {item}")

            while True:
                try:
                    number = int(input("\nEnter number: "))
                    if 1 <= number <= len(INTENTS):
                        break
                except ValueError:
                    pass
                print("Invalid number. Try again.")

            row["intent"] = INTENTS[number - 1]

            row["expected_resolution"] = input(
                "Expected resolution: "
            ).strip()

            row["should_escalate"] = input(
                "Should escalate? (y/n): "
            ).strip().lower()

            if row["should_escalate"] == "y":
                row["escalation_reason"] = input(
                    "Escalation reason: "
                ).strip()
            else:
                row["escalation_reason"] = ""

        elif choice == "e":
            row["intent"] = intent
            row["expected_resolution"] = input(
                "Expected resolution: "
            ).strip()

            row["should_escalate"] = input(
                "Should escalate? (y/n): "
            ).strip().lower()

            if row["should_escalate"] == "y":
                row["escalation_reason"] = input(
                    "Escalation reason: "
                ).strip()
            else:
                row["escalation_reason"] = ""

        save(rows)

    save(rows)
    print("\n========================================")
    print("ALL 200 EXAMPLES REVIEWED!")
    print("========================================")

if __name__ == "__main__":
    main()
