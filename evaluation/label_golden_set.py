import csv
from pathlib import Path

INPUT_FILE = Path("data/golden/golden_set_v2.csv")
OUTPUT_FILE = Path("data/golden/golden_set_labeled.csv")

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


def get_input(prompt):
    """Read input and allow quitting."""
    value = input(prompt).strip()

    if value.lower() == "quit":
        print("\nLabeling paused. Your progress has been saved.")
        raise SystemExit

    return value


def show_intents():
    print("\nChoose intent:")

    for i, intent in enumerate(INTENTS, start=1):
        print(f"{i:2}. {intent}")

    while True:
        choice = get_input("\nEnter number: ")

        try:
            number = int(choice)

            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1]

        except ValueError:
            pass

        print("Invalid choice. Enter a number from 1 to 11.")


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    # ---------------------------------------------------------
    # Read existing golden set
    # ---------------------------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # ---------------------------------------------------------
    # Read previous progress if available
    # ---------------------------------------------------------

    if OUTPUT_FILE.exists():

        print("Previous labeling progress found.")

        with open(OUTPUT_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            labeled_rows = list(reader)

        # Use previously labeled version
        rows = labeled_rows

    print("\n========================================")
    print(" TRUSTSUPPORT GOLDEN SET LABELER")
    print("========================================")

    print(f"\nTotal examples: {len(rows)}")
    print("\nCommands:")
    print("  quit  = save and exit")

    # ---------------------------------------------------------
    # Find first unlabeled example
    # ---------------------------------------------------------

    start_index = 0

    for i, row in enumerate(rows):

        if not row.get("intent", "").strip():
            start_index = i
            break

    else:
        print("\nAll examples are already labeled!")
        return

    # ---------------------------------------------------------
    # Label examples
    # ---------------------------------------------------------

    for index in range(start_index, len(rows)):

        row = rows[index]

        print("\n")
        print("=" * 70)
        print(f"Example {index + 1} / {len(rows)}")
        print("=" * 70)

        print("\nCUSTOMER MESSAGE:")
        print("-" * 70)
        print(row["customer_message"])

        print("\nHISTORICAL VIRGIN TRAINS RESPONSE:")
        print("-" * 70)
        print(row["historical_response"])

        # -----------------------------------------------------
        # Intent
        # -----------------------------------------------------

        intent = show_intents()

        # -----------------------------------------------------
        # Expected resolution
        # -----------------------------------------------------

        print("\nWhat should a good support agent do?")
        print("Write a short expected resolution.")
        print("Example:")
        print("Explain Delay Repay eligibility and provide the appropriate claim process.")

        expected_resolution = get_input("\nExpected resolution: ")

        # -----------------------------------------------------
        # Escalation
        # -----------------------------------------------------

        while True:

            escalate = get_input(
                "\nShould this case be escalated? (y/n): "
            ).lower()

            if escalate in ["y", "n"]:
                break

            print("Please enter y or n.")

        should_escalate = "yes" if escalate == "y" else "no"

        # -----------------------------------------------------
        # Escalation reason
        # -----------------------------------------------------

        if should_escalate == "yes":

            print("\nWhy should it be escalated?")

            escalation_reason = get_input(
                "Escalation reason: "
            )

        else:

            escalation_reason = ""

        # -----------------------------------------------------
        # Save labels
        # -----------------------------------------------------

        row["intent"] = intent
        row["expected_resolution"] = expected_resolution
        row["should_escalate"] = should_escalate
        row["escalation_reason"] = escalation_reason

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
            writer.writerows(rows)

        print("\n✓ Saved.")

    print("\n========================================")
    print(" ALL 200 EXAMPLES LABELED!")
    print("========================================")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()