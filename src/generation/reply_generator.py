class GroundedReplyGenerator:

    def generate(
        self,
        message,
        intent,
        evidence_result,
        decision,
    ):

        evidence = evidence_result.get(
            "evidence",
            []
        )

        actions = [
            item["action"]
            for item in evidence
            if isinstance(item, dict)
            and item.get("action")
        ]

        decision_type = decision.get(
            "decision",
            "ESCALATE"
        )

        # =====================================================
        # NO USEFUL EVIDENCE
        # =====================================================

        if not actions:

            return (
                "Thanks for getting in touch. "
                "We need a little more information about "
                "your request so that we can provide the "
                "right assistance."
            )

        # =====================================================
        # REFUND / COMPENSATION
        # =====================================================

        if intent == "refund_compensation":

            if "delay_repay" in actions:

                return (
                    "Thanks for getting in touch. "
                    "Historical VirginTrains support responses "
                    "directed similar delayed-journey cases to "
                    "Delay Repay. Please check the relevant "
                    "Delay Repay process for your journey. "
                    "Eligibility should be confirmed against "
                    "your journey details."
                )

            if "refund" in actions:

                return (
                    "Thanks for getting in touch. "
                    "Historical VirginTrains responses directed "
                    "similar cases to the refund process. "
                    "Your ticket and journey details should be "
                    "checked to determine the appropriate process."
                )

        # =====================================================
        # TICKET CHANGE
        # =====================================================

        if intent == "ticket_change":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains responses directed "
                "similar ticket amendment requests to the "
                "Aftersales team. They can review the booking "
                "and advise on the available options."
            )

        # =====================================================
        # WIFI
        # =====================================================

        if intent == "wifi_connectivity":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains support responses "
                "provided guidance for onboard Wi-Fi issues. "
                "Please check the onboard connection and follow "
                "the available connection or login steps."
            )

        # =====================================================
        # SEAT RESERVATION
        # =====================================================

        if intent == "seat_reservation":

            return (
                "Thanks for getting in touch. "
                "This appears to relate to your seat reservation. "
                "Historical support responses handled similar "
                "requests by checking the reservation and the "
                "affected service."
            )

        # =====================================================
        # LOST PROPERTY
        # =====================================================

        if intent == "lost_property":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains responses directed "
                "lost-property enquiries to the Lost Property team. "
                "Please provide your journey and item details."
            )

        # =====================================================
        # FARE / PRICE
        # =====================================================

        if intent == "fare_price":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains responses provided fare "
                "information based on the journey and ticket type. "
                "Please provide your journey details and travel date."
            )

        # =====================================================
        # BOOKING / TICKET
        # =====================================================

        if intent == "booking_ticket":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains responses handled similar "
                "ticket and booking questions by checking the ticket "
                "type and journey details."
            )

        # =====================================================
        # STATION
        # =====================================================

        if intent == "station_facilities":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains responses provided "
                "information about station facilities. "
                "Please provide the station and facility you "
                "are asking about."
            )

        # =====================================================
        # COMPLAINT / FEEDBACK
        # =====================================================

        if intent == "complaint_feedback":

            return (
                "Thanks for getting in touch and sharing your "
                "feedback. Historical VirginTrains responses "
                "directed complaints and feedback to the "
                "appropriate support or customer-relations team."
            )

        # =====================================================
        # TRAIN STATUS
        # =====================================================

        if intent == "train_status":

            return (
                "Thanks for getting in touch. "
                "Historical VirginTrains responses handled similar "
                "service-status enquiries by providing information "
                "about delays, cancellations, or service disruption."
            )

        # =====================================================
        # OTHER / UNKNOWN
        # =====================================================

        return (
            "Thanks for getting in touch. "
            "We need a little more information about your request "
            "so that we can provide the appropriate assistance."
        )


# =============================================================
# SIMPLE TEST
# =============================================================

if __name__ == "__main__":

    generator = GroundedReplyGenerator()

    example_evidence = {

        "evidence": [

            {
                "action": "delay_repay",
                "support_count": 2,
                "agreement": 0.40,
            },

            {
                "action": "website_form",
                "support_count": 2,
                "agreement": 0.40,
            },

        ]
    }

    example_decision = {
        "decision": "AUTO-HANDLE"
    }

    reply = generator.generate(

        message=(
            "My train is delayed by two hours, "
            "can I claim compensation?"
        ),

        intent="refund_compensation",

        evidence_result=example_evidence,

        decision=example_decision,
    )

    print("=" * 60)
    print("GROUNDED REPLY GENERATOR")
    print("=" * 60)

    print()

    print("Draft reply:")

    print()

    print(reply)