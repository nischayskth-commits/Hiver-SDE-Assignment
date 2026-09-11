import re


class InputQualityDetector:

    def __init__(self):

        # =====================================================
        # NON-ACTIONABLE / CONVERSATIONAL MESSAGES
        # =====================================================

        self.non_actionable_phrases = [
            "thanks",
            "thank you",
            "great thanks",
            "many thanks",
            "how are you",
            "good night",
            "good morning",
            "good evening",
            "well done",
            "love travelling",
            "yay",
            "resurrection",
        ]

        # =====================================================
        # SUPPORT SIGNALS
        # =====================================================

        self.support_signals = [
            "refund",
            "compensation",
            "delay",
            "delayed",
            "cancel",
            "cancelled",
            "ticket",
            "tickets",
            "book",
            "booking",
            "reserve",
            "reserved",
            "seat",
            "wifi",
            "internet",
            "fare",
            "price",
            "cost",
            "lost",
            "missing",
            "change",
            "amend",
            "station",
            "platform",
            "train",
            "service",
            "journey",
            "travel",
            "accepted",
            "valid",
            "claim",
            "charge",
            "charged",
            "complaint",
            "problem",
            "issue",
            "help",
            "working",
            "available",
            "wrong",
            "date",
        ]

        # =====================================================
        # EXPLICIT SUPPORT ACTIONS
        # =====================================================

        self.action_patterns = [
            r"\bcan i\b",
            r"\bcould i\b",
            r"\bdo i\b",
            r"\bwill i\b",
            r"\bhow do i\b",
            r"\bhow can i\b",
            r"\bwhere is\b",
            r"\bwhere are\b",
            r"\bwhen is\b",
            r"\bwhen will\b",
            r"\bwhy is\b",
            r"\bwhy are\b",
            r"\bi want\b",
            r"\bi need\b",
            r"\bi need help\b",
            r"\bplease help\b",
            r"\bplease check\b",
            r"\bwhat happens\b",
            r"\bwhat can i\b",
            r"\bis it\b",
            r"\bis there\b",
            r"\bcan you\b",
            r"\bcould you\b",
        ]

    # =========================================================
    # URL / IDENTIFIER HELPERS
    # =========================================================

    def _is_url_only(self, text):

        return bool(
            re.fullmatch(
                r"https?://\S+",
                text
            )
        )

    def _is_reference_only(self, text):

        compact = re.sub(
            r"[\s\-]+",
            "",
            text
        )

        return bool(
            re.fullmatch(
                r"[a-z]{2,8}\d{5,20}",
                compact
            )
        )

    def _is_number_only(self, text):

        compact = re.sub(
            r"[\s\-]+",
            "",
            text
        )

        return bool(
            re.fullmatch(
                r"\d{7,15}",
                compact
            )
        )

    # =========================================================
    # JOURNEY FRAGMENT DETECTOR
    # =========================================================

    def _is_journey_fragment(self, text):

        words = text.split()

        # Examples:
        #
        # 05.56 WBQ to EUS
        # 17.56 Crewe Euston
        # MCR - BHM
        # 8am one from Euston
        #
        # These often represent thread fragments rather
        # than complete customer requests.

        time_pattern = (
            r"\b\d{1,2}[:.]?\d{2}\b"
            r"|\b\d{1,2}\s?(?:am|pm)\b"
        )

        has_time = bool(
            re.search(
                time_pattern,
                text
            )
        )

        has_route = bool(
            re.search(
                r"\b(?:to|from|via)\b",
                text
            )
        )

        has_dash_route = bool(
            re.search(
                r"\b[a-z]{2,5}\s*-\s*[a-z]{2,5}\b",
                text
            )
        )

        # Very short route/time-only messages

        if len(words) <= 5:

            if has_time and (
                has_route
                or len(words) <= 4
            ):
                return True

            if has_dash_route:
                return True

        return False

    # =========================================================
    # CONVERSATIONAL MESSAGE DETECTOR
    # =========================================================

    def _is_conversational(self, text):

        # Short conversational statements without
        # a recognizable support request.

        conversational_patterns = [

            r"^his name'?s\b",
            r"^her name'?s\b",
            r"^their name'?s\b",

            r"^i'?ve screamed\b",
            r"^i'?m laughing\b",
            r"^lol\b",
            r"^haha\b",
            r"^many thanks\b",
            r"^thanks\b",
            r"^thank you\b",

            r"^please check your dms$",

        ]

        for pattern in conversational_patterns:

            if re.search(
                pattern,
                text
            ):
                return True

        return False

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(self, message):

        text = (
            message
            .strip()
            .lower()
        )

        # -----------------------------------------------------
        # EMPTY
        # -----------------------------------------------------

        if not text:

            return {
                "is_insufficient": True,
                "reason": "empty_message",
            }

        # -----------------------------------------------------
        # REMOVE TWITTER MENTIONS
        # -----------------------------------------------------

        cleaned = re.sub(
            r"@\w+",
            "",
            text
        ).strip()

        # -----------------------------------------------------
        # URL ONLY
        # -----------------------------------------------------

        if self._is_url_only(cleaned):

            return {
                "is_insufficient": True,
                "reason": "url_only",
            }

        # -----------------------------------------------------
        # REFERENCE ONLY
        # -----------------------------------------------------

        if self._is_reference_only(cleaned):

            return {
                "is_insufficient": True,
                "reason": "reference_only",
            }

        # -----------------------------------------------------
        # NUMBER ONLY
        # -----------------------------------------------------

        if self._is_number_only(cleaned):

            return {
                "is_insufficient": True,
                "reason": "number_only",
            }

        # -----------------------------------------------------
        # EXACT NON-ACTIONABLE PHRASES
        # -----------------------------------------------------

        for phrase in self.non_actionable_phrases:

            if cleaned == phrase:

                return {
                    "is_insufficient": True,
                    "reason": "non_actionable_message",
                }

        # -----------------------------------------------------
        # CONVERSATIONAL TWITTER FRAGMENT
        # -----------------------------------------------------

        if self._is_conversational(cleaned):

            return {
                "is_insufficient": True,
                "reason": "conversational_fragment",
            }

        # -----------------------------------------------------
        # JOURNEY / TIME FRAGMENT
        # -----------------------------------------------------

        if self._is_journey_fragment(cleaned):

            return {
                "is_insufficient": True,
                "reason": "journey_fragment",
            }

        # -----------------------------------------------------
        # SUPPORT SIGNAL
        # -----------------------------------------------------

        has_support_signal = any(
            signal in cleaned
            for signal in self.support_signals
        )

        # -----------------------------------------------------
        # EXPLICIT SUPPORT ACTION
        # -----------------------------------------------------

        has_action_pattern = any(
            re.search(
                pattern,
                cleaned
            )
            for pattern in self.action_patterns
        )

        # -----------------------------------------------------
        # VERY SHORT MESSAGE
        # -----------------------------------------------------

        words = cleaned.split()

        if len(words) <= 2:

            if not has_support_signal:

                return {
                    "is_insufficient": True,
                    "reason": "very_short_message",
                }

        # -----------------------------------------------------
        # QUESTION WITHOUT SUPPORT CONTEXT
        # -----------------------------------------------------

        if (
            cleaned.endswith("?")
            and not has_support_signal
            and not has_action_pattern
        ):

            return {
                "is_insufficient": True,
                "reason": "non_support_question",
            }

        # -----------------------------------------------------
        # OTHERWISE USABLE
        # -----------------------------------------------------

        return {
            "is_insufficient": False,
            "reason": "sufficient_information",
        }


# =============================================================
# TEST PROGRAM
# =============================================================

if __name__ == "__main__":

    detector = InputQualityDetector()

    test_messages = [

        # Short / insufficient
        "Manchester",
        "Thanks HP",
        "Great, thanks",
        "How are you?",
        "His name's Alex!",
        "I've screamed at the computer!",
        "please check your DMs",

        # Journey fragments
        "05.56 WBQ to EUS",
        "17.56 Crewe-Euston",
        "MCR - BHM",
        "8am one from Euston",
        "1013 from Milton Keynes to Glasgow",

        # Valid support requests
        "WiFi not working",
        "Need refund",
        "Train cancelled",
        "Wrong date",
        "My train is delayed by two hours",
        "I want to change my ticket",
        "How do I change my ticket?",
        "Where is my train?",
        "Can I get a refund?",
        "When will my refund arrive?",

        # Identifier / URL
        "https://t.co/example",
        "VTN171005BPCH",
        "03445565650",
    ]

    print("=" * 60)
    print("INPUT QUALITY DETECTOR V3")
    print("=" * 60)

    for message in test_messages:

        result = detector.analyze(
            message
        )

        print()
        print(
            f"Message: {message}"
        )

        print(
            f"Result: {result}"
        )