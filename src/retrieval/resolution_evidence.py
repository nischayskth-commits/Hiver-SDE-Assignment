import re
from collections import Counter


class ResolutionEvidenceExtractor:

    # ---------------------------------------------------------
    # Useful phrases that represent common support actions.
    #
    # These are evidence categories, NOT current policies.
    # ---------------------------------------------------------

    ACTION_PATTERNS = {

        "delay_repay": [
            r"delay repay",
            r"delayrepay",
            r"claim.*delay",
            r"compensation.*delay",
        ],

        "refund": [
            r"refund",
            r"money back",
            r"reimburse",
            r"repayment",
        ],

        "contact_aftersales": [
            r"aftersales",
            r"after sales",
            r"amend.*booking",
            r"amend.*ticket",
        ],

        "contact_customer_resolutions": [
            r"customer resolutions",
            r"customer relations",
            r"friends at @",
        ],

        "direct_message": [
            r"\bdm\b",
            r"direct message",
            r"send us a",
            r"private message",
            r"contact us",
        ],

        "station_staff": [
            r"station staff",
            r"staff at the station",
            r"train manager",
        ],

        "website_form": [
            r"website",
            r"form",
            r"online",
        ],

        "ticket_validity": [
            r"ticket.*valid",
            r"valid.*ticket",
            r"ticket.*accepted",
            r"accepted.*ticket",
        ],

        "wifi_support": [
            r"wifi",
            r"wi-fi",
            r"internet",
        ],

        "lost_property_team": [
            r"lost property",
            r"lost property team",
        ],
    }


    def _normalize(self, text):

        text = text.lower()

        text = re.sub(
            r"https?://\S+",
            " ",
            text
        )

        text = re.sub(
            r"@\w+",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    def extract(self, retrieved_cases):

        evidence_counts = Counter()

        supporting_cases = {}

        for case in retrieved_cases:

            response = self._normalize(
                case["brand_response"]
            )

            for action, patterns in (
                self.ACTION_PATTERNS.items()
            ):

                matched = False

                for pattern in patterns:

                    if re.search(
                        pattern,
                        response
                    ):

                        evidence_counts[action] += 1

                        supporting_cases.setdefault(
                            action,
                            []
                        ).append(case)

                        matched = True
                        break

        total_cases = len(
            retrieved_cases
        )

        evidence = []

        for action, count in (
            evidence_counts.most_common()
        ):

            agreement = (
                count / total_cases
                if total_cases
                else 0
            )

            evidence.append({
                "action": action,
                "support_count": count,
                "agreement": round(
                    agreement,
                    3
                ),
            })

        return {
            "total_cases": total_cases,
            "evidence": evidence,
            "supporting_cases": supporting_cases,
        }


def print_evidence(result):

    print()
    print("=" * 60)
    print("RESOLUTION EVIDENCE")
    print("=" * 60)

    print(
        f"Historical cases analysed: "
        f"{result['total_cases']}"
    )

    if not result["evidence"]:

        print(
            "\nNo recognizable resolution "
            "patterns were found."
        )

        return

    for item in result["evidence"]:

        print()
        print(
            f"Action: {item['action']}"
        )

        print(
            f"Supporting cases: "
            f"{item['support_count']}"
        )

        print(
            f"Agreement: "
            f"{item['agreement']:.0%}"
        )


if __name__ == "__main__":

    # Small standalone demonstration.
    #
    # In the final pipeline this class will receive
    # results directly from HistoricalRetriever.

    from historical_retriever import (
        HistoricalRetriever
    )

    retriever = HistoricalRetriever()

    extractor = (
        ResolutionEvidenceExtractor()
    )

    query = input(
        "Enter a customer message: "
    )

    cases = retriever.search(
        query,
        top_k=5
    )

    result = extractor.extract(
        cases
    )

    print_evidence(result)