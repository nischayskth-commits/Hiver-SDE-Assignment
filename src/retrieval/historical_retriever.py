import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path(
    "data/processed/virgintrains_support_pairs.csv"
)


class HistoricalRetriever:

    def __init__(self, data_file=DATA_FILE):

        self.data_file = Path(data_file)

        self.rows = []
        self.vectorizer = None
        self.matrix = None

        self._load()
        self._build_index()

    def _load(self):

        with self.data_file.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                message = row[
                    "customer_message"
                ].strip()

                response = row[
                    "brand_response"
                ].strip()

                if not message or not response:
                    continue

                self.rows.append({
                    "customer_tweet_id":
                        row["customer_tweet_id"],

                    "brand_tweet_id":
                        row["brand_tweet_id"],

                    "customer_message":
                        message,

                    "brand_response":
                        response,

                    "created_at":
                        row["created_at"],
                })

    def _clean(self, text):

        text = text.lower()

        # Remove URLs
        text = re.sub(
            r"https?://\S+",
            " ",
            text
        )

        # Remove Twitter mentions
        text = re.sub(
            r"@\w+",
            " ",
            text
        )

        # Normalize whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def _build_index(self):

        documents = [
            self._clean(
                row["customer_message"]
            )
            for row in self.rows
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        self.matrix = self.vectorizer.fit_transform(
            documents
        )

    def search(
        self,
        query,
        top_k=5,
        exclude_ids=None
    ):

        """
        Search historical customer-support interactions.

        Parameters
        ----------
        query : str
            New customer message.

        top_k : int
            Number of historical cases to return.

        exclude_ids : set/list/None
            Tweet IDs that must not appear in the results.
            Used during evaluation to prevent test leakage.
        """

        # Convert to a set for fast membership checks
        if exclude_ids is None:
            exclude_ids = set()
        else:
            exclude_ids = set(exclude_ids)

        cleaned_query = self._clean(query)

        query_vector = (
            self.vectorizer.transform(
                [cleaned_query]
            )
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        # Rank all cases from highest to lowest similarity
        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices:

            row = self.rows[index]

            # -------------------------------------------------
            # LEAKAGE PROTECTION
            # -------------------------------------------------
            # Skip this historical case if either its customer
            # tweet ID or brand response tweet ID is excluded.
            #
            # During evaluation, this prevents the golden
            # example itself from being retrieved.
            # -------------------------------------------------

            if (
                row["customer_tweet_id"] in exclude_ids
                or row["brand_tweet_id"] in exclude_ids
            ):
                continue

            results.append({
                "customer_message":
                    row["customer_message"],

                "brand_response":
                    row["brand_response"],

                "similarity":
                    float(similarities[index]),

                "customer_tweet_id":
                    row["customer_tweet_id"],

                "brand_tweet_id":
                    row["brand_tweet_id"],
            })

            # Stop once enough valid results have been collected
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":

    print(
        "Building historical retrieval index..."
    )

    retriever = HistoricalRetriever()

    print(
        f"Indexed {len(retriever.rows):,} "
        "historical support interactions."
    )

    query = input(
        "\nEnter a customer message: "
    )

    results = retriever.search(
        query,
        top_k=5
    )

    print()
    print("=" * 60)
    print("TOP HISTORICAL CASES")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"CASE {i} "
            f"(similarity="
            f"{result['similarity']:.4f})"
        )

        print(
            "Customer:"
        )

        print(
            result["customer_message"]
        )

        print(
            "\nVirginTrains response:"
        )

        print(
            result["brand_response"]
        )

        print(
            "\nCustomer tweet ID:"
        )

        print(
            result["customer_tweet_id"]
        )

        print(
            "Brand tweet ID:"
        )

        print(
            result["brand_tweet_id"]
        )