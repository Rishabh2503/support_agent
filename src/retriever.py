import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = "data/working/train_pairs.csv"


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


class AppleSupportRetriever:

    def __init__(self, data_file=DATA_FILE):

        print("Loading historical AppleSupport conversations...")

        self.df = pd.read_csv(data_file)

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        required_columns = {
            "customer_text",
            "brand_response",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        # ----------------------------------------------------
        # Optional labels
        #
        # These are used if your training data contains them.
        # We do NOT invent labels if they are missing.
        # ----------------------------------------------------

        self.has_intent = "intent" in self.df.columns
        self.has_escalation = "escalate" in self.df.columns

        if self.has_intent:
            print("Historical intent labels detected.")

        if self.has_escalation:
            print("Historical escalation labels detected.")

        # ----------------------------------------------------
        # Clean customer text
        # ----------------------------------------------------

        self.df["clean_text"] = (
            self.df["customer_text"]
            .fillna("")
            .apply(clean_text)
        )

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            min_df=2,
            max_features=30_000,
        )

        print("Building retrieval index...")

        self.matrix = self.vectorizer.fit_transform(
            self.df["clean_text"]
        )

        print(
            f"Indexed {len(self.df):,} conversations."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(self, query, k=5):

        query_clean = clean_text(query)

        query_vector = self.vectorizer.transform(
            [query_clean]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        # Highest similarity first
        indices = scores.argsort()[-k:][::-1]

        results = self.df.iloc[indices].copy()

        results["similarity"] = scores[indices]

        # ----------------------------------------------------
        # Return useful historical information
        # ----------------------------------------------------

        columns = [
            "customer_text",
            "brand_response",
            "similarity",
        ]

        if self.has_intent:
            columns.append("intent")

        if self.has_escalation:
            columns.append("escalate")

        return results[
            columns
        ].reset_index(drop=True)


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = AppleSupportRetriever()

    while True:

        query = input(
            "\nCustomer message "
            "(or 'exit'): "
        )

        if query.lower() == "exit":
            break

        results = retriever.search(
            query,
            k=5,
        )

        print("\n" + "=" * 80)
        print("TOP HISTORICAL MATCHES")
        print("=" * 80)

        for i, row in results.iterrows():

            print(
                f"\n[{i + 1}] "
                f"similarity={row['similarity']:.3f}"
            )

            print(
                f"Customer: {row['customer_text']}"
            )

            print(
                f"AppleSupport: {row['brand_response']}"
            )

            if "intent" in results.columns:
                print(
                    f"Intent: {row['intent']}"
                )

            if "escalate" in results.columns:
                print(
                    f"Escalate: {row['escalate']}"
                )