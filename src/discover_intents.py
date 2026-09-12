import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


INPUT = "data/working/apple_pairs.csv"

N_CLUSTERS = 10
RANDOM_STATE = 42


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove "RT"
    text = re.sub(r"\bRT\b", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


print("Loading data...")

df = pd.read_csv(INPUT)

df["clean_text"] = df["customer_text"].apply(clean_text)

print(f"Examples: {len(df):,}")
print("Creating TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])

print(f"TF-IDF shape: {X.shape}")
print("Running KMeans...")

model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

df["cluster"] = model.fit_predict(X)

terms = vectorizer.get_feature_names_out()

print("\n" + "=" * 80)
print("INTENT CLUSTERS")
print("=" * 80)

for cluster_id in range(N_CLUSTERS):

    cluster_indices = (df["cluster"] == cluster_id).values

    # Important terms for this cluster
    center = model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[-12:][::-1]

    top_terms = [
        terms[i]
        for i in top_indices
    ]

    cluster_df = df[df["cluster"] == cluster_id]

    print("\n")
    print("-" * 80)
    print(
        f"CLUSTER {cluster_id} "
        f"({len(cluster_df):,} examples)"
    )
    print("-" * 80)

    print("TOP TERMS:")
    print(", ".join(top_terms))

    print("\nEXAMPLES:")

    examples = cluster_df.sample(
        min(5, len(cluster_df)),
        random_state=RANDOM_STATE
    )

    for _, row in examples.iterrows():
        print(f"  • {row['customer_text'][:300]}")

# Save for later inspection
df.to_csv(
    "data/working/apple_clustered.csv",
    index=False
)

print("\n")
print("=" * 80)
print("Saved:")
print("data/working/apple_clustered.csv")
print("=" * 80)