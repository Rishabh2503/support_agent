import pandas as pd


ALL_DATA = "data/working/apple_pairs.csv"
GOLDEN = "evaluation/golden_set.csv"
OUTPUT = "data/working/train_pairs.csv"


df = pd.read_csv(ALL_DATA)
golden = pd.read_csv(GOLDEN)

if "customer_tweet_id" in golden.columns:
    golden_ids = set(golden["customer_tweet_id"].dropna().astype(int))
    train = df[
        ~df["customer_tweet_id"].astype(int).isin(golden_ids)
    ].copy()
else:
    golden_pairs = set(
        zip(
            golden["customer_text"].fillna(""),
            golden["brand_response"].fillna(""),
        )
    )
    train = df[
        ~df.apply(
            lambda row: (row["customer_text"], row["brand_response"])
            in golden_pairs,
            axis=1,
        )
    ].copy()

print(f"All working examples: {len(df):,}")
print(f"Golden examples:      {len(golden):,}")

print(f"Training examples:     {len(train):,}")

train.to_csv(
    OUTPUT,
    index=False
)

print(f"\nSaved: {OUTPUT}")