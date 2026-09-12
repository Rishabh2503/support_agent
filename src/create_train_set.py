import pandas as pd


ALL_DATA = "data/working/apple_pairs.csv"
GOLDEN = "evaluation/golden_set.csv"
OUTPUT = "data/working/train_pairs.csv"


df = pd.read_csv(ALL_DATA)
golden = pd.read_csv(GOLDEN)

golden_ids = set(
    golden["customer_tweet_id"].astype(int)
)

print(f"All working examples: {len(df):,}")
print(f"Golden examples:      {len(golden_ids):,}")


# Remove every golden example from training.
train = df[
    ~df["customer_tweet_id"].astype(int).isin(golden_ids)
].copy()

print(f"Training examples:     {len(train):,}")

train.to_csv(
    OUTPUT,
    index=False
)

print(f"\nSaved: {OUTPUT}")