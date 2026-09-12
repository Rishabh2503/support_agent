import pandas as pd

INPUT = "data/archive/twcs/twcs.csv"
OUTPUT = "data/working/apple_pairs.csv"

BRAND = "AppleSupport"

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

print("Loading dataset...")

# ---------------------------------------------------------
# Pass 1:
# Build a mapping:
#
# customer tweet ID -> AppleSupport response
# ---------------------------------------------------------

responses = {}

for chunk in pd.read_csv(
    INPUT,
    usecols=USECOLS,
    chunksize=100_000,
    low_memory=False,
):
    outbound = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"] == BRAND)
        & (chunk["in_response_to_tweet_id"].notna())
    ]

    for _, row in outbound.iterrows():
        customer_id = int(row["in_response_to_tweet_id"])

        responses[customer_id] = {
            "response_tweet_id": int(row["tweet_id"]),
            "response": str(row["text"]),
            "response_created_at": row["created_at"],
        }


print(f"AppleSupport responses found: {len(responses):,}")


# ---------------------------------------------------------
# Pass 2:
# Find the customer tweets corresponding to those responses
# ---------------------------------------------------------

pairs = []

for chunk in pd.read_csv(
    INPUT,
    usecols=USECOLS,
    chunksize=100_000,
    low_memory=False,
):
    inbound = chunk[chunk["inbound"] == True]

    for _, row in inbound.iterrows():

        tweet_id = int(row["tweet_id"])

        if tweet_id not in responses:
            continue

        response = responses[tweet_id]

        text = str(row["text"]).strip()

        if not text or text == "nan":
            continue

        pairs.append({
            "customer_tweet_id": tweet_id,
            "customer_id": row["author_id"],
            "customer_text": text,
            "customer_created_at": row["created_at"],
            "response_tweet_id": response["response_tweet_id"],
            "brand_response": response["response"],
            "response_created_at": response["response_created_at"],
        })


df = pd.DataFrame(pairs)

print(f"Total AppleSupport pairs: {len(df):,}")

# Remove exact duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_text", "brand_response"]
)

# Remove extremely short messages
df = df[df["customer_text"].str.len() >= 5]

# Randomize deterministically
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# We don't need 100K examples for our experiment.
# Keep a manageable working dataset.
df = df.head(20_000)

df.to_csv(
    OUTPUT,
    index=False
)

print(f"Saved {len(df):,} examples to:")
print(OUTPUT)

print("\nSample:")
print(
    df[
        ["customer_text", "brand_response"]
    ].head(10).to_string(index=False)
)