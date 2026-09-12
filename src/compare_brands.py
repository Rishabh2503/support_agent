import pandas as pd
from collections import defaultdict

PATH = "data/archive/twcs/twcs.csv"

TOP_BRANDS = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
}

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# ---------------------------------------------------------
# PASS 1
# Find outbound support responses from our candidate brands
# ---------------------------------------------------------

print("Pass 1/2: finding customer -> support response links...")

response_map = defaultdict(set)

for chunk in pd.read_csv(
    PATH,
    usecols=USECOLS,
    chunksize=100_000,
    low_memory=False,
):
    outbound = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"].isin(TOP_BRANDS))
        & (chunk["in_response_to_tweet_id"].notna())
    ]

    for _, row in outbound.iterrows():
        parent_id = str(int(row["in_response_to_tweet_id"]))
        response_map[parent_id].add(row["author_id"])


print(f"Found {len(response_map):,} customer tweets with direct support responses.")


# ---------------------------------------------------------
# PASS 2
# Find those customer tweets and collect statistics
# ---------------------------------------------------------

print("Pass 2/2: matching customer messages...")

stats = {
    brand: {
        "customer_messages": 0,
        "resolved_pairs": 0,
        "unique_customers": set(),
    }
    for brand in TOP_BRANDS
}

# Also store response counts per customer tweet
for chunk in pd.read_csv(
    PATH,
    usecols=USECOLS,
    chunksize=100_000,
    low_memory=False,
):
    inbound = chunk[
        (chunk["inbound"] == True)
        & chunk["tweet_id"].astype(str).isin(response_map)
    ]

    for _, row in inbound.iterrows():
        tweet_id = str(int(row["tweet_id"]))

        brands = response_map.get(tweet_id, set())

        for brand in brands:
            stats[brand]["resolved_pairs"] += 1
            stats[brand]["unique_customers"].add(str(row["author_id"]))


# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

rows = []

for brand, s in stats.items():
    rows.append({
        "brand": brand,
        "resolved_pairs": s["resolved_pairs"],
        "unique_customers": len(s["unique_customers"]),
    })

result = pd.DataFrame(rows)

result = result.sort_values(
    "resolved_pairs",
    ascending=False
)

print("\n")
print("=" * 65)
print("BRAND COMPARISON")
print("=" * 65)

print(result.to_string(index=False))

print("=" * 65)