import pandas as pd
from collections import defaultdict

PATH = "data/archive/twcs/twcs.csv"

# We only need the columns required for brand/conversation analysis
usecols = [
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

print("Reading dataset in chunks...")

brand_stats = defaultdict(lambda: {
    "outbound": 0,
    "customer_messages": 0,
    "paired_customer_messages": 0,
    "paired_responses": 0,
})

# First collect support-account tweets and their IDs
support_tweet_ids = set()
support_authors = set()

for chunk in pd.read_csv(
    PATH,
    usecols=usecols,
    chunksize=100_000,
    low_memory=False,
):
    outbound = chunk[chunk["inbound"] == False]

    for author in outbound["author_id"].dropna():
        support_authors.add(author)

print(f"Found {len(support_authors)} potential support accounts.")

# Count statistics per support account
for chunk in pd.read_csv(
    PATH,
    usecols=usecols,
    chunksize=100_000,
    low_memory=False,
):
    inbound = chunk[chunk["inbound"] == True]
    outbound = chunk[chunk["inbound"] == False]

    # Support tweets
    for author, count in outbound["author_id"].value_counts().items():
        brand_stats[author]["outbound"] += int(count)

    # Customer tweets which have a response
    paired = inbound[inbound["response_tweet_id"].notna()]

    for author in paired["response_tweet_id"].dropna():
        # response_tweet_id can contain multiple IDs separated by commas
        ids = str(author).split(",")
        for response_id in ids:
            # We don't know the brand yet here, so count globally later
            pass

    # Customer messages
    for author, count in inbound["author_id"].value_counts().items():
        # We don't need customer authors in final ranking
        pass

print("\nTop support accounts by number of outbound tweets:\n")

rows = []

for brand, stats in brand_stats.items():
    rows.append({
        "brand": brand,
        "support_tweets": stats["outbound"],
    })

result = pd.DataFrame(rows)

result = result.sort_values(
    "support_tweets",
    ascending=False
).head(40)

print(result.to_string(index=False))