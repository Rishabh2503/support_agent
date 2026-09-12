import pandas as pd

INPUT = "data/working/apple_pairs.csv"
OUTPUT = "evaluation/golden_set.csv"

N = 150
RANDOM_STATE = 42

print("Loading AppleSupport pairs...")

df = pd.read_csv(INPUT)

# Clean obvious duplicates
df = df.drop_duplicates(
    subset=["customer_text", "brand_response"]
).reset_index(drop=True)

# We don't want extremely tiny messages in the evaluation set.
df = df[df["customer_text"].str.len() >= 10]

# Deterministic random sample.
golden = df.sample(
    n=N,
    random_state=RANDOM_STATE
).copy()

# Keep only fields useful for manual evaluation.
golden = golden[
    [
        "customer_tweet_id",
        "customer_text",
        "brand_response",
    ]
].copy()

# Human annotation columns.
golden["intent"] = ""
golden["escalation"] = ""
golden["escalation_reason"] = ""

# Add an ID that is easier to reference in the report.
golden.insert(
    0,
    "example_id",
    range(1, len(golden) + 1)
)

golden.to_csv(
    OUTPUT,
    index=False
)

print(f"\nCreated {len(golden)} golden examples.")
print(f"Saved to: {OUTPUT}")

print("\nColumns:")
print(golden.columns.tolist())

print("\nFirst 10 examples:")
print(
    golden[
        [
            "example_id",
            "customer_text",
            "brand_response",
        ]
    ].head(10).to_string(index=False)
)