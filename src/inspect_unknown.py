import pandas as pd

INPUT = "data/working/apple_labeled.csv"

df = pd.read_csv(INPUT)

unknown = df[df["intent"] == "unknown_other"]

print(f"Unknown examples: {len(unknown):,}")

print("\n" + "=" * 80)
print("RANDOM UNKNOWN EXAMPLES")
print("=" * 80)

samples = unknown.sample(
    min(100, len(unknown)),
    random_state=42
)

for i, (_, row) in enumerate(samples.iterrows(), 1):
    print(f"\n{i}. {row['customer_text']}")
    print(f"   RESPONSE: {row['brand_response']}")