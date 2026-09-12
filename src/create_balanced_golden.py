import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = Path("data/working/apple_labeled.csv")
OUTPUT = Path("evaluation/golden_candidates.csv")

TARGET_PER_INTENT = 18
RANDOM_STATE = 42


# ============================================================
# WEAK LABEL -> FINAL AGENT INTENT
# ============================================================

INTENT_MAP = {
    "apple_id_account": "apple_id_account",
    "apple_pay_billing": "payments_billing",
    "apps_features": "apps_features",
    "battery": "battery",
    "device_performance": "device_performance",
    "orders_sales": "orders_sales",
    "repair_support": "repair_support",
    "software_update": "software_update",
    "unknown_other": "other_unclear",
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading labeled dataset...")

df = pd.read_csv(INPUT)

print(f"Total rows: {len(df)}")


# ============================================================
# NORMALIZE INTENTS
# ============================================================

df["intent"] = df["intent"].map(INTENT_MAP)

# Remove rows with unknown/unmapped intent.
df = df.dropna(subset=["intent"]).copy()


print("\nAvailable examples:")
print(
    df["intent"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# CREATE BALANCED SAMPLE
# ============================================================

samples = []

print("\nSampling examples...")

for intent in sorted(INTENT_MAP.values()):

    subset = df[df["intent"] == intent].copy()

    if len(subset) == 0:
        print(f"WARNING: No examples available for {intent}")
        continue

    n = min(TARGET_PER_INTENT, len(subset))

    sample = subset.sample(
        n=n,
        random_state=RANDOM_STATE,
    )

    samples.append(sample)

    print(
        f"{intent:<22} -> {n} examples"
    )


# ============================================================
# COMBINE SAMPLES
# ============================================================

if not samples:
    raise RuntimeError(
        "No examples were available to create the golden candidate set."
    )

golden = pd.concat(
    samples,
    ignore_index=True,
)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

required_columns = [
    "customer_text",
    "brand_response",
    "intent",
]

missing_columns = [
    column
    for column in required_columns
    if column not in golden.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing required columns: {missing_columns}"
    )

golden = golden[required_columns].copy()


# ============================================================
# MANUAL ANNOTATION COLUMNS
# ============================================================

# You will manually fill these after opening the CSV.
golden["escalation"] = ""
golden["notes"] = ""


# ============================================================
# SHUFFLE
# ============================================================

golden = golden.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

golden.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# VALIDATION / SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BALANCED GOLDEN CANDIDATE SET")
print("=" * 70)

print(f"\nTotal examples: {len(golden)}")

print("\nIntent distribution:")
print(
    golden["intent"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nColumns:")
print(golden.columns.tolist())

print(f"\nSaved to:")
print(OUTPUT)

print("\nNext step:")
print(
    "Open evaluation/golden_candidates.csv and manually fill "
    "the 'escalation' column with yes/no."
)