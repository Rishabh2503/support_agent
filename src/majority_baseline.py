import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


GOLDEN_FILE = "evaluation/golden_set.csv"

golden = pd.read_csv(GOLDEN_FILE)

golden = golden[
    golden["intent"].notna()
    & (golden["intent"] != "")
].copy()

y_true = golden["intent"]

# Majority class from the GOLDEN evaluation distribution.
majority_intent = y_true.value_counts().idxmax()

predictions = [majority_intent] * len(y_true)

accuracy = accuracy_score(
    y_true,
    predictions,
)

macro_f1 = f1_score(
    y_true,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    predictions,
    average="weighted",
    zero_division=0,
)

print("=" * 70)
print("MAJORITY CLASS BASELINE")
print("=" * 70)

print(f"\nMajority intent: {majority_intent}")
print(f"Examples:        {len(y_true)}")

print(f"\nAccuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        predictions,
        zero_division=0,
    )
)