import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from agent import run_agent


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_SET = Path("evaluation/golden_set.csv")
OUTPUT_FILE = Path("evaluation/agent_predictions.csv")

# Set to None to evaluate ALL examples.
# Example: 10 for a quick test.
MAX_EXAMPLES = 162

# Reproducible ordering/sampling
RANDOM_STATE = 42


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("Loading historical AppleSupport conversations...")
print("Building retrieval index...")

# run_agent creates the retriever internally, so no need to
# create another retriever here.

df = pd.read_csv(GOLDEN_SET)

print()
print("=" * 70)
print("APPLE SUPPORT — AGENT EVALUATION")
print("=" * 70)

print(f"\nGolden set rows: {len(df)}")

required_columns = {
    "customer_text",
    "intent",
    "escalation",
}

missing = required_columns - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns in golden_set.csv: {sorted(missing)}"
    )


# ============================================================
# NORMALIZE GOLDEN SET
# ============================================================

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["intent"] = (
    df["intent"]
    .fillna("other_unclear")
    .astype(str)
    .str.strip()
)

df["escalation"] = (
    df["escalation"]
    .fillna("no")
    .astype(str)
    .str.strip()
    .str.lower()
)

# Normalize possible boolean values
df["escalation"] = df["escalation"].replace(
    {
        "true": "yes",
        "false": "no",
        "1": "yes",
        "0": "no",
    }
)

# Validate escalation values
invalid_escalation = sorted(
    set(df["escalation"]) - {"yes", "no"}
)

if invalid_escalation:
    raise RuntimeError(
        "Invalid escalation values found: "
        f"{invalid_escalation}. Use only yes/no."
    )


# ============================================================
# EXAMPLE SELECTION
# ============================================================

if MAX_EXAMPLES is not None and MAX_EXAMPLES < len(df):

    eval_df = df.sample(
        n=MAX_EXAMPLES,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

else:

    eval_df = df.reset_index(drop=True)


print(f"Examples to evaluate: {len(eval_df)}")
print()


# ============================================================
# EVALUATION
# ============================================================

predictions = []

successful_calls = 0
failed_calls = 0


for index, row in eval_df.iterrows():

    customer_text = row["customer_text"]

    print(
        f"[{index + 1}/{len(eval_df)}] "
        f"{customer_text[:120]}"
    )

    try:

        result = run_agent(
            customer_text,
            k=3,
        )

        successful_calls += 1

        predicted_intent = result.get(
            "intent",
            "other_unclear",
        )

        predicted_escalation = result.get(
            "escalate",
            False,
        )

        # Normalize escalation to yes/no
        if isinstance(predicted_escalation, bool):

            predicted_escalation = (
                "yes"
                if predicted_escalation
                else "no"
            )

        else:

            predicted_escalation = str(
                predicted_escalation
            ).strip().lower()

            if predicted_escalation in {
                "true",
                "1",
                "yes",
            }:
                predicted_escalation = "yes"

            else:
                predicted_escalation = "no"


        predictions.append(
            {
                "example_id": index + 1,
                "customer_text": customer_text,

                "gold_intent": row["intent"],
                "predicted_intent": predicted_intent,

                "gold_escalation": row["escalation"],
                "predicted_escalation": predicted_escalation,

                "reply": result.get(
                    "reply",
                    "",
                ),

                "evidence": json.dumps(
                    result.get(
                        "evidence",
                        [],
                    ),
                    ensure_ascii=False,
                ),

                "reason": result.get(
                    "reason",
                    "",
                ),

                "status": "success",
            }
        )

    except Exception as e:

        failed_calls += 1

        print(
            f"ERROR: {type(e).__name__}: {e}"
        )

        # IMPORTANT:
        # Do not crash the entire evaluation if one
        # Groq request fails.

        predictions.append(
            {
                "example_id": index + 1,
                "customer_text": customer_text,

                "gold_intent": row["intent"],
                "predicted_intent": "ERROR",

                "gold_escalation": row["escalation"],
                "predicted_escalation": "ERROR",

                "reply": "",
                "evidence": "",
                "reason": str(e),

                "status": "error",
            }
        )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(predictions)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

predictions_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8",
)

print()
print("=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)

print(f"\nSaved:")
print(OUTPUT_FILE)


# ============================================================
# ONLY EVALUATE SUCCESSFUL API CALLS
# ============================================================

successful_df = predictions_df[
    predictions_df["status"] == "success"
].copy()


if len(successful_df) == 0:

    print()
    print("No successful API calls.")
    print("Cannot calculate evaluation metrics.")
    raise SystemExit(1)


# ============================================================
# INTENT EVALUATION
# ============================================================

y_true_intent = successful_df[
    "gold_intent"
].tolist()

y_pred_intent = successful_df[
    "predicted_intent"
].tolist()


print()
print("=" * 70)
print("INTENT RESULTS")
print("=" * 70)

intent_accuracy = accuracy_score(
    y_true_intent,
    y_pred_intent,
)

intent_macro_f1 = f1_score(
    y_true_intent,
    y_pred_intent,
    average="macro",
    zero_division=0,
)

intent_weighted_f1 = f1_score(
    y_true_intent,
    y_pred_intent,
    average="weighted",
    zero_division=0,
)


print(
    f"\nAccuracy:    {intent_accuracy:.4f}"
)

print(
    f"Macro F1:    {intent_macro_f1:.4f}"
)

print(
    f"Weighted F1: {intent_weighted_f1:.4f}"
)


print("\nClassification report:")

print(
    classification_report(
        y_true_intent,
        y_pred_intent,
        zero_division=0,
    )
)


# ============================================================
# INTENT CONFUSION MATRIX
# ============================================================

labels = sorted(
    set(y_true_intent) |
    set(y_pred_intent)
)

cm = confusion_matrix(
    y_true_intent,
    y_pred_intent,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"actual:{x}"
        for x in labels
    ],
    columns=[
        f"pred:{x}"
        for x in labels
    ],
)

print()
print("=" * 70)
print("INTENT CONFUSION MATRIX")
print("=" * 70)

print()
print(cm_df.to_string())


# ============================================================
# ESCALATION EVALUATION
# ============================================================

y_true_escalation = successful_df[
    "gold_escalation"
].tolist()

y_pred_escalation = successful_df[
    "predicted_escalation"
].tolist()


print()
print("=" * 70)
print("ESCALATION RESULTS")
print("=" * 70)


escalation_precision = precision_score(
    y_true_escalation,
    y_pred_escalation,
    pos_label="yes",
    zero_division=0,
)

escalation_recall = recall_score(
    y_true_escalation,
    y_pred_escalation,
    pos_label="yes",
    zero_division=0,
)

escalation_f1 = f1_score(
    y_true_escalation,
    y_pred_escalation,
    pos_label="yes",
    zero_division=0,
)


print(
    f"\nPrecision: {escalation_precision:.4f}"
)

print(
    f"Recall:    {escalation_recall:.4f}"
)

print(
    f"F1:        {escalation_f1:.4f}"
)


true_escalations = sum(
    x == "yes"
    for x in y_true_escalation
)

predicted_escalations = sum(
    x == "yes"
    for x in y_pred_escalation
)


print(
    f"\nTrue escalations: {true_escalations}"
)

print(
    f"Predicted escalations: "
    f"{predicted_escalations}"
)


# ============================================================
# ESCALATION CONFUSION MATRIX
# ============================================================

esc_labels = ["no", "yes"]

esc_cm = confusion_matrix(
    y_true_escalation,
    y_pred_escalation,
    labels=esc_labels,
)

esc_cm_df = pd.DataFrame(
    esc_cm,
    index=[
        "actual:no",
        "actual:yes",
    ],
    columns=[
        "pred:no",
        "pred:yes",
    ],
)

print()
print("=" * 70)
print("ESCALATION CONFUSION MATRIX")
print("=" * 70)

print()
print(esc_cm_df.to_string())


# ============================================================
# ERROR SUMMARY
# ============================================================

intent_errors = sum(
    successful_df["gold_intent"]
    != successful_df["predicted_intent"]
)

escalation_errors = sum(
    successful_df["gold_escalation"]
    != successful_df["predicted_escalation"]
)


print()
print("=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    f"\nIntent errors:       {intent_errors}"
)

print(
    f"Escalation errors:   {escalation_errors}"
)

print(
    f"Successful API calls: "
    f"{successful_calls} / {len(eval_df)}"
)

print(
    f"Failed API calls:     "
    f"{failed_calls} / {len(eval_df)}"
)


# ============================================================
# SHOW ACTUAL MISCLASSIFICATIONS
# ============================================================

intent_mistakes = successful_df[
    successful_df["gold_intent"]
    != successful_df["predicted_intent"]
]

if len(intent_mistakes) > 0:

    print()
    print("=" * 70)
    print("INTENT MISCLASSIFICATIONS")
    print("=" * 70)

    for _, row in intent_mistakes.iterrows():

        print()
        print(
            f"Example {row['example_id']}"
        )

        print(
            f"Customer: "
            f"{row['customer_text'][:200]}"
        )

        print(
            f"Expected: "
            f"{row['gold_intent']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted_intent']}"
        )


# ============================================================
# SHOW ESCALATION MISTAKES
# ============================================================

escalation_mistakes = successful_df[
    successful_df["gold_escalation"]
    != successful_df["predicted_escalation"]
]

if len(escalation_mistakes) > 0:

    print()
    print("=" * 70)
    print("ESCALATION MISCLASSIFICATIONS")
    print("=" * 70)

    for _, row in escalation_mistakes.iterrows():

        print()
        print(
            f"Example {row['example_id']}"
        )

        print(
            f"Customer: "
            f"{row['customer_text'][:200]}"
        )

        print(
            f"Expected escalation: "
            f"{row['gold_escalation']}"
        )

        print(
            f"Predicted escalation: "
            f"{row['predicted_escalation']}"
        )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)