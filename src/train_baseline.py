import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


TRAIN_FILE = "data/working/train_labeled.csv"
GOLDEN_FILE = "evaluation/golden_set.csv"


print("=" * 70)
print("APPLE SUPPORT — TF-IDF BASELINE")
print("=" * 70)


# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
golden = pd.read_csv(GOLDEN_FILE)

print(f"\nTraining examples: {len(train):,}")
print(f"Golden examples:   {len(golden):,}")


# ---------------------------------------------------------
# Remove examples without usable weak labels
# ---------------------------------------------------------

train = train[
    train["intent"].notna()
    & (train["intent"] != "unknown_other")
].copy()

print(f"Training after removing unknown_other: {len(train):,}")

print("\nTraining distribution:")
print(train["intent"].value_counts())


# ---------------------------------------------------------
# Golden set
# ---------------------------------------------------------

golden = golden[
    golden["intent"].notna()
    & (golden["intent"] != "")
].copy()

X_train = train["customer_text"].fillna("")
y_train = train["intent"]

X_test = golden["customer_text"].fillna("")
y_test = golden["intent"]


# ---------------------------------------------------------
# TF-IDF + Logistic Regression
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20_000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    ),
])


print("\nTraining model...")

model.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

predictions = model.predict(X_test)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions,
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nAccuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

labels = sorted(
    set(y_test) | set(predictions)
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

print(cm_df)


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

golden_results = golden.copy()

golden_results["predicted_intent"] = predictions

# Confidence
probabilities = model.predict_proba(X_test)

golden_results["confidence"] = probabilities.max(axis=1)

golden_results.to_csv(
    "evaluation/baseline_predictions.csv",
    index=False,
)

print(
    "\nSaved predictions to "
    "evaluation/baseline_predictions.csv"
)