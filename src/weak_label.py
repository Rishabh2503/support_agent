import re
import pandas as pd


INPUT = "data/working/train_pairs.csv"
OUTPUT = "data/working/train_labeled.csv"


def clean(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    return text


RULES = {
    "software_update": [
        "ios update",
        "ios 11",
        "ios11",
        "update",
        "updated",
        "updating",
        "software update",
        "latest ios",
    ],

    "battery": [
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "charging",
        "charge",
        "won't charge",
        "not charging",
    ],

    "device_performance": [
        "freezing",
        "freeze",
        "frozen",
        "restarting",
        "restart",
        "reboot",
        "lag",
        "slow",
        "crash",
        "crashing",
        "keeps restarting",
    ],

    "apple_id_account": [
        "apple id",
        "icloud",
        "password",
        "forgot password",
        "sign in",
        "login",
        "locked account",
        "account",
    ],

    "apple_pay_billing": [
        "apple pay",
        "payment",
        "charged",
        "charge",
        "billing",
        "credit card",
        "debit card",
        "payment information",
        "purchase",
    ],

    "orders_sales": [
        "order",
        "ordered",
        "reservation",
        "buy",
        "bought",
        "purchase",
        "sales",
        "shipping",
        "delivery",
    ],

    "repair_support": [
        "repair",
        "replacement",
        "replace",
        "service",
        "genius",
        "store",
        "broken",
        "damage",
        "screen",
    ],

    "apps_features": [
        "app",
        "application",
        "keyboard",
        "notification",
        "notifications",
        "carplay",
        "itunes",
        "music",
        "facetime",
        "imessage",
        "maps",
        "siri",
        "autocorrect",
    ],
}


def classify(text):
    text = clean(text)

    scores = {}

    for intent, keywords in RULES.items():

        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    # No matching signal
    if best_score == 0:
        return "unknown_other"

    # Ambiguous: multiple intents with same score
    top_scores = [
        intent
        for intent, score in scores.items()
        if score == best_score
    ]

    if len(top_scores) > 1:
        return "unknown_other"

    return best_intent


print("Loading dataset...")

df = pd.read_csv(INPUT)

print(f"Examples: {len(df):,}")

df["intent"] = df["customer_text"].apply(classify)

print("\nIntent distribution:")
print(df["intent"].value_counts())

df.to_csv(
    OUTPUT,
    index=False
)

print(f"\nSaved to {OUTPUT}")