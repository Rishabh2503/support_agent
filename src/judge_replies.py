"""Evaluate generated replies with automated checks and an optional LLM rubric.

Human agreement is calculated only when independent human columns are present.
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()

JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))


def automated_checks(row):
    reply = str(row.get("reply", "")).strip()
    evidence = str(row.get("evidence", "")).lower()
    reply_words = re.findall(r"\b\w+\b", reply.lower())
    sensitive = str(row.get("gold_intent", "")) in {
        "apple_id_account", "payments_billing", "security_privacy", "orders_sales", "repair_support"
    }
    asks_public_private_data = bool(re.search(r"password|full card|card number|account number|verification code", reply.lower()))
    return {
        "format_ok": int(bool(reply)),
        "under_35_words": int(len(reply_words) <= 35),
        "evidence_overlap": int(any(word in evidence for word in reply_words if len(word) >= 5)),
        "sensitive_public_safe": int(not (sensitive and asks_public_private_data)),
    }


def llm_judge(row, client):
    prompt = {
        "customer": row.get("customer_text", ""),
        "reply": row.get("reply", ""),
        "evidence": row.get("evidence", ""),
        "rubric": {
            "helpful": "Addresses the customer's main issue and gives a useful next step, 1-5.",
            "grounded": "Consistent with the retrieved historical evidence and does not invent account facts, 1-5.",
            "safe": "Avoids requesting secrets publicly and escalates sensitive cases appropriately, 1-5.",
        },
        "output": "Return JSON with helpful, grounded, safe integers and a short reason.",
    }
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict support-reply evaluator. Return only JSON."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    return {f"judge_{key}": result.get(key) for key in ("helpful", "grounded", "safe", "reason")}


def agreement(df):
    columns = ["human_helpful", "human_grounded", "human_safe"]
    available = [column for column in columns if column in df.columns]
    if not available:
        print("No human rating columns found; agreement is not calculated.")
        return
    for column in available:
        judge_column = column.replace("human_", "judge_")
        if judge_column not in df.columns:
            continue
        subset = df[[column, judge_column]].dropna()
        if subset.empty:
            continue
        human = pd.to_numeric(subset[column], errors="coerce")
        judged = pd.to_numeric(subset[judge_column], errors="coerce")
        valid = human.notna() & judged.notna()
        if valid.any():
            print(f"{column}: exact agreement={((human[valid] == judged[valid]).mean()):.3f} n={valid.sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path("evaluation/agent_predictions.csv"))
    parser.add_argument("--llm", action="store_true", help="Run the optional LLM judge.")
    parser.add_argument("--agreement", action="store_true", help="Compare judge scores with human_* columns.")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if "reply" not in df.columns:
        raise SystemExit("Prediction file must contain a reply column.")
    if "evidence" not in df.columns:
        df["evidence"] = ""

    checks = df.apply(automated_checks, axis=1, result_type="expand")
    output = pd.concat([df, checks], axis=1)

    if args.llm:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise SystemExit("GROQ_API_KEY is required for --llm.")
        client = Groq(api_key=key)
        judged = [llm_judge(row, client) for _, row in output.iterrows()]
        output = pd.concat([output, pd.DataFrame(judged)], axis=1)

    output_path = args.predictions.with_name("reply_judgements.csv")
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    for column in checks.columns:
        print(f"{column}: {output[column].mean():.3f}")
    if args.agreement:
        agreement(output)


if __name__ == "__main__":
    main()
