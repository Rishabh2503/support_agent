import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from retriever import AppleSupportRetriever


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAQ_API_KEY = os.getenv("MAQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

MAQ_MODEL = os.getenv(
    "MAQ_MODEL",
    "gemma-4-31b"
)

MAQ_BASE_URL = os.getenv(
    "MAQ_BASE_URL",
    "https://llm.maqsoftware.net/v1"
)


# ============================================================
# CLIENTS
# ============================================================

groq_client = None
maq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

if MAQ_API_KEY:
    maq_client = OpenAI(
        base_url=MAQ_BASE_URL,
        api_key=MAQ_API_KEY,
    )


# ============================================================
# PROVIDER STATE
# ============================================================

# Start with Groq.
# Once switched to MAQ, stay on MAQ for this process.
ACTIVE_PROVIDER = "groq"


def switch_to_maq():
    global ACTIVE_PROVIDER

    if not MAQ_API_KEY:
        raise RuntimeError(
            "Groq failed and MAQ_API_KEY is not configured."
        )

    ACTIVE_PROVIDER = "maq"

    print(
        "\nWARNING: Switching from Groq to MAQ for the rest of this run.\n"
    )


# ============================================================
# RETRIEVER
# ============================================================

retriever = AppleSupportRetriever()


# ============================================================
# INTENTS
# ============================================================

ALLOWED_INTENTS = {
    "software_update",
    "battery",
    "device_performance",
    "apps_features",
    "apple_id_account",
    "orders_sales",
    "payments_billing",
    "photos_storage",
    "security_privacy",
    "repair_support",
    "other_unclear",
}


SENSITIVE_INTENTS = {
    "apple_id_account",
    "payments_billing",
    "security_privacy",
    "orders_sales",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You classify Apple customer support messages.

Choose exactly ONE intent.

INTENT DEFINITIONS:

software_update:
The main issue is caused by, related to, or asking about an OS/software update.
Examples: iOS update problems, update bugs, update installation, new update behavior.

battery:
Battery life, battery drain, charging, battery health, or device not holding charge.

device_performance:
Device freezing, crashing, restarting, lagging, slowing down, overheating, or becoming unusable.
Do NOT choose this only because an update is mentioned. If the customer clearly blames an update, prefer software_update.

apps_features:
The customer is asking how to use, configure, access, or understand a specific app or feature.
Examples: how to turn on shuffle, where a feature is located, how a feature works.

apple_id_account:
Apple ID, iCloud account, password, login, account recovery, account access, account settings, or account/device association.

orders_sales:
Buying a product, product availability, purchase decisions, product purchase problems, or sales-related questions.

payments_billing:
Charges, billing, payment methods, Apple Pay, payment authorization, or payment-related problems.

photos_storage:
Photos, camera roll, photo syncing, photo duplication, photo deletion, iCloud Photos, or storage specifically involving photos.

security_privacy:
Unauthorized access, suspicious account/device activity, privacy, security vulnerabilities, compromised accounts, or security concerns.

repair_support:
Physical repair, defective hardware, warranty/service center/store support, replacement/service requests, or hardware repair.

other_unclear:
Use when the message does not clearly fit another intent or there is insufficient information.

IMPORTANT CLASSIFICATION RULES:

1. Identify the PRIMARY problem, not just keywords.
2. "Update" + software behavior/problem → usually software_update.
3. Freezing/crashing/restarting/slow device → device_performance unless the update itself is clearly the primary issue.
4. Questions about using a feature → apps_features.
5. Apple ID/iCloud/account access → apple_id_account.
6. Physical defect, repair, replacement, warranty, or service → repair_support.
7. Purchase/availability/product buying → orders_sales.
8. Payment/charge/Apple Pay → payments_billing.
9. Photos/photo synchronization/photo storage → photos_storage.
10. If uncertain between several categories, use other_unclear rather than apps_features.

ESCALATION:

Escalate when the customer needs account-specific, payment-specific,
security-specific, order-specific, repair/service-specific, or private investigation.

Routine troubleshooting should normally NOT escalate.

OUTPUT:

Return ONLY valid JSON.

{
  "intent": "...",
  "reply": "...",
  "escalate": false,
  "reason": "..."
}

Reply must be under 35 words.
Reason must be under 10 words.
"""


# ============================================================
# VALIDATION
# ============================================================

def validate_result(result):

    if not isinstance(result, dict):
        raise ValueError("LLM result is not an object")

    required = {
        "intent",
        "reply",
        "escalate",
        "reason",
    }

    missing = required - set(result.keys())

    if missing:
        raise ValueError(
            f"Missing fields: {missing}"
        )

    intent = str(result["intent"]).strip()

    if intent not in ALLOWED_INTENTS:
        raise ValueError(
            f"Invalid intent: {intent}"
        )

    result["intent"] = intent

    result["reply"] = str(
        result["reply"]
    ).strip()

    result["reason"] = str(
        result["reason"]
    ).strip()

    # Strict boolean handling
    escalate = result["escalate"]

    if isinstance(escalate, bool):
        result["escalate"] = escalate

    elif isinstance(escalate, str):
        result["escalate"] = (
            escalate.lower().strip() == "true"
        )

    else:
        result["escalate"] = bool(escalate)

    return result


# ============================================================
# PROMPT
# ============================================================

def build_prompt(customer_message, evidence):

    examples = []

    for _, row in evidence.iterrows():

        examples.append(
            "Customer: "
            + str(row["customer_text"])
            + "\nResponse: "
            + str(row["brand_response"])
        )

    evidence_text = "\n\n".join(examples)

    return f"""
Customer:
{customer_message}

Historical examples:
{evidence_text}

Return ONLY one JSON object:

{{
"intent":"one allowed label",
"reply":"short response",
"escalate":false,
"reason":"short reason"
}}
"""


# ============================================================
# GROQ CALL
# ============================================================

def call_groq(prompt):

    if not groq_client:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=120,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Groq returned empty response."
        )

    return validate_result(
        json.loads(content)
    )


# ============================================================
# MAQ CALL
# ============================================================

def call_maq(prompt):

    if not maq_client:
        raise RuntimeError(
            "MAQ_API_KEY is not configured."
        )

    response = maq_client.chat.completions.create(
        model=MAQ_MODEL,
        temperature=0,
        max_tokens=120,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "MAQ returned empty response."
        )

    # Some OpenAI-compatible models may wrap JSON
    # in markdown fences.
    content = content.strip()

    if content.startswith("```"):
        content = content.replace(
            "```json", ""
        ).replace(
            "```", ""
        ).strip()

    return validate_result(
        json.loads(content)
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_with_llm(customer_message, evidence):

    global ACTIVE_PROVIDER

    prompt = build_prompt(
        customer_message,
        evidence,
    )

    # --------------------------------------------------------
    # If already switched, NEVER go back to Groq.
    # --------------------------------------------------------

    if ACTIVE_PROVIDER == "maq":

        try:
            return call_maq(prompt)

        except Exception as error:

            raise RuntimeError(
                f"MAQ failed: {error}"
            )


    # --------------------------------------------------------
    # Try Groq first.
    # --------------------------------------------------------

    try:

        return call_groq(prompt)

    except Exception as groq_error:

        print(
            f"WARNING: Groq failed: {groq_error}"
        )

        # ----------------------------------------------------
        # One-way switch to MAQ.
        # ----------------------------------------------------

        switch_to_maq()

        try:

            return call_maq(prompt)

        except Exception as maq_error:

            raise RuntimeError(
                f"Groq failed: {groq_error}. "
                f"MAQ failed: {maq_error}"
            )


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(customer_message, k=3):

    # --------------------------------------------------------
    # Retrieve historical examples
    # --------------------------------------------------------

    evidence = retriever.search(
        customer_message,
        k=k,
    )

    # --------------------------------------------------------
    # LLM classification
    # --------------------------------------------------------

    result = classify_with_llm(
        customer_message,
        evidence,
    )

    # --------------------------------------------------------
    # Safety-first escalation
    # --------------------------------------------------------

    if result["intent"] in SENSITIVE_INTENTS:

        result["escalate"] = True

        result["reason"] = (
            "Sensitive account, payment, security, or order issue"
        )

    # --------------------------------------------------------
    # Add retrieval evidence
    # --------------------------------------------------------

    result["evidence"] = []

    for _, row in evidence.iterrows():

        result["evidence"].append(
            {
                "customer": row["customer_text"],
                "response": row["brand_response"],
                "similarity": round(
                    float(row["similarity"]),
                    3,
                ),
            }
        )

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        customer_message = " ".join(
            sys.argv[1:]
        )

    else:

        customer_message = input(
            "Customer message: "
        )

    print(
        "\nRunning Apple Support agent...\n"
    )

    try:

        result = run_agent(
            customer_message,
            k=3,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        sys.exit(1)