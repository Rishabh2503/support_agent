# AppleSupport support agent

A small retrieval-augmented support agent built from the Customer Support on Twitter dataset. The selected brand is `AppleSupport`.

## What the agent does

For each customer message it returns:

- one intent from the nine labels defined in `src/agent.py`;
- a short reply based on retrieved historical AppleSupport responses;
- an escalation decision and reason;
- the retrieved examples used as evidence.

This is a take-home experiment, not a production support system. It does not authenticate users, access orders, issue refunds, or send messages.

## Reproduce

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`) and place it at `data/archive/twcs/twcs.csv`. The raw dataset is intentionally ignored by git. Then run:

```powershell
python src/extract_apple.py
python src/weak_label.py
python src/create_golden_set.py
python src/create_train_set.py
python src/train_baseline.py
python src/majority_baseline.py
```

Before evaluation, manually label `evaluation/golden_set.csv`. The committed file contains 162 examples, but its labels and escalation decisions must be independently checked before being treated as gold data.

Create a `.env` file with either provider credentials:

```text
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
```

Run the agent on one message:

```powershell
python src/agent.py "My iPhone battery is draining very quickly"
```

Run the model evaluation:

```powershell
python src/evaluate_agent.py
```

The full pipeline is designed for a 20,000-row AppleSupport working sample and normally completes in under 15 minutes excluding API rate limits. For a quick smoke test, set `MAX_EXAMPLES = 10` in `src/evaluate_agent.py`.

## Reply evaluation

The reply harness scores every prediction for format, length, evidence overlap, and whether the response asks for sensitive information publicly. It can also call an LLM judge using the same Groq credentials:

```powershell
python src/judge_replies.py --predictions evaluation/agent_predictions.csv
python src/judge_replies.py --predictions evaluation/agent_predictions.csv --llm
```

For human agreement, add independent columns `human_helpful`, `human_grounded`, and `human_safe` to `evaluation/agent_predictions.csv` using `0`/`1` values, then run:

```powershell
python src/judge_replies.py --predictions evaluation/agent_predictions.csv --agreement
```

The judge rubric is in `src/judge_replies.py`. Human agreement must be measured on a separately rated sample; it must not be inferred from model labels.

## Current evidence

The committed golden set has 162 rows, evenly distributed across nine intents. The majority intent baseline scores 11.1% accuracy and 2.2% macro F1. No valid agent headline score is committed because the raw dataset, API credentials, and independently verified human labels are not included in this repository.

See [report.md](report.md) for framing, limitations, failure modes, and next steps. See [decision_log.md](decision_log.md) for non-obvious design decisions.
