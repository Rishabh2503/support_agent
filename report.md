# AppleSupport support agent report

## Problem framing

The goal is a triage assistant for public AppleSupport-style messages. A good system identifies the main issue, drafts a short response consistent with historical support behavior, and routes private or high-risk cases to a human. I selected AppleSupport and defined nine intents: software updates, battery, device performance, apps/features, Apple ID/account, orders/sales, payments/billing, repair/support, and other/unclear.

The system is deliberately narrow. It does not authenticate users, inspect device state, look up orders, issue refunds, diagnose hardware with certainty, or send replies. Those omissions are safety boundaries, not missing product features.

## System

`extract_apple.py` makes direct customer-to-AppleSupport pairs from the Kaggle TWCS data. `retriever.py` indexes customer text with character TF-IDF and retrieves the three closest historical examples. `agent.py` gives those examples to an LLM with the intent definitions and returns structured JSON. Sensitive intents are escalated after generation as a deterministic safety policy.

The working sample is capped at 20,000 pairs. The evaluation set contains 162 examples across nine intents, 18 per intent. The intended split excludes golden examples from training.

## Evaluation plan

Intent quality is measured with accuracy, macro F1, weighted F1, per-class reports, and a confusion matrix. Escalation quality is measured with precision, recall, F1, and a confusion matrix. Reply quality is measured separately by `src/judge_replies.py`: format validity, length, evidence overlap, sensitive-public-request checks, and optional LLM rubric scores for helpfulness, grounding, and safety.

The trivial majority baseline is reproducible from the committed CSV:

| Baseline | Accuracy | Macro F1 |
|---|---:|---:|
| Majority intent (`software_update`) | 0.111 | 0.022 |

A TF-IDF + logistic regression baseline is implemented in `src/train_baseline.py`. Its result must be generated after the raw dataset is downloaded and weak labels are created; it is not claimed here because those generated files and a model run are not committed.

Likewise, no agent headline score is claimed. The repository does not contain the raw Kaggle file or API credentials, and the existing labels still require independent manual verification. This is preferable to presenting an unreproducible number as evidence.
## Evaluation Results

The agent was evaluated on the 162-example golden set.

### Intent Classification

| Metric | Result |
|---|---:|
| Accuracy | 49.38% |
| Macro F1 | 42.07% |
| Weighted F1 | 51.42% |

The majority-intent baseline achieved 11.11% accuracy and 2.22% macro F1, so the retrieval + LLM agent substantially outperformed the simple baseline.

The main classification failures occurred between semantically related support categories, particularly software updates vs. device performance, battery vs. software updates, iCloud/account vs. photo storage, and app functionality vs. repair/support.

### Escalation

| Metric | Result |
|---|---:|
| Precision | 66.04% |
| Recall | 64.81% |
| F1 | 65.42% |

There were 54 gold escalation cases and the agent predicted escalation for 53 examples.

### Reliability

| Metric | Result |
|---|---:|
| Successful API calls | 162/162 |
| Failed API calls | 0/162 |
| Intent errors | 82 |
| Escalation errors | 37 |

### Failure Analysis

The agent generally handled common battery, device-performance, payment, and software-update queries, but struggled when a customer message contained multiple overlapping symptoms.

Examples include battery problems following an iOS update, iCloud/photo synchronization issues, and hardware/support requests that also mentioned purchasing or warranty context.

The golden set also contains some potentially ambiguous labels. Therefore, classification errors should be interpreted together with manual review of the gold labels rather than assuming every mismatch represents a model failure.

### Limitations

The evaluation uses a relatively small 162-example golden set. Some labels are difficult to distinguish semantically and may contain annotation ambiguity. The retrieval system also uses lexical TF-IDF similarity rather than a semantic embedding model.

The agent is intended as an experimental support assistant and does not authenticate users, access orders, process refunds, or perform account actions.

## Golden-set quality

The committed set has the required size, but it is not yet submission-grade evidence. Its annotation notes are blank, and there is no recorded second annotator or adjudication pass. Before submission, a human should review every intent and escalation label, record sampling and adjudication notes, and rate a separate subset of generated replies. The judge script computes agreement only after those human fields are supplied.

## Failure analysis

The current dataset reveals five likely failure modes that should be checked in the final run:

1. **Overlapping primary intents.** Battery drain after an update can be battery or software update; the prompt uses the customer's primary complaint, but this remains ambiguous.
2. **Noisy or underspecified tweets.** Messages such as “fix it” do not contain enough information and should become `other_unclear`, not a confident product diagnosis.
3. **Weak historical resolution evidence.** A nearest neighbor may be lexically similar but have a generic “DM us” response, which limits how actionable the draft can be.
4. **Dataset label noise.** Historical support replies do not always reveal whether a case was actually resolved, and the original conversations can contain multiple brands or turns.
5. **Public-vs-private routing errors.** Account, payment, order, security, and repair cases can require private handling even when the text looks routine; the deterministic sensitive-intent rule reduces false non-escalations but can over-escalate.

The final report should replace these hypotheses with five concrete examples from `agent_predictions.csv`, including the expected label, prediction, retrieved evidence, and a proposed fix.

## What is misleading about my headline number?

A single intent accuracy number hides label ambiguity, balanced sampling, possible annotation noise, and the fact that reply usefulness and escalation safety are separate tasks. Retrieval can also make a model look strong on near-duplicates while failing on new wording. API failures are not successes, and LLM-judge scores are not human truth. For those reasons, the report should show per-intent metrics, escalation confusion matrices, reply rubric distributions, failure examples, and judge-human agreement rather than one score alone.

## One more week

I would complete a two-person annotation pass with adjudication, freeze a versioned 200-example test set, add temporal and near-duplicate leakage checks, run the TF-IDF and agent baselines on the same split, and rate at least 30 replies independently. I would then improve retrieval with intent-aware filtering and add abstention when similarity is low or evidence conflicts.
