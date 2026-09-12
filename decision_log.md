# Decision log

- Selected `AppleSupport` because it has enough directly paired customer-to-brand responses for retrieval and a recognizable support domain.
- Used customer tweets paired with direct outbound replies rather than treating every tweet as an independent support example.
- Limited the working extraction to 20,000 pairs so experiments are reproducible on a laptop.
- Removed exact duplicate customer/response pairs to reduce memorization from repeated tweets.
- Used character n-gram TF-IDF retrieval because tweets contain misspellings, abbreviations, URLs, and product names.
- Defined nine intents from observed AppleSupport themes instead of importing Banking77 labels.
- Added `other_unclear` because noisy tweets often lack enough context for a reliable fine-grained label.
- Kept classification, response drafting, and escalation in one structured JSON output so each prediction is auditable.
- Forced sensitive account, payment, security, and order intents to escalate as a conservative safety policy.
- Kept replies below 35 words to match the short public-support setting.
- Removed golden examples from training by ID when available and by exact text/response pair as a fallback.
- Reported macro F1 in addition to accuracy because the assignment is about multiple intents, not only the largest class.
- Added automated reply checks for length, evidence overlap, and public handling of sensitive requests.
- Treat LLM judging as a separate measurement layer; it cannot replace an independently hand-rated agreement sample.
- Do not publish a model headline score until the golden labels, raw data version, and API run are all recorded.
