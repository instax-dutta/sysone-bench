# Purpose
- Owns all benchmark states, ground truth labels, and labeling rules.
- Every model answers these exact bytes; fairness depends on this file not changing per model.

# Ownership
- Cases curated by repo owner. Additions require ground truth from two agreeing labelers or a cited public dataset.

# Local Contracts
- Case format: `(state_dict, {question_id: expected})` where expected is a label string for
  `choice` questions, 0/1 for `noul` questions, or an int level for `score` questions.
- `SUITES` maps suite name to `(questions_fn_name, cases)`. Question dicts come from the
  `laya` package presets so both runners share identical wording.
- `public_cases.json` holds public-source suites with inline question dicts:
  agnews (100), emotion (100), banking77_12 (96), mnli (60), sst5 (60, score),
  multilingual_intent (25). Built by `build_public.py`, SEED=42.
- `SEED = 42`. Case order is fixed; runners must not shuffle.
- Never edit an existing case to favor a model. Add new cases instead.

# Work Guidance
- Keep suites balanced across labels. Triage intents: 8 per class across 40 states.
- Guardrails/moderation: 50/50 positive/negative per question where possible.

# Verification
- `python3 -c "from datasets.cases import SUITES; print({k: len(v[1]) for k, v in SUITES.items()})"`
  must print `{'triage': 40, 'guardrails': 30, 'moderation': 30}`.

# Child DOX Index
- None. Single file scope.
