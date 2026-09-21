# sysone-bench

Laya and Jev answer the same 100 states, same questions, same seed. This repo records
who does better. As far as we can tell, nobody has run both models on identical inputs
before. The Laya author never had Jev API access, and Jev's published numbers come from
different prompts, so this is the first comparison where the inputs match byte for byte
(the harness checks question hashes before comparing).

## Results

v2 runs from 2026-09-21: 751 states, 9 suites. Laya is the `convaiinnovations/laya`
English checkpoint on a local M2 CPU. Jev is pinned `jev-1.13.0` through the TypeSafe
API. Full Jev run cost $0.008.

| Suite | n | Laya | Jev | Gap |
|---|---|---|---|---|
| triage (curated) | 160 | 0.800 | 0.888 | +0.088 |
| guardrails (curated) | 60 | 0.883 | 0.967 | +0.084 |
| moderation (curated) | 90 | 0.833 | 0.989 | +0.156 |
| agnews (4 labels) | 100 | 0.940 | 0.910 | -0.030 |
| emotion (6 labels) | 100 | 0.540 | 0.550 | +0.010 |
| banking77, 12 intents | 96 | 0.802 | 0.906 | +0.104 |
| mnli (3-way NLI) | 60 | 0.983 | 0.867 | -0.117 |
| sst5 (score, 5 levels) | 60 | 0.367 | 0.617 | +0.250 |
| multilingual intent (5 langs) | 25 | 0.360 | 1.000 | +0.640 |

v3: same 751 states through `laya.Router` reproduce all English numbers exactly and
lift multilingual intent to 0.840 (Jev 1.000). Router keys on script: 15 of 590 calls
went multilingual, Spanish/French/German stayed on English. Run files in `results/`.

The Jev lead sits in multi-class and non-English questions: 6-way `intent`
(Laya 0.725, Jev 0.975), `toxic` (0.767 vs 1.000), multilingual intent (0.360 vs 1.000).
Laya wins agnews and mnli at $0 self-hosted, and takes `churn_risk` (0.800 vs 0.750).
Emotion is weak on both (0.54-0.55, ECE near 0.3), and score questions miscalibrate
more than choice or noul on either model.

Gating at confidence 0.85 keeps 58% of traffic at 0.878 accuracy on Laya and 78% at
0.917 on Jev. Latency per 5-question call: Laya 180-660 ms local, Jev 925-1068 ms
over the API. The benchmark-cum-feedback report for TypeSafe is `FEEDBACK_REPORT.md`.
Machine specs are in `MACHINES.md`.

## Run it

1. `python3 -m venv .venv && .venv/bin/python -m ensurepip`
2. `./.venv/bin/python -m pip install laya requests python-dotenv`
3. Laya only: `USE_TF=0 ./.venv/bin/python run.py --models laya`
4. Both: `TYPESAFE_API_KEY=... ./.venv/bin/python run.py --models laya,jev`
5. Compare: `./.venv/bin/python compare.py results/<run_a>.json results/<run_b>.json`

## Layout

- `datasets/cases.py` - states + ground truth, fixed seed
- `runners/base.py` - runner interface every model adapter implements
- `runners/laya_runner.py` - local Laya inference
- `runners/jev_runner.py` - TypeSafe Decisions API (needs key)
- `run.py` - executes suites, writes `results/run_<model>_<timestamp>.json`
- `compare.py` - accuracy, ECE, latency deltas across runs
- `PLAN.md` - full plan
