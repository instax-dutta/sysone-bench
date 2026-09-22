# sysone-bench

**The first independent head-to-head benchmark of System One decision models on
byte-identical inputs.**

Laya (open weights, Apache-2.0) and Jev (TypeSafe, closed API) answer the exact same
states and typed questions in the same run, with the same seed. Question SHA hashes are
verified identical before any comparison, so the inputs match byte for byte. Vendor
numbers are published from different prompts and are not directly comparable; this repo
is.

## Why this exists

Before this repo there was no comparison where Laya and Jev saw the same input. The Laya
author never had Jev API access, and Jev's published numbers come from different prompts,
so cross-vendor claims could not be checked. This benchmark closes that gap.

## Fairness rules (binding)

- Same states, same question dicts, same seed for every model. No per-model prompt tuning.
- Pinned, versioned model ids recorded in every result file.
- Question hashes verified identical before comparing.
- Raw and temperature-fitted ECE reported separately, never mixed.
- Results are append-only records: a published run is never overwritten.

## Results

### v2 - 751 states, 9 suites (2026-09-21)

Laya is the `convaiinnovations/laya` English checkpoint on a local M2 CPU. Jev is pinned
`jev-1.13.0` through the TypeSafe API. The full Jev run cost $0.008.

| Suite | n | Laya | Jev | Qwen-PCD | Best |
|---|---|---|---|---|---|
| triage (curated) | 160 | 0.800 | 0.888 | 0.825 | Jev |
| guardrails (curated) | 60 | 0.883 | 0.967 | 0.900 | Jev |
| moderation (curated) | 90 | 0.833 | 0.989 | 0.700 | Jev |
| agnews (4 labels) | 100 | 0.940 | 0.910 | 0.730 | Laya |
| emotion (6 labels) | 100 | 0.540 | 0.550 | 0.540 | tie |
| banking77, 12 intents | 96 | 0.802 | 0.906 | 0.500 | Jev |
| mnli (3-way NLI) | 60 | 0.983 | 0.867 | 0.733 | Laya |
| sst5 (score, 5 levels) | 60 | 0.367 | 0.617 | 0.617 | Jev/Qwen tie |
| multilingual intent (5 langs) | 25 | 0.360 | 1.000 | 0.880 | Jev |

Qwen-PCD is a secondary open baseline: stock Qwen2.5-1.5B-Instruct with parallel
constrained decoding (same code as `harshatheg/Qwen-2.5-1B-RLCD`, which ships no
fine-tuned weights), local MLX on M2. It collapses on 12-option enums (banking77 0.500)
but matches Jev on sst5 and trails only Jev on multilingual (0.880).

The Jev lead concentrates in multi-class and non-English questions: 6-way `intent`
(Laya 0.725, Jev 0.975), `toxic` (0.767 vs 1.000), multilingual intent (0.360 vs
1.000). Laya wins agnews and mnli at $0 self-hosted, and takes `churn_risk`
(0.800 vs 0.750). Emotion is weak on both (0.54-0.55, ECE near 0.3), and `score`
questions miscalibrate more than `choice` or `noul` on either model.

Gating at confidence 0.85 keeps 58% of traffic at 0.878 accuracy on Laya and 78% at
0.917 on Jev. Latency per 5-question call: Laya 180-660 ms local, Jev 925-1068 ms over
the API.

### v3 - routed Laya

The same 751 states through `laya.Router` reproduce all English numbers exactly and lift
multilingual intent to 0.840 (Jev 1.000). The Router keys on script: 15 of 590 calls went
multilingual, while Spanish/French/German stayed on the English checkpoint.

### Supporting documents

- `REPORT.md` - accuracy, ECE, gating, latency for the head-to-head.
- `FEEDBACK_REPORT.md` - benchmark-cum-feedback report sent to TypeSafe.
- `PLAN.md` - the benchmark plan and binding fairness rules.
- `MACHINES.md` - machine specs for every recorded run.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/python -m ensurepip
./.venv/bin/python -m pip install laya requests python-dotenv

# Laya only (open weights, no key needed)
USE_TF=0 ./.venv/bin/python run.py --models laya

# Laya and Jev (needs a TypeSafe key in .env as TYPESAFE_API_KEY)
TYPESAFE_API_KEY=... ./.venv/bin/python run.py --models laya,jev

# Compare two runs
./.venv/bin/python compare.py results/<run_a>.json results/<run_b>.json
```

## Repository layout

- `datasets/cases.py` - curated states and ground truth, fixed seed.
- `datasets/public_cases.json` - public-suite states and questions.
- `runners/base.py` - the runner interface every model adapter implements.
- `runners/laya_runner.py` - local Laya inference.
- `runners/jev_runner.py` - TypeSafe Decisions API (needs `TYPESAFE_API_KEY`).
- `run.py` - executes suites, writes `results/run_<model>_<timestamp>.json`.
- `compare.py` - accuracy, ECE, and latency deltas across runs.
- `results/` - append-only run outputs and comparisons.

## Citation

```bibtex
@misc{sysonebench2026,
  title  = {sysone-bench: independent head-to-head benchmark of System One decision models},
  year   = {2026},
  author = {instax-dutta},
  url    = {https://github.com/instax-dutta/sysone-bench}
}
```

## License

MIT - see `LICENSE`. Model weights and datasets keep their own licenses (Laya is
Apache-2.0; public suites follow their upstream terms).
