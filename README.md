# sysone-bench

Laya and Jev answer the same 100 states, same questions, same seed. This repo records
who does better. As far as we can tell, nobody has run both models on identical inputs
before. The Laya author never had Jev API access, and Jev's published numbers come from
different prompts, so this is the first comparison where the inputs match byte for byte
(the harness checks question hashes before comparing).

## Results

Runs from 2026-09-21. Laya is the `convaiinnovations/laya` English checkpoint on a local
M2 CPU. Jev is pinned `jev-1.13.0` through the TypeSafe API.

| Suite | Laya | Jev | Gap |
|---|---|---|---|
| triage (160 decisions) | 0.800 | 0.894 | +0.094 |
| guardrails (60 decisions) | 0.883 | 0.950 | +0.067 |
| moderation (90 decisions) | 0.833 | 0.989 | +0.156 |

95% confidence intervals: triage Laya 0.738-0.862, Jev 0.846-0.942; guardrails Laya
0.802-0.965, Jev 0.895-1.000; moderation Laya 0.756-0.910, Jev 0.967-1.000. The
guardrails gap is inside the noise.

Per question, the gap sits in multi-class judgments: 6-way `intent` (Laya 0.725, Jev
0.975) and `toxic` (0.767 vs 1.000). The one Laya win is `churn_risk` (0.800 vs 0.750).
Binary flags are close: `refund_requested` 0.925 vs 1.000, `prompt_injection` 0.867 vs 0.900.

Calibration is solid on both sides: ECE 0.065-0.082 for Laya, 0.034-0.055 for Jev.
Gating at confidence 0.85 keeps 69% of traffic at 0.953 accuracy on Laya and 83% at
0.988 on Jev.

Latency per 5-question call: Laya 375-476 ms local, Jev 885-1017 ms over the API.
The Jev run cost under $1. Laya costs nothing after the download. Full writeup with
limits and takeaways is in `REPORT.md`. Machine specs are in `MACHINES.md`.

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
