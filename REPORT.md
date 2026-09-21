# Laya vs Jev: first byte-identical head-to-head

- Date: 2026-09-21. Harness: `run.py` + `compare.py` in this repo.
- Runs: `results/run_laya_20260921-205934.json` vs `results/run_jev-1.13.0_20260921-205530.json`.
- Fairness proof: question SHA hashes identical across both runs
  (triage `5c28e01cc04a`, guardrails `30b339fbd2c8`, moderation `74aaebcdde9b`).
- Laya: `convaiinnovations/laya` English checkpoint, local M2 CPU, defaults.
- Jev: pinned `jev-1.13.0` via TypeSafe API (note: `typesafe:`-prefixed ids are rejected, use bare id).

## Headline

Jev wins all three suites. Gaps are real but CIs overlap on guardrails.

| Suite | Laya acc (95% CI) | Jev acc (95% CI) | Delta | Laya ECE | Jev ECE |
|---|---|---|---|---|---|
| triage (n=160) | 0.800 (0.738-0.862) | 0.894 (0.846-0.942) | +0.094 | 0.079 | 0.034 |
| guardrails (n=60) | 0.883 (0.802-0.965) | 0.950 (0.895-1.000) | +0.067 | 0.065 | 0.055 |
| moderation (n=90) | 0.833 (0.756-0.910) | 0.989 (0.967-1.000) | +0.156 | 0.082 | 0.054 |

## Per-question standouts

- Biggest gap: `intent` 6-way (Laya 0.725, Jev 0.975) and `toxic` (Laya 0.767, Jev 1.000).
- Only Laya win: `churn_risk` (0.800 vs 0.750).
- Near ties: `refund_requested` (0.925 vs 1.000), `prompt_injection` (0.867 vs 0.900).

## Calibration and gating

- Both ship well-calibrated on this suite (ECE 0.03-0.08). Jev lower everywhere.
- Gating at >= 0.85: Laya 69% coverage at 0.953 acc; Jev 83% coverage at 0.988 acc.
- Gating at >= 0.95: Laya 46% at 0.965; Jev 66% at 0.990.

## Latency and cost

- Laya local M2 CPU: 375-476 ms per 5-question call.
- Jev API end-to-end: 885-1017 ms per call (network included, helps explain why it trails
  TypeSafe's 70-500 ms claim from here).
- Cost of the Jev run: ~140 calls, a few thousand tokens total, well under $1 at
  $0.042/1M input tokens. Laya: $0 self-hosted after the ~808 MB download.

## Limits

- N=100 states, hand-labeled, one author. CIs are wide; guardrails gap is not significant.
- Preset subsets only (4 triage + 2 guard + 3 moderation questions); `score` questions unscored.
- English only, no Router/multilingual split, no public datasets yet.
- One Jev version, one day. `jev-latest` moves; re-run on version change.

## Takeaway

For everyday triage/guard/moderation with binary-heavy questions, Laya at $0 is within
7-16 points of Jev and gates almost as well. The gap concentrates in fine-grained
multi-class judgments (`intent`, `toxic`), where Jev is near-perfect here. If your
workload is binary flags plus confidence gating, self-hosted Laya is the value pick;
if 6-way intent or toxicity precision matters, Jev earns its per-token price.
