# Plan: first independent Laya vs Jev benchmark

## Goal
Publish the first benchmark where Laya and Jev answer byte-identical states and
questions in the same harness. Everything else available today compares vendor-published
numbers collected on different prompts.

## Why it wins
1. Genuinely unclaimed (Laya author never had Jev API access).
2. Half built already (100 states, Laya data from a full run on the shared Linux host).
3. Benchmark repos get cited by both camps.

## Steps
1. DONE: model-agnostic harness with Laya runner; Jev runner live (model id `jev-1.13.0`).
2. DONE: 100 curated states with ground truth (`datasets/cases.py`).
3. DONE: live Jev run 2026-09-21, same states, same seed, question hashes verified identical.
4. DONE: `compare.py` head-to-head (see `results/compare_laya_vs_jev-1.13.0.json`).
5. TODO: publish GitHub repo + writeup.

## Fairness rules (binding)
- Same states, same question dicts, same seed for both models. No per-model prompt tuning.
- Laya runs `convaiinnovations/laya` English checkpoint at defaults.
- Jev runs pinned versioned id (not `jev-latest`), recorded in the result file.
- Report raw and temperature-fitted ECE separately; never compare fitted-vs-raw.
- Sample sizes: 100 states / ~310 decisions v1. Note CIs are wide; v2 adds public
  datasets (AG News, Banking77 subset, XNLI subset).

## Risks
1. Jev early-access API shape may drift. Mitigation: pin version, record raw responses.
2. Small N. Mitigation: state it plainly, ship CIs in compare.py.
3. Cost: 310 decisions is a few thousand tokens, under $1 on Jev pricing.

## Time estimate
- Harness + docs: done today (~1 hr).
- Jev wiring + live run once key lands: ~30 min.
- Writeup + publish: ~1 hr.
