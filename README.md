# sysone-bench: Laya vs Jev head-to-head

First independent benchmark of System One decision models on byte-identical inputs.

## Status
- [x] Laya runner + 100 curated states (triage 40, guardrails 30, moderation 30)
- [x] Full Laya run on pelican CPU (310 decisions, see `results/`)
- [x] Jev API key + live Jev run (2026-09-21, pinned `jev-1.13.0`)
- [x] Head-to-head comparison (`results/compare_laya_vs_jev-1.13.0.json`, see `REPORT.md`)
- [ ] Publish to GitHub

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
