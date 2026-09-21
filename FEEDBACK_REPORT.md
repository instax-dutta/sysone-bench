# Benchmark-cum-feedback report: Jev 1.13.0 vs Laya, byte-identical inputs

- To: TypeSafe team. From: independent benchmarker, no affiliation.
- Date: 2026-09-21. Harness: https://github.com/instax-dutta/sysone-bench
- Runs: `results/run_laya_20260921-213504.json` vs `results/run_jev-1.13.0_20260921-212125.json`.
- Fairness: same 751 states, same question dicts, seed 42, question hashes verified
  identical before comparing. Laya is `convaiinnovations/laya` English, local CPU, defaults.
  Jev is pinned `jev-1.13.0`. Full Jev run cost $0.008 (192K input tokens).

## Headline numbers

| Suite | n | Laya | Jev | Gap | Laya ECE | Jev ECE |
|---|---|---|---|---|---|---|
| triage (curated) | 160 | 0.800 | 0.888 | +0.088 | 0.079 | 0.053 |
| guardrails (curated) | 60 | 0.883 | 0.967 | +0.084 | 0.065 | 0.049 |
| moderation (curated) | 90 | 0.833 | 0.989 | +0.156 | 0.082 | 0.052 |
| agnews (4 labels) | 100 | 0.940 | 0.910 | -0.030 | 0.223 | 0.091 |
| emotion (6 labels) | 100 | 0.540 | 0.550 | +0.010 | 0.302 | 0.274 |
| banking77, 12 intents | 96 | 0.802 | 0.906 | +0.104 | 0.200 | 0.087 |
| mnli (3-way NLI) | 60 | 0.983 | 0.867 | -0.117 | 0.236 | 0.153 |
| sst5 (score, 5 levels) | 60 | 0.367 | 0.617 | +0.250 | 0.220 | 0.257 |
| multilingual intent (5 langs) | 25 | 0.360 | 1.000 | +0.640 | 0.306 | 0.004 |

## Feedback for TypeSafe

1. Emotion is the weak primitive on both models (0.54-0.55, ECE ~0.3). Six overlapping
   affect labels with single-word criteria under-specify the question. Richer per-option
   criteria would likely lift both accuracy and calibration. Worth a docs recipe.
2. Score questions miscalibrate more than choice/noul on both models (sst5 ECE 0.22-0.26).
   Users gating automation on score confidence should be warned explicitly.
3. MNLI goes against the trend: Laya 0.983 vs Jev 0.867. Laya's bidirectional encoder
   seems to suit premise-hypothesis comparison. If Jev underperforms NLI-shaped
   questions generally, that is worth knowing before selling RAG-relevance use cases.
4. API nit: `typesafe:jev-1.13.0` (your own Pydantic naming) is rejected with a bare 400.
   The error message does not say the model id is the problem. Accept the prefixed form
   or name the offending field.
5. Latency from our region measured 925-1068 ms per call, above the 70-500 ms claim.
   Fine for batch triage, too slow for inline request paths. Report the measurement
   location dependence in docs.
6. Multilingual is Jev's clearest lead: 1.000 vs 0.360 across Hindi, Spanish, French,
   German, Arabic. This deserves headline placement in your own comparisons.

## Notes on Laya (for the record)

- Laya matches or beats Jev on agnews (0.94) and mnli (0.983) at $0 self-hosted.
- It collapses exactly where its author predicts: non-English intent (0.36) and sst5
  score (0.367). The honest-limits section of BENCHMARKS.md held up in our runs.

## v3 addendum: routed Laya (2026-09-22)

Ran the same 751 states through `laya.Router` (english + multilingual preloaded).
English suites reproduce the base numbers exactly. Multilingual intent moves
0.360 to 0.840 (Jev 1.000). Router sent 15 of 590 calls to multilingual;
routing keys on script, so Spanish/French/German stayed on the English checkpoint.
Run file: `results/run_laya-router_20260922-001411.json`.

## v4 addendum: Qwen2.5-1.5B-Instruct + parallel constrained decoding (2026-09-22)

Fourth column, same 751 states. The harshatheg/Qwen-2.5-1B-RLCD repo ships no weights,
so this is stock `mlx-community/Qwen2.5-1.5B-Instruct-4bit` under the vendored parallel
engine, local MLX on M2. Score questions asked as enums over rubric levels.

Triage 0.825, guardrails 0.900, moderation 0.700, agnews 0.730, emotion 0.540,
banking77_12 0.500, mnli 0.733, sst5 0.617 (ties Jev), multilingual 0.880.
Latency ~700-1150 ms per call on M2, below the author's M4 Max figures as expected.

Two findings: first-token logit slicing degrades as option count grows (banking77
12-way at 0.500 while 4-6 way stays competitive), and emotion is now a three-model
wall at 0.54-0.55. Run file: `results/run_qwen-pcd-15b_20260922-003839.json`.

## Limits of this report

- Curated suites are hand-labeled by one author. Public subsets are random draws,
  seed 42, sizes 25-100. CIs are wide on the small suites.
- English Laya checkpoint only; no Router, no typed-decisions checkpoint.
- One Jev version, one day, one region.
