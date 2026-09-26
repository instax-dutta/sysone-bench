# Purpose
- Append-only store of run outputs, head-to-head comparisons, and published reports.

# Ownership
- Own all run outputs, head-to-head comparisons, and published reports, including `results/v2/`.
- Result files are records. Never edit or overwrite one; write a new file.
- `benchmark/` provides exclusive storage primitives, but this contract governs all result paths and records.

# Local Contracts
- Legacy run file: `run_<runner-name>_<YYYYMMDD-HHMMSS>.json` with keys
  `meta`, `suites`, `speed_scaling`, `gating`; never rewrite it through the v2 path.
- V2 run directory: one exclusive run-ID directory containing `metadata.json`,
  `predictions.jsonl`, `summary.json`, `usage.json`, and `checksums.sha256`.
- `predictions.jsonl` binds exactly once to metadata suite IDs, case IDs, order indices, and split membership in suite order with calibration before evaluation. Each row's confidence must equal the recomputed normalized-answer confidence through type-sensitive JSON equality, and row plus suite latency values must be finite and non-negative. Summary `cases` and `decisions` fields are exact non-negative JSON integers; summary accuracies are finite numeric values, while integer `0` and `1` remain valid. Summary metrics use evaluation only; calibration remains in predictions and separate usage counters.
- V2 comparison directory: one exclusive comparison-ID directory under a requested output root that is disjoint from both source run directories, containing `comparison.json` and `checksums.sha256`. Source trees are rejected before metadata or descendant reads if any file or directory is symlinked. Legacy root-level `compare_<a>_vs_<b>.json` files remain historical and are never replaced.
- `meta` must include: runner name, model/version pin, timestamp, device,
  seed, question-source hash (to prove byte-identical questions).
- Legacy `compare.py` output: `compare_<a>_vs_<b>.json`. Never hand-edit.
- V2 report directory: one exclusive report directory such as `results/v2/report-20260926/` containing `REPORT.md` (prose), `report.json` (the multi-model comparison document), and `figures/` with `figure-source.json`, one CSV per figure family under `figures/data/`, and the rendered PNGs. Every figure in the directory must come from that directory's own `figure-source.json`.
- `REPORT.md` is prose and may be rewritten for clarity, but every number in it must match `report.json`, and `report.json` must match the source runs. `report.json` is the record; `REPORT.md` and the PNGs are derived views.

# Work Guidance
- Metadata and summary identity is exact: JSON integer schema `2`, JSON integer metric schema `2`, JSON integer seed `42`, the manifest dataset version, and `metadata.run_id` equal to the run directory basename. Benchmark-suite usage must report zero warmup and speed calls unless a future explicit source field is introduced.
- Resolved release blocker: row-to-metadata model identity comparison uses recursive type-sensitive equality, so boolean and numeric identities cannot alias.
- Never synthesize, repair, or migrate a result record. A failed v2 process may leave an
  incomplete append-only directory and must not reuse it on retry.

# Verification
- Every legacy run file remains byte-identical to its pre-v2 state.
- V2 run and comparison checksums are verified by `benchmark.orchestrator.compare_v2`
  before metrics are read; mismatches create no comparison artifact.
- A report directory is only publishable when `benchmark.report` wrote it, which requires every source run's `checksums.sha256` to verify and every recomputed case count, decision count, and per-suite accuracy to equal that run's sealed `summary.json`.

# Child DOX Index
- None.
