# Purpose
- Append-only store of run outputs, head-to-head comparisons, and published reports.

# Ownership
- Own all run outputs, head-to-head comparisons, and published reports, including `results/v2/`.
- Result files are records. Never edit or overwrite one; write a new file.
- `benchmark/` provides exclusive storage primitives, but this contract governs all result paths and records.
- `results/raw/` holds the raw V2 run directories and is **gitignored**. 52 verified run directories
  (99 including the 47 `-v210` successors), CPU under `cpu/runs/` and GPU under `gpu/`, plus
  `quarantine/` and the partial `colab/` salvage. The `-v210` successors sit as siblings of the run
  they supersede, never nested inside it, or the auditor stops preferring them and the site silently
  reverts to 2.0.0 numbers.
- It is gitignored because git retains every blob permanently. Under the append-only rule, committing
  it would add roughly 2MiB of unrecoverable history per future run, and deleting it later would not
  reclaim the space. `.git` is under 3MiB and must stay that way.
- Point tooling at that root with `--results-root results/raw` or `SYSONE_RESULTS_ROOT`. Never
  hardcode a home-directory path into a script.

# Local Contracts
- Legacy run file: `run_<runner-name>_<YYYYMMDD-HHMMSS>.json` with keys
  `meta`, `suites`, `speed_scaling`, `gating`; never rewrite it through the v2 path.
- V2 run directory: one exclusive run-ID directory on the results root, containing `metadata.json`,
  `predictions.jsonl`, `summary.json`, `usage.json`, and `checksums.sha256`.
- `predictions.jsonl` binds exactly once to metadata suite IDs, case IDs, order indices, and split membership in suite order with calibration before evaluation. Each row's confidence must equal the recomputed normalized-answer confidence through type-sensitive JSON equality, and row plus suite latency values must be finite and non-negative. Summary `cases` and `decisions` fields are exact non-negative JSON integers; summary accuracies are finite numeric values, while integer `0` and `1` remain valid. Summary metrics use evaluation only; calibration remains in predictions and separate usage counters.
- V2 comparison directory: one exclusive comparison-ID directory under a requested output root that is disjoint from both source run directories, containing `comparison.json` and `checksums.sha256`. Source trees are rejected before metadata or descendant reads if any file or directory is symlinked. Legacy root-level `compare_<a>_vs_<b>.json` files remain historical and are never replaced.
- `results/raw/` is the working copy; the authoritative mirror is the private dataset
  `saidutta69/sysone-bench-gpu-results`, one top-level entry per run. Verified file-by-file at
  578/579 byte-identical with zero mismatches on 2026-10-09, covering all 52 run directories, all 47
  `-v210` successors, `quarantine/` and the `colab/` salvage. The single non-matching local file is
  the generator's scratch `rescore.json`. `gpu/from-bundle` is deliberately absent because it failed
  checksum verification.
- **A v2.1.0 successor record is only as durable as its mirror.** The `-v210` directories were never
  uploaded, and when the working copy was lost the whole published panel had to be regenerated from
  stored predictions. Upload them in the same operation that writes them. `tools/write_runs_v210.py`
  reproduces them deterministically, and regenerating is a valid recovery path because the rescore is
  pure arithmetic over stored answers.
- Every run directory must exist in the mirror before the working copy may be discarded.
- Verify the mirror byte-for-byte, not by presence alone: compare each local file's sha256 against
  the remote `resolve/main` object before trusting it.
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
- **A measurement whose artifacts cannot be reproduced is withdrawn, not published from a superseded
  run.** Move the run directory to `results/raw/quarantine/`, which the auditor does not scan, keep the
  record intact and append-only, and list the entry in `webpage/build-data.py` `EXCLUDED_NOTES` with
  the reason. Publishing the number from a superseded or broken run misstates the model; publishing an
  unreproducible one misstates the evidence. `jeff` is the worked example: it measured 0.6855 on a
  DGX Spark that was never mirrored, so the row was withdrawn rather than dropped to the 0.5492 of the
  broken earlier run.

# Verification
- Every legacy run file remains byte-identical to its pre-v2 state.
- V2 run and comparison checksums are verified by `benchmark.orchestrator.compare_v2`
  before metrics are read; mismatches create no comparison artifact.
- A report directory is only publishable when `benchmark.report` wrote it, which requires every source run's `checksums.sha256` to verify and every recomputed case count, decision count, and per-suite accuracy to equal that run's sealed `summary.json`.

# Child DOX Index
- None.
