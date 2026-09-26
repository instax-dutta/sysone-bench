# Benchmark core package

## Purpose
- Own the framework-independent v2 benchmark core package.

## Ownership
- Own canonical v2 JSON serialization and content digests in `benchmark/canonical.py`.
- Own question and answer validation in `benchmark/contracts.py`.
- Own manifest loading, validation, summaries, and digesting in `benchmark/manifest.py`.
- Own append-only run-directory and artifact creation in `benchmark/storage.py`.
- Own exact per-phase usage accounting in `benchmark/usage.py`.
- Own injected call timing and model-agnostic speed-scaling orchestration in `benchmark/timing.py`; this module must not initialize or invoke model SDKs.
- Own sealed-manifest execution, v2 artifact schemas, phase-split accounting, and fail-closed comparison in `benchmark/orchestrator.py`.
- Own task-specific primitive metrics, calibration summaries, and raw metric inputs in `benchmark/metrics.py`.
- Own paired clustered uncertainty, permutation inference, Holm correction, and binary temperature calibration in `benchmark/statistics.py`.
- Own deterministic, provenance-bound figure source rows and exclusive CSV/JSON exports in `benchmark/figure_data.py`.
- Own multi-model report assembly from sealed runs, including checksum verification, sealed-summary cross-checks, and paired inference, in `benchmark/report.py`.
- Own deterministic figure rendering from figure source data only, in `benchmark/graphics.py`.
- Keep model adapters under `runners/`.
- `results/`, including `results/v2/`, is owned by `results/AGENTS.md`; this package provides storage primitives but does not change the records contract.
- No module may mutate legacy artifacts under `results/`.

## Local Contracts
- Expose the v2 package version as `benchmark.__version__ == "2.0.0"`.
- Keep core modules free of model inference and network access.
- Serialize canonical JSON as UTF-8 with sorted object keys, compact separators, and rejected non-finite numbers; expose full lowercase SHA-256 digests.
- Validate each question before matching its answer; reject unsupported types or fields, duplicate question IDs, missing or extra answers, mismatched types, and invalid ranges instead of normalizing or dropping data.
- Reject malformed or non-standard manifest JSON, duplicate object keys, non-v2 records, missing or unknown fields, invalid questions, duplicate case or suite-order identities, and suite-count mismatches; digest validated records in their original order.
- The v2 loader rejects symlinked manifest or checksum components, reads each input once, parses strict JSONL or the existing JSON-list format from the captured manifest bytes, and computes the raw digest from those same bytes. It requires ordered zero-based suite records, validates choice labels, integer 0/1 noul labels, and finite numeric score levels within `max_score`, and completes all record, identity, and canonical-digest checks before claiming output. When a checksum path is supplied, require exactly `<raw-file-sha256>  manifest.jsonl\n` from its single captured read.
- `run_suite_v2(runner, suite_id, records, counter)` runs one split only, validates every answer ID without key coercion through `benchmark.contracts.validate_answers`, digests the sealed state before a potentially mutating call, measures one call with `benchmark.timing.measure_call`, and records usage only after validation succeeds.
- `run_all_v2(runner_names, manifest_path, output_root)` resets Python random seed `42` independently before constructing each model, uses UTC-plus-UUID run IDs when omitted, rejects normalized or case-folded target collisions before runner construction or output creation, enforces symlink-free containment, lazily imports model adapters, keeps calibration and evaluation `UsageCounter` snapshots separate, and reports evaluation-only headline metrics. Each row retains ordered question definitions and IDs, expected and full normalized answers, confidence, per-call usage, latency, split phase, the pre-call state digest, a stable model-identity projection, and redacted optional metadata.
- `summarize_predictions` accepts ordered v2 prediction rows with `execution_phase == "benchmark"` and `phase == split`, groups choice, Noul, and score decisions, uses evaluation rows for headline metrics, fits probability calibration only from calibration rows, returns stable primitive keys with `None` for unavailable values, and preserves raw per-decision inputs without coercing valid numeric forms. NLL is `None` when a valid zero probability makes the aggregate undefined; raw inputs remain available. AUPRC is tie-safe, row-order invariant, and `None` for one-class cases. One-class calibration returns `None` for fitted ECE. Choice uses exact labels and top-label probability; Noul thresholds P(true) at `0.5`; score uses finite nonnegative half-up levels and never emits probability ECE or risk-coverage.
- `benchmark/statistics.py` validates finite numeric inputs, equal lengths, nonempty data, non-empty hashable cluster IDs, positive integer replicates, and first-seen cluster order. Paired primary effects use `b_rows - a_rows` (Jev minus Laya), weight clusters by row count, and use paired whole-cluster resampling with linear 2.5%/97.5% percentile interpolation. Permutation uses independent cluster sign flips with a finite add-one two-sided convention; Holm restores input order and enforces monotonicity; temperature functions accept only finite binary probabilities and reject invalid, one-class, or constant-probability calibration inputs.
- Noul calibration in `summarize_predictions` uses `benchmark.statistics.fit_temperature` only on calibration rows and `apply_temperature` only on evaluation rows. Each raw Noul record retains the raw `probability` and `raw_probability`, exposes the fitted `temperature`, and stores `fitted_probability` only for evaluation rows; all values remain canonical-JSON-safe.
- `benchmark/figure_data.py` projects actual pairwise v2 comparisons and the release comparison fixture into accuracy, paired-effects, calibration, risk-coverage, efficiency, multilingual, and Router families. Every nonempty row binds aligned source run ID/checksum pairs, dataset version, full manifest digest, split, metric definition, the stable `paired_state_cluster_bootstrap_20000_seed_42` uncertainty method, an explicit sample count or `null`, a context-derived `source_kind` of `base` or `router`, and finite numeric plotted/value fields. Generic rows require complete, nonconflicting model or explicit run/checksum attribution and never infer all base sources for an unknown model. Probability families accept only choice and Noul primitives. Router identity may be bound through comparison models, sources, or explicit Router run/checksum metadata and remains separate from base-model families; route reasons and counts are preserved. Generic Router-family rows that carry run/checksum or comparison-model attribution while no Router-attributed source run exists are rejected with a `ValueError` instead of being dropped into an empty family; absent Router data still produces an empty family. Unavailable rich metrics remain absent, and empty-family CSV exports carry one stable reason instead of fabricated zeros.
- Figure source rows sort deterministically with the complete canonical row as final tie-breaker. `write_figure_data` revalidates every row before creating output, requires Router-family rows to carry `source_kind == "router"` and all other families to carry `source_kind == "base"` even without a comparison context, sorts every family, and writes one newline-terminated CSV per family plus canonical newline-terminated `figure-source.json` through exclusive, symlink-free paths without replacing any existing artifact.
- A normal v2 run directory contains `metadata.json`, `predictions.jsonl`, `summary.json`, `usage.json`, and `checksums.sha256`. Every pre-existing normal target is rejected, including an otherwise empty or sentinel-only directory. Task 5's exact `.sysone-owner` marker is the sole reuse exception only when the existing directory basename equals the requested run ID and no required v2 artifact exists; it defers the checksum so the launcher can finalize it after `postflight.json`.
- `compare_v2(path_a, path_b, output_root)` requires the comparison root to be disjoint from both source run paths, validates symlink-free paths and the complete source-run trees before reading metadata or descendants, checksums, exact JSON integer schema/metric versions and seed `42`, manifest dataset identity, `metadata.run_id` equal to each source directory basename, suite/case/order/question/split identity, exact calibration-then-evaluation row sequence, stable row-to-metadata model identity, recursive type-sensitive recomputed confidence and metric equality, finite non-negative row and suite latency, exact derived summary counts and finite accuracy, benchmark-only zero warmup/speed usage, and metric reproducibility. Any mismatch creates no comparison artifact.
- Resolved release blocker: row-to-metadata model identity uses recursive type-sensitive equality, so JSON booleans cannot compare equal to numeric values.
- The module CLI requires `--models`, `--manifest`, `--manifest-checksum`, `--output-root`, and `--run-id`. Root `run.py --models ...` supplies the sealed manifest/checksum/output defaults and fails clearly while the draft is unsealed; explicit flags override them. Root `compare.py path_a path_b` defaults to `results/v2/comparisons`. Both print only model/suite progress without state, answers, or secrets.
- Create v2 run directories and artifacts exclusively, serialize JSON and JSONL canonically with final newlines, validate the complete original lexical component stack before normalization so missing components cannot hide `..` or a symlink, reject symlinks before reading or enumeration, use recursive JSON equality that never equates booleans with numbers, and emit deterministic checksums without replacing an existing path.
- `benchmark/report.py` reads only finished run directories and the sealed manifest. It verifies every artifact named in each run's `checksums.sha256` before use, requires `metadata.run_id` to equal the directory basename, recomputes per-decision correctness through the public `metrics.decision_correct` rule, and refuses to emit a report unless the recomputed case counts, decision counts, and per-suite accuracies reproduce each run's own sealed `summary.json` exactly.
- `benchmark/report.py` refuses to compare runs produced against different manifest digests or dataset versions, and the paired Jev-minus-Laya contrast requires identical decision keys and complete correctness coverage in both runs. Paired effects cluster on case ID because several questions in one case share a state, use 20,000 bootstrap and permutation replicates at seed `42`, and Holm-adjust p-values across suites.
- `benchmark/graphics.py` renders only from `figure-source.json`, so a chart cannot disagree with validated rows. It never reads a run directory, never invents a value, and refuses to overwrite an existing figure file. Suite accuracy and multilingual accuracy always start at zero; latency uses a log axis because the observed spread exceeds 40x. No radar, 3D, or composite cross-suite score is produced.
- Preserve append-only behavior for v2 artifacts and historical results.
- Require run IDs for run-directory creation to be safe single path components under the results root.
- `UsageCounter` exposes exactly `warmup`, `benchmark`, and `speed`, each with independent `input_tokens`, `output_tokens`, and `calls`; present token values must be non-negative integers, bools and malformed types fail, and snapshots are detached JSON-serializable dictionaries.
- `measure_call` takes exactly one clock reading immediately before and one immediately after the callable, preserves the callable result, and returns the unrounded elapsed float.
- `speed_scaling` requires unique positive sizes and positive repetitions, uses exactly two untimed warmups plus the requested timed repetitions per size, accounts warmup and speed calls separately through `UsageCounter`, and returns unrounded linear-interpolation p50, p95, and sample count from sorted samples in request order.

## Work Guidance
- Keep additions typed, deterministic, framework-independent, and independent of runner implementations.
- Keep model SDK imports behind the default runner factory and only after sealed-input validation; manifest parsing and comparison must remain importable in a clean image without Laya, Torch, or Transformers.
- Exercise orchestration, timing, and speed-scaling behavior only with fake `BaseRunner` implementations; never perform local model inference.
- Preserve record order, canonical bytes, full digests, exclusive writes, and stable public interfaces.
- Do not repair, migrate, or otherwise mutate legacy result artifacts as part of this package.

## Verification
- `uv run ruff check benchmark tests`
- `uv run --offline mypy --python-version 3.12 --allow-subclassing-any --follow-imports=skip --ignore-missing-imports benchmark`
- `uv run --offline pytest tests/test_run_compare_v2.py -q`
- `uv run --offline pytest tests/test_metrics.py -q`
- `uv run --offline pytest tests/test_statistics.py -q`
- `uv run --offline pytest tests/test_figure_data.py -q`
- `uv run --offline pytest tests/test_report.py -q`
- `uv run --offline pytest tests/test_graphics.py -q`
- `uv run --offline pytest tests/test_runner_contracts.py tests/test_timing.py -q`
- Scoped mypy command: `uv run --offline mypy --python-version 3.12 --allow-subclassing-any --follow-imports=skip --ignore-missing-imports benchmark/orchestrator.py benchmark/manifest.py benchmark/storage.py run.py compare.py tests/test_run_compare_v2.py ops/remote/remote_jev_entrypoint.py`.
- `uv run pytest`
- Whole-tree Ruff remains outside this task because unrelated legacy runner modules retain separate scoped contracts.

## Child DOX Index
- No child DOX files yet.
