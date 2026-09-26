# Sysone-bench v2 Finalization Design

- Date: 2026-09-24
- Status: Approved in interactive design review
- Scope: Benchmark integrity, corrected model runs, audited reporting, and Qwen 3.5-style benchmark graphics
- Implementation policy: Do not commit, push, publish externally, or overwrite historical results unless explicitly requested

## 1. Objective

Produce a publication-grade sysone-bench release that:

1. Uses a frozen, independently reviewed, provenance-complete dataset.
2. Runs Laya, Jev, Qwen-PCD, and Laya Router through strict, auditable adapters.
3. Preserves every historical result record unchanged.
4. Publishes paired statistical comparisons without a composite leaderboard score.
5. Generates clean Qwen 3.5-style benchmark images suitable for GitHub and Hugging Face model cards.
6. Can be reproduced from documented commands and verified from immutable artifacts.

## 2. Non-goals

- No vendor-published cross-comparison.
- No training or fine-tuning.
- No per-model benchmark prompt tuning.
- No new model families beyond the existing Laya, Jev, Qwen-PCD, and Router scope.
- No composite normalized score across heterogeneous tasks.
- No radar charts, decorative dashboards, 3D charts, or truncated accuracy axes.
- No direct modification of existing result JSON files.

## 3. Binding decisions

- Use fresh corrected runs.
- Label v2 with the `human-reviewed-ai-assisted-v1` protocol: one human reviewer reads every case and corrects an AI draft, with no second independent review and no adjudication. The two-reviewer seal path stays in the repository for a future upgrade, but the released dataset does not claim it.
- Use Qwen 3.5 small-multiple aesthetics for final benchmark graphics.
- Show Laya, Jev, and Qwen-PCD as three equal model series in main figures.
- Show Router in a dedicated deployment-variant figure rather than as a fourth base-model series.
- Do not publish a single overall leaderboard score.
- Publish assets in this GitHub repository and make them Hugging Face-ready.
- Run open models on `tejes@pelican` without affecting existing workloads.
- Stream `TYPESAFE_API_KEY` transiently and never store it in the repository or on pelican.
- Use dark mode for future visual brainstorming pages.
- Treat Laya and Jev as the primary manifest-identical comparison. Qwen-PCD is a secondary baseline with a documented schema transform, and Router is a deployment variant of Laya.

## 4. Visual reference pattern

The final image system follows patterns observed in current Hugging Face model cards and release assets from Qwen, DeepSeek, Mistral, Kimi, Google, Microsoft, AllenAI, and OpenAI.

The selected Qwen 3.5 pattern uses:

- White image canvases.
- Equal small-multiple panels.
- Bold benchmark names.
- Gray metric subtitles.
- Direct value labels.
- Restrained, colorblind-accessible series colors.
- Consistent model order and shared legend.
- One principal message per image.
- Explicit metric, sample-size, and provenance footers.

The final design does not copy any lab logo, proprietary asset, or brand styling.

## 5. Repository architecture

### 5.1 New durable paths

```text
benchmark/
  contracts.py
  canonical.py
  manifest.py
  storage.py
  metrics.py
  statistics.py
  reporting.py
  graphics.py

datasets/v2/
  manifest.jsonl
  manifest.provisional.jsonl
  provenance.json
  provenance.provisional.json
  manifest.sha256
  labels/
    assistant_confirmed.json
  labeling/
    assistant_confirmed.html

results/v2/
  runs/<run_id>/
    metadata.json
    predictions.jsonl
    summary.json
    usage.json
    checksums.sha256
  comparisons/<comparison_id>/
    comparison.json
    checksums.sha256

graphs/v2/
  png/
  svg/
  pdf/
  data/
  captions/
  figure-manifest.json

reports/archive/
ops/pelican/
tests/
.github/workflows/ci.yml
pyproject.toml
uv.lock
```

### 5.2 Thin entry points

- `run.py` becomes a thin CLI over `benchmark` orchestration.
- `compare.py` becomes a thin CLI over validated comparison logic.
- Adapter-specific behavior remains under `runners/`.
- Dataset construction and validation live under `datasets/v2/` and `benchmark/`.
- Graph generation never imports model adapters or network clients.

### 5.3 Legacy preservation

Existing files remain unchanged as historical evidence:

- `datasets/cases.py`
- `datasets/public_cases.json`
- Every file currently under `results/`
- Existing report content, copied into `reports/archive/` before root reports are updated

A release test records hashes of all legacy result JSONs before implementation and verifies them at closeout.

## 6. Canonical identity and hashing

### 6.1 Canonical JSON

Canonical JSON uses:

- UTF-8 encoding.
- Compact separators.
- Object keys sorted lexicographically.
- Array order preserved.
- Unicode code points preserved without normalization in the hashed representation.
- Rejection of NaN and infinity.
- Explicit schema version.

Question order is represented as an ordered list and therefore affects the hash.

### 6.2 Full manifest digest

The full SHA-256 manifest digest covers:

- Dataset schema version and dataset version.
- Suite ID, case ID, and order index.
- Calibration or evaluation split.
- Exact state value.
- Ordered question objects containing each question ID, type, instructions, criteria, allowed responses, and score range.
- Final reviewed expected answers.
- Source and provenance identifiers.
- Label packet and review-protocol version.

The release stores both per-case digests and one full-manifest digest.

### 6.3 Claim language

The primary Laya-versus-Jev benchmark may claim that both models receive the same ordered logical manifest.

It must not claim byte-identical model-internal prompts unless an adapter can prove the exact rendered and tokenized bytes. Jev records the canonical HTTP body digest. Laya records its package revision, model revision, adapter version, and canonical input digest.

Qwen-PCD is labeled as a secondary constrained-decoding baseline because its adapter converts typed questions into an enum or boolean schema. Router is labeled as a Laya deployment variant. Neither is included in the primary manifest-identity claim.

## 7. Dataset v2

### 7.1 Size and split

Dataset v2 contains 1,190 state cases and 1,550 scored decisions.

A deterministic, stratified 20% calibration split contains 238 cases. The untouched 80% evaluation split contains 952 cases and 1,240 scored decisions.

| Suite | Cases | Decisions | Calibration cases | Evaluation cases | Construction |
|---|---:|---:|---:|---:|---|
| Triage | 60 | 240 | 12 | 48 | 12 states per intent; binary labels balanced 30/30 |
| Guardrails | 60 | 120 | 12 | 48 | 30 positive and 30 negative jailbreak/prompt-injection cases |
| Moderation | 60 | 180 | 12 | 48 | 30/30 toxic; balanced threat and spam coverage with overlap |
| AG News | 200 | 200 | 40 | 160 | 50 cases per topic |
| Emotion | 240 | 240 | 48 | 192 | 40 cases per emotion |
| Banking77 | 120 | 120 | 24 | 96 | 10 cases for each of 12 intents |
| MNLI | 150 | 150 | 30 | 120 | 50 cases per relation |
| SST-5 | 150 | 150 | 30 | 120 | 30 cases per sentiment level |
| Multilingual intent | 150 | 150 | 30 | 120 | 5 languages × 6 intents × 5 cases, including `other` |

### 7.2 Manifest record

Each manifest record contains:

```json
{
  "schema_version": 2,
  "dataset_version": "2.0.0",
  "suite_id": "triage",
  "case_id": "triage-0001",
  "order_index": 0,
  "split": "calibration",
  "state": {},
  "questions": [
    {"qid": "intent", "type": "choice", "instructions": "Classify the request", "criteria": {"refund": "a refund request"}}
  ],
  "expected": {},
  "provenance_id": "source-record",
  "label_provenance": "human-reviewed-ai-assisted-v1"
}
```

Question order is explicit and may not be reconstructed with `sort_keys=True`.

### 7.3 Public-source provenance

Every public case records:

- Canonical dataset.
- Dataset wrapper or mirror.
- Dataset revision or commit.
- Split and original row index.
- Original label.
- Local modification.
- License.
- Citation.
- Output checksum.

Source labels are retained even when human review changes the final benchmark label.

## 8. Human labeling workflow

### 8.1 Assisted packets

A single reviewer receives one self-contained HTML packet.

Packets include:

- Case ID.
- State.
- Ordered questions and allowed responses.
- Neutral labeling instructions.
- The AI draft answer and confidence for each question, pre-filled and editable.
- Approve and re-pass controls, so the reviewer either accepts a draft or overrides it.
- Resumable progress and local draft storage.
- Keyboard navigation.
- JSON export with schema validation.

Packets exclude:

- Existing benchmark labels.
- Source labels.
- Prior model results.
- Model identity claims for the draft answers beyond the recorded assistant model id.

### 8.2 Agreement

Not applicable to this protocol. There is no second reviewer, so inter-annotator agreement, Cohen's kappa, and weighted kappa are not computed and must not be reported for this dataset.

### 8.3 Review record

`provenance.json` records the review facts rather than an adjudication trail:

- `protocol`: `human-reviewed-ai-assisted-v1`.
- `reviewer_count`: 1.
- `independent_human_review`: false.
- `adjudication`: false.
- `assistant_model`: the model that produced the draft.
- `assistant_export_sha256`: digest of the exact reviewed export bytes.
- `entries_applied` and `questions_reviewed`: must both equal the manifest decision count.

The method string states plainly that one human read every case and corrected the AI draft, with no second independent review and no adjudication.

### 8.4 Multilingual review: known limitation

The `multilingual_intent` suite was reviewed under the same single-reviewer protocol. It therefore has no native-speaker review and no second independent intent label, and the suite-level inter-reviewer agreement figures the two-reviewer design called for do not exist for it. Results for that suite must be reported with this caveat attached, and any claim that depends on native-speaker validation is out of scope for this release.

### 8.5 Dataset gates

A dataset is sealed only when:

- Expected suite and class counts pass.
- Every case has final labels.
- Calibration and evaluation splits are disjoint.
- No normalized exact duplicates exist.
- Near-duplicate clusters are documented.
- Every public case has complete provenance and licensing.
- Every multilingual case passes language review.
- The full manifest hash is reproducible.

## 9. Runner and answer contracts

### 9.1 Common answer shape

Choice answers require:

- Exact question ID.
- `type=choice`.
- Legal option label.
- Full normalized probability map.
- Finite confidence in `[0, 1]`.

Noul answers require:

- Exact question ID.
- `type=noul`.
- Finite P(true) in `[0, 1]`.

Score answers require:

- Exact question ID.
- `type=score`.
- Finite score inside the declared range.
- Explicit confidence semantics or `confidence=null` with a recorded reason.

Missing IDs, extra IDs, wrong types, invalid labels, non-finite values, malformed probability maps, and unknown answer types fail the run.

### 9.2 Seed semantics

- Dataset split seed: `42`.
- Open-model inference seed: `42` where supported.
- Jev seed: sent only if the API supports it; otherwise recorded as unsupported.
- Closed-API stochasticity is measured through predeclared repeat runs and is not described as fixed-seed determinism.

### 9.3 Model identities

- Laya: pinned package version, model repository, and immutable checkpoint revision.
- Router: pinned Laya revisions plus Router configuration and preload list.
- Jev: bare pinned `jev-1.13.0`; returned model ID must match the requested ID.
- Qwen-PCD: pinned stock `Qwen/Qwen2.5-1.5B-Instruct` revision and vendored PCD source revision.

### 9.4 Qwen-PCD portability

The Torch backend is implemented and tested on Linux CPU. The Apple MLX backend remains available for local development but is not used for the final pelican run.

The Torch backend must not fabricate confidence on token-collision paths. Invalid constrained generations fail explicitly.

## 10. Remote execution on pelican

### 10.1 Isolation

Open-model runs execute inside a dedicated short-lived container with:

- 4 CPU quota.
- 12 GB memory limit.
- No exposed ports.
- No changes to existing containers.
- No global host package installation.
- `nice 19`.
- Idle I/O priority.
- A dedicated model cache and workspace.
- A cleanup trap that removes only resources created by the job.

### 10.2 Preflight

A run starts only when:

- One-minute load average per available CPU is at most `0.50`.
- Five-minute load average per available CPU is at most `0.75`.
- At least 16 GiB memory is available.
- At least 100 GiB free disk is available.
- The host exposes AVX2 and the dedicated container provides Python 3.12.
- `/home/tejes/sysone-bench-v2/runs/<run_id>` and container name `sysone-bench-<run_id>` are unused.
- The dedicated model cache is `/home/tejes/.cache/sysone-bench-v2`.

The preflight snapshot and post-run snapshot are stored in run metadata.

### 10.3 Jev secret handling

- The user places the key in the local gitignored `.env`.
- A one-shot pelican process receives the key through protected SSH stdin.
- The key is never placed in a command line, log, result, remote file, or graph.
- The process exits after the run.
- Logs are scanned and redacted before publication.

### 10.4 Run protocol

- Laya, Qwen-PCD, Router, and Jev run sequentially.
- Open models run in the isolated pelican container.
- Jev calls originate from pelican for consistent network context.
- Warmups are separated from benchmark and speed-scaling calls.
- Every call is counted by phase.
- The first valid Jev run is the predeclared primary run.
- Two additional full Jev runs measure closed-API stability.
- Laya and Qwen repeat a deterministic 10% smoke sample; answers must match exactly.

## 11. Result storage

Each v2 run is an append-only directory identified by UTC timestamp and UUID.

Required artifacts:

- `metadata.json`
- `predictions.jsonl`
- `summary.json`
- `usage.json`
- `checksums.sha256`

Metadata includes:

- Run ID and UTC timestamp.
- Git revision and dirty-state flag.
- Dataset version and full manifest hash.
- Runner and adapter version.
- Model repository and immutable revision.
- Requested and returned API model.
- Python and dependency lock hash.
- Host, CPU, memory, OS, and runtime limits.
- Seed and sampling settings.
- Question and adapter-payload digests.
- Usage by phase.
- Raw-response digest or redacted raw-response reference.
- Timing protocol and system-load context.

Writers use exclusive creation. An existing run or comparison path is never reopened.

## 12. Comparison rules

A comparison fails before metrics are emitted when any of these differ:

- Dataset version or full manifest hash.
- Case IDs, order, or state digests.
- Question definitions or order.
- Suite set or case counts.
- Evaluation split.
- Metric schema version.
- Model or adapter identity.
- Calibration method.

A valid comparison stores:

- Exact source run IDs and checksums.
- Per-suite and per-question task-specific metrics.
- Paired effects and uncertainty.
- Raw and fitted calibration metrics.
- Risk-coverage curves.
- Timing and usage summaries.
- Complete figure source data.

## 13. Metrics and statistics

### 13.1 Choice

- Exact accuracy.
- Macro-F1.
- Balanced accuracy.
- Per-class recall.
- Brier score.
- Negative log likelihood.
- Raw ECE.
- Temperature-fitted ECE.

### 13.2 Noul

- Accuracy at P(true) >= 0.5.
- Balanced accuracy.
- AUROC.
- AUPRC.
- Brier score.
- Negative log likelihood.
- Raw ECE.
- Fitted ECE.

### 13.3 Score

- MAE and RMSE against the reviewed integer level.
- Nearest-level accuracy using deterministic half-up rounding, `floor(prediction + 0.5)` for nonnegative ordinal scores.
- Within-one accuracy using the same half-up rounded level.
- Exact raw equality rate, reported only as a diagnostic and never labeled ordinary accuracy.
- Spearman correlation.
- Ordinal confusion matrix using half-up rounded levels.

Score outputs are not forced into probability ECE or probability gating. The former within-0.5 accuracy is removed.

### 13.4 Decision correctness and suite accuracy

A decision is correct under one explicit primitive-specific rule:

- Choice: predicted label exactly equals the reviewed label.
- Noul: P(true) >= 0.5 equals the reviewed Boolean.
- Score: half-up rounded prediction equals the reviewed integer level.

Suite accuracy is micro accuracy across the suite's scored decisions. It is always accompanied by primitive-specific metrics in the report and source data. No cross-suite mean is calculated.

### 13.5 Primary paired effect

The primary endpoint is Jev accuracy minus Laya accuracy in percentage points within each suite.

Uncertainty uses 20,000 paired state-cluster bootstrap samples with seed `42`.

- Curated states are resampled as clusters containing all their questions.
- Public one-decision cases are resampled as individual case clusters.
- The same sampled clusters are used for both models in each replicate.
- Reports contain the point estimate and 95% percentile interval.
- Two-sided p-values use 20,000 paired state-cluster permutation tests.
- Holm adjustment covers the nine primary suite tests and remains in the machine-readable comparison.

No overall composite score is calculated.

### 13.6 Calibration

- Choice probabilities use temperature scaling fitted only on the calibration split.
- Noul probabilities use logit temperature or Platt scaling fitted only on the calibration split.
- Raw and fitted values are stored and displayed separately.
- Evaluation metrics use only the evaluation split.
- Score calibration remains ordinal and is not mixed with probability ECE.

### 13.7 Gating

- Thresholding uses full-precision probabilities.
- Choice gating uses the top-label probability, not vendor-specific confidence semantics.
- Noul gating uses P(true).
- Raw and fitted gating curves remain separate.
- Choice and Noul gating is reported separately.
- Answer-level and state-level coverage are both reported.
- Score questions are excluded from probability risk-coverage plots.
- Missing probabilities fail the run rather than becoming zero.

### 13.8 Latency and resource use

The shared pelican host makes timing a controlled diagnostic, not a production hardware claim.

Record:

- Warmup calls.
- Repetitions.
- p50 and p95.
- Throughput.
- CPU seconds.
- Peak process memory where available.
- Host load before and after.
- CPU quota and priority.
- Local versus API execution path.

Missing cost is never plotted as zero.

## 14. Benchmark graph release

### 14.1 Style contract

- White 16:9 canvases.
- Python 3.12 with matplotlib, NumPy, and pandas as the deterministic plotting stack.
- PNG at 3200×1800, sRGB.
- Matching SVG and PDF.
- Inter Regular and SemiBold bundled from a pinned official release with checksum and OFL license.
- Colorblind-accessible palette:
  - Laya: `#009E73`
  - Jev: `#0072B2`
  - Qwen-PCD: `#E69F00`
  - Router-only: `#CC79A7`
- Primary text: `#111827`
- Secondary text: `#6B7280`
- Grid: `#E5E7EB`
- Consistent model order and shared legend.
- Direct labels on marks.
- Accuracy axes fixed at 0-100.
- No logos, gradients, 3D, decorative backgrounds, or radar charts.

### 14.2 Image suite

1. `01-accuracy-by-suite`
   - 3×3 small multiples.
   - One suite per panel.
   - Laya, Jev, and Qwen-PCD bars.
2. `02-paired-effects`
   - 3×3 Jev-minus-Laya panels.
   - 95% state-cluster intervals.
3. `03-calibration`
   - 2×2 raw and fitted reliability panels.
   - Choice and Noul separated.
4. `04-risk-coverage`
   - 2×2 answer-level and state-level curves.
5. `05-efficiency`
   - 2×2 p50, p95, throughput, and measured resource use.
6. `06-multilingual`
   - Five equal language panels.
   - Laya, Jev, and Qwen-PCD series.
7. `07-router-deployment`
   - English parity.
   - Script-routed gain.
   - Route counts and reasons.

Every image includes:

- Benchmark or metric title.
- Metric definition.
- Evaluation n.
- Uncertainty method.
- Run IDs.
- Dataset version.
- Full manifest short hash.
- Caption-ready alt text.

### 14.3 Graph artifacts

For every image:

- PNG.
- SVG.
- PDF.
- Plotting CSV.
- Figure specification JSON.
- Caption Markdown.
- Source run IDs and checksums.

The complete graph set is described by `graphs/v2/figure-manifest.json`.

Graph generation is deterministic. Timestamps, random state, SVG hash salt, fonts, and metadata are fixed.

## 15. Documentation release

### 15.1 README

The root README becomes the current release entry point:

- One-scope statement without unsupported byte-identical or fixed-seed claims.
- Qwen-style benchmark images.
- Current results.
- Reproduction commands.
- Dataset and labeling summary.
- Model and hardware context.
- Artifact index.
- Limitations and citation.

### 15.2 REPORT.md

The canonical report contains:

- Benchmark scope.
- Dataset construction and review.
- Model identities and run protocol.
- Paired statistics.
- Calibration and gating.
- Latency and resource use.
- Provenance and checksums.
- Limitations.
- Links to immutable artifacts.

### 15.3 FEEDBACK_REPORT.md

The feedback report is regenerated from final validated results and contains concise model-specific findings. It does not mix historical run values.

### 15.4 Archive

Existing report content is preserved under `reports/archive/` and labeled historical before root reports are replaced.

### 15.5 Third-party notices

Add:

- Vendored PCD upstream source and revision.
- PCD modifications.
- Inter font license.
- Dataset citations and licenses.
- Model references.
- No-secret and data-egress notes.

## 16. Testing and quality gates

### 16.1 Automated tests

- Canonical JSON and Unicode behavior.
- Per-case and full-manifest hash reproducibility.
- Dataset counts, balance, split isolation, labels, and provenance.
- Exact and normalized duplicate scans.
- Answer-contract validation.
- Jev environment loading, URL validation, retries, and redaction.
- Laya, Router, and Qwen metadata identity.
- Torch PCD token mapping and malformed-generation handling.
- Metric formulas against golden fixtures.
- Bootstrap determinism and paired resampling.
- Calibration split isolation.
- Full-precision gating.
- Append-only and no-clobber writes.
- Comparison mismatch rejection.
- Usage accounting across all phases.
- Graph data equality, deterministic output, dimensions, and unclipped labels.

### 16.2 Static quality

- Ruff.
- mypy.
- pytest.
- `uv sync --frozen` from the checked-in `uv.lock`.
- Python 3.11 and 3.12 CI.
- Secret scan.
- JSON and Markdown link checks.

### 16.3 Release audit

The release is blocked unless:

- Legacy result hashes are unchanged.
- Dataset gates pass.
- All run artifacts validate.
- All comparisons validate.
- All tests, lint, and type checks pass.
- No secret appears in tracked or generated files.
- Graph source data exactly matches validated comparison data.
- Report values exactly match generated summaries.
- Pelican job cleanup leaves existing workloads unchanged.

## 17. DOX updates

Implementation must update the DOX hierarchy for new durable boundaries:

- Add `benchmark/AGENTS.md`.
- Add `graphs/AGENTS.md`.
- Add `reports/AGENTS.md`.
- Add `ops/AGENTS.md`.
- Add `tests/AGENTS.md` when the test framework exists.
- Update root Child DOX Index.
- Reconcile `datasets/AGENTS.md` with dataset v2 and the two-reviewer contract.
- Reconcile `runners/AGENTS.md` with all four adapters and strict answer validation.
- Reconcile `results/AGENTS.md` with append-only v2 run directories.
- Remove stale statements that results are empty or datasets are single-file.

Record the stable user preference that future visual brainstorming sites use dark mode.

## 18. Execution blockers

Two inputs block final model runs:

1. A completed single-reviewer `human-reviewed-ai-assisted-v1` label export.
2. A valid Jev API key supplied transiently through stdin.

Open-model implementation, dataset preparation, tests, and graph development may proceed before those inputs are available. No final comparison or release claim may be published before both are resolved.

## 19. Acceptance criteria

The work is complete when:

1. Dataset v2 contains 1,190 sealed cases and 1,550 decisions.
2. Calibration and evaluation splits contain 238 and 952 cases.
3. The review record in `provenance.json` states the protocol, reviewer count, and absence of adjudication, and `manifest.sha256` holds the raw manifest-file SHA-256.
4. The full manifest hash reproduces from a clean environment.
5. Laya, Jev, Qwen-PCD, and Router produce validated v2 records.
6. Every raw prediction, model identity, environment fact, and usage record is retained.
7. Comparisons fail closed on any identity mismatch.
8. Paired metrics and calibration are generated without a composite score.
9. Seven Qwen-style image sheets and all vector/source artifacts pass graph tests.
10. README, REPORT, FEEDBACK_REPORT, MACHINES, and third-party notices reflect only v2 facts.
11. All legacy results remain byte-for-byte unchanged.
12. Ruff, mypy, pytest, CI, secret scan, and release audit pass.
13. Pelican cleanup is verified without modifying existing containers or workloads.
14. No commit, push, or external publication occurs without explicit user instruction.

## 20. Approved design outcome

Use versioned hardening in the existing repository. Preserve history, build a strict v2 data and result layer, run only isolated jobs on pelican, and publish a Qwen 3.5-style benchmark release grounded in paired statistics and complete provenance.
