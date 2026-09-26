# sysone-bench

An independent head-to-head benchmark of System One decision models. Laya (open weights),
Jev (closed TypeSafe API) and Qwen with parallel constrained decoding (PCD) answer the
same 1,190 cases and 1,550 typed questions in one run, from one sealed manifest, at one
seed. Inputs are verified byte identical before any number is compared, so the models are
graded on the same bytes.

There was no way to check cross-vendor claims before this. Laya's author had no Jev API
access, and the published vendor numbers come from different prompts. Two numbers from
different prompts are not a comparison.

## Result

Evaluation split, 952 cases and 1,240 scored decisions. Raw manifest bytes
`a938cc2483a592dc84e0d5baac12594491bcaa5b4ceb6b7c3b0def71b36297bd`, logical digest
`4272a7a25ebcb324235696ad808544e157a31e3f9c04421da731a08dfb9d5768`.

| model | accuracy | what it is |
|---|---|---|
| Jev 1.13.0 | 0.9065 | closed API at `api.typesafe.ai` |
| Laya 0.3.11 | 0.6863 | open weights, commit `55cf4c4e`, CPU |
| Qwen2.5-1.5B PCD | 0.6048 | open weights, revision `989aa798`, CPU |

Jev leads Laya on all nine suites and on 8 of 9 the paired difference survives Holm
correction. Triage is the exception: +0.057 with a permutation p of 0.0627, so we do not
claim it. Per-suite numbers, paired intervals and figures are in
[the report](results/v2/report-20260926/REPORT.md).

![Evaluation accuracy by suite](results/v2/report-20260926/figures/suite-accuracy.png)

## How the ground truth was made

This release says plainly what it did and what it did not do.

One human reviewer read all 1,190 cases and corrected an AI draft of every answer, the draft
having been produced by the assistant model `opencode/mimo-v2.6-flash-free`. The protocol id
is `human-reviewed-ai-assisted-v1`. There was no second independent reviewer and no
adjudication, so this dataset has no inter-annotator agreement, no kappa and no adjudication
artifact, and the two-reviewer seal path in `datasets/v2/validate.py` was never exercised.
`datasets/v2/provenance.json` records all of it, including `"independent_human_review": false`
and `"adjudication": false`.

The `multilingual_intent` suite carries the sharpest version of that caveat, with no
native-speaker review and no second label. Read its per-language numbers in
[the report](results/v2/report-20260926/REPORT.md) as one reviewer's judgment rather than a
consensus.

## Fairness rules

- One sealed manifest, one seed (42), the same question dicts and the same order for every
  model. No per-model prompt tuning.
- Pinned model identities recorded in every run file, including the Laya commit and the
  Qwen revision.
- Manifest checksum verified before a run starts and again inside the container.
- Raw and calibration-fitted ECE reported separately and never mixed.
- Run artifacts are append-only. A published run is never overwritten.
- Vendor output is projected onto the answer contract at the adapter edge, never by
  loosening the contract.

## Reproduce

Model runs need the sealed manifest and its checksum, which are in this repo. Open models
run CPU only in a container, pinned to 4 CPUs and 12 GB. The launcher derives its paths from
the checkout, so it needs no configuration:

```bash
ops/remote/run_open_model.sh --run-id <run-id> --model laya   # or qwen
```

Jev needs a key, and the key never touches disk. It arrives on stdin and lives only in the
child process environment. The destination is deployment configuration, so point the sender
at your own host and checkout:

```bash
export SYSONE_BENCH_SSH_HOST="user@host"
export SYSONE_BENCH_REMOTE_ROOT="/path/to/sysone-bench"
printf '%s' "$TYPESAFE_API_KEY" | .venv/bin/python ops/remote/remote_jev.py
```

Optional overrides for the open-model launcher: `SYSONE_BENCH_WORKSPACE_ROOT`,
`SYSONE_BENCH_MODEL_CACHE` and `SYSONE_BENCH_IMAGE`.

Comparisons and figures, from finished run directories:

```bash
./.venv/bin/python compare.py <run-a> <run-b>          # pairwise, with bootstrap and tests
./.venv/bin/python -m benchmark.report \
  --laya <laya-run-dir> --jev <jev-run-dir> --qwen <qwen-run-dir> \
  --manifest datasets/v2/manifest.jsonl --output-root <new-report-dir>
./.venv/bin/python -m benchmark.graphics \
  --figure-source <new-report-dir>/figures/figure-source.json \
  --output-root <new-report-dir>/figures
```

`benchmark.report` refuses to emit anything unless its recomputed counts and per-suite
accuracies reproduce each run's own sealed `summary.json`. `benchmark.graphics` reads only
`figure-source.json`, so a chart cannot disagree with the rows that were validated. Both
refuse to write into an existing directory, and two runs over the same runs produce
byte-identical output, figures included.

## Layout

- `datasets/v2/manifest.jsonl`, `manifest.sha256`, `provenance.json` - the sealed dataset
  and its review record. `*.provisional.*` are the preserved pre-seal originals.
- `datasets/v2/labeling.py` - packet generator for either review protocol.
- `datasets/v2/assistant.py` - local review UI with approve and override.
- `datasets/v2/assisted_labels.py` - the single-review merge and seal.
- `runners/` - one adapter per model, all behind `BaseRunner`.
- `benchmark/orchestrator.py` - sealed-manifest execution and v2 artifacts.
- `benchmark/metrics.py`, `benchmark/statistics.py` - per-primitive metrics, paired
  clustered bootstrap, permutation tests, Holm correction.
- `benchmark/report.py`, `benchmark/figure_data.py`, `benchmark/graphics.py` - report
  assembly and figure rendering.
- `ops/remote/` - capacity preflight, container image, launchers, Jev credential path.
- `results/` - append-only run outputs, comparisons and the published report.

## License

MIT, see `LICENSE`. Model weights and public datasets keep their own terms: Laya is
Apache-2.0, and the public suites follow their upstream licenses recorded in
`datasets/v2/sources.lock.json`.

```bibtex
@misc{sysonebench2026,
  title  = {sysone-bench: independent head-to-head benchmark of System One decision models},
  year   = {2026},
  author = {instax-dutta},
  url    = {https://github.com/instax-dutta/sysone-bench}
}
```
