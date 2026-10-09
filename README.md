# sysone-bench

An independent head-to-head benchmark of System One decision models. Laya (open weights),
Jev (closed TypeSafe API) and Qwen with parallel constrained decoding (PCD) answer the
same 1,190 cases and 1,550 typed questions in one run, from one sealed manifest, at one
seed. Inputs are verified byte identical before any number is compared, so the models are
graded on the same bytes.

There was no way to check cross-vendor claims before this. Laya's author had no Jev API
access, and the published vendor numbers come from different prompts. Two numbers from
different prompts are not a comparison.

## Expanded panel: 49 of 50 measured

The three-model result below is the sealed v2.0.0 release. Since then the scope was widened to the
Decision Index 0.2.1 panel, and **49 distinct measurements** are complete against the same sealed
manifest, re-scored onto the corrected dataset 2.1.0 gold.

Counts are derived by `ops/panel_coverage.py` from finished run directories, never from a
hand-kept tally. It verifies every artifact against its own `checksums.sha256` and fingerprints
the evaluation-phase answers, so repeat runs and identical runs collapse instead of inflating the
number. It found 52 verified directories but only **49** distinct measurements:

| Runner | Directories | Counted once because |
| :--- | :--- | :--- |
| `decider-2b` | `decider-2b-t4-c0`, `decider-2b-t4-d0` | one retried run, both 0.7895 |
| `tev1-08b` | `tev1-08b-20261002`, `tev1-08b-t4-g1` | the CPU fp32 and T4 fp16 cross-host pair, 0.7629 and 0.7734 |
| `mini-jev`, `openvons` | two separate runners | byte-identical on all 1240 evaluation decisions |

### Reference: the closed API

| model | entry kind | accuracy | notes |
|---|---|---:|---|
| **`Jev 1.13.0`** | closed API | **0.9065** | the reference every open model below is measured against; not a panel entry |

Jev is the vendor's hosted model behind `api.typesafe.ai`, so it is not one of the 50 panel
entries — the panel is open-weights only. It is listed first because it is the number to beat.
Its 0.9065 comes from the sealed v2.0.0 release below, measured on the same manifest digest and
1,240 evaluation decisions as every row in the table underneath.

### Strongest open-weights results

| model | params | entry kind | accuracy | notes |
|---|---:|---|---:|---|
| `kev-4b` | 4.66B | LoRA + head | **0.8556** | best open model; runs on 1x T4 |
| `jpt-9b` | 9.65B | LoRA | 0.8548 | needed a 2-GPU shard, 18 GiB |
| `jet` | 4.66B | LoRA | 0.8524 | letter-logit readout over a scalar scorer |
| `jevk5` | 4.66B | LoRA | 0.8508 | letter-logit readout |
| `intern-decision-4b` | 4.66B | full fine-tune | 0.8460 | remote code manages its own precision |
| `tev1-4b` | 4.66B | full fine-tune | 0.8460 | see the `tev1-08b` CPU/T4 pair below |
| `decider-4b` | 4.66B | full fine-tune | 0.8411 | eager path, avoids T4 compile stalls |
| `neohorse-4b` | 4.66B | head / adapter | 0.8266 | reproduced exactly on a second run |
| `decision-nox` | 4.66B | head / adapter | 0.8048 | vendor-managed precision |
| `hopper-g` | 4.66B | LoRA | 0.7911 | matches the vendor's self-reported board score |

The best open model trails the closed reference by 0.0509. The gap to the top three open models
is 0.0048, so those three are effectively tied at this sample size.

Every row is bf16 on a T4 except `intern-decision-4b`, whose remote code selects its own numerics.

`tev1-08b` is the only model measured on two hosts, which makes it the one direct read on precision
cost: **0.7629 on CPU fp32 against 0.7734 on T4 fp16**, a 0.0105 gap on identical bytes and one
pinned revision. `decider-2b` differed 0.7895 on Kaggle against 0.7927 on Colab, but those runs were
not byte-identical in configuration, so it is a weaker signal.

Full 49-row table with scoring method and provenance flags:
[`docs/decision-index-panel-coverage.md`](docs/decision-index-panel-coverage.md).

Two caveats carried in the table rather than hidden:

- Rows marked **technique reimpl.** are our reading of a published method over a public base
  checkpoint, not the authors' code. `mini-jev` and `openvons` share a base and agree on every
  decision, which is what an insensitive prompt looks like.
- `pngwn` scored 0.2734 but its vendor never published the prompt format its scorer was trained on.
  Treat it as unresolved rather than as a measurement of the model.

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

<!-- Figures are withheld from the README while the expanded panel is measured. Numbers only for
     now. The rendered charts and their source rows stay in
     results/v2/report-20260926/figures/, regenerated from validated rows by
     benchmark/graphics.py; nothing there is hand-drawn. -->

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

## Thanks

Special thanks to [MrDragonFox](https://huggingface.co/MrDragonFox) for lending us a DGX Spark,
on which the larger models in the panel were benchmarked.
