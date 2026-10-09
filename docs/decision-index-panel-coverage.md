# Decision Index 0.2.1 Panel Coverage

Scope record for the expanded benchmark: which panel models are measured, which are out of scope,
and why the unreachable ones are unreachable.

Source of truth is the panel extract of `multimodalart/jev-decision-index` at suite 0.2.1
(70 models). Every row here was checked against that extract, not inferred from it.

## Scope decision

| Tier | Count | Disposition |
| :--- | ---: | :--- |
| Decision Index 0.2.1 panel | 70 | source list |
| Dropped - exceeds 2x T4 (29.12 GiB) | 9 | 48-67 GiB bf16; see `panel-hardware-requirements.md` |
| Dropped - will not fit 2x T4 in fp16 | 1 | out of scope, precision confound |
| Dropped - weights never published | 2 | out of scope, unreproducible |
| **In scope for 2x T4** | **49** | target (was 51; `akash-gemma` and `semif` removed, no public weights) |
| Added by author request | 1 | `seb-9b` (`ironbcc/seb-9b`), 9.65B, pending measurement |

The 9 dropped models are every entry at or above 25.8B served parameters with public weights. None fits 2x T4 in fp16
(29.1 GiB usable), and the operator declined PRO 6000 spend. Seven further names had no public weights anywhere and were removed from scope entirely.

`winnow-12b` is dropped for a different reason. Its fp16 weights are 34.2 GiB against 29.1 GiB
usable, so it cannot load on 2x T4 at the precision every other row uses. An 8-bit load would fit at
roughly 17 GiB, but it would then be the only quantized row in the comparison, which trades a
coverage gap for a precision confound. A controlled benchmark should take the gap.

Its sibling `winnow-e4b` stays in scope at 22.4 GiB.

## Coverage

Derived by `ops/panel_coverage.py` from finished run directories, not from a hand-kept tally.
Run every artifact through `sha256sum -c` before counting, and fingerprint the evaluation-phase
answers so identical runs collapse.

| Measure | Count |
| :--- | ---: |
| Run directories verified | 46 |
| Distinct runner names | 44 |
| **Distinct measurements** | **49 of 50** |
| Not measured | 8 |

Three directories are correctly counted once each:

| Runner | Directories | Why collapsed |
| :--- | :--- | :--- |
| `decider-2b` | `decider-2b-t4-c0`, `decider-2b-t4-d0` | one retried run, both 0.7895 |
| `tev1-08b` | `tev1-08b-20261002`, `tev1-08b-t4-g1` | the CPU fp32 and T4 fp16 cross-host pair, 0.7629 and 0.7734 |
| `mini-jev` and `openvons` | two runners | byte-identical on all 1240 evaluation decisions; see below |

### The `mini-jev` and `openvons` collision

Both are `kind: "inference technique"` over the same stock checkpoint,
`Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`, differing
only in prompt wording. Their evaluation-phase answers hash identically
(`952:544fc02392c51e9070d6f4945333125d`): **all 1240 decisions agree, including the full
probability payloads.** Changing the prompt moved nothing, which means the option-logprob ranking is
insensitive to it.

They are therefore one measurement, and the panel would be overstated by one model if both were
counted. Both carry `technique_reimplementation: true` and `vendor_code_executed: false`, so neither
row is a measurement of the authors' system.

### Results

| # | Model | Accuracy | Provenance |
|---:| :--- | ---: | :--- |
| 1 | `kev-4b` | 0.8556 | vendor readout |
| 2 | `jpt-9b` | 0.8548 | vendor readout |
| 3 | `jet` | 0.8524 | vendor readout |
| 4 | `jevk5` | 0.8508 | vendor readout |
| 5 | `intern-decision-4b` | 0.8460 | vendor readout |
| 6 | `tev1-4b` | 0.8460 | vendor readout |
| 7 | `decider-4b` | 0.8411 | vendor readout |
| 8 | `neohorse-4b` | 0.8266 | vendor readout |
| 9 | `decision-nox` | 0.8048 | vendor readout |
| 10 | `this-that-12` | 0.8008 | vendor readout |
| 11 | `hopper-g` | 0.7911 | vendor readout |
| 12 | `decider-2b` | 0.7895 | vendor readout |
| 13 | `decision-sol` | 0.7863 | vendor readout |
| 14 | `mini-jev` | 0.7831 | technique reimpl. |
| 15 | `nimble-v2` | 0.7823 | 2-GPU shard |
| 16 | `intern-decision-2b` | 0.7790 | vendor readout |
| 17 | `tev1-08b` | 0.7734 | vendor readout |
| 18 | `kev-08b` | 0.7726 | vendor readout |
| 19 | `metask` | 0.7710 | vendor readout |
| 20 | `jpt-08b` | 0.7702 | vendor readout |
| 21 | `decision-eos` | 0.7685 | vendor readout |
| 22 | `bosun-17b` | 0.7315 | vendor readout |
| 23 | `intern-decision-08b` | 0.7113 | vendor readout |
| 24 | `decision-kai` | 0.7105 | vendor readout |
| 25 | `gliner-decide` | 0.7065 | vendor readout |
| 26 | `lavoir` | 0.7040 | vendor readout |
| 27 | `bosun-06b` | 0.6976 | vendor readout |
| 28 | `laya` | 0.6863 | vendor readout |
| 29 | `gliner-base` | 0.6363 | vendor readout |
| 30 | `gliner-multi` | 0.5863 | vendor readout |
| 31 | `jobe` | 0.5855 | technique reimpl. |
| 32 | `decision-lex` | 0.5823 | vendor readout |
| 33 | `lev` | 0.5613 | vendor readout |
| 34 | `verdict` | 0.5532 | vendor readout |
| 35 | `gliner-small` | 0.5419 | vendor readout |
| 36 | `mojev` | 0.5395 | vendor readout |
| 37 | `julia-1` | 0.5016 | vendor readout |
| 38 | `lumma-fev-06b` | 0.4815 | vendor readout |
| 39 | `harsha` | 0.4339 | technique reimpl. |
| 40 | `lumma-fev-01b` | 0.4194 | vendor readout |
| 41 | `lfm2600` | 0.3855 | vendor readout |
| 42 | `lfm350` | 0.3669 | vendor readout |
| 43 | `pngwn` | 0.2734 | prompt unresolved |


## The two models whose weights were never published

Both are dropped from scope after a final re-verification of the author accounts and repositories.
Neither is a compute limit; neither can be reproduced by any party.

`akash-gemma` (0.27B, `kind: LoRA + head`, `google/gemma-3-270m` base).

- Panel `weights_repo` is null.
- The author account `akash-kamat` publishes **zero** models on Hugging Face.
- A name search returns only unrelated third-party repos, not the author's work.
- The code is public at `github.com/akash-kamat/system-one-gemma`.

`semif` (4.66B, SemIf protocol over `Qwen3.5-4B-Base`).

- `huggingface.co/TheoLeeCJ/SemIf-OpenJev` returns **404**.
- `huggingface.co/TheoLeeCJ/openjev` returns **404**.
- The author account `TheoLeeCJ` publishes **zero** models on Hugging Face.
- The repository was observed earlier in this work and has since been deleted or made private.

Code without published weights, and for `semif` not even reachable code. Neither can be independently
reproduced, which is a finding about the panel rather than a gap this benchmark can close.

## Correction: inference techniques are not missing weights

The panel distinguishes `kind: "model"` from `kind: "inference technique"`. A technique entry has
`served_checkpoint` pointing at a **stock base model** and its own repository under
`code_url`. There is nothing to download because nothing was trained; the entry is a prompting or
serving method applied to someone else's checkpoint.

Six remaining panel models are techniques, and all six have live, public, permissively licensed code:

| Panel model | Code | Base checkpoint |
| :--- | :--- | :--- |
| `jeff-uncapped` | `github.com/logan-markewich/jeff` (MIT) | `knowledgator/gliformer-large-v1` |
| `mini-jev` | `github.com/r-ms/mini-jev` (MIT) | `Qwen/Qwen3-4B-Instruct-2507` |
| `openvons` | `github.com/genai-craft/openvons` | `Qwen/Qwen3-4B-Instruct-2507` |
| `jobe` | `github.com/MantisShrimpdev/jobe` (MIT) | `Qwen3.5-4B` |

`semif` is dropped from scope with the two models above: its repository and author account are gone.

**`harsha` is not a technique entry.** A final re-verification found
`huggingface.co/harshatheg/Qwen-2.5-1B-RLCD` returns **200**, and the `harshatheg` account publishes
two public models. It is a real published fine-tuned checkpoint over `Qwen/Qwen2.5-1.5B`, so it is
adapterable like any other fine-tune. An earlier revision of this document recorded it as shipping no
weight file at all; that was wrong and is corrected here.

Reading a null `weights_repo` as "no weights" conflates these two kinds and understates the
reachable set.

## Fit caveats inside the in-scope tier

- `winnow-e4b` (22.4 GiB) and `jev-omni` (22.3 GiB) fit in fp16 but with only about 7 GiB spare
  across 2x T4. Either will fail rather than degrade if a sequence runs long.
- Several in-scope entries are LoRA-only and need a base checkpoint fetched alongside them
  (`kev`, `pngwn`, `lev`, `clm-v0.1-8b`, `nimble-v2`). Disk cost is trivial; the dependency is not.

## Models that cannot run on 2x T4 at all

Three further panel models are in no position to be measured on this hardware. They are not counted
as unreachable, because their weights exist and are public; they are counted as hardware-infeasible
and are excluded from the 51-model target for the same reason `winnow-12b` is.

| Model | Served params | Reason |
| :--- | ---: | :--- |
| `kev-9b` | 8.9B | bf16 weights about 18 GiB against 29.1 GiB usable across 2x T4, and about 36 GiB of host RAM to stage, against 31 GiB present |
| `decision-lux-9b` | 9B | same |
| `winnow-12b` | 12B | 34.2 GiB bf16, exceeds one card's 15.4 GiB outright |

`kev-9b` and `decision-lux-9b` were each attempted once and both failed on load *on the 2x T4 hardware*. Both have since been measured on a GB10 Spark (117 GB unified): `decision-lux-9b` 0.8484, `kev-9b` 0.8290. A RAM staging gate
and a VRAM denylist now refuse them automatically so the failure is not repeated.

## Verification

- Panel size and per-model records come from the 0.2.1 extract; counts in this document are derived
  from it by weight-repo membership, not by name similarity.
- Every "unreachable" claim requires that the author account publishes no models and that a
  repository-name search surfaces nothing first-party.