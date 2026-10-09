# Panel hardware requirements

Every figure is **weights only**, from the Decision Index 0.2.1 extract's `served_params`.
Multiply by **1.15** for a working estimate that leaves room for activations, KV cache and
allocator fragmentation. Reference hardware measured in this project:

| Device | Usable | Notes |
| --- | ---: | --- |
| Kaggle T4 (1x) | 14.56 GiB | bf16 only, no native bf16 |
| Kaggle T4 (2x) | 29.12 GiB | PCIe split; ~7x slower per token than 1 card |
| RTX PRO 6000 | ~96 GiB | the tier the 16 largest entries need |

## Why LoRA entries are listed at full base size

A LoRA or head adapter does not shrink the resident model. `kev-9b` is a 9.65B adapter over
`Qwen3.5-9B`; the base must be resident in full, so its footprint is 18.0 GiB in bf16 regardless of
how small the adapter file is. Ranking by `trained_bytes` instead of `served_params` would
badly understate these.

## Banding

### Fits 1x T4 (under 14.56 GiB) — 44 of 51 in scope

| Model | Served | bf16 | fp32 |
| --- | ---: | ---: | ---: |
| `gliner-small` | 0.07B | 0.1 | 0.3 |
| `julia-1` | 0.14B | 0.3 | 0.5 |
| `verdict` | 0.15B | 0.3 | 0.6 |
| `lumma-fev-0.1b` | 0.15B | 0.3 | 0.6 |
| `gliner-base` | 0.19B | 0.4 | 0.7 |
| `akash-gemma` | 0.27B | 0.5 | 1.0 |
| `gliner-multi` | 0.29B | 0.5 | 1.1 |
| `decision-kai`, `decision-lex` | 0.31B | 0.6 | 1.1 |
| `lfm350` | 0.35B | 0.7 | 1.3 |
| `lavoir` | 0.40B | 0.7 | 1.5 |
| `laya` | 0.42B | 0.8 | 1.6 |
| `gliner-decide` | 0.49B | 0.9 | 1.8 |
| `jeff` | 0.58B | 1.1 | 2.1 |
| `bosun-v3.1-0.6b` | 0.60B | 1.1 | 2.2 |
| `lumma-fev-0.6b` | 0.65B | 1.2 | 2.4 |
| `decision-eos-0.8b`, `kev-0.8b`, `mojev`, `tev1-0.8b`, `intern-decision-0.8b`, `jpt-0.8b` | 0.87B | 1.6 | 3.3 |
| `harsha` | 1.54B | 2.9 | 5.8 |
| `bosun-v3.1-1.7b` | 1.72B | 3.2 | 6.4 |
| `this-that-1.2` | 1.88B | 3.5 | 7.0 |
| `decider-2b`, `decision-sol`, `intern-decision-2b` | 2.27B | 4.2 | 8.5 |
| `lfm2600` | 2.70B | 5.0 | 10.0 |
| `mini-jev`, `openvons` | 4.02B | 7.5 | 15.0 |
| the 4.66B band: `decider-4b`, `decision-nox`, `hopper-g`, `jevk5`, `jobe`, `kev-4b`, `metask`, `pngwn`, `semif`, `tev1-4b`, `lev`, `intern-decision-4b`, `neohorse`, `jpt-4b`, `jet` | 4.66B | 8.7 | 17.4 |

### Needs 2x T4 (14.56-29.12 GiB) — reachable by sharding

Proven for `nimble-v2`: 18.0 GiB bf16 ran 10.4 GB + 9.2 GB across two T4s, 2191 s, with
`low_cpu_mem_usage=True` and `device_map="auto"`, no precision confound.

| Model | Served | bf16 | fp32 | Extra blocker beyond VRAM |
| --- | ---: | ---: | ---: | --- |
| `winnow-e4b` | 8.00B | 14.9 | 29.8 | **GGUF only**; needs `llama.cpp` |
| `clm-v0.1-8b` | 8.19B | 15.3 | 30.5 | `.pt` pickle; contrastive **reranker**, different task shape |
| `decision-lux-9b` | 9.65B | 18.0 | 36.0 | **measured**, 0.8484 (GB10 Spark, fits whole-device) |
| `kev-9b` | 9.65B | 18.0 | 36.0 | **measured**, 0.8290 (GB10 Spark, vendor server self-loads on one device) |
| `nimble-v2` | 9.65B | 18.0 | 36.0 | **measured**, 0.7823 |
| `jpt-9b` | 9.65B | 18.0 | 36.0 | measured, 0.8548 |
| `winnow-12b` | 11.96B | 22.3 | 44.6 | **GGUF only**; 22.3 GiB fits a GB10 whole, runtime still required |
| `jev-omni` | 11.96B | 22.3 | 44.6 | Gemma-4-12B LoRA; needs the shard path exercised |

### Needs more than 2x T4 — the 9 out-of-scope entries with public weights

Grouped by requirement. The `need N` column is bf16 x 1.15, the working estimate.
Seven further names from the original 16 (`joshua-diffusion`, `razorback-one-read`,
`vllm-pr57250`, `jevfire-uncapped`, `reflex-27b-v2`, `solomon-v11`, `xor`) have no
public weights on HF or GitHub and were removed from scope entirely rather than
carried as exclusions.

| Requirement | Models | Served | bf16 | Approx GPU |
| ---: | --- | ---: | ---: | --- |
| ~55 GiB | `djev`, `rune-26b-a4b-v3` | 25.8B | 48.1 | RTX 6000 Ada 48 GB, or A6000 with offload |
| ~60 GiB | `autojev-27b`, `decider-chat-qwen3.6-27b`, `simple-jev-qwen3.8-27b`, `jebadiah-27b`, `eikos-27b-fp8` | 27.8B | 51.7 | RTX PRO 6000 96 GB, or H100 80 GB |
| ~70 GiB | `decider-chat-gemma4-31b` | 32.7B | 60.9 | RTX PRO 6000 96 GB |
| ~77 GiB | `decider-35b-nvfp4` | 35.9B | 67.0 | RTX PRO 6000 96 GB |

At fp32 every entry in this last band doubles and needs 96-134 GiB, which puts them beyond a
single 96 GB card.

## Two measurement caveats a contributor should not inherit

- **`decision-lux-9b` is not a VRAM problem.** Its fp32 encoders need ~36 GiB of *host* RAM to
  stage, against 31 GiB on the Kaggle box. More VRAM alone will not fix it; it needs more host RAM
  or a vendor-side fp16 path.
- **`kev-9b` is not a VRAM problem either.** `kev.serve` is a vendored HTTP server that performs its
  own load, so accelerate's `device_map` sharding never engages. Upstream support is required.

## Installing the vendor SDKs

Adapters import their SDKs lazily, so `uv sync` alone leaves the suite unable to import most of
them. Install the extra:

```
uv sync --extra vendor
```

That covers `gliner2`, `gliclass`, `thisthat`, `llm2jev` and `peft`, all resolvable on PyPI.
Two adapters need more than a wheel:

- **`lavoir`** has no PyPI package under any obvious name. The `lavoir` adapter, its score of 0.7040,
  and `ops/remote/vendor_check.py` were produced against an SDK installed from its vendor source on
  the run host. A fresh checkout must install it the same way, or skip that one adapter.
- **`flash-linear-attention`** is needed by the Qwen3.5-based adapters (`jpt`, `jet`, `decider`,
  `decision1`, `neohorse`, `nimble`, `metask`). Without it those models fall back to a reference
  PyTorch path, which is correct but much slower. `nimble-v2` warns about this on load.

The remaining SDK-backed rows need no extra: `kev` (vendored HTTP server), `intern-decision`,
`tev1`, `lumma-fev`, `mojev`, `julia` and `bosun` all run through `transformers` with
`trust_remote_code`.

## Contributing a measurement

A run is acceptable if it records, per row:

1. `runner`, `model`, and the **full 40-character** `revision`
2. `base_model` and `base_revision` for any LoRA, head, or technique entry
3. `device`, `dtype`, and `serving`
4. `scoring` — the readout used, and whether it is the vendor's own path or a reimplementation
5. `checksums.sha256` over `metadata.json`, `predictions.jsonl`, `summary.json`, `usage.json`,
   `gpu_provenance.json`, `.sysone-owner`
6. 1190 prediction rows against manifest digest
   `0d21a64c6b61b6be7b49050dbd89f89f934912a7b586734e475308e9c9422ea2` (dataset 2.1.0)

Technique entries must additionally set `technique_reimplementation: true` and
`vendor_code_executed: false`. Two panel rows currently satisfy the letter of the contract but not
its intent: `openvons` and `mini-jev` produce **byte-identical predictions across all 1240
evaluation decisions** while sharing a base checkpoint, so they are one measurement rather than two.
A coverage count that includes both overstates the panel by one model.
