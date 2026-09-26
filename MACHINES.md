# Benchmark machines

## machine-a (head-to-head runs)
- Role: ran both `run_laya_*` and `run_jev-1.13.0_*` in `results/`
- CPU: Apple M2
- RAM: 8 GB
- OS: macOS 26.6.2
- Python: 3.14.6 (project `.venv`)
- Laya serving: local, `convaiinnovations/laya` English checkpoint, CPU, defaults
- Jev serving: TypeSafe API, pinned `jev-1.13.0`, network path US/gcc to `api.typesafe.ai`

## machine-b (shared Linux CPU host)
- Role: earlier full Laya-only run that motivated this repo (`bench_full.py` era), and the
  host the current v2 runs and the report were produced on
- CPU: Intel Xeon E5-2620 v3 @ 2.40GHz, 12 cores
- RAM: 47 GB
- OS: Ubuntu 24.04.4 LTS
- Python: 3.12.3
- Laya serving: local, same checkpoint, 4 torch threads, nice 19
- Accuracy matched machine-a exactly (triage 0.800, guardrails 0.883, moderation 0.833);
  latency was ~4-5x slower per call (~400 ms/question vs ~40-95 ms/question on M2),
  consistent with an older server CPU.

## Reproducing on your hardware
- Latency numbers are hardware-specific; accuracy/ECE should reproduce within noise.
- GPU serving changes latency by an order of magnitude; see the v2 report under
  `results/v2/report-20260926/`.
- The v2.0.0 result was produced on a CPU-only host with each run container limited to
  4 CPUs and 12 GB. No hostname or account is recorded here, and none is needed to reproduce
  the accuracy numbers.
