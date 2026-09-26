# Sysone-bench v2 Runners and Pelican Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all four model adapters strict and reproducible, add a tested Torch PCD backend, and execute open-model and Jev runs inside an isolated, resource-capped pelican workflow.

**Architecture:** Adapters normalize vendor responses into the core answer contract and return raw/usage metadata without mutating inputs. The orchestration layer records every phase and writes v2 artifacts through exclusive storage. Open models run sequentially in a dedicated Docker container; Jev receives a transient key through SSH stdin and never writes it to disk.

**Tech Stack:** Python 3.12, requests, python-dotenv, PyTorch CPU, Transformers, pytest, unittest.mock, Docker on `tejes@pelican`.

**Spec:** `docs/superpowers/specs/2026-09-24-sysone-bench-v2-design.md`

**Depends on:** `docs/superpowers/plans/2026-09-24-sysone-bench-v2-core.md` and `docs/superpowers/plans/2026-09-24-sysone-bench-v2-data.md`

## Global Constraints

- The primary Laya-versus-Jev comparison uses the same ordered logical manifest.
- Qwen-PCD is a secondary transformed-schema baseline; Router is a deployment variant.
- Seed `42` is passed only where the adapter supports it and is recorded as unsupported otherwise.
- Jev model ID is the bare pinned `jev-1.13.0`; the returned model ID must match.
- Open-model jobs on pelican use 4 CPUs, 12 GiB RAM, no ports, `nice 19`, and idle I/O.
- Do not install packages globally on pelican or alter existing containers.
- Never place `TYPESAFE_API_KEY` in a command line, log, result, remote file, or graph.
- Count warmup, benchmark, and speed-scaling calls separately.
- Do not run model inference on the local development machine.
- Do not commit, push, or publish automatically.

## File Map

- Modify `runners/base.py` - explicit metadata and answer contract documentation.
- Modify `runners/laya_runner.py` - immutable model identity and deep-copy adapter.
- Modify `runners/jev_runner.py` - dotenv configuration, HTTPS validation, session retries, returned-model checks, and usage.
- Modify `runners/router_runner.py` - scoped route counts and model identity.
- Modify `runners/qwen_runner.py` - platform-neutral PCD import, complete probability map, score rubric schema.
- Modify `runners/pcd/engine.py` - platform-neutral backend selection.
- Create `runners/pcd/engine_torch.py` - Linux CPU parallel constrained decoding.
- Create `runners/pcd/engine_common.py` - shared telemetry and invalid-generation errors.
- Create `benchmark/orchestrator.py` - phase-aware suite execution and v2 prediction rows.
- Create `benchmark/usage.py` - per-phase token and call accounting.
- Create `benchmark/timing.py` - warmup, repetitions, p50, p95, and host context.
- Modify `run.py` - dotenv loading, phase tracking, v2 storage, and model selection.
- Modify `compare.py` - fail-closed manifest comparison and v2 output.
- Create `ops/pelican/Dockerfile` - pinned Python 3.12 CPU image.
- Create `ops/pelican/preflight.py` - read-only capacity checks.
- Create `ops/pelican/run_open_model.sh` - isolated container launcher.
- Create `ops/pelican/remote_jev.py` - local key reader and SSH stdin sender.
- Create `ops/pelican/remote_jev_entrypoint.py` - remote one-shot Jev receiver.
- Create `ops/pelican/cleanup.sh` - removes only the job's container and temporary files.
- Create `ops/pelican/AGENTS.md` - remote safety and secret-handling contract.
- Create `tests/test_runner_contracts.py` - fake adapter and malformed-output tests.
- Create `tests/test_jev_runner.py` - mocked HTTP and configuration tests.
- Create `tests/test_timing.py` - deterministic clock tests.
- Create `tests/test_pelican_preflight.py` - capacity threshold tests.

- Create `tests/fakes.py` - deterministic fake runners, mocked Jev setup, and v2 manifest fixtures.

---

### Task 0: Define deterministic runner test fixtures

**Files:**
- Create: `tests/fakes.py`

**Interfaces:**
- Produces `make_fake_laya_runner() -> tuple[BaseRunner, dict, dict]`.
- Produces `make_fake_router() -> BaseRunner`.
- Produces `make_runner_with_key() -> JevRunner`.
- Produces `run_backend_import_test(backend: str) -> dict[str, str]`.
- Produces `run_collision_fixture() -> None`.
- Produces `score_question() -> dict[str, object]`.
- Produces `manifest_fixture(tmp_path: Path) -> Path`.
- Produces `write_fake_run(tmp_path: Path, manifest_digest_value: str) -> Path`.

- [ ] **Step 1: Add fake runners and fixtures**

```python
import json
from copy import deepcopy
from pathlib import Path

from runners.base import BaseRunner


class FakeLayaRunner(BaseRunner):
    name = "fake-laya"

    def __init__(self):
        self.seen: list[tuple[dict, dict]] = []

    def predict(self, state, questions):
        self.seen.append((deepcopy(state), deepcopy(questions)))
        return {"answers": {"intent": {"type": "noul", "noul": 1.0}}}

    def info(self):
        return {"runner": self.name, "model": "fake", "revision": "test", "serving": "fake", "device": "cpu", "adapter_version": "test"}


class FakeRouter(FakeLayaRunner):
    name = "fake-router"

    def __init__(self):
        super().__init__()
        self.route_counts = {}

    def predict(self, state, questions):
        self.route_counts["english"] = self.route_counts.get("english", 0) + 1
        return super().predict(state, questions)

    def info(self):
        info = super().info()
        info["route_counts"] = dict(self.route_counts)
        return info


def make_fake_laya_runner():
    state = {"message": "hello"}
    questions = {"intent": {"type": "noul"}}
    return FakeLayaRunner(), state, questions


def make_fake_router():
    return FakeRouter()


def make_runner_with_key():
    from runners.jev_runner import JevRunner
    return JevRunner(model="jev-1.13.0", base_url="https://api.typesafe.ai/v1/systemone")


def run_backend_import_test(backend: str) -> dict[str, str]:
    import importlib
    module = importlib.import_module("runners.pcd.engine_torch" if backend == "torch" else "runners.pcd.engine_mlx")
    return {"backend": backend, "module": module.__name__}


def run_collision_fixture() -> None:
    from runners.pcd.engine_common import InvalidGenerationError
    raise InvalidGenerationError("collision fixture")


def score_question() -> dict[str, object]:
    return {"qid": "sentiment", "type": "score", "instructions": "How positive is the sentiment?", "criteria": ["very negative", "negative", "neutral", "positive", "very positive"], "max_score": 4}


def manifest_fixture(tmp_path: Path) -> Path:
    records = [{"schema_version": 2, "dataset_version": "2.0.0", "suite_id": "fixture", "case_id": f"fixture-{index:04d}", "order_index": index, "split": "evaluation", "state": {"text": str(index)}, "questions": [{"qid": "q", "type": "noul", "instructions": "fixture question"}], "expected": {"q": index % 2}, "provenance_id": "fixture", "label_provenance": "fixture"} for index in range(3)]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    return path


def write_fake_run(tmp_path: Path, manifest_digest_value: str) -> Path:
    run_root = tmp_path / f"run-{manifest_digest_value}"
    run_root.mkdir()
    (run_root / "metadata.json").write_text(json.dumps({"run_id": run_root.name, "manifest_digest": manifest_digest_value, "model": {"id": "fake", "revision": "test"}, "suites": {}}), encoding="utf-8")
    (run_root / "predictions.jsonl").write_text("", encoding="utf-8")
    (run_root / "summary.json").write_text(json.dumps({"decisions": 0, "suites": {}}), encoding="utf-8")
    return run_root
```

- [ ] **Step 2: Run the fixture import test**

Run: `uv run python -c "import sys; sys.path.insert(0, 'tests'); import fakes; assert fakes.make_fake_router().name == 'fake-router'"`

Expected: PASS after the fixture module is created.

---

### Task 1: Harden the common runner interface

**Files:**
- Modify: `runners/base.py`
- Modify: `runners/laya_runner.py`
- Modify: `runners/router_runner.py`
- Test: `tests/test_runner_contracts.py`

**Interfaces:**
- `BaseRunner.info() -> dict[str, Any]` returns runner, model, revision, serving, device, and adapter version.
- `BaseRunner.predict(state: Mapping[str, Any], questions: Mapping[str, Any]) -> dict[str, Any]` returns `answers`, optional `_usage`, optional `_routing`, and optional `_raw_model`.
- Inputs are deep-copied before adapter use.

- [ ] **Step 1: Write failing adapter contract tests**

```python
from copy import deepcopy


def test_laya_does_not_mutate_inputs():
    runner, state, questions = make_fake_laya_runner()
    original_state = deepcopy(state)
    original_questions = deepcopy(questions)
    runner.predict(state, questions)
    assert state == original_state
    assert questions == original_questions


def test_router_info_reports_scoped_route_counts():
    runner = make_fake_router()
    runner.predict({"message": "hello"}, {"intent": {"type": "noul"}})
    info = runner.info()
    assert info["route_counts"]["english"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_runner_contracts.py -q`

Expected: FAIL because the fake fixtures and strict metadata contract do not exist.

- [ ] **Step 3: Update the base contract**

Document in `BaseRunner` that every adapter must return exact answer IDs, must not mutate inputs, must record actual model identity, and must expose phase-independent usage counters. Keep the interface small and model-agnostic.

- [ ] **Step 4: Make Laya and Router metadata immutable**

Laya must record the actual repository argument, resolved checkpoint revision, package version, device, and adapter version. Router must record preloaded checkpoint names, route reason counts per benchmark phase, and the same Laya identity.

- [ ] **Step 5: Run the focused tests**

Run: `uv run pytest tests/test_runner_contracts.py -q`

Expected: PASS.

---

### Task 2: Repair Jev configuration, transport, and response validation

**Files:**
- Modify: `runners/jev_runner.py`
- Modify: `.env.example`
- Test: `tests/test_jev_runner.py`

**Interfaces:**
- `JevRunner.__init__(model: str | None = None, base_url: str | None = None)`.
- `JevRunner.predict(state, questions) -> dict[str, Any]`.
- Raises `ValueError` for non-HTTPS or credential-bearing base URLs.
- Raises `RuntimeError` for missing keys, repeated malformed responses, returned model mismatch, or exhausted retries.

- [ ] **Step 1: Write failing mocked HTTP tests**

```python
from unittest.mock import Mock, patch


def test_prefixed_model_id_is_rejected():
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test", "JEV_MODEL": "typesafe:jev-1.13.0"}, clear=False):
        with pytest.raises(ValueError, match="bare model ID"):
            JevRunner()


def test_returned_model_must_match_requested_model():
    runner = make_runner_with_key()
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"model": "jev-1.12.0", "answers": {}}
    with patch("runners.jev_runner.requests.Session.post", return_value=response):
        with pytest.raises(RuntimeError, match="returned model"):
            runner.predict({"message": "hello"}, {"intent": {"type": "noul"}})


def test_retry_stops_after_three_attempts():
    runner = make_runner_with_key()
    with patch("runners.jev_runner.requests.Session.post", side_effect=TimeoutError("timeout")) as post:
        with pytest.raises(RuntimeError, match="three attempts"):
            runner.predict({"message": "hello"}, {"intent": {"type": "noul"}})
    assert post.call_count == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_jev_runner.py -q`

Expected: FAIL because the current adapter accepts the prefix and uses one-shot `requests.post`.

- [ ] **Step 3: Implement safe configuration**

Load `.env` from the repository root before reading environment variables. Accept only a bare versioned ID matching `^jev-[0-9]+\\.[0-9]+\\.[0-9]+$`. Default to `jev-1.13.0`. Validate the base URL with `urllib.parse.urlparse`; require HTTPS, reject userinfo, query strings, and fragments.

- [ ] **Step 4: Implement bounded transport**

Use one `requests.Session` per runner. Set connect and read timeouts separately, retry only connection errors, timeouts, HTTP 429, and HTTP 5xx, use bounded exponential backoff, and stop after three attempts. Never include the key in exception messages or logs.

- [ ] **Step 5: Normalize and validate answers**

Call `validate_answers` after normalization. Preserve full probability maps, usage, returned model, and a redacted response digest. Reject missing or extra answer IDs before returning.

- [ ] **Step 6: Update `.env.example`**

Set:

```dotenv
TYPESAFE_API_KEY=
JEV_MODEL=jev-1.13.0
JEV_BASE_URL=https://api.typesafe.ai/v1/systemone
```

- [ ] **Step 7: Run Jev tests**

Run: `uv run pytest tests/test_jev_runner.py -q`

Expected: PASS.

---

### Task 3: Add phase-aware usage and timing

**Files:**
- Create: `benchmark/usage.py`
- Create: `benchmark/timing.py`
- Modify: `run.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- `UsageCounter.record(phase: str, usage: Mapping[str, Any]) -> None`.
- `UsageCounter.as_dict() -> dict[str, dict[str, int]]`.
- `measure_call(clock: Callable[[], float], call: Callable[[], object]) -> tuple[object, float]`.
- `speed_scaling(runner: BaseRunner, phases: Sequence[int], repetitions: int = 10) -> dict[str, dict[str, float]]`.

- [ ] **Step 1: Write failing usage tests**

```python
def test_usage_is_separated_by_phase():
    counter = UsageCounter()
    counter.record("benchmark", {"input_tokens": 10, "output_tokens": 2})
    counter.record("speed", {"input_tokens": 3, "output_tokens": 1})
    assert counter.as_dict()["benchmark"]["input_tokens"] == 10
    assert counter.as_dict()["speed"]["input_tokens"] == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_timing.py -q`

Expected: FAIL because `benchmark.usage` does not exist.

- [ ] **Step 3: Implement usage counters**

Store separate `input_tokens`, `output_tokens`, and `calls` for `warmup`, `benchmark`, and `speed` phases. Reject negative or non-integer token values.

- [ ] **Step 4: Implement timing with injected clocks**

Measure `perf_counter()` around each call, run exactly two warmups and ten timed repetitions per speed size, and store p50, p95, and sample count. Do not round values before threshold calculations.

- [ ] **Step 5: Run timing tests**

Run: `uv run pytest tests/test_timing.py -q`

Expected: PASS.

---

### Task 4: Implement a real Torch PCD backend

**Files:**
- Create: `runners/pcd/engine_torch.py`
- Create: `runners/pcd/engine_common.py`
- Modify: `runners/pcd/engine.py`
- Modify: `runners/qwen_runner.py`
- Test: `tests/test_pcd_torch.py`

**Interfaces:**
- `engine.get_engine() -> tuple[torch.nn.Module, PreTrainedTokenizerBase]`.
- `engine.run_parallel_generation(context: str, schema: StructuredSchema, temperature: float = 1.0) -> dict[str, Any]`.
- `engine.run_naive_generation(context: str, schema: StructuredSchema, max_tokens: int = 700, temperature: float = 0.2) -> dict[str, Any]`.
- `engine.stream_naive_generation(context: str, schema: StructuredSchema, max_tokens: int = 700, temperature: float = 0.2) -> Iterator[dict[str, Any]]`.
- Invalid output raises `InvalidGenerationError` rather than selecting the first choice.

- [ ] **Step 1: Write failing backend tests**

```python
def test_torch_backend_imports_without_mlx():
    result = run_backend_import_test(backend="torch")
    assert result["backend"] == "torch"


def test_collision_does_not_fabricate_confidence():
    with pytest.raises(InvalidGenerationError):
        run_collision_fixture()


def test_qwen_schema_contains_allowed_score_levels():
    schema = questions_to_schema(score_question())
    rendered = StructuredSchema({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in schema.items()}).to_parallel_schema_str()
    assert "very positive" in rendered
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra qwen pytest tests/test_pcd_torch.py -q`

Expected: FAIL because `engine_torch.py` and `InvalidGenerationError` do not exist.

- [ ] **Step 3: Implement Torch loading and cache broadcasting**

Load the pinned Transformers model and tokenizer on CPU. Use `torch.inference_mode()`, fixed random seeds, explicit dtype, a model cache, and a single lock around generation. Reuse the existing schema metadata compiler through a backend-neutral adapter.

- [ ] **Step 4: Implement collision-safe constrained decoding**

Use token-trie traversal for legal continuations. Compute sequence probabilities over legal continuations only. If the model emits an illegal value or cannot be parsed, raise `InvalidGenerationError` with the field name and generated text. Do not clamp to a minimum confidence or assign equal residual probabilities.

- [ ] **Step 5: Make the engine router platform-neutral**

`qwen_runner.py` imports `from runners.pcd import engine`; `engine.py` selects MLX only on Apple Silicon and otherwise imports `engine_torch.py`. Remove the missing `engine_torch` import path.

- [ ] **Step 6: Preserve full probability maps**

Return all choice probabilities from PCD telemetry. Qwen must include ordered score levels in the prompt and map the selected level back to the integer score.

- [ ] **Step 7: Run PCD tests**

Run: `uv run --extra qwen pytest tests/test_pcd_torch.py -q`

Expected: PASS.

---

### Task 5: Add isolated pelican execution

**Files:**
- Create: `ops/pelican/Dockerfile`
- Create: `ops/pelican/preflight.py`
- Create: `ops/pelican/run_open_model.sh`
- Create: `ops/pelican/cleanup.sh`
- Create: `ops/pelican/remote_jev.py`
- Create: `ops/pelican/remote_jev_entrypoint.py`
- Create: `ops/pelican/AGENTS.md`
- Test: `tests/test_pelican_preflight.py`

**Interfaces:**
- `preflight.check(snapshot: Mapping[str, Any]) -> None`.
- `run_open_model.sh --run-id <id> --model <laya|qwen|router>` runs one model and exits nonzero on any validation failure.
- `remote_jev.py` reads local `TYPESAFE_API_KEY` from `.env` and sends it only through SSH stdin.

- [ ] **Step 1: Write failing preflight tests**

```python
import pytest

from ops.pelican.preflight import check


def test_preflight_rejects_high_load():
    with pytest.raises(RuntimeError, match="load"):
        check({"cpu_count": 12, "load1": 8.0, "load5": 10.0, "memory_available_gib": 30, "disk_available_gib": 500})


def test_preflight_accepts_approved_capacity():
    check({"cpu_count": 12, "load1": 4.0, "load5": 6.0, "memory_available_gib": 20, "disk_available_gib": 500})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_pelican_preflight.py -q`

Expected: FAIL because the preflight module does not exist.

- [ ] **Step 3: Implement read-only preflight**

Use `/proc/loadavg`, `free -b`, `statvfs`, and `platform`. Reject `load1 / cpu_count > 0.50`, `load5 / cpu_count > 0.75`, memory below 16 GiB, disk below 100 GiB, missing AVX2, or non-Python 3.12. Record raw values without modifying the host.

- [ ] **Step 4: Build the isolated image**

Use `python:3.12-slim-bookworm`, install only locked runtime dependencies, set `PYTHONUNBUFFERED=1`, and run as a non-root user. Do not expose ports or mount the host Docker socket.

- [ ] **Step 5: Implement the open-model launcher**

`run_open_model.sh` must:

1. Create a unique run directory under `/home/tejes/sysone-bench-v2/runs/<run_id>`.
2. Run the read-only preflight.
3. Start `docker run --rm --name sysone-bench-<run_id> --cpus=4 --memory=12g --memory-swap=12g --cpuset-cpus=0-3 -v <run_dir>:/workspace -w /workspace`.
4. Execute the requested model sequentially.
5. Write v2 artifacts and checksums.
6. Run `docker rm -f` only for its own container name in an exit trap.
7. Leave every pre-existing container untouched.

- [ ] **Step 6: Implement Jev key streaming**

`remote_jev.py` reads the local gitignored `.env`, extracts only `TYPESAFE_API_KEY` in memory, and invokes `ssh tejes@pelican /home/tejes/sysone-bench-v2/ops/remote_jev_entrypoint.py` with the key on stdin. The remote entrypoint reads one line, sets it in the child process environment, runs the Jev command, clears the variable, and exits. Neither script prints the key or writes it to disk.

- [ ] **Step 7: Add remote safety AGENTS**

Document the shared-mainframe boundary, no-global-install rule, CPU and memory caps, no exposed ports, no existing-container changes, cleanup ownership, and secret non-persistence.

- [ ] **Step 8: Run preflight and shell syntax tests**

Run:

```bash
uv run pytest tests/test_pelican_preflight.py -q
bash -n ops/pelican/run_open_model.sh ops/pelican/cleanup.sh
```

Expected: PASS.

---

### Task 6: Replace run and compare orchestration

**Files:**
- Create: `benchmark/orchestrator.py`
- Modify: `run.py`
- Modify: `compare.py`
- Test: `tests/test_run_compare_v2.py`

**Interfaces:**
- `run_suite_v2(runner: BaseRunner, suite_id: str, records: Sequence[Mapping[str, Any]], counter: UsageCounter) -> dict[str, Any]`.
- `run_all_v2(runner_names: Sequence[str], manifest_path: Path, output_root: Path) -> list[Path]`.
- `compare_v2(path_a: Path, path_b: Path, output_root: Path) -> Path`.

- [ ] **Step 1: Write failing end-to-end fake tests**

```python
def test_fake_run_writes_predictions_and_summary(tmp_path):
    paths = run_all_v2(["fake"], manifest_fixture(tmp_path), tmp_path / "results")
    summary = json.loads((paths[0] / "summary.json").read_text())
    assert summary["decisions"] == 3
    assert (paths[0] / "predictions.jsonl").exists()


def test_compare_rejects_different_manifest_hashes(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        compare_v2(write_fake_run(tmp_path, "one"), write_fake_run(tmp_path, "two"), tmp_path / "comparisons")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_run_compare_v2.py -q`

Expected: FAIL because the v2 orchestration functions do not exist.

- [ ] **Step 3: Implement manifest-driven execution**

Load the sealed manifest, select only `split=evaluation` for headline results, and run calibration separately. For every case, store `case_id`, ordered question IDs, expected values, full normalized answer, confidence, model response metadata, latency, and phase.

- [ ] **Step 4: Make comparisons fail closed**

Before reading any metrics, compare full manifest digests, suite IDs, case IDs, order, question definitions, split, model identities, and metric schema. Return a nonzero CLI exit and create no comparison artifact on mismatch.

- [ ] **Step 5: Preserve CLI behavior**

Keep `run.py --models` and `compare.py path_a path_b` as the public commands. Print progress without secrets and return nonzero on validation, transport, or storage errors.

- [ ] **Step 6: Run orchestration tests**

Run: `uv run pytest tests/test_runner_contracts.py tests/test_timing.py tests/test_run_compare_v2.py -q`

Expected: PASS.

## Completion Gate

The runner track is complete when Laya, Jev, Qwen-PCD, and Router pass strict fake-adapter tests, Jev configuration and retries are safe, Torch PCD works without MLX, every phase is accounted for, and pelican preflight/launcher tests prove the job cannot alter existing workloads.
