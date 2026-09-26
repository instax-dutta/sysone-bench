# Purpose
- Owns the runner interface: the contract every model adapter implements.
- `benchmark/orchestrator.py` owns the v2 BaseRunner boundary; root `run.py` is a thin CLI while its tested legacy timing helpers remain available.

# Ownership
- Interface changes require a corresponding update for every adapter claiming the v2 contract; all current production adapters are called through that boundary.

# Local Contracts
- The v2 `BaseRunner.predict(state, questions, *, phase="benchmark") -> dict` contract accepts mappings, validates `phase` as `warmup`, `benchmark`, or `speed`, deep-copies both inputs before adapter use, and never mutates caller-owned input. Adapters claiming this contract must support the keyword.
- Legacy adapters may still expose only `predict(state, questions)`. The legacy root `run.py` `_call_predict` helper inspects the bound method signature and forwards `phase` only to a keyword-capable `phase` parameter or `**kwargs`; otherwise it calls the two-argument form. It never retries or masks `TypeError`s raised inside a model call.
- Usage accounting must record the requested phase for both phase-aware and legacy methods.
- The v2 orchestrator calls phase-aware adapters with `phase="benchmark"`, keeps calibration and evaluation counters separate, and requires each adapter response to contain only `answers` plus the optional metadata fields listed below. Response answer maps must have exact non-empty string keys before contract validation.
- Qwen is a phase-capable v2 adapter. `QwenRunner.predict` accepts the optional `phase` keyword, validates it, deep-copies inputs, and returns only normalized answers.
- Results contain `answers` keyed by exactly the requested question IDs, plus optional `_usage`, `_routing`, and `_raw_model` fields; Jev additionally returns `_response_digest` as redacted response metadata.
- Routing metadata is optional; valid `routing` or `_routing` is preserved under `_routing`, while missing metadata is counter-neutral and malformed or placeholder model metadata fails clearly without counting a route.
- Adapters validate missing or extra answer IDs at their edge and normalize vendor output to the laya answer shape:
  - choice: `{"type": "choice", "choice": <label>, "probabilities": {...}, "confidence": <0-1>}`
  - noul: `{"type": "noul", "noul": <P(true) 0-1>}`
  - score: `{"type": "score", "score": <float>, "confidence": <0-1>}`
- Jev rejects any non-mapping question object before response-answer validation; a matching answer ID set cannot bypass malformed-question validation.
- `BaseRunner.name` is the run-file identifier (`laya`, `jev-1.13.0`, ...).
- `BaseRunner.info()` always exposes `runner`, `model`, `revision`, `serving`, `device`, and `adapter_version`, and returns detached snapshots.
- Qwen metadata reports Torch CPU as `device="cpu"` and exposes the immutable vendored upstream/base PCD revision `6345318eea8cf81b68fa21962aae3ef688a50284` as `pcd_source_revision`, with `pcd_source_revision_role="vendored_upstream_base_revision"`.
- Qwen metadata reports a deterministic SHA-256 `pcd_implementation_digest` over `engine_common.py`, `engine_torch.py`, `engine.py`, `prompt_builder.py`, and `schema.py`; the digest excludes `qwen_runner.py` and does not attribute the Torch implementation to the base revision.
- Laya keeps the requested repository separate from resolved metadata and fails closed when revision, device, checkpoint, or package version is missing.
- Laya resolves identity from the loaded checkpoint, not from attributes the package may omit:
  - `revision` comes from the agent when exposed, otherwise from the commit directory of the already-cached `rl_agent_config.json` for the requested repository and subfolder, read offline through the Hugging Face cache. An unresolvable revision is a hard failure, never `unknown`.
  - `checkpoint` comes from the agent when exposed, otherwise from the subfolder the adapter passed to the loader; the default repository-root checkpoint is recorded as `root`.
  - `resolved_repo` may fall back to the agent's `model_id` when no explicit resolved-repository attribute exists.
- Laya's `laya` package returns richer answers than the benchmark contract allows: every answer carries `action`, score answers add `legend` and `probabilities`, and noul answers add `confidence`. The adapter projects each answer onto the exact declared field set for the question type and drops the rest rather than loosening the contract.
- Laya rounds every probability to four decimals, so raw choice probabilities drift up to `len(labels) * 5e-5` away from one. The adapter divides by the observed total and absorbs the residual on the largest entry, which is order-preserving, so the reported `choice` stays the argmax and the probabilities sum to one within the contract tolerance. Non-numeric, negative, non-finite, empty, zero-sum, or criteria-mismatched probability payloads fail the run.
- Router records verified identity for every preloaded checkpoint, exposes phase-scoped route counts/reason collections/counts plus detached aggregate compatibility snapshots, and never invents a model or uses one unchecked agent as the whole identity. Prediction rows store a deep stable-identity projection with all `route_*` telemetry excluded; final metadata may retain that live telemetry.
- Phase-scoped Router metadata uses the exact `phase` supplied by the caller; aggregate route fields remain available for compatibility.
- Runners must not mutate the input questions dict.
- Secrets: only `jev_runner.py` may read `TYPESAFE_API_KEY` from env. Never log it.
- Jev configuration loads the repository-root `.env` through dotenv, accepts only bare versioned model IDs matching `^jev-[0-9]+\.[0-9]+\.[0-9]+$`, and requires HTTPS base URLs without credentials, queries, or fragments.
- Jev uses one `requests.Session` per runner with separate connect/read timeouts and at most three attempts, retrying only connection errors, timeouts, HTTP 429, and HTTP 5xx; malformed successful payloads and returned-model mismatches are not retried.
- Jev results retain validated answers, full usage, the returned model ID, and a redacted response digest; adapter errors never expose credentials, authorization headers, or raw response bodies.
- The Jev response digest redacts authorization, access, and secret token fields while preserving material usage counters such as `input_tokens`, `output_tokens`, and other `*_tokens` fields.
- PCD exposes the common `get_engine`, `run_parallel_generation`, `run_naive_generation`, and `stream_naive_generation` interfaces. The platform-neutral router always selects Torch on every platform and never imports MLX in the normal path; `BACKEND=torch` remains supported.
- The Torch PCD backend is lazy, CPU-only, seeded with `42`, uses explicit `float32` loading for stock `Qwen/Qwen2.5-1.5B-Instruct` at immutable revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, caches one model and tokenizer, and serializes generation with one lock. Tests must use fake models and tokenizers and must not load weights.
- Torch PCD cached scoring uses one longest-common-prefix prefill and one batched suffix forward across all legal-choice rows; each field carries its suffix IDs and suffix logit offset, and in-place cache expansion is supported.
- Torch PCD transition scoring uses raw model probabilities for legal children plus stop mass at terminal trie nodes with children, and normalizes only among legal children at non-terminal nodes. Invalid, empty, unparseable, or colliding generation raises `InvalidGenerationError` with field context; adapters never select a fallback candidate, clamp confidence, or distribute residual probability.
- The Apple MLX implementation remains available only as a compatibility source; normal v2 execution uses Torch and does not import it. `qwen_runner.py` imports the platform-neutral `runners.pcd.engine` module and never imports `engine_mlx` directly.
- Offline Jev fixtures expose a fail-closed default `session.post`; tests may replace that instance method explicitly, and autouse guards block top-level and session transport entry points including `Session.send`.

# Work Guidance
- Keep vendor normalization and exact answer-ID checks at the adapter edge so the v2 orchestrator and `compare.py` stay model-agnostic; root `run.py` must not contain a second v2 execution path. Root import must remain possible without Laya, while legacy timing helpers retain patchable module-level behavior.
- Keep metadata, route counts, and route reasons detached from adapter internals.
- Sort reason collections and reason-count maps deterministically; update the legacy aggregate reason only from the complete sorted collection.
- Keep `qwen_runner.py` on the platform-neutral PCD interface and preserve ordered score levels, complete choice probability maps, and full validated probability values when normalizing telemetry.

# Verification
- `uv run --offline pytest tests/test_timing.py -q`
- `uv run --offline pytest tests/test_run_compare_v2.py -q`
- `./.venv/bin/python -m pytest tests/test_runner_contracts.py tests/test_fakes.py -q`
- `./.venv/bin/ruff check runners/base.py runners/laya_runner.py runners/router_runner.py tests/test_runner_contracts.py tests/fakes.py`
- `./.venv/bin/ruff format --check runners/base.py runners/laya_runner.py runners/router_runner.py tests/test_runner_contracts.py tests/fakes.py`
- `./.venv/bin/mypy --allow-subclassing-any --follow-imports=skip --ignore-missing-imports runners/base.py runners/laya_runner.py runners/router_runner.py tests/test_runner_contracts.py tests/fakes.py`
- `uv run pytest tests/test_jev_runner.py -q`
- `uv run ruff check runners/jev_runner.py tests/test_jev_runner.py`
- `uv run ruff format --check runners/jev_runner.py tests/test_jev_runner.py`
- `uv run mypy --allow-subclassing-any --follow-imports=skip --ignore-missing-imports runners/jev_runner.py tests/test_jev_runner.py`
- `uv run --offline pytest tests/test_pcd_torch.py -q`
- `uv run --offline ruff check runners/pcd/engine_common.py runners/pcd/engine_torch.py runners/pcd/engine.py runners/qwen_runner.py tests/test_pcd_torch.py`
- `uv run --offline ruff format --check runners/pcd/engine_common.py runners/pcd/engine_torch.py runners/pcd/engine.py runners/qwen_runner.py tests/test_pcd_torch.py`
- `uv run --offline mypy --allow-subclassing-any --follow-imports=skip --ignore-missing-imports runners/pcd/engine_common.py runners/pcd/engine_torch.py runners/pcd/engine.py runners/qwen_runner.py tests/test_pcd_torch.py`
- CI installs `--extra dev --extra qwen` and runs the scoped PCD Ruff, Ruff format, and mypy checks; it does not run whole-tree legacy Ruff.

# Child DOX Index
- `pcd/` - vendored parallel constrained decoding. Owns the common engine interface, Torch CPU backend, MLX compatibility path, schema compiler, and collision-safe telemetry.
- `base.py`, `laya_runner.py`, `jev_runner.py`, `router_runner.py`, and `qwen_runner.py` remain under this contract.
