from __future__ import annotations

import builtins
import importlib
import math
import platform
import sys
import threading
import time
from collections.abc import Mapping
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

PINNED_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PCD_SOURCE_REVISION = "6345318eea8cf81b68fa21962aae3ef688a50284"

from runners.pcd.engine_common import (
    InvalidGenerationError,
    build_field_telemetry,
    validate_probability_map,
)
from runners.pcd.engine_torch import (
    MODEL_ID,
    SEED,
    get_engine,
    run_naive_generation,
    run_parallel_generation,
    stream_naive_generation,
)
from runners.pcd.schema import StructuredSchema


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, values: Mapping[str, int] | None = None) -> None:
        self.values = dict(values or {})
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        if text in self.values:
            return [self.values[text]]
        return [ord(text[-1])] if text else [0]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        inverse = {value: key for key, value in self.values.items()}
        return "".join(inverse.get(token_id, "") for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.values.get(token)


class FakeModel:
    def __init__(self, logits: Any, *, generated: list[int] | None = None) -> None:
        self.logits = logits
        self.generated = list(generated or [])
        self.calls = 0
        self.devices: list[Any] = []
        self.evaluated = False

    def to(self, device: Any) -> FakeModel:
        self.devices.append(device)
        return self

    def eval(self) -> FakeModel:
        self.evaluated = True
        return self

    def __call__(self, input_ids: Any) -> Any:
        self.calls += 1
        if self.generated:
            token_id = self.generated.pop(0)
            values = torch.full((1, 1, 8), -10.0)
            values[0, 0, token_id] = 10.0
            return values
        return self.logits.clone()


def _parallel_logits(token_scores: Mapping[int, float], *, batch_size: int = 2) -> torch.Tensor:
    logits = torch.full((batch_size, 3, 8), -10.0)
    for token_id, score in token_scores.items():
        logits[:, :, token_id] = score
    return logits


def _schema() -> StructuredSchema:
    return StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "flag"},
            "color": {"type": "enum", "description": "color", "choices": ["red", "blue"]},
        }
    )


def test_torch_backend_imports_without_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKEND", "torch")
    real_import = builtins.__import__
    imported: list[str] = []

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        imported.append(name)
        if name == "mlx" or name.startswith("mlx."):
            raise AssertionError("MLX must not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("runners.pcd.engine", None)
    engine = importlib.import_module("runners.pcd.engine")

    assert engine.USE_MLX is False
    assert engine.get_engine.__module__ == "runners.pcd.engine_torch"
    assert not any(name == "mlx" or name.startswith("mlx.") for name in imported)
    assert not hasattr(engine, "engine_mlx")


def test_platform_router_uses_torch_by_default_on_apple_silicon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runners import pcd

    fake_mlx = ModuleType("runners.pcd.engine_mlx")
    fake_mlx.__dict__.update(
        {
            "get_engine": object(),
            "run_parallel_generation": object(),
            "run_naive_generation": object(),
            "stream_naive_generation": object(),
            "run_rlcd_generation": object(),
        }
    )
    monkeypatch.setitem(sys.modules, "runners.pcd.engine_mlx", fake_mlx)
    monkeypatch.setattr(pcd, "engine_mlx", fake_mlx, raising=False)
    monkeypatch.delenv("BACKEND", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    sys.modules.pop("runners.pcd.engine", None)

    engine = importlib.import_module("runners.pcd.engine")

    assert engine.USE_MLX is False
    assert engine.BACKEND == "torch"
    assert engine.get_engine is engine._torch_engine.get_engine


def test_get_engine_uses_cpu_seed_explicit_dtype_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    calls: list[tuple[str, str, dict[str, Any]]] = []
    tokenizer = FakeTokenizer({"x": 1})
    model = FakeModel(_parallel_logits({}))

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeTokenizer:
            calls.append(("tokenizer", model_id, kwargs))
            return tokenizer

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeModel:
            calls.append(("model", model_id, kwargs))
            return model

    monkeypatch.setattr(backend, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(backend, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(backend, "_model", None)
    monkeypatch.setattr(backend, "_tokenizer", None)
    manual_seed = torch.manual_seed
    seeds: list[int] = []

    def record_seed(seed: int) -> None:
        seeds.append(seed)
        manual_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", record_seed)

    first = get_engine()
    second = get_engine()

    assert first == second
    assert len(calls) == 2
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[1] == "Qwen/Qwen2.5-1.5B-Instruct" for call in calls)
    assert calls[0][2]["revision"] == PINNED_REVISION
    assert calls[1][2]["revision"] == PINNED_REVISION
    assert all(call[2].get("revision") != "main" for call in calls)
    assert calls[1][2]["dtype"] is torch.float32
    assert model.devices == [torch.device("cpu")]
    assert model.evaluated is True
    assert seeds == [42]
    assert SEED == 42
    assert MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"


def test_qwen_and_torch_use_the_immutable_model_revision() -> None:
    import runners.pcd.engine_torch as backend
    import runners.qwen_runner as qwen

    info = qwen.QwenRunner().info()
    assert backend.MODEL_REVISION == PINNED_REVISION
    assert qwen.MODEL_REVISION == PINNED_REVISION
    assert qwen.PCD_SOURCE_REVISION == PCD_SOURCE_REVISION
    assert info["revision"] == PINNED_REVISION
    assert info["device"] == "cpu"
    assert info["pcd_source_revision"] == PCD_SOURCE_REVISION
    assert "main" not in {backend.MODEL_REVISION, qwen.MODEL_REVISION}


def test_qwen_metadata_separates_pcd_base_revision_from_local_digest() -> None:
    import runners.qwen_runner as qwen

    expected_files = (
        "engine_common.py",
        "engine_torch.py",
        "engine.py",
        "prompt_builder.py",
        "schema.py",
    )
    digest = qwen.pcd_implementation_digest()

    assert qwen.PCD_IMPLEMENTATION_FILES == expected_files
    assert qwen.PCD_IMPLEMENTATION_DIGEST_ALGORITHM == "sha256"
    assert digest == qwen.pcd_implementation_digest()
    assert len(digest) == 64
    assert qwen.PCD_IMPLEMENTATION_DIGEST == digest
    assert qwen.PCD_SOURCE_REVISION == PCD_SOURCE_REVISION
    assert qwen.PCD_SOURCE_REVISION_ROLE == "vendored_upstream_base_revision"
    assert "qwen_runner.py" not in qwen.PCD_IMPLEMENTATION_FILES

    info = qwen.QwenRunner().info()
    assert info["pcd_source_revision"] == PCD_SOURCE_REVISION
    assert info["pcd_source_revision_role"] == "vendored_upstream_base_revision"
    assert info["pcd_implementation_digest"] == digest


def test_probability_helpers_reject_invalid_maps() -> None:
    with pytest.raises(InvalidGenerationError, match="probabilit"):
        validate_probability_map({"a": 0.2, "b": 0.2}, ["a", "b"], "answer")
    with pytest.raises(InvalidGenerationError, match="answer"):
        validate_probability_map({"a": 0.5}, ["a", "b"], "answer")

    telemetry = build_field_telemetry(
        "answer",
        "a",
        "enum",
        ["a", "b"],
        {"a": 0.25, "b": 0.75},
    )
    assert telemetry["probabilities"] == {"a": 0.25, "b": 0.75}
    assert [item["choice"] for item in telemetry["top_choices"]] == ["b", "a"]


def test_parallel_probabilities_sum_over_all_legal_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    import runners.pcd.engine_torch as backend

    tokenizer = FakeTokenizer({"true": 1, "false": 2, " red": 3, " blue": 4, "red": 3, "blue": 4})
    model = FakeModel(_parallel_logits({1: 3.0, 2: 1.0, 3: 2.0, 4: 0.0, 7: 100.0}, batch_size=4))
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))

    result = run_parallel_generation("context", _schema())

    assert result["mode"] == "parallel_constrained_calibrated"
    assert result["sequential_forward_passes"] == 1
    assert result["field_telemetry"]["color"]["probabilities"] == pytest.approx(
        {"red": 0.8807970779778823, "blue": 0.11920292202211755},
        rel=1e-3,
    )
    assert sum(result["field_telemetry"]["color"]["probabilities"].values()) == pytest.approx(1.0)
    assert len(result["field_telemetry"]["color"]["top_choices"]) == 2


def test_parallel_scoring_disambiguates_shared_token_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class MultiTokenTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            values = {"a": [1, 2], "b": [1, 3]}
            return values.get(text, [9])

    logits = torch.full((2, 3, 8), -10.0)
    logits[:, 0, 1] = 1.0
    logits[0, 1, 2] = 3.0
    logits[0, 1, 3] = 0.0
    logits[1, 1, 2] = 0.0
    logits[1, 1, 3] = 1.0
    monkeypatch.setattr(backend, "get_engine", lambda: (FakeModel(logits), MultiTokenTokenizer()))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"choice": {"type": "enum", "choices": ["a", "b"]}}),
    )

    probabilities = result["field_telemetry"]["choice"]["probabilities"]
    assert probabilities["a"] > probabilities["b"]
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_prefix_terminal_mass_tracks_continuation_logit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class PrefixTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            values = {"a": [1], " a": [1], "ab": [1, 2], " ab": [1, 2]}
            return values.get(text, [9])

    tokenizer = PrefixTokenizer()

    def run(continuation_logit: float) -> dict[str, Any]:
        logits = torch.full((2, 3, 8), -10.0)
        logits[:, 0, 1] = 1.0
        logits[0, 1, 2] = -10.0
        logits[1, 1, 2] = continuation_logit
        monkeypatch.setattr(backend, "get_engine", lambda: (FakeModel(logits), tokenizer))
        return run_parallel_generation(
            "context",
            StructuredSchema({"choice": {"type": "enum", "choices": ["a", "ab"]}}),
        )

    low = run(-10.0)["field_telemetry"]["choice"]["probabilities"]
    high = run(8.0)["field_telemetry"]["choice"]["probabilities"]

    assert low["a"] > low["ab"]
    assert high["ab"] > high["a"]
    assert high["ab"] > low["ab"]
    assert sum(high.values()) == pytest.approx(1.0)


def test_parallel_scoring_normalizes_nonterminal_legal_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class UnequalTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            values = {"a": [1], " a": [1], "bb": [2, 3], " bb": [2, 3]}
            return values.get(text, [0])

    logits = torch.full((2, 3, 8), -10.0)
    logits[:, 0, 1] = 1.0
    logits[:, 0, 2] = 1.0
    logits[:, 0, 7] = 100.0
    logits[1, 1, 3] = 1.0
    logits[1, 1, 7] = 100.0
    monkeypatch.setattr(backend, "get_engine", lambda: (FakeModel(logits), UnequalTokenizer()))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"choice": {"type": "enum", "choices": ["a", "bb"]}}),
    )

    probabilities = result["field_telemetry"]["choice"]["probabilities"]
    assert probabilities == pytest.approx({"a": 0.5, "bb": 0.5}, abs=1e-6)


def test_torch_cache_broadcast_scores_all_legal_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class FakeCache:
        def __init__(self, batch: int) -> None:
            self.batch = batch
            self.expansions: list[int] = []

        def batch_repeat_interleave(self, repeats: int) -> FakeCache:
            self.expansions.append(repeats)
            return FakeCache(self.batch * repeats)

    class CacheModel:
        def __init__(self) -> None:
            self.cache_expansions: list[int] = []

        def forward(
            self,
            input_ids: Any,
            past_key_values: Any = None,
            use_cache: bool = False,
        ) -> Any:
            if past_key_values is None:
                cache = FakeCache(1)
                prefill = torch.full((1, 1, 8), -1.0)
                prefill[0, 0, 1] = 2.0
                return SimpleNamespace(
                    logits=prefill,
                    past_key_values=cache,
                )
            self.cache_expansions.append(past_key_values.batch)
            logits = torch.full((2, 1, 8), -1.0)
            logits[0, 0, 1] = 2.0
            logits[1, 0, 2] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=None)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    tokenizer = FakeTokenizer({"a": 1, "b": 2})
    model = CacheModel()
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"choice": {"type": "enum", "choices": ["a", "b"]}}),
    )

    assert model.cache_expansions == [2]
    assert result["sequential_forward_passes"] == 2
    assert result["field_telemetry"]["choice"]["probabilities"]["a"] > 0.5


def test_cached_parallel_uses_one_shared_prefill_and_batched_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class TwoFieldTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if "Field: first" in text:
                return [10, 11, 5]
            if "Field: second" in text:
                return [10, 11, 6]
            values = {"a": 1, " a": 1, "b": 2, " b": 2, "c": 3, " c": 3, "d": 4, " d": 4}
            return [values.get(text, 0)]

    class FakeCache:
        def __init__(self, batch: int = 1) -> None:
            self.batch = batch

        def batch_repeat_interleave(self, repeats: int) -> FakeCache:
            return FakeCache(self.batch * repeats)

    class SharedCacheModel:
        def __init__(self) -> None:
            self.prefill_inputs: list[list[list[int]]] = []
            self.suffix_inputs: list[list[list[int]]] = []
            self.cache_batches: list[int] = []

        def forward(
            self,
            input_ids: Any,
            past_key_values: Any = None,
            use_cache: bool = False,
        ) -> Any:
            ids = input_ids.detach().cpu().tolist()
            if past_key_values is None:
                self.prefill_inputs.append(ids)
                logits = torch.full((1, len(ids[0]), 8), -10.0)
                logits[0, -1, 1] = 4.0
                logits[0, -1, 2] = 0.0
                return SimpleNamespace(logits=logits, past_key_values=FakeCache())
            self.suffix_inputs.append(ids)
            self.cache_batches.append(past_key_values.batch)
            logits = torch.full((len(ids), max(len(row) for row in ids), 8), -10.0)
            for row_index, row in enumerate(ids):
                if row[0] == 5:
                    logits[row_index, 0, 1] = 6.0
                    logits[row_index, 0, 2] = 0.0
                elif row[0] == 6:
                    logits[row_index, 0, 3] = 6.0
                    logits[row_index, 0, 4] = 0.0
            return SimpleNamespace(logits=logits, past_key_values=None)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    model = SharedCacheModel()
    monkeypatch.setattr(backend, "get_engine", lambda: (model, TwoFieldTokenizer()))

    result = run_parallel_generation(
        "context",
        StructuredSchema(
            {
                "first": {"type": "enum", "choices": ["a", "b"]},
                "second": {"type": "enum", "choices": ["c", "d"]},
            }
        ),
    )

    assert model.prefill_inputs == [[[10, 11]]]
    assert len(model.suffix_inputs) == 1
    assert model.suffix_inputs[0] == [[5, 1], [5, 2], [6, 3], [6, 4]]
    assert model.cache_batches == [4]
    assert result["sequential_forward_passes"] == 2
    assert result["field_telemetry"]["first"]["probabilities"]["a"] > 0.9
    assert (
        result["field_telemetry"]["first"]["probabilities"]["a"]
        > result["field_telemetry"]["first"]["probabilities"]["b"]
    )
    assert result["field_telemetry"]["second"]["probabilities"]["c"] > 0.9
    assert (
        result["field_telemetry"]["second"]["probabilities"]["c"]
        > result["field_telemetry"]["second"]["probabilities"]["d"]
    )


def test_cached_path_accepts_in_place_transformers_cache_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class InPlaceCache:
        def __init__(self) -> None:
            self.batch = 1

        def batch_repeat_interleave(self, repeats: int) -> None:
            self.batch *= repeats

    class InPlaceCacheModel:
        def __init__(self) -> None:
            self.seen_batches: list[int] = []

        def forward(
            self,
            input_ids: Any,
            past_key_values: Any = None,
            use_cache: bool = False,
        ) -> Any:
            if past_key_values is None:
                prefill = torch.full((1, 1, 8), -1.0)
                prefill[0, 0, 1] = 3.0
                prefill[0, 0, 2] = 1.0
                return SimpleNamespace(
                    logits=prefill,
                    past_key_values=InPlaceCache(),
                )
            self.seen_batches.append(past_key_values.batch)
            logits = torch.full((2, 1, 8), -1.0)
            logits[0, 0, 1] = 3.0
            logits[1, 0, 2] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=None)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    model = InPlaceCacheModel()
    tokenizer = FakeTokenizer({"a": 1, "b": 2})
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"choice": {"type": "enum", "choices": ["a", "b"]}}),
    )

    assert model.seen_batches == [2]
    assert result["field_telemetry"]["choice"]["probabilities"]["a"] > 0.5


def test_cached_scoring_uses_prefill_for_the_first_token_and_suffix_for_later_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class FakeCache:
        def __init__(self, batch: int = 1) -> None:
            self.batch = batch

        def batch_repeat_interleave(self, repeats: int) -> FakeCache:
            return FakeCache(self.batch * repeats)

    class MultiTokenTokenizer(FakeTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            values = {"a": [1, 3], "b": [2, 4]}
            return values.get(text, [9])

    class CausalCacheModel:
        def __init__(self) -> None:
            self.forward_calls = 0
            self.cache_batch: int | None = None

        def forward(
            self,
            input_ids: Any,
            past_key_values: Any = None,
            use_cache: bool = False,
        ) -> Any:
            self.forward_calls += 1
            if past_key_values is None:
                prefill = torch.full((1, 1, 8), -10.0)
                prefill[0, 0, 1] = 6.0
                prefill[0, 0, 2] = 0.0
                return SimpleNamespace(logits=prefill, past_key_values=FakeCache())
            self.cache_batch = past_key_values.batch
            suffix = torch.full((2, 2, 8), -10.0)
            suffix[0, 0, 1] = 0.0
            suffix[1, 0, 2] = 6.0
            suffix[0, 1, 3] = 2.0
            suffix[1, 1, 4] = 0.0
            return SimpleNamespace(logits=suffix, past_key_values=None)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    model = CausalCacheModel()
    tokenizer = MultiTokenTokenizer()
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"choice": {"type": "enum", "choices": ["a", "b"]}}),
    )

    probabilities = result["field_telemetry"]["choice"]["probabilities"]
    assert model.forward_calls == 2
    assert model.cache_batch == 2
    assert probabilities["a"] > probabilities["b"]


def test_cached_parallel_telemetry_counts_and_measures_both_forward_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class FakeCache:
        def __init__(self, batch: int = 1) -> None:
            self.batch = batch

        def batch_repeat_interleave(self, repeats: int) -> FakeCache:
            return FakeCache(self.batch * repeats)

    class TimedCacheModel:
        def __init__(self) -> None:
            self.forward_calls = 0

        def forward(
            self,
            input_ids: Any,
            past_key_values: Any = None,
            use_cache: bool = False,
        ) -> Any:
            self.forward_calls += 1
            if past_key_values is None:
                return SimpleNamespace(
                    logits=torch.full((1, 1, 8), -1.0),
                    past_key_values=FakeCache(),
                )
            return SimpleNamespace(
                logits=torch.full((2, 1, 8), -1.0),
                past_key_values=None,
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

    model = TimedCacheModel()
    tokenizer = FakeTokenizer({"true": 1, "false": 2})
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))
    clock_value = 0.0

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    monkeypatch.setattr(backend, "time", SimpleNamespace(perf_counter=clock))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"flag": {"type": "boolean"}}),
    )

    assert model.forward_calls == 2
    assert result["sequential_forward_passes"] == 2
    assert result["prefill_ms"] >= 0.0
    assert result["suffix_eval_ms"] >= 0.0
    assert result["prefill_ms"] > 0.0
    assert result["suffix_eval_ms"] > 0.0
    assert result["prefill_ms"] + result["suffix_eval_ms"] <= result["elapsed_ms"]


def test_uncached_parallel_telemetry_does_not_report_zero_for_unavailable_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class TimedModel:
        def __init__(self) -> None:
            self.forward_calls = 0
            self.logits = _parallel_logits({1: 2.0, 2: 1.0}, batch_size=2)

        def __call__(self, input_ids: Any) -> Any:
            self.forward_calls += 1
            return self.logits.clone()

    model = TimedModel()
    tokenizer = FakeTokenizer({"true": 1, "false": 2})
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))
    clock_value = 0.0

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    monkeypatch.setattr(backend, "time", SimpleNamespace(perf_counter=clock))

    result = run_parallel_generation(
        "context",
        StructuredSchema({"flag": {"type": "boolean"}}),
    )

    timing_values = [
        result.get("prefill_ms"),
        result.get("suffix_eval_ms"),
        result.get("forward_ms"),
    ]
    assert model.forward_calls == 1
    assert result["sequential_forward_passes"] == 1
    assert all(value is None or value >= 0.0 for value in timing_values)
    assert any(value is not None and value > 0.0 for value in timing_values)


def test_empty_model_logits_raise_with_field_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    tokenizer = FakeTokenizer({"true": 1, "false": 2})
    model = FakeModel(torch.empty((1, 0, 8)))
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))

    with pytest.raises(InvalidGenerationError, match="flag"):
        run_parallel_generation("context", StructuredSchema({"flag": {"type": "boolean"}}))


def test_out_of_vocabulary_legal_token_raises_field_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    tokenizer = FakeTokenizer({"a": 9})
    model = FakeModel(torch.full((1, 2, 8), -1.0))
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))

    with pytest.raises(InvalidGenerationError, match="answer"):
        run_parallel_generation(
            "context",
            StructuredSchema({"answer": {"type": "enum", "choices": ["a"]}}),
        )


def test_collision_and_illegal_output_raise_with_field_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    collision_tokenizer = FakeTokenizer({"a": 5, "b": 5})
    collision_schema = StructuredSchema({"choice": {"type": "enum", "choices": ["a", "b"]}})
    monkeypatch.setattr(
        backend,
        "get_engine",
        lambda: (FakeModel(_parallel_logits({5: 1.0}, batch_size=1)), collision_tokenizer),
    )

    with pytest.raises(InvalidGenerationError, match="choice"):
        run_parallel_generation("context", collision_schema)

    illegal_tokenizer = FakeTokenizer({"illegal": 6})
    illegal_model = FakeModel(_parallel_logits({6: 10.0}, batch_size=1), generated=[6])
    monkeypatch.setattr(backend, "get_engine", lambda: (illegal_model, illegal_tokenizer))
    with pytest.raises(InvalidGenerationError, match="choice"):
        run_naive_generation("context", collision_schema, max_tokens=1)


def test_naive_and_stream_interfaces_return_valid_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    import runners.pcd.engine_torch as backend

    tokenizer = FakeTokenizer({'"choice": "a"\n}': 1, "": 2})
    model = FakeModel(_parallel_logits({}, batch_size=1), generated=[1, 2])
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))
    schema = StructuredSchema({"choice": {"type": "enum", "choices": ["a"]}})

    result = run_naive_generation("context", schema, max_tokens=8)
    model.generated = [1, 2]
    events = list(stream_naive_generation("context", schema, max_tokens=8))

    assert result["is_valid_json"] is True
    assert result["schema_match"] is True
    assert result["parsed_json"] == {"choice": "a"}
    assert [event["type"] for event in events] == ["token", "done"]
    assert events[-1]["result"]["parsed_json"] == {"choice": "a"}


def test_seeded_temperature_sampling_is_used_by_naive_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.pcd.engine_torch as backend

    class TemperatureModel:
        def __call__(self, input_ids: Any) -> Any:
            logits = torch.full((1, 1, 8), -10.0)
            logits[0, 0, 0] = math.log(0.4)
            logits[0, 0, 1] = math.log(0.36)
            logits[0, 0, 2] = math.log(0.24)
            return logits

    tokenizer = FakeTokenizer({"bad": 0, "worse": 1, '"choice": "a"\n}': 2})
    tokenizer.eos_token_id = 3
    monkeypatch.setattr(backend, "get_engine", lambda: (TemperatureModel(), tokenizer))
    schema = StructuredSchema({"choice": {"type": "enum", "choices": ["a"]}})

    def generate(temperature: float, streaming: bool) -> dict[str, Any]:
        if streaming:
            events = list(
                stream_naive_generation("context", schema, max_tokens=1, temperature=temperature)
            )
            return cast(dict[str, Any], events[-1]["result"])
        return run_naive_generation("context", schema, max_tokens=1, temperature=temperature)

    for streaming in (False, True):
        result = generate(1.0, streaming)
        assert result["parsed_json"] == {"choice": "a"}
        with pytest.raises(InvalidGenerationError):
            generate(0.05, streaming)


def test_each_generation_resets_seed_under_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    import runners.pcd.engine_torch as backend

    tokenizer = FakeTokenizer({"true": 1, "false": 2})
    model = FakeModel(_parallel_logits({1: 2.0, 2: 1.0}, batch_size=2))
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))
    seeds: list[int] = []
    monkeypatch.setattr(backend, "_set_seed", lambda: seeds.append(42))

    run_parallel_generation("context", StructuredSchema({"flag": {"type": "boolean"}}))
    run_parallel_generation("context", StructuredSchema({"flag": {"type": "boolean"}}))

    assert seeds == [42, 42]


def test_generation_lock_serializes_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import runners.pcd.engine_torch as backend

    active = 0
    maximum = 0
    guard = threading.Lock()

    class BlockingModel(FakeModel):
        def __call__(self, input_ids: Any) -> Any:
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with guard:
                active -= 1
            return super().__call__(input_ids)

    tokenizer = FakeTokenizer({"true": 1, "false": 2})
    model = BlockingModel(_parallel_logits({1: 2.0, 2: 1.0}, batch_size=2))
    monkeypatch.setattr(backend, "get_engine", lambda: (model, tokenizer))
    schema = StructuredSchema({"flag": {"type": "boolean"}})
    threads = [
        threading.Thread(target=run_parallel_generation, args=("context", schema)) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert maximum == 1


def test_qwen_schema_and_answer_mapping_preserve_order_and_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.qwen_runner as qwen
    from runners.qwen_runner import QwenRunner

    questions: dict[str, dict[str, Any]] = {
        "choice": {"type": "choice", "criteria": {"left": "L", "right": "R"}},
        "score": {
            "type": "score",
            "instructions": "How positive?",
            "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
        },
    }
    schema = qwen.questions_to_schema(questions)
    rendered = StructuredSchema(
        {
            key: {name: value for name, value in spec.items() if not name.startswith("_")}
            for key, spec in schema.items()
        }
    ).to_parallel_schema_str()
    assert rendered.index("very negative") < rendered.index("very positive")

    result = {
        "field_telemetry": {
            "choice": {
                "value": "left",
                "type": "enum",
                "confidence": 0.75,
                "probabilities": {"left": 0.75, "right": 0.25},
            },
            "score": {
                "value": "very positive",
                "type": "enum",
                "confidence": 0.8,
                "probabilities": {
                    "very negative": 0.05,
                    "negative": 0.05,
                    "neutral": 0.05,
                    "positive": 0.05,
                    "very positive": 0.8,
                },
            },
        }
    }
    monkeypatch.setattr(qwen.engine, "run_parallel_generation", lambda *args, **kwargs: result)
    runner = QwenRunner()
    state = {"message": "hello"}
    original = {"message": "hello"}
    answer = runner.predict(state, questions, phase="speed")

    assert state == original
    assert answer == {
        "answers": {
            "choice": {
                "type": "choice",
                "choice": "left",
                "probabilities": {"left": 0.75, "right": 0.25},
                "confidence": 0.75,
            },
            "score": {"type": "score", "score": 4.0, "confidence": 0.8},
        }
    }
    assert not hasattr(qwen, "engine_mlx")


def test_qwen_normalizes_legacy_noul_values(monkeypatch: pytest.MonkeyPatch) -> None:
    import runners.qwen_runner as qwen
    from runners.qwen_runner import QwenRunner

    questions = {"intent": {"type": "noul"}}
    for value in (True, False, "True", "False"):
        monkeypatch.setattr(
            qwen.engine,
            "run_parallel_generation",
            lambda *args, value=value, **kwargs: {
                "field_telemetry": {
                    "intent": {
                        "value": value,
                        "type": "boolean",
                        "probabilities": {"true": 0.75, "false": 0.25},
                    }
                }
            },
        )

        result = QwenRunner().predict({"message": "hello"}, questions)

        assert result["answers"]["intent"] == {"type": "noul", "noul": 0.75}


def test_qwen_preserves_full_validated_probability_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.qwen_runner as qwen
    from runners.qwen_runner import QwenRunner

    questions = {"choice": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
    probabilities = {"a": 0.00005, "b": 0.99995}
    monkeypatch.setattr(
        qwen.engine,
        "run_parallel_generation",
        lambda *args, **kwargs: {
            "field_telemetry": {
                "choice": {
                    "value": "b",
                    "type": "enum",
                    "probabilities": probabilities,
                }
            }
        },
    )

    answer = QwenRunner().predict({"message": "hello"}, questions)["answers"]["choice"]

    assert answer["probabilities"] == probabilities
    assert answer["confidence"] == probabilities["b"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_qwen_accepts_complete_legacy_choice_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.qwen_runner as qwen
    from runners.qwen_runner import QwenRunner

    questions = {"choice": {"type": "choice", "criteria": {"left": "L", "right": "R"}}}
    monkeypatch.setattr(
        qwen.engine,
        "run_parallel_generation",
        lambda *args, **kwargs: {
            "field_telemetry": {
                "choice": {
                    "value": "left",
                    "type": "enum",
                    "confidence": 0.75,
                    "top_choices": [
                        {"choice": "left", "probability": 0.75},
                        {"choice": "right", "probability": 0.25},
                    ],
                }
            }
        },
    )

    result = QwenRunner().predict({"message": "hello"}, questions)

    assert result["answers"]["choice"]["probabilities"] == {"left": 0.75, "right": 0.25}


def test_qwen_rejects_incomplete_legacy_choice_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runners.qwen_runner as qwen
    from runners.qwen_runner import QwenRunner

    questions = {"choice": {"type": "choice", "criteria": {"left": "L", "right": "R"}}}
    monkeypatch.setattr(
        qwen.engine,
        "run_parallel_generation",
        lambda *args, **kwargs: {
            "field_telemetry": {
                "choice": {
                    "value": "left",
                    "type": "enum",
                    "confidence": 0.75,
                    "top_choices": [{"choice": "left", "probability": 1.0}],
                }
            }
        },
    )

    with pytest.raises(InvalidGenerationError, match="choice"):
        QwenRunner().predict({"message": "hello"}, questions)
