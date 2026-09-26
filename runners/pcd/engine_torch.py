from __future__ import annotations

import copy
import importlib
import inspect
import json
import math
import random
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

np: Any = importlib.import_module("numpy")

try:
    import torch
except (ImportError, ModuleNotFoundError, OSError):
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except (ImportError, ModuleNotFoundError, OSError):
    AutoModelForCausalLM = None
    AutoTokenizer = None

from runners.pcd.engine_common import (
    InvalidGenerationError,
    TrieNode,
    build_field_telemetry,
    build_legal_trie,
    compile_legal_choices,
    parse_json_value,
)
from runners.pcd.prompt_builder import build_naive_json_prompt, build_parallel_field_prompts
from runners.pcd.schema import StructuredSchema

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEED = 42

_model: Any | None = None
_tokenizer: Any | None = None
_generation_lock = threading.RLock()


@dataclass
class _ForwardStats:
    prefill_ms: float = 0.0
    suffix_ms: float = 0.0
    forward_ms: float = 0.0
    forward_calls: int = 0
    prefill_calls: int = 0
    suffix_calls: int = 0


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("The Torch PCD backend requires torch")
    return torch


def _set_seed() -> None:
    torch_module = _require_torch()
    torch_module.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def _from_pretrained(cls: Any, **kwargs: Any) -> Any:
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise RuntimeError("The Torch PCD backend requires transformers")
    return cls.from_pretrained(MODEL_ID, **kwargs)


def get_engine() -> tuple[Any, Any]:
    global _model, _tokenizer
    with _generation_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer
        torch_module = _require_torch()
        _set_seed()
        tokenizer = _from_pretrained(AutoTokenizer, revision=MODEL_REVISION)
        model = _from_pretrained(
            AutoModelForCausalLM,
            revision=MODEL_REVISION,
            dtype=torch_module.float32,
        )
        device_factory = getattr(torch_module, "device", None)
        device = device_factory("cpu") if callable(device_factory) else "cpu"
        moved_model = model.to(device)
        model = moved_model if moved_model is not None else model
        evaluated_model = model.eval()
        model = evaluated_model if evaluated_model is not None else model
        _model = model
        _tokenizer = tokenizer
        return _model, _tokenizer


def _inference_context() -> Any:
    if torch is not None and callable(getattr(torch, "inference_mode", None)):
        return torch.inference_mode()
    return nullcontext()


def _tensor(values: Sequence[Sequence[int]], *, dtype: Any = None) -> Any:
    if torch is not None:
        try:
            return torch.tensor(values, dtype=dtype if dtype is not None else torch.long)
        except (AttributeError, TypeError, RuntimeError):
            pass
    return np.asarray(values, dtype=np.int64)


def _single_tensor(values: Sequence[int], *, dtype: Any = None) -> Any:
    return _tensor([values], dtype=dtype)[0]


def _invoke_model(model: Any, input_ids: Any, **kwargs: Any) -> Any:
    with _inference_context():
        try:
            if kwargs:
                return model(input_ids, **kwargs)
            return model(input_ids)
        except TypeError as positional_error:
            try:
                if kwargs:
                    return model(input_ids=input_ids, **kwargs)
                return model(input_ids=input_ids)
            except TypeError:
                raise positional_error


def _timed_invoke_model(
    model: Any,
    input_ids: Any,
    stats: _ForwardStats,
    phase: str,
    **kwargs: Any,
) -> Any:
    stats.forward_calls += 1
    if phase == "prefill":
        stats.prefill_calls += 1
    elif phase == "suffix":
        stats.suffix_calls += 1
    started = time.perf_counter()
    try:
        return _invoke_model(model, input_ids, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stats.forward_ms += elapsed_ms
        if phase == "prefill":
            stats.prefill_ms += elapsed_ms
        elif phase == "suffix":
            stats.suffix_ms += elapsed_ms


def _logits_array(output: Any, expected_batch: int) -> np.ndarray:
    value = getattr(output, "logits", output)
    if isinstance(value, (tuple, list)) and not hasattr(value, "dtype"):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise InvalidGenerationError("model logits are not numeric") from exc
    if array.ndim == 1:
        array = array.reshape(1, 1, -1)
    elif array.ndim == 2:
        if expected_batch == 1:
            array = array.reshape(1, 1, -1)
        elif array.shape[0] == expected_batch:
            array = array.reshape(expected_batch, 1, -1)
        else:
            array = array.reshape(1, array.shape[0], -1)
    if array.ndim != 3:
        raise InvalidGenerationError("model logits must have a batch and sequence dimension")
    if array.shape[0] != expected_batch:
        raise InvalidGenerationError("model logits batch does not match legal continuations")
    if array.shape[1] == 0 or array.shape[2] == 0:
        raise InvalidGenerationError("model logits are empty")
    return cast(np.ndarray, array.astype(np.float64, copy=False))


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    try:
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
    except TypeError:
        encoded = tokenizer.encode(prompt)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, int):
        encoded = [encoded]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise InvalidGenerationError(
            "tokenizer returned an invalid prompt encoding", generated_text=prompt
        )
    values: list[int] = []
    for token_id in encoded:
        if isinstance(token_id, bool):
            raise InvalidGenerationError(
                "tokenizer returned an invalid prompt token", generated_text=prompt
            )
        try:
            values.append(int(token_id))
        except (TypeError, ValueError) as exc:
            raise InvalidGenerationError(
                "tokenizer returned an invalid prompt token", generated_text=prompt
            ) from exc
    if not values:
        raise InvalidGenerationError("prompt encoding is empty", generated_text=prompt)
    return values


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        decoded = tokenizer.decode([token_id])
    if not isinstance(decoded, str):
        decoded = str(decoded)
    return decoded


def _stop_ids(tokenizer: Any) -> set[int]:
    result: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        result.add(eos)
    elif isinstance(eos, Sequence) and not isinstance(eos, (str, bytes)):
        result.update(
            int(token_id)
            for token_id in eos
            if isinstance(token_id, int) and not isinstance(token_id, bool)
        )
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        for token in ("<end_of_turn>", "<|im_end|>", "<eos>"):
            try:
                token_id = converter(token)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if isinstance(token_id, int) and token_id > 0:
                result.add(token_id)
    return result


def _validate_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("temperature must be a finite non-negative number")
    return value if value > 0.0 else 1e-4


def _validate_max_tokens(max_tokens: int) -> int:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    return max_tokens


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    if not math.isfinite(maximum):
        raise InvalidGenerationError("model produced non-finite legal logits")
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _field_plans(context: str, schema: StructuredSchema, tokenizer: Any) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for field_name, field_definition, prompt in build_parallel_field_prompts(context, schema):
        prompt = f"{prompt} "
        choices = compile_legal_choices(
            tokenizer,
            field_definition.choices,
            field_name,
            boolean=field_definition.field_type == "boolean",
        )
        prompt_ids = _encode_prompt(tokenizer, prompt)
        plans.append(
            {
                "field_name": field_name,
                "field_definition": field_definition,
                "tokenizer": tokenizer,
                "choices": choices,
                "trie": build_legal_trie(choices),
                "prompt_ids": prompt_ids,
            }
        )
    return plans


def _expand_cache_value(value: Any, repeats: int) -> Any:
    if hasattr(value, "repeat_interleave"):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, tuple):
        return tuple(_expand_cache_value(item, repeats) for item in value)
    if isinstance(value, list):
        return [_expand_cache_value(item, repeats) for item in value]
    return value


def _broadcast_cache(cache: Any, repeats: int) -> Any:
    if cache is None:
        return None
    repeater = getattr(cache, "batch_repeat_interleave", None)
    if callable(repeater):
        try:
            expanded = repeater(repeats)
        except (AttributeError, TypeError, ValueError):
            return None
        return cache if expanded is None else expanded
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        expanded_cache = copy.copy(cache)
        expanded_cache.key_cache = [_expand_cache_value(value, repeats) for value in key_cache]
        expanded_cache.value_cache = [_expand_cache_value(value, repeats) for value in value_cache]
        return expanded_cache
    if isinstance(cache, (tuple, list)):
        expanded_layers = tuple(_expand_cache_value(layer, repeats) for layer in cache)
        return expanded_layers if isinstance(cache, tuple) else list(expanded_layers)
    return None


def _supports_kv_cache(model: Any) -> bool:
    forward = getattr(type(model), "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    return "past_key_values" in parameters


def _common_token_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
    if not sequences:
        return 0
    prefix_length = len(sequences[0])
    first = sequences[0]
    for sequence in sequences[1:]:
        limit = min(prefix_length, len(sequence))
        index = 0
        while index < limit and first[index] == sequence[index]:
            index += 1
        prefix_length = index
        if prefix_length == 0:
            break
    return prefix_length


def _score_cached_plans(
    model: Any,
    plans: Sequence[Mapping[str, Any]],
    temperature: float,
    stats: _ForwardStats,
) -> list[dict[str, Any]] | None:
    if not plans:
        return None
    prompt_sequences = [list(plan["prompt_ids"]) for plan in plans]
    prefix_length = _common_token_prefix_length(prompt_sequences)
    if prefix_length == 0:
        return None
    common_prefix = prompt_sequences[0][:prefix_length]
    cached_plans: list[dict[str, Any]] = []
    suffix_rows: list[list[int]] = []
    for plan, prompt_ids in zip(plans, prompt_sequences):
        suffix_ids = prompt_ids[prefix_length:]
        cached_plan = dict(plan)
        cached_plan["suffix_ids"] = suffix_ids
        cached_plan["suffix_logit_offset"] = len(suffix_ids)
        cached_plans.append(cached_plan)
        suffix_rows.extend(list(suffix_ids) + list(choice.token_ids) for choice in plan["choices"])
    row_count = len(suffix_rows)
    maximum_length = max(len(row) for row in suffix_rows)
    pad_token_id = getattr(plans[0]["tokenizer"], "pad_token_id", None)
    if not isinstance(pad_token_id, int):
        pad_token_id = 0
    padded_rows = [row + [pad_token_id] * (maximum_length - len(row)) for row in suffix_rows]
    try:
        prefill = _timed_invoke_model(
            model,
            _tensor([common_prefix]),
            stats,
            "prefill",
            use_cache=True,
        )
        prefill_logits = _logits_array(prefill, 1)
        cache = getattr(prefill, "past_key_values", None)
        expanded_cache = _broadcast_cache(cache, row_count)
        if expanded_cache is None:
            return None
        suffix_output = _timed_invoke_model(
            model,
            _tensor(padded_rows),
            stats,
            "suffix",
            past_key_values=expanded_cache,
            use_cache=True,
        )
        suffix_logits = _logits_array(suffix_output, row_count)
    except InvalidGenerationError as exc:
        if exc.field is None:
            raise InvalidGenerationError(
                exc.reason,
                field=str(plans[0]["field_name"]),
                generated_text=exc.generated_text,
            ) from exc
        raise
    if prefill_logits.shape[2] != suffix_logits.shape[2]:
        raise InvalidGenerationError(
            "model logits vocabulary changed between prefill and suffix",
            field=str(plans[0]["field_name"]),
        )
    prefill_tail = np.broadcast_to(
        prefill_logits[0, -1, :][None, :],
        (row_count, 1, prefill_logits.shape[2]),
    )
    combined_logits = np.concatenate((prefill_tail, suffix_logits), axis=1)
    logit_bases = [int(plan["suffix_logit_offset"]) for plan in cached_plans]
    return _score_plans(combined_logits, cached_plans, temperature, logit_bases)


def _transition_log_probs(
    row: np.ndarray,
    legal_children: Mapping[int, TrieNode],
    temperature: float,
    field_name: str,
    generated_text: str,
    *,
    include_stop: bool = False,
) -> tuple[dict[int, float], float]:
    if np.any(np.isnan(row)) or np.any(np.isposinf(row)):
        raise InvalidGenerationError(
            "model produced non-finite legal logits",
            field=field_name,
            generated_text=generated_text,
        )
    scaled_row = row / temperature
    legal_values = [float(scaled_row[child_id]) for child_id in legal_children]
    if not all(math.isfinite(value) for value in legal_values):
        raise InvalidGenerationError(
            "model produced non-finite legal logits",
            field=field_name,
            generated_text=generated_text,
        )
    try:
        normalizer = _logsumexp(scaled_row.tolist() if include_stop else legal_values)
    except InvalidGenerationError as exc:
        raise InvalidGenerationError(
            exc.reason,
            field=field_name,
            generated_text=generated_text,
        ) from exc
    child_log_probs = {
        child_id: float(scaled_row[child_id]) - normalizer for child_id in legal_children
    }
    if not include_stop:
        return child_log_probs, float("-inf")
    child_log_mass = _logsumexp(list(child_log_probs.values()))
    if child_log_mass >= 0.0:
        stop_log_prob = float("-inf")
    else:
        stop_mass = -math.expm1(child_log_mass)
        stop_log_prob = math.log(stop_mass) if stop_mass > 0.0 else float("-inf")
    return child_log_probs, stop_log_prob


def _score_plans(
    logits: np.ndarray,
    plans: Sequence[Mapping[str, Any]],
    temperature: float,
    logit_bases: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    row_offset = 0
    for plan_index, plan in enumerate(plans):
        field_name = str(plan["field_name"])
        choices = list(plan["choices"])
        prompt_length = len(plan["prompt_ids"])
        logit_base = logit_bases[plan_index] if logit_bases is not None else prompt_length - 1
        scores: list[float] = []
        for choice_index, choice in enumerate(choices):
            node: TrieNode = plan["trie"]
            score = 0.0
            for position, token_id in enumerate(choice.token_ids):
                legal_children = node.children
                if not legal_children:
                    raise InvalidGenerationError(
                        "legal continuation is incomplete",
                        field=field_name,
                        generated_text=choice.value,
                    )
                row_index = row_offset + choice_index
                logit_position = logit_base + position
                if logit_position < 0 or logit_position >= logits.shape[1]:
                    raise InvalidGenerationError(
                        "model logits do not cover legal continuation",
                        field=field_name,
                        generated_text=choice.value,
                    )
                row = logits[row_index, logit_position]
                vocab_size = row.shape[0]
                if (
                    token_id < 0
                    or token_id >= vocab_size
                    or any(child_id < 0 or child_id >= vocab_size for child_id in legal_children)
                ):
                    raise InvalidGenerationError(
                        "legal token is outside the model vocabulary",
                        field=field_name,
                        generated_text=choice.value,
                    )
                child_log_probs, _ = _transition_log_probs(
                    row,
                    legal_children,
                    temperature,
                    field_name,
                    choice.value,
                    include_stop=bool(node.terminals),
                )
                score += child_log_probs[token_id]
                node = legal_children[token_id]
            if not node.terminals:
                raise InvalidGenerationError(
                    "legal continuation is incomplete",
                    field=field_name,
                    generated_text=choice.value,
                )
            if node.children:
                terminal_position = logit_base + len(choice.token_ids)
                if terminal_position < 0 or terminal_position >= logits.shape[1]:
                    raise InvalidGenerationError(
                        "model logits do not cover legal continuation",
                        field=field_name,
                        generated_text=choice.value,
                    )
                terminal_row = logits[row_offset + choice_index, terminal_position]
                vocab_size = terminal_row.shape[0]
                if any(child_id < 0 or child_id >= vocab_size for child_id in node.children):
                    raise InvalidGenerationError(
                        "legal token is outside the model vocabulary",
                        field=field_name,
                        generated_text=choice.value,
                    )
                _, stop_log_prob = _transition_log_probs(
                    terminal_row,
                    node.children,
                    temperature,
                    field_name,
                    choice.value,
                    include_stop=True,
                )
                score += stop_log_prob
            scores.append(score)
        maximum = max(scores)
        if not math.isfinite(maximum):
            raise InvalidGenerationError(
                "model produced no usable legal continuation", field=field_name
            )
        weights = [math.exp(score - maximum) for score in scores]
        denominator = math.fsum(weights)
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise InvalidGenerationError(
                "legal probabilities could not be normalized", field=field_name
            )
        probabilities = [weight / denominator for weight in weights]
        selected_index = max(range(len(choices)), key=lambda index: probabilities[index])
        results.append(
            {
                "field_name": field_name,
                "field_definition": plan["field_definition"],
                "choices": choices,
                "probabilities": probabilities,
                "selected_index": selected_index,
            }
        )
        row_offset += len(choices)
    return results


def _build_parallel_result(
    context: str,
    schema: StructuredSchema,
    temperature: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    if len(schema) == 0:
        return {
            "mode": "parallel_constrained_calibrated",
            "elapsed_ms": 0.0,
            "prefill_ms": 0.0,
            "suffix_eval_ms": 0.0,
            "forward_ms": 0.0,
            "total_tokens_generated": 0,
            "sequential_forward_passes": 0,
            "is_valid_json": True,
            "schema_match": True,
            "parsed_json": {},
            "field_telemetry": {},
            "has_calibrated_probabilities": True,
            "num_fields": 0,
        }
    model, tokenizer = get_engine()
    plans = _field_plans(context, schema, tokenizer)
    if not plans:
        return {
            "mode": "parallel_constrained_calibrated",
            "elapsed_ms": 0.0,
            "prefill_ms": 0.0,
            "suffix_eval_ms": 0.0,
            "forward_ms": 0.0,
            "total_tokens_generated": 0,
            "sequential_forward_passes": 0,
            "is_valid_json": True,
            "schema_match": True,
            "parsed_json": {},
            "field_telemetry": {},
            "has_calibrated_probabilities": True,
            "num_fields": 0,
        }
    stats = _ForwardStats()
    scored = (
        _score_cached_plans(model, plans, temperature, stats) if _supports_kv_cache(model) else None
    )
    if scored is None:
        flattened = [
            list(plan["prompt_ids"]) + list(choice.token_ids)
            for plan in plans
            for choice in plan["choices"]
        ]
        maximum_length = max(len(sequence) for sequence in flattened)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_token_id, int):
            pad_token_id = 0
        batch_rows = [
            list(sequence) + [pad_token_id] * (maximum_length - len(sequence))
            for sequence in flattened
        ]
        input_ids = _tensor(batch_rows)
        try:
            output = _timed_invoke_model(model, input_ids, stats, "combined")
            logits = _logits_array(output, len(flattened))
        except InvalidGenerationError as exc:
            if exc.field is None and plans:
                raise InvalidGenerationError(
                    exc.reason,
                    field=str(plans[0]["field_name"]),
                    generated_text=exc.generated_text,
                ) from exc
            raise
        scored = _score_plans(logits, plans, temperature)
    parsed_json: dict[str, Any] = {}
    field_telemetry: dict[str, Any] = {}
    for result in scored:
        field_definition = result["field_definition"]
        choices = result["choices"]
        selected_index = result["selected_index"]
        selected = choices[selected_index]
        probability_map = {
            choice.value: result["probabilities"][index] for index, choice in enumerate(choices)
        }
        value = selected.value
        field_telemetry[result["field_name"]] = build_field_telemetry(
            result["field_name"],
            value,
            field_definition.field_type,
            [choice.value for choice in choices],
            probability_map,
        )
        parsed_json[result["field_name"]] = {"value": value, "prob": probability_map[value]}
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    prefill_ms = stats.prefill_ms if stats.prefill_calls else None
    suffix_eval_ms = stats.suffix_ms if stats.suffix_calls else None
    return {
        "mode": "parallel_constrained_calibrated",
        "elapsed_ms": round(elapsed_ms, 2),
        "prefill_ms": prefill_ms,
        "suffix_eval_ms": suffix_eval_ms,
        "forward_ms": stats.forward_ms,
        "total_tokens_generated": 0,
        "sequential_forward_passes": stats.forward_calls,
        "is_valid_json": True,
        "schema_match": True,
        "parsed_json": parsed_json,
        "field_telemetry": field_telemetry,
        "has_calibrated_probabilities": True,
        "num_fields": len(schema),
    }


def run_parallel_generation(
    context: str,
    schema: StructuredSchema,
    temperature: float = 1.0,
) -> dict[str, Any]:
    value = _validate_temperature(temperature)
    with _generation_lock:
        _set_seed()
        return _build_parallel_result(context, schema, value)


def _validate_naive_payload(
    parsed: object, schema: StructuredSchema, raw_text: str
) -> dict[str, Any]:
    if not isinstance(parsed, Mapping):
        raise InvalidGenerationError("generated JSON is not an object", generated_text=raw_text)
    expected = set(schema.fields)
    actual = set(parsed)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        field_name = missing[0] if missing else str(extra[0])
        raise InvalidGenerationError(
            f"generated JSON field IDs do not match schema: missing={missing}, extra={extra}",
            field=field_name,
            generated_text=raw_text,
        )
    normalized: dict[str, Any] = {}
    for field_name, field_definition in schema.fields.items():
        normalized[field_name] = parse_json_value(
            parsed[field_name],
            field_definition.choices,
            field_name,
            boolean=field_definition.field_type == "boolean",
        )
    return normalized


def _parse_generated_json(text: str, schema: StructuredSchema) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise InvalidGenerationError(
            "generated output does not contain a JSON object",
            field=next(iter(schema.fields), None),
            generated_text=text,
        )
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError) as exc:
        raise InvalidGenerationError(
            "generated output is unparseable JSON",
            field=next(iter(schema.fields), None),
            generated_text=text,
        ) from exc
    return _validate_naive_payload(parsed, schema, text)


def _sample_token_id(
    logits: np.ndarray,
    field_name: str | None = None,
    temperature: float = 1.0,
    rng: Any = None,
) -> int:
    value = _validate_temperature(temperature)
    if logits.ndim == 3:
        if logits.shape[0] == 0:
            raise InvalidGenerationError("model returned empty logits", field=field_name)
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise InvalidGenerationError("model returned empty logits", field=field_name)
    row = logits[-1]
    if not np.all(np.isfinite(row)):
        raise InvalidGenerationError("model returned non-finite logits", field=field_name)
    if value <= 1e-4:
        return int(np.argmax(row))
    scaled = row / value
    maximum = float(np.max(scaled))
    if not math.isfinite(maximum):
        raise InvalidGenerationError("model produced non-finite legal logits", field=field_name)
    weights = np.exp(scaled - maximum)
    denominator = math.fsum(float(weight) for weight in weights)
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise InvalidGenerationError("model sampling probabilities are invalid", field=field_name)
    probabilities = weights / denominator
    generator = rng if rng is not None else np.random.default_rng(SEED)
    try:
        return int(generator.choice(len(probabilities), p=probabilities))
    except (TypeError, ValueError) as exc:
        raise InvalidGenerationError("model sampling failed", field=field_name) from exc


def _next_token_id(
    logits: np.ndarray,
    field_name: str | None = None,
    temperature: float = 1.0,
    rng: Any = None,
) -> int:
    return _sample_token_id(logits, field_name, temperature, rng)


def _naive_events(
    context: str,
    schema: StructuredSchema,
    max_tokens: int,
    temperature: float,
) -> Iterator[dict[str, Any]]:
    model, tokenizer = get_engine()
    prompt = build_naive_json_prompt(context, schema)
    prompt_ids = _encode_prompt(tokenizer, prompt)
    current_ids = list(prompt_ids)
    current_text = "{\n  "
    generated_count = 0
    started = time.perf_counter()
    stop_ids = _stop_ids(tokenizer)
    rng = np.random.default_rng(SEED)
    for _ in range(max_tokens):
        output = _invoke_model(model, _tensor([current_ids]))
        logits = _logits_array(output, 1)
        next_token = _next_token_id(logits, next(iter(schema.fields), None), temperature, rng)
        if next_token in stop_ids:
            break
        token_text = _decode_token(tokenizer, next_token)
        current_text += token_text
        current_ids.append(next_token)
        generated_count += 1
        yield {
            "type": "token",
            "token": token_text,
            "accumulated": current_text,
            "token_count": generated_count,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
        if current_text.count("{") == current_text.count("}"):
            break
    if generated_count == 0:
        raise InvalidGenerationError(
            "generation produced no output tokens",
            field=next(iter(schema.fields), None),
            generated_text=current_text,
        )
    parsed = _parse_generated_json(current_text, schema)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    tokens_per_second = generated_count / (elapsed_ms / 1000.0) if elapsed_ms else 0.0
    yield {
        "type": "done",
        "result": {
            "mode": "naive_autoregressive",
            "elapsed_ms": round(elapsed_ms, 2),
            "total_tokens": generated_count,
            "tokens_per_second": round(tokens_per_second, 1),
            "sequential_forward_passes": generated_count,
            "is_valid_json": True,
            "schema_match": True,
            "raw_text": current_text,
            "parsed_json": parsed,
            "parse_error": None,
            "missing_keys": [],
            "invalid_enums": [],
            "has_calibrated_probabilities": False,
        },
    }


def run_naive_generation(
    context: str,
    schema: StructuredSchema,
    max_tokens: int = 700,
    temperature: float = 0.2,
) -> dict[str, Any]:
    limit = _validate_max_tokens(max_tokens)
    value = _validate_temperature(temperature)
    with _generation_lock:
        _set_seed()
        result: dict[str, Any] | None = None
        for event in _naive_events(context, schema, limit, value):
            if event["type"] == "done":
                result = event["result"]
        if result is None:
            raise InvalidGenerationError("naive generation did not produce a result")
        return result


def stream_naive_generation(
    context: str,
    schema: StructuredSchema,
    max_tokens: int = 700,
    temperature: float = 0.2,
) -> Iterator[dict[str, Any]]:
    limit = _validate_max_tokens(max_tokens)
    value = _validate_temperature(temperature)
    with _generation_lock:
        _set_seed()
        yield from _naive_events(context, schema, limit, value)


run_rlcd_generation = run_parallel_generation


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "SEED",
    "get_engine",
    "run_naive_generation",
    "run_parallel_generation",
    "run_rlcd_generation",
    "stream_naive_generation",
]
