from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any

from benchmark.usage import UsageCounter

if TYPE_CHECKING:
    from runners.base import BaseRunner


def measure_call(clock: Callable[[], float], call: Callable[[], object]) -> tuple[object, float]:
    started = clock()
    result = call()
    return result, float(clock() - started)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validated_sizes(phases: Sequence[int]) -> tuple[int, ...]:
    if isinstance(phases, (str, bytes, bytearray)) or not isinstance(phases, Sequence):
        raise TypeError("size sequence must contain positive integers")
    sizes = tuple(_positive_integer(size, "size") for size in phases)
    if not sizes:
        raise ValueError("size sequence must not be empty")
    if len(set(sizes)) != len(sizes):
        raise ValueError("size sequence must not contain duplicates")
    return sizes


def _request(size: int) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    state = {"size": size}
    questions = {
        f"q{index}": {"type": "noul", "instructions": "Return yes or no."} for index in range(size)
    }
    return state, questions


def _record_result(counter: UsageCounter, phase: str, result: object) -> None:
    if not isinstance(result, Mapping):
        raise TypeError("runner result must be a mapping")
    usage = result.get("_usage")
    if usage is None:
        usage = {}
    counter.record(phase, usage)


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def speed_scaling(
    runner: BaseRunner,
    phases: Sequence[int],
    repetitions: int = 10,
) -> dict[str, dict[str, float]]:
    sizes = _validated_sizes(phases)
    repetition_count = _positive_integer(repetitions, "repetitions")
    predict = getattr(runner, "predict", None)
    if not callable(predict):
        raise TypeError("runner must provide a callable predict method")
    counter = UsageCounter()
    result: dict[str, dict[str, float]] = {}
    for size in sizes:
        state, questions = _request(size)
        for _ in range(2):
            response = predict(
                deepcopy(state),
                deepcopy(questions),
                phase="warmup",
            )
            _record_result(counter, "warmup", response)
        speed_calls_before = counter.as_dict()["speed"]["calls"]
        elapsed_samples: list[float] = []
        for _ in range(repetition_count):
            call_state = deepcopy(state)
            call_questions = deepcopy(questions)
            response, elapsed = measure_call(
                perf_counter,
                partial(predict, call_state, call_questions, phase="speed"),
            )
            _record_result(counter, "speed", response)
            elapsed_samples.append(elapsed)
        sample_count = counter.as_dict()["speed"]["calls"] - speed_calls_before
        result[str(size)] = {
            "p50": _percentile(elapsed_samples, 0.50),
            "p95": _percentile(elapsed_samples, 0.95),
            "sample_count": float(sample_count),
        }
    return result
