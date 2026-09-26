from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, cast

_LOGIT_BOUND = 745.0
_LOG_TEMPERATURE_LOW = -10.0
_LOG_TEMPERATURE_HIGH = 10.0


def _materialize(value: object, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be an array-like sequence")
    if isinstance(value, Sequence):
        return list(cast(Sequence[object], value))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        materialized = cast(object, tolist())
        if isinstance(materialized, Sequence) and not isinstance(
            materialized, (str, bytes, bytearray)
        ):
            return list(cast(Sequence[object], materialized))
    raise TypeError(f"{name} must be an array-like sequence")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _numeric_array(value: object, name: str) -> list[float]:
    return [
        _finite_number(item, f"{name}[{index}]")
        for index, item in enumerate(_materialize(value, name))
    ]


def _probability_array(value: object, name: str) -> list[float]:
    values = _numeric_array(value, name)
    for index, probability in enumerate(values):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name}[{index}] must be in [0, 1]")
    return values


def _label_array(value: object, name: str) -> list[int]:
    labels: list[int] = []
    for index, item in enumerate(_materialize(value, name)):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"{name}[{index}] must be integer 0 or 1")
        label = int(item)
        if label not in (0, 1):
            raise ValueError(f"{name}[{index}] must be integer 0 or 1")
        labels.append(label)
    return labels


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    _integer_result_number(number, name)
    return number


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer")
    result = int(value)
    _integer_result_number(result, "seed")
    return result


def _cluster_groups(value: object) -> list[list[int]]:
    values = _materialize(value, "cluster_ids")
    groups: dict[object, list[int]] = {}
    for index, identifier in enumerate(values):
        _validate_cluster_id(identifier, f"cluster_ids[{index}]")
        try:
            groups.setdefault(identifier, []).append(index)
        except TypeError as error:
            raise TypeError(f"cluster_ids[{index}] must be hashable") from error
    return list(groups.values())


def _validate_cluster_id(value: object, name: str) -> None:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-empty identifier")
    if isinstance(value, (str, bytes, bytearray)) and (not value or not value.strip()):
        raise ValueError(f"{name} must be a non-empty identifier")
    if isinstance(value, Real):
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"{name} must be a non-empty identifier") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a non-empty identifier")
    try:
        empty = len(cast(Any, value)) == 0
    except (AttributeError, TypeError):
        empty = False
    if empty:
        raise ValueError(f"{name} must be a non-empty identifier")
    try:
        hash(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be hashable") from error


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one observation is required")
    divisor = float(len(values))
    try:
        result = math.fsum(value / divisor for value in values)
    except (OverflowError, ValueError) as error:
        raise ValueError("numeric values do not produce a finite mean") from error
    if not math.isfinite(result):
        raise ValueError("numeric values do not produce a finite mean")
    return result


def _delta(a_values: Sequence[float], b_values: Sequence[float]) -> float:
    result = _mean(b_values) - _mean(a_values)
    if not math.isfinite(result):
        raise ValueError("numeric values do not produce a finite delta")
    return result


def _cluster_values(
    a_rows: object, b_rows: object, cluster_ids: object
) -> tuple[list[float], list[float], list[list[int]], float]:
    a_values = _numeric_array(a_rows, "a_rows")
    b_values = _numeric_array(b_rows, "b_rows")
    groups = _cluster_groups(cluster_ids)
    if not a_values or not b_values:
        raise ValueError("a_rows and b_rows must not be empty")
    if len(a_values) != len(b_values) or len(a_values) != sum(map(len, groups)):
        raise ValueError("a_rows, b_rows, and cluster_ids must have equal lengths")
    return a_values, b_values, groups, _delta(a_values, b_values)


def _cluster_statistics(
    a_rows: object, b_rows: object, cluster_ids: object
) -> tuple[list[float], list[int], float]:
    a_values, b_values, groups, point_delta = _cluster_values(a_rows, b_rows, cluster_ids)
    cluster_deltas = [
        _delta([a_values[index] for index in indices], [b_values[index] for index in indices])
        for indices in groups
    ]
    sizes = [len(indices) for indices in groups]
    return cluster_deltas, sizes, point_delta


def _weighted_delta(
    cluster_deltas: Sequence[float],
    sizes: Sequence[int],
    indices: Sequence[int],
    signs: Sequence[int] | None = None,
) -> float:
    total_size = sum(sizes[index] for index in indices)
    if total_size <= 0:
        raise ValueError("cluster sizes must be positive")
    if signs is not None and len(signs) != len(indices):
        raise ValueError("permutation signs must match cluster count")
    try:
        if signs is None:
            result = math.fsum(
                cluster_deltas[index] * (sizes[index] / total_size) for index in indices
            )
        else:
            result = math.fsum(
                cluster_deltas[index] * (sizes[index] / total_size) * sign
                for index, sign in zip(indices, signs, strict=True)
            )
    except (OverflowError, ValueError) as error:
        raise ValueError("cluster values do not produce a finite delta") from error
    if not math.isfinite(result):
        raise ValueError("cluster values do not produce a finite delta")
    return result


def _percentile(samples: Sequence[float], quantile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("at least one replicate is required")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    result = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    if not math.isfinite(result):
        raise ValueError("replicate values do not produce a finite percentile")
    return float(result)


def _result_number(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _integer_result_number(value: int, name: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    return _result_number(result, name)


def paired_cluster_bootstrap(
    a_rows: Sequence[float],
    b_rows: Sequence[float],
    cluster_ids: Sequence[object],
    replicates: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    cluster_deltas, sizes, point_delta = _cluster_statistics(a_rows, b_rows, cluster_ids)
    replicate_count = _positive_integer(replicates, "replicates")
    seed_value = _seed(seed)
    generator = random.Random(seed_value)
    cluster_count = len(cluster_deltas)
    sampled_indices = generator.choices(range(cluster_count), k=replicate_count * cluster_count)
    replicate_deltas = [
        _weighted_delta(
            cluster_deltas,
            sizes,
            sampled_indices[start : start + cluster_count],
        )
        for start in range(0, len(sampled_indices), cluster_count)
    ]
    return {
        "point_delta": _result_number(point_delta, "point_delta"),
        "ci_low": _result_number(_percentile(replicate_deltas, 0.025), "ci_low"),
        "ci_high": _result_number(_percentile(replicate_deltas, 0.975), "ci_high"),
        "replicates": _integer_result_number(replicate_count, "replicates"),
        "seed": _integer_result_number(seed_value, "seed"),
    }


def paired_cluster_permutation(
    a_rows: Sequence[float],
    b_rows: Sequence[float],
    cluster_ids: Sequence[object],
    replicates: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    cluster_deltas, sizes, point_delta = _cluster_statistics(a_rows, b_rows, cluster_ids)
    replicate_count = _positive_integer(replicates, "replicates")
    seed_value = _seed(seed)
    generator = random.Random(seed_value)
    extreme_count = 0
    observed = abs(point_delta)
    cluster_count = len(cluster_deltas)
    for _ in range(replicate_count):
        signs = [generator.choice((-1, 1)) for _ in range(cluster_count)]
        replicate_delta = _weighted_delta(cluster_deltas, sizes, range(cluster_count), signs)
        if abs(replicate_delta) >= observed:
            extreme_count += 1
    p_value = (extreme_count + 1) / (replicate_count + 1)
    return {
        "p_value": _result_number(float(p_value), "p_value"),
        "replicates": _integer_result_number(replicate_count, "replicates"),
        "seed": _integer_result_number(seed_value, "seed"),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = _numeric_array(p_values, "p_values")
    for index, value in enumerate(values):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p_values[{index}] must be in [0, 1]")
    if not values:
        return []
    count = len(values)
    order = sorted(range(count), key=lambda index: (values[index], index))
    adjusted: list[float] = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * float(count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _logit(probability: float) -> float:
    if probability <= 0.0:
        return -_LOGIT_BOUND
    if probability >= 1.0:
        return _LOGIT_BOUND
    return math.log(probability) - math.log1p(-probability)


def _softplus(value: float) -> float:
    if value >= 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _binary_nll(logit: float, label: int, temperature: float) -> float:
    scaled = logit / temperature
    return _softplus(scaled) - label * scaled


def fit_temperature(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    probability_values = _probability_array(probabilities, "probabilities")
    label_values = _label_array(labels, "labels")
    if not probability_values:
        raise ValueError("probabilities and labels must not be empty")
    if len(probability_values) != len(label_values):
        raise ValueError("probabilities and labels must have equal lengths")
    if len(set(label_values)) < 2:
        raise ValueError("temperature calibration requires both binary classes")
    logits = [_logit(probability) for probability in probability_values]
    if len(set(logits)) < 2:
        raise ValueError("temperature calibration is degenerate when probabilities are constant")
    ordered = sorted(zip(logits, label_values, strict=True), key=lambda pair: (pair[0], pair[1]))

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return math.fsum(_binary_nll(logit, label, temperature) for logit, label in ordered)

    best_log_temperature = 0.0
    best_objective = objective(best_log_temperature)
    left = _LOG_TEMPERATURE_LOW
    right = _LOG_TEMPERATURE_HIGH
    for _ in range(100):
        first = (2.0 * left + right) / 3.0
        second = (left + 2.0 * right) / 3.0
        first_objective = objective(first)
        second_objective = objective(second)
        if first_objective < best_objective:
            best_log_temperature = first
            best_objective = first_objective
        if second_objective < best_objective:
            best_log_temperature = second
            best_objective = second_objective
        if first_objective <= second_objective:
            right = second
        else:
            left = first
    final_log_temperature = (left + right) / 2.0
    final_objective = objective(final_log_temperature)
    if final_objective < best_objective:
        best_log_temperature = final_log_temperature
    temperature = math.exp(best_log_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature calibration did not produce a finite positive value")
    return temperature


def apply_temperature(probabilities: Sequence[float], temperature: float) -> list[float]:
    probability_values = _probability_array(probabilities, "probabilities")
    if not probability_values:
        raise ValueError("probabilities must not be empty")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be a finite positive number")
    try:
        temperature_value = float(cast(Real, temperature))
    except (OverflowError, ValueError) as error:
        raise ValueError("temperature must be a finite positive number") from error
    if not math.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("temperature must be a finite positive number")
    if temperature_value == 1.0:
        return probability_values
    result: list[float] = []
    for probability in probability_values:
        if probability == 0.0:
            result.append(0.0)
            continue
        if probability == 1.0:
            result.append(1.0)
            continue
        scaled = _logit(probability) / temperature_value
        if scaled >= _LOGIT_BOUND:
            result.append(1.0)
        elif scaled <= -_LOGIT_BOUND:
            result.append(0.0)
        elif scaled >= 0.0:
            result.append(1.0 / (1.0 + math.exp(-scaled)))
        else:
            exponential = math.exp(scaled)
            result.append(exponential / (1.0 + exponential))
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in result):
        raise ValueError("temperature application did not produce finite probabilities")
    return result
