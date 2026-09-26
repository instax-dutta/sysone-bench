from __future__ import annotations

import csv
import io
import math
import re
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from benchmark.canonical import canonical_json
from benchmark.storage import write_json_exclusive

FIGURE_FAMILIES = (
    "accuracy",
    "paired_effects",
    "calibration",
    "risk_coverage",
    "efficiency",
    "multilingual",
    "router",
)
EMPTY_FAMILY_REASON = "source comparison contains no attributable plottable rows for this family"
UNCERTAINTY_METHOD = "paired_state_cluster_bootstrap_20000_seed_42"
_MODEL_ORDER = ("laya", "jev", "qwen-pcd", "router", "a", "b")
_METRIC_ORDER = (
    "p50_ms",
    "p95_ms",
    "throughput",
    "p50_seconds",
    "p95_seconds",
    "english_delta",
    "multilingual_delta",
    "route_count",
)
_BASE_CSV_COLUMNS = (
    "checksum",
    "dataset_version",
    "manifest_digest",
    "metric_definition",
    "model",
    "plotted_values",
    "run_id",
    "sample_count",
    "source_checksums",
    "source_kind",
    "source_run_ids",
    "split",
    "suite_id",
    "uncertainty_method",
    "unit",
    "value",
)
_GENERIC_METADATA_KEYS = frozenset(
    {
        "cases",
        "checksum",
        "checksums",
        "dataset_version",
        "decisions",
        "manifest_digest",
        "metric",
        "metric_definition",
        "model",
        "n",
        "primitive",
        "route",
        "route_reason",
        "route_reason_count",
        "route_reason_counts",
        "route_reasons",
        "run_id",
        "sample_count",
        "source_checksums",
        "source_kind",
        "source_run_ids",
        "split",
        "suite_id",
        "uncertainty_method",
        "unit",
        "value",
    }
)
_PRIMITIVES = frozenset({"choice", "noul", "score"})
_PROBABILITY_PRIMITIVES = frozenset({"choice", "noul"})
_ROUTER_MARKERS = frozenset({"laya-router", "local-router", "router"})


@dataclass(frozen=True)
class _Source:
    label: str
    run_id: str
    checksum: str
    model: dict[str, Any]
    is_router: bool


@dataclass(frozen=True)
class _Context:
    manifest_digest: str
    dataset_version: str
    sources: tuple[_Source, ...]
    synthetic: bool
    evaluation_cases: int | None
    evaluation_decisions: int | None


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result: dict[str, Any] = {}
    for key, child in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        result[key] = deepcopy(child)
    return result


def _sequence(value: object, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return [deepcopy(child) for child in cast(Sequence[object], value)]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    text = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a full lowercase SHA-256 digest")
    return text


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name)


def _number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _ordered_labels(labels: set[str]) -> tuple[str, ...]:
    preferred = tuple(label for label in _MODEL_ORDER if label in labels)
    return (*preferred, *tuple(sorted(labels - set(preferred))))


def _model_is_router(model: Mapping[str, Any], name: str) -> bool:
    if "is_router" in model:
        value = model["is_router"]
        if not isinstance(value, bool):
            raise TypeError(f"{name}.is_router must be a boolean")
        if value:
            return True
    for field in ("model", "runner", "serving", "variant"):
        if field not in model:
            continue
        marker = _text(model[field], f"{name}.{field}").casefold()
        if marker in _ROUTER_MARKERS or marker.endswith("-router"):
            return True
    if "identity_scope" in model:
        scope = _text(model["identity_scope"], f"{name}.identity_scope")
        if scope == "per-checkpoint":
            return True
    return False


def _explicit_identity(value: Mapping[str, Any], name: str) -> tuple[str, str] | None:
    containers = [value]
    for key in ("provenance", "source"):
        nested = value.get(key)
        if nested is not None:
            containers.append(_mapping(nested, f"{name}.{key}"))
    run_ids: list[str] = []
    checksums: list[str] = []
    for container in containers:
        for key in ("run_id", "source_run_id"):
            if key in container:
                run_ids.append(_text(container[key], f"{name}.{key}"))
        for key in ("checksum", "checksums_sha256", "source_checksum"):
            if key in container:
                checksums.append(_digest(container[key], f"{name}.{key}"))
    if not run_ids and not checksums:
        return None
    if not run_ids or not checksums:
        raise ValueError(f"{name} must contain both router run_id and checksum")
    if len(set(run_ids)) != 1 or len(set(checksums)) != 1:
        raise ValueError(f"{name} contains conflicting router run or checksum identities")
    return run_ids[0], checksums[0]


def _comparison_models(comparison: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    value = comparison.get("models")
    if value is None:
        return {}
    models = _mapping(value, "comparison models")
    result: dict[str, dict[str, Any]] = {}
    for raw_label in sorted(models):
        label = _text(raw_label, "comparison model label")
        result[label] = _mapping(models[raw_label], f"comparison model {label}")
    return result


def _context(comparison: Mapping[str, Any]) -> _Context:
    runs_value = comparison.get("runs")
    sources_value = comparison.get("sources")
    if runs_value is not None and sources_value is not None:
        raise ValueError("comparison must use exactly one source identity schema")
    if runs_value is None and sources_value is None:
        raise ValueError("comparison must contain runs or sources")
    synthetic = runs_value is not None
    source_values = _mapping(runs_value if synthetic else sources_value, "comparison sources")
    models = _comparison_models(comparison)
    missing_models = set(models) - set(source_values)
    if missing_models:
        raise ValueError("comparison models must be bound to comparison sources")
    sources: list[_Source] = []
    seen_run_ids: set[str] = set()
    for raw_label in sorted(source_values):
        label = _text(raw_label, "comparison source label")
        source = _mapping(source_values[raw_label], f"comparison source {label}")
        if synthetic:
            run_id = _text(source.get("run_id"), f"source {label} run_id")
            checksum = _digest(source.get("checksum"), f"source {label} checksum")
        else:
            run_id = _text(source.get("run_id"), f"source {label} run_id")
            checksum = _digest(source.get("checksums_sha256"), f"source {label} checksums_sha256")
        if run_id in seen_run_ids:
            raise ValueError("comparison source run IDs must be unique")
        seen_run_ids.add(run_id)
        model = source if synthetic else models.get(label, {})
        model_identity = _explicit_identity(model, f"comparison model {label}")
        if model_identity is not None and model_identity != (run_id, checksum):
            raise ValueError(f"comparison model {label} source identity differs from sources")
        sources.append(
            _Source(
                label,
                run_id,
                checksum,
                model,
                label == "router" or _model_is_router(model, f"comparison model {label}"),
            )
        )
    if not sources:
        raise ValueError("comparison sources must not be empty")
    router_value = comparison.get("router")
    router_identity = (
        _explicit_identity(_mapping(router_value, "comparison router"), "router")
        if isinstance(router_value, Mapping)
        else None
    )
    if router_identity is not None:
        router_run_id, router_checksum = router_identity
        matching = [source for source in sources if source.run_id == router_run_id]
        checksum_matches = [source for source in sources if source.checksum == router_checksum]
        if matching and matching[0].checksum != router_checksum:
            raise ValueError("router checksum differs from the comparison source")
        if checksum_matches and (not matching or checksum_matches[0].run_id != router_run_id):
            raise ValueError("router run identity differs from the comparison source")
        if matching:
            sources = [
                replace(source, is_router=True) if source.run_id == router_run_id else source
                for source in sources
            ]
        else:
            sources.append(_Source("router", router_run_id, router_checksum, {}, True))
    router_sources = [source for source in sources if source.is_router]
    if len(router_sources) > 1:
        raise ValueError("comparison must identify at most one router source")
    ordered = _ordered_labels({source.label for source in sources})
    by_label = {source.label: source for source in sources}
    sources = [by_label[label] for label in ordered]
    if synthetic:
        dataset = _mapping(comparison.get("dataset"), "comparison dataset")
        manifest_digest = _digest(dataset.get("manifest_digest"), "dataset manifest digest")
        dataset_version = _text(dataset.get("version"), "dataset version")
        evaluation_cases = _optional_nonnegative_integer(
            dataset.get("evaluation_cases"), "dataset evaluation_cases"
        )
        evaluation_decisions = _optional_nonnegative_integer(
            dataset.get("evaluation_decisions"), "dataset evaluation_decisions"
        )
    else:
        manifest = _mapping(comparison.get("manifest"), "comparison manifest")
        manifest_digest = _digest(manifest.get("digest"), "comparison manifest digest")
        dataset_version = _text(comparison.get("dataset_version"), "dataset version")
        evaluation_cases = None
        evaluation_decisions = None
    return _Context(
        manifest_digest,
        dataset_version,
        tuple(sources),
        synthetic,
        evaluation_cases,
        evaluation_decisions,
    )


def _source_map(context: _Context) -> dict[str, _Source]:
    return {source.label: source for source in context.sources}


def _base_sources(context: _Context) -> tuple[_Source, ...]:
    return tuple(source for source in context.sources if not source.is_router)


def _source_kind(sources: Sequence[_Source]) -> str:
    if all(source.is_router for source in sources):
        return "router"
    if all(not source.is_router for source in sources):
        return "base"
    raise ValueError("figure row source kinds must be homogeneous")


def _base_row(
    context: _Context,
    source_labels: Sequence[str],
    metric_definition: str,
    sample_count: int | None,
    split: str,
) -> dict[str, Any]:
    if split not in {"calibration", "evaluation"}:
        raise ValueError("figure row split must be calibration or evaluation")
    sources = _source_map(context)
    try:
        selected = [sources[label] for label in source_labels]
    except KeyError as error:
        raise ValueError(f"figure row references unknown source: {error.args[0]}") from None
    if not selected:
        raise ValueError("figure row must reference at least one source run")
    if len({source.label for source in selected}) != len(selected):
        raise ValueError("figure row source runs must be unique")
    row: dict[str, Any] = {
        "dataset_version": context.dataset_version,
        "manifest_digest": context.manifest_digest,
        "metric_definition": _text(metric_definition, "metric_definition"),
        "sample_count": _optional_nonnegative_integer(sample_count, "sample_count"),
        "source_checksums": [source.checksum for source in selected],
        "source_kind": _source_kind(selected),
        "source_run_ids": [source.run_id for source in selected],
        "split": split,
        "uncertainty_method": UNCERTAINTY_METHOD,
    }
    if len(selected) == 1:
        row["checksum"] = selected[0].checksum
        row["run_id"] = selected[0].run_id
    return row


def _sample_count(
    value: Mapping[str, Any], name: str, keys: Sequence[str] = ("sample_count", "n")
) -> int | None:
    for key in keys:
        if key in value:
            return _nonnegative_integer(value[key], f"{name} {key}")
    return None


def _accuracy_rows(comparison: Mapping[str, Any], context: _Context) -> list[dict[str, Any]]:
    suites = _mapping(comparison.get("suites"), "comparison suites")
    rows: list[dict[str, Any]] = []
    for suite_id in sorted(suites):
        suite = _mapping(suites[suite_id], f"suite {suite_id}")
        if context.synthetic:
            sample_count = _nonnegative_integer(
                suite.get("decisions"), f"suite {suite_id} decisions"
            )
            for source in _base_sources(context):
                if source.label not in suite:
                    raise ValueError(f"suite {suite_id} is missing source value {source.label}")
                if suite[source.label] is None:
                    continue
                value = _number(suite[source.label], f"suite {suite_id} {source.label}")
                row = _base_row(
                    context,
                    [source.label],
                    "Micro accuracy across reviewed decisions in the evaluation split",
                    sample_count,
                    "evaluation",
                )
                row.update(
                    {
                        "metric": "accuracy",
                        "model": source.label,
                        "plotted_values": {"value": value},
                        "suite_id": suite_id,
                        "unit": "fraction",
                        "value": value,
                    }
                )
                rows.append(row)
            continue
        for source in _base_sources(context):
            if source.label not in suite:
                raise ValueError(f"suite {suite_id} is missing source summary {source.label}")
            summary = _mapping(suite[source.label], f"suite {suite_id} {source.label}")
            if "accuracy" not in summary:
                raise ValueError(f"suite {suite_id} {source.label} is missing accuracy")
            if summary["accuracy"] is None:
                continue
            value = _number(summary["accuracy"], f"suite {suite_id} {source.label} accuracy")
            sample_count = _nonnegative_integer(
                summary.get("decisions"), f"suite {suite_id} {source.label} decisions"
            )
            row = _base_row(
                context,
                [source.label],
                "Micro accuracy across reviewed decisions in the evaluation split",
                sample_count,
                "evaluation",
            )
            row.update(
                {
                    "metric": "accuracy",
                    "model": source.label,
                    "plotted_values": {"value": value},
                    "suite_id": suite_id,
                    "unit": "fraction",
                    "value": value,
                }
            )
            rows.append(row)
    return rows


def _paired_effects_rows(comparison: Mapping[str, Any], context: _Context) -> list[dict[str, Any]]:
    suites = _mapping(comparison.get("suites"), "comparison suites")
    source_labels = tuple(source.label for source in _base_sources(context))
    pair: tuple[str, ...]
    if context.synthetic:
        pair = tuple(label for label in ("laya", "jev") if label in source_labels)
        point_key = "delta"
    else:
        pair = tuple(label for label in ("a", "b") if label in source_labels)
        point_key = "accuracy_delta_b_minus_a"
    if len(pair) != 2:
        return []
    rows: list[dict[str, Any]] = []
    for suite_id in sorted(suites):
        suite = _mapping(suites[suite_id], f"suite {suite_id}")
        if point_key not in suite:
            continue
        if suite[point_key] is None:
            continue
        point = _number(suite[point_key], f"suite {suite_id} {point_key}")
        if context.synthetic:
            sample_count = _nonnegative_integer(
                suite.get("decisions"), f"suite {suite_id} decisions"
            )
            definition = "Jev evaluation accuracy minus Laya evaluation accuracy"
        else:
            sample_count = _nonnegative_integer(
                _mapping(suite[pair[0]], f"suite {suite_id} {pair[0]}").get("decisions"),
                f"suite {suite_id} decisions",
            )
            definition = "Model B evaluation accuracy minus model A evaluation accuracy"
        plotted_values: dict[str, Any] = {"estimate": point}
        row = _base_row(context, pair, definition, sample_count, "evaluation")
        row.update(
            {
                "contrast": f"{pair[1]}_minus_{pair[0]}",
                "metric": point_key,
                "model": f"{pair[1]}_minus_{pair[0]}",
                "plotted_values": plotted_values,
                "suite_id": suite_id,
                "unit": "fraction",
                "value": point,
            }
        )
        for key in ("ci_low", "ci_high", "p_value", "p_value_holm"):
            if key in suite:
                value = _number(suite[key], f"suite {suite_id} {key}")
                plotted_values[key] = value
                row[key] = value
        rows.append(row)
    return rows


def _metric_definition(primitive: str, metric: str) -> str:
    if metric == "ece_raw":
        return f"Expected calibration error for {primitive} using raw evaluation probabilities"
    if metric == "ece_fitted":
        return (
            f"Expected calibration error for {primitive} using calibration-fitted evaluation "
            "probabilities"
        )
    return f"Reported {metric} for {primitive}"


def _source_labels(
    raw: Mapping[str, Any],
    context: _Context,
    family: str,
    index: int | str,
    *,
    default_base: bool = False,
) -> tuple[str, tuple[str, ...]]:
    sources = _source_map(context)
    by_run_id = {source.run_id: source for source in context.sources}
    single_run_present = "run_id" in raw or "checksum" in raw
    list_run_present = "source_run_ids" in raw or "source_checksums" in raw
    selected: tuple[_Source, ...] = ()
    if single_run_present:
        if "run_id" not in raw or "checksum" not in raw:
            raise ValueError(f"{family} row {index} attribution requires run_id and checksum")
        run_id = _text(raw["run_id"], f"{family} row {index} run_id")
        checksum = _digest(raw["checksum"], f"{family} row {index} checksum")
        try:
            source = by_run_id[run_id]
        except KeyError:
            raise ValueError(f"{family} row {index} references an unknown source run") from None
        if source.checksum != checksum:
            raise ValueError(f"{family} row {index} checksum conflicts with source run")
        selected = (source,)
    if list_run_present:
        if "source_run_ids" not in raw or "source_checksums" not in raw:
            raise ValueError(
                f"{family} row {index} attribution requires source_run_ids and source_checksums"
            )
        raw_run_ids = _sequence(raw["source_run_ids"], f"{family} row {index} source_run_ids")
        run_ids = [_text(run_id, f"{family} row {index} source_run_ids") for run_id in raw_run_ids]
        checksums = [
            _digest(checksum, f"{family} row {index} source_checksums")
            for checksum in _sequence(
                raw["source_checksums"], f"{family} row {index} source_checksums"
            )
        ]
        if not run_ids or len(run_ids) != len(checksums):
            raise ValueError(
                f"{family} row {index} source attribution must be nonempty and aligned"
            )
        listed: list[_Source] = []
        for run_id, checksum in zip(run_ids, checksums, strict=True):
            try:
                source = by_run_id[run_id]
            except KeyError:
                raise ValueError(f"{family} row {index} references an unknown source run") from None
            if source.checksum != checksum:
                raise ValueError(f"{family} row {index} source checksum conflicts with source run")
            listed.append(source)
        if len({source.run_id for source in listed}) != len(listed):
            raise ValueError(f"{family} row {index} source attribution contains duplicate runs")
        listed_sources = tuple(listed)
        if selected and selected != listed_sources:
            raise ValueError(f"{family} row {index} source attribution conflicts")
        selected = listed_sources
    model = _text(raw["model"], f"{family} row {index} model") if "model" in raw else "comparison"
    explicit = single_run_present or list_run_present
    if not explicit:
        if model in sources:
            selected = (sources[model],)
        elif default_base and model == "comparison":
            selected = _base_sources(context)
        else:
            raise ValueError(f"{family} row {index} is missing explicit source attribution")
    if model in sources and selected != (sources[model],):
        raise ValueError(f"{family} row {index} model conflicts with source attribution")
    if not selected:
        raise ValueError(f"{family} row {index} is missing source attribution")
    if family == "router":
        if any(not source.is_router for source in selected):
            raise ValueError("router family cannot attribute a base-model source")
    elif any(source.is_router for source in selected):
        raise ValueError("router source cannot be attributed to a base-model family")
    return model, tuple(source.label for source in selected)


def _probability_primitive(value: object, family: str, index: int | str) -> str:
    primitive = _text(value, f"{family} row {index} primitive")
    if primitive not in _PROBABILITY_PRIMITIVES:
        raise ValueError(f"{family} primitive must be choice or noul, not {primitive}")
    return primitive


def _calibration_rows(comparison: Mapping[str, Any], context: _Context) -> list[dict[str, Any]]:
    value = comparison.get("calibration")
    if value is None:
        return []
    if isinstance(value, list):
        return _generic_rows(value, context, "calibration")
    records = _mapping(value, "comparison calibration")
    rows: list[dict[str, Any]] = []
    ignored = {
        "cases",
        "checksum",
        "dataset_version",
        "decisions",
        "manifest_digest",
        "metric_definition",
        "model",
        "n",
        "primitive",
        "run_id",
        "sample_count",
        "source_checksums",
        "source_kind",
        "source_run_ids",
        "split",
        "suite_id",
        "uncertainty_method",
        "unit",
    }
    for raw_primitive in sorted(records):
        primitive = _probability_primitive(raw_primitive, "calibration", raw_primitive)
        record = _mapping(records[raw_primitive], f"calibration {primitive}")
        if "primitive" in record:
            record_primitive = _probability_primitive(record["primitive"], "calibration", primitive)
            if record_primitive != primitive:
                raise ValueError("calibration primitive conflicts with its record family")
        split = _text(record.get("split", "evaluation"), f"calibration {primitive} split")
        if split not in {"calibration", "evaluation"}:
            raise ValueError(f"calibration {primitive} split must be calibration or evaluation")
        sample_count = _sample_count(
            record, f"calibration {primitive}", ("sample_count", "n", "decisions")
        )
        model, source_labels = _source_labels(
            record, context, "calibration", primitive, default_base=True
        )
        custom_metric_definition = (
            _text(record["metric_definition"], f"calibration {primitive} metric_definition")
            if "metric_definition" in record
            else None
        )
        suite_id = _text(record.get("suite_id", primitive), f"calibration {primitive} suite_id")
        unit = _text(record.get("unit", "fraction"), f"calibration {primitive} unit")
        for metric in sorted(record):
            if metric in ignored:
                continue
            if record[metric] is None:
                continue
            metric_value = _number(record[metric], f"calibration {primitive} {metric}")
            row = _base_row(
                context,
                source_labels,
                custom_metric_definition or _metric_definition(primitive, metric),
                sample_count,
                split,
            )
            row.update(
                {
                    "metric": metric,
                    "model": model,
                    "plotted_values": {metric: metric_value},
                    "primitive": primitive,
                    "suite_id": suite_id,
                    "unit": unit,
                    "value": metric_value,
                }
            )
            rows.append(row)
    return rows


def _metric_order(value: str) -> tuple[int, str]:
    try:
        return (_METRIC_ORDER.index(value), value)
    except ValueError:
        return (len(_METRIC_ORDER), value)


def _unit(metric: str) -> str:
    if metric.endswith("_ms"):
        return "milliseconds"
    if metric.endswith("_seconds"):
        return "seconds"
    if metric == "throughput":
        return "reported throughput"
    if metric.endswith("_mb"):
        return "megabytes"
    return "reported units"


def _efficiency_definition(metric: str) -> str:
    definitions = {
        "p50_ms": "Median end-to-end latency in milliseconds",
        "p95_ms": "95th-percentile end-to-end latency in milliseconds",
        "p50_seconds": "Median end-to-end latency in seconds",
        "p95_seconds": "95th-percentile end-to-end latency in seconds",
        "throughput": "Measured end-to-end throughput",
    }
    return definitions.get(metric, f"Measured resource value: {metric}")


def _efficiency_rows(comparison: Mapping[str, Any], context: _Context) -> list[dict[str, Any]]:
    value = comparison.get("efficiency")
    if isinstance(value, list):
        return _generic_rows(value, context, "efficiency")
    rows: list[dict[str, Any]] = []
    if value is not None:
        records = _mapping(value, "comparison efficiency")
        for model in _ordered_labels(set(records)):
            if model not in {source.label for source in _base_sources(context)}:
                continue
            metrics = _mapping(records[model], f"efficiency {model}")
            for metric in sorted(metrics, key=_metric_order):
                if metrics[metric] is None:
                    continue
                metric_value = _number(metrics[metric], f"efficiency {model} {metric}")
                row = _base_row(
                    context,
                    [model],
                    _efficiency_definition(metric),
                    context.evaluation_cases,
                    "evaluation",
                )
                row.update(
                    {
                        "metric": metric,
                        "model": model,
                        "plotted_values": {"value": metric_value},
                        "suite_id": "comparison",
                        "unit": _unit(metric),
                        "value": metric_value,
                    }
                )
                rows.append(row)
        return rows
    if context.synthetic:
        return rows
    suites = _mapping(comparison.get("suites"), "comparison suites")
    for suite_id in sorted(suites):
        suite = _mapping(suites[suite_id], f"suite {suite_id}")
        for source in _base_sources(context):
            if source.label not in suite:
                continue
            summary = _mapping(suite[source.label], f"suite {suite_id} {source.label}")
            sample_count = _nonnegative_integer(
                summary.get("cases"), f"suite {suite_id} {source.label} cases"
            )
            latency = _mapping(summary.get("latency"), f"suite {suite_id} {source.label} latency")
            for metric in ("p50_seconds", "p95_seconds"):
                if metric not in latency:
                    continue
                metric_value = _number(latency[metric], f"suite {suite_id} {source.label} {metric}")
                row = _base_row(
                    context,
                    [source.label],
                    _efficiency_definition(metric),
                    sample_count,
                    "evaluation",
                )
                row.update(
                    {
                        "metric": metric,
                        "model": source.label,
                        "plotted_values": {"value": metric_value},
                        "suite_id": suite_id,
                        "unit": _unit(metric),
                        "value": metric_value,
                    }
                )
                rows.append(row)
    return rows


def _multilingual_rows(comparison: Mapping[str, Any], context: _Context) -> list[dict[str, Any]]:
    value = comparison.get("multilingual")
    if value is None:
        return []
    if isinstance(value, list):
        return _generic_rows(value, context, "multilingual")
    records = _mapping(value, "comparison multilingual")
    rows: list[dict[str, Any]] = []
    for language in sorted(records):
        record = _mapping(records[language], f"multilingual {language}")
        sample_count = _sample_count(
            record, f"multilingual {language}", ("sample_count", "n", "cases", "decisions")
        )
        for model in _ordered_labels(set(record)):
            if model not in {source.label for source in _base_sources(context)}:
                continue
            if record[model] is None:
                continue
            metric_value = _number(record[model], f"multilingual {language} {model}")
            row = _base_row(
                context,
                [model],
                "Classification accuracy for the stated language in the evaluation split",
                sample_count,
                "evaluation",
            )
            row.update(
                {
                    "language": language,
                    "metric": "accuracy",
                    "model": model,
                    "plotted_values": {"value": metric_value},
                    "suite_id": "multilingual_intent",
                    "unit": "fraction",
                    "value": metric_value,
                }
            )
            rows.append(row)
    return rows


def _router_telemetry(
    router: Mapping[str, Any], source: _Source
) -> tuple[
    dict[str, int],
    dict[str, str],
    dict[str, dict[str, int]],
]:
    count_metadata = router if "route_counts" in router else source.model
    reason_metadata = router if "route_reasons" in router else source.model
    reason_count_metadata = router if "route_reason_counts" in router else source.model
    raw_counts = _mapping(count_metadata.get("route_counts", {}), "router route_counts")
    route_counts = {
        _text(route, "router route"): _nonnegative_integer(count, f"router route count {route}")
        for route, count in raw_counts.items()
    }
    raw_reasons = _mapping(reason_metadata.get("route_reasons", {}), "router route_reasons")
    route_reasons = {
        _text(route, "router route"): _text(reason, f"router route reason {route}")
        for route, reason in raw_reasons.items()
    }
    raw_reason_counts = _mapping(
        reason_count_metadata.get("route_reason_counts", {}),
        "router route_reason_counts",
    )
    route_reason_counts: dict[str, dict[str, int]] = {}
    for raw_route, raw_counts_for_route in raw_reason_counts.items():
        route = _text(raw_route, "router route")
        counts = _mapping(raw_counts_for_route, f"router route_reason_counts {route}")
        route_reason_counts[route] = {
            _text(reason, f"router reason for {route}"): _nonnegative_integer(
                count, f"router route reason count {route} {reason}"
            )
            for reason, count in counts.items()
        }
    return route_counts, route_reasons, route_reason_counts


def _router_reason_fields(
    route: str,
    reasons: Mapping[str, str],
    reason_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    reason = reasons.get(route)
    if reason is not None:
        fields["route_reason"] = reason
        count = reason_counts.get(route, {}).get(reason)
        if count is not None:
            fields["route_reason_count"] = count
    elif len(reason_counts.get(route, {})) == 1:
        only_reason, count = next(iter(reason_counts[route].items()))
        fields["route_reason"] = only_reason
        fields["route_reason_count"] = count
    route_reason_names = set(reason_counts.get(route, {}))
    if reason is not None:
        route_reason_names.add(reason)
    if route_reason_names:
        fields["route_reasons"] = sorted(route_reason_names)
    if route in reason_counts:
        fields["route_reason_counts"] = {
            reason_name: count for reason_name, count in sorted(reason_counts[route].items())
        }
    return fields


def _router_rows_without_router_source(value: object, context: _Context) -> list[dict[str, Any]]:
    labels = {source.label for source in context.sources}
    for raw_value in _sequence(value, "comparison router"):
        raw = _mapping(raw_value, "router source row")
        model = _text(raw.get("model", "comparison"), "router source row model")
        if model in labels or any(
            key in raw for key in ("run_id", "checksum", "source_run_ids", "source_checksums")
        ):
            raise ValueError("router family rows require a Router-attributed source run")
    return []


def _router_rows(comparison: Mapping[str, Any], context: _Context) -> list[dict[str, Any]]:
    value = comparison.get("router")
    if value is None:
        return []
    router_sources = [source for source in context.sources if source.is_router]
    if not router_sources:
        if isinstance(value, list):
            return _router_rows_without_router_source(value, context)
        return []
    source = router_sources[0]
    if isinstance(value, list):
        return _generic_rows(value, context, "router")
    router = _mapping(value, "comparison router")
    rows: list[dict[str, Any]] = []
    sample_count = _sample_count(router, "router", ("sample_count", "n", "cases", "decisions"))
    route_counts, route_reasons, route_reason_counts = _router_telemetry(router, source)
    for metric in sorted((key for key in router if key.endswith("_delta")), key=_metric_order):
        if router[metric] is None:
            continue
        metric_value = _number(router[metric], f"router {metric}")
        routing_scope = metric.removesuffix("_delta")
        row = _base_row(
            context,
            [source.label],
            "Router evaluation accuracy delta for the stated routing scope",
            sample_count,
            "evaluation",
        )
        row.update(
            {
                "metric": metric,
                "model": "router",
                "plotted_values": {"value": metric_value},
                "routing_scope": routing_scope,
                "suite_id": "router",
                "unit": "fraction",
                "value": metric_value,
            }
        )
        row.update(_router_reason_fields(routing_scope, route_reasons, route_reason_counts))
        rows.append(row)
    for route in sorted(route_counts):
        count = route_counts[route]
        row = _base_row(
            context,
            [source.label],
            "Evaluation decisions assigned to the stated route",
            count,
            "evaluation",
        )
        row.update(
            {
                "metric": "route_count",
                "model": "router",
                "plotted_values": {"value": count},
                "route": route,
                "suite_id": "router",
                "unit": "decisions",
                "value": count,
            }
        )
        row.update(_router_reason_fields(route, route_reasons, route_reason_counts))
        rows.append(row)
    for route in sorted(set(route_reason_counts) - set(route_counts)):
        for reason, count in sorted(route_reason_counts[route].items()):
            row = _base_row(
                context,
                [source.label],
                "Evaluation decisions assigned to the stated route reason",
                count,
                "evaluation",
            )
            row.update(
                {
                    "metric": "route_reason_count",
                    "model": "router",
                    "plotted_values": {"value": count},
                    "route": route,
                    "route_reason": reason,
                    "route_reason_count": count,
                    "route_reason_counts": {reason: count},
                    "route_reasons": [reason],
                    "suite_id": "router",
                    "unit": "decisions",
                    "value": count,
                }
            )
            rows.append(row)
    return rows


def _generic_value(record: Mapping[str, Any], family: str, index: int) -> int | float:
    has_value = "value" in record
    has_risk = family == "risk_coverage" and "risk" in record
    if not has_value and not has_risk:
        raise ValueError(f"{family} source row is missing a plotted value")
    value = _number(record["value"], f"{family} row {index} value") if has_value else None
    risk = _number(record["risk"], f"{family} row {index} risk") if has_risk else None
    if value is not None and risk is not None and value != risk:
        raise ValueError(f"{family} row {index} value conflicts with risk")
    return cast(int | float, value if value is not None else risk)


def _generic_metric(record: Mapping[str, Any], family: str) -> str:
    if "metric" in record:
        return _text(record["metric"], f"{family} metric")
    if family == "risk_coverage" and "risk" in record:
        return "risk"
    raise ValueError(f"{family} source row is missing metric")


def _generic_rows(value: object, context: _Context, family: str) -> list[dict[str, Any]]:
    raw_rows = _sequence(value, f"comparison {family}")
    rows: list[dict[str, Any]] = []
    definitions = {
        "calibration": "Reported calibration value",
        "efficiency": "Reported efficiency value",
        "multilingual": "Reported multilingual evaluation value",
        "risk_coverage": "Selective risk at the stated coverage",
        "router": "Reported router deployment value",
    }
    for index, raw_value in enumerate(raw_rows):
        raw = _mapping(raw_value, f"{family} row {index}")
        if (
            "dataset_version" in raw
            and _text(raw["dataset_version"], f"{family} row {index} dataset_version")
            != context.dataset_version
        ):
            raise ValueError(f"{family} row {index} dataset_version conflicts with comparison")
        if (
            "manifest_digest" in raw
            and _digest(raw["manifest_digest"], f"{family} row {index} manifest_digest")
            != context.manifest_digest
        ):
            raise ValueError(f"{family} row {index} manifest_digest conflicts with comparison")
        if (
            "uncertainty_method" in raw
            and _text(raw["uncertainty_method"], f"{family} row {index} uncertainty_method")
            != UNCERTAINTY_METHOD
        ):
            raise ValueError(f"{family} row {index} uncertainty_method conflicts with protocol")
        model, source_labels = _source_labels(raw, context, family, index)
        metric = _generic_metric(raw, family)
        metric_value = _generic_value(raw, family, index)
        split = _text(raw.get("split", "evaluation"), f"{family} row {index} split")
        if split not in {"calibration", "evaluation"}:
            raise ValueError(f"{family} row {index} split must be calibration or evaluation")
        plotted_values = {
            key: _number(child, f"{family} row {index} plotted_values.{key}")
            for key, child in raw.items()
            if key not in _GENERIC_METADATA_KEYS
        }
        if not plotted_values:
            plotted_values = {"value": metric_value}
        canonical_json(plotted_values)
        sample_count = _sample_count(
            raw, f"{family} row {index}", ("sample_count", "n", "cases", "decisions")
        )
        metric_definition = (
            _text(raw["metric_definition"], f"{family} row {index} metric_definition")
            if "metric_definition" in raw
            else definitions[family]
        )
        suite_id = _text(raw.get("suite_id", "comparison"), f"{family} row {index} suite_id")
        default_unit = (
            "decisions"
            if metric.endswith("count")
            else _unit(metric)
            if family == "efficiency"
            else "fraction"
        )
        unit = _text(raw.get("unit", default_unit), f"{family} row {index} unit")
        row = _base_row(
            context,
            source_labels,
            metric_definition,
            sample_count,
            split,
        )
        row.update(
            {
                "metric": metric,
                "model": model,
                "plotted_values": plotted_values,
                "suite_id": suite_id,
                "unit": unit,
                "value": metric_value,
            }
        )
        if family in {"calibration", "risk_coverage"}:
            if "primitive" not in raw:
                raise ValueError(f"{family} row {index} is missing primitive")
            row["primitive"] = _probability_primitive(raw["primitive"], family, index)
        elif "primitive" in raw:
            primitive = _text(raw["primitive"], f"{family} row {index} primitive")
            if primitive not in _PRIMITIVES:
                raise ValueError(f"{family} row {index} primitive is invalid")
            row["primitive"] = primitive
        for key in ("language", "route", "route_reason"):
            if key in raw:
                row[key] = _text(raw[key], f"{family} row {index} {key}")
        if "route_reason_count" in raw:
            row["route_reason_count"] = _nonnegative_integer(
                raw["route_reason_count"], f"{family} row {index} route_reason_count"
            )
        if "route_reasons" in raw:
            route_reasons = _sequence(raw["route_reasons"], f"{family} row {index} route_reasons")
            row["route_reasons"] = [
                _text(reason, f"{family} row {index} route_reasons") for reason in route_reasons
            ]
        if "route_reason_counts" in raw:
            raw_reason_counts = _mapping(
                raw["route_reason_counts"], f"{family} row {index} route_reason_counts"
            )
            row["route_reason_counts"] = {
                _text(reason, f"{family} row {index} route_reason_counts"): _nonnegative_integer(
                    count, f"{family} row {index} route_reason_counts.{reason}"
                )
                for reason, count in raw_reason_counts.items()
            }
        rows.append(row)
    return rows


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {label: index for index, label in enumerate(_MODEL_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            cast(str, row["suite_id"]),
            cast(str, row.get("language", "")),
            cast(str, row.get("route", "")),
            cast(str, row.get("route_reason", "")),
            cast(str, row.get("primitive", "")),
            order.get(cast(str, row["model"]), len(order)),
            cast(str, row.get("metric", "")),
            canonical_json(row),
        ),
    )


def build_figure_data(comparison: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    value = _mapping(comparison, "comparison")
    context = _context(value)
    families = {
        "accuracy": _accuracy_rows(value, context),
        "paired_effects": _paired_effects_rows(value, context),
        "calibration": _calibration_rows(value, context),
        "risk_coverage": (
            _generic_rows(value["risk_coverage"], context, "risk_coverage")
            if value.get("risk_coverage") is not None
            else []
        ),
        "efficiency": _efficiency_rows(value, context),
        "multilingual": _multilingual_rows(value, context),
        "router": _router_rows(value, context),
    }
    result = {family: _sort_rows(families[family]) for family in FIGURE_FAMILIES}
    _validate_rows(result, context)
    return result


def _validate_rows(data: Mapping[str, Any], context: _Context | None) -> None:
    if set(data) != set(FIGURE_FAMILIES):
        raise ValueError("figure data families do not match the required family set")
    known_sources = (
        {(source.run_id, source.checksum) for source in context.sources}
        if context is not None
        else set[tuple[str, str]]()
    )
    source_by_run = (
        {source.run_id: source for source in context.sources} if context is not None else {}
    )
    required = {
        "dataset_version",
        "manifest_digest",
        "metric_definition",
        "model",
        "plotted_values",
        "sample_count",
        "source_checksums",
        "source_kind",
        "source_run_ids",
        "split",
        "suite_id",
        "uncertainty_method",
        "value",
    }
    identity: tuple[str, str, str] | None = None
    checksum_by_run: dict[str, str] = {}
    for family in FIGURE_FAMILIES:
        raw_rows = data[family]
        if not isinstance(raw_rows, list):
            raise TypeError(f"figure data {family} must be a list")
        for index, raw_row in enumerate(cast(list[object], raw_rows)):
            prefix = f"{family}[{index}]"
            row = _mapping(raw_row, prefix)
            missing = required - set(row)
            if missing:
                raise ValueError(f"{prefix} is missing {', '.join(sorted(missing))}")
            dataset_version = _text(row["dataset_version"], f"{prefix}.dataset_version")
            manifest_digest = _digest(row["manifest_digest"], f"{prefix}.manifest_digest")
            uncertainty_method = _text(row["uncertainty_method"], f"{prefix}.uncertainty_method")
            if uncertainty_method != UNCERTAINTY_METHOD:
                raise ValueError(f"{prefix}.uncertainty_method is not the stable protocol")
            row_identity = (dataset_version, manifest_digest, uncertainty_method)
            if identity is None:
                identity = row_identity
            elif row_identity != identity:
                raise ValueError("figure data rows must share one dataset and uncertainty identity")
            if context is not None and (
                dataset_version != context.dataset_version
                or manifest_digest != context.manifest_digest
            ):
                raise ValueError(f"{prefix} dataset identity differs from the comparison")
            _text(row["metric_definition"], f"{prefix}.metric_definition")
            _text(row["suite_id"], f"{prefix}.suite_id")
            _text(row["model"], f"{prefix}.model")
            source_kind = _text(row["source_kind"], f"{prefix}.source_kind")
            expected_source_kind = "router" if family == "router" else "base"
            if source_kind != expected_source_kind:
                raise ValueError(
                    f"{prefix}.source_kind must be {expected_source_kind} for the {family} family"
                )
            split = _text(row["split"], f"{prefix}.split")
            if split not in {"calibration", "evaluation"}:
                raise ValueError(f"{prefix}.split must be calibration or evaluation")
            if row["sample_count"] is not None:
                _nonnegative_integer(row["sample_count"], f"{prefix}.sample_count")
            run_ids = [
                _text(run_id, f"{prefix}.source_run_ids")
                for run_id in _sequence(row["source_run_ids"], f"{prefix}.source_run_ids")
            ]
            checksums = [
                _digest(checksum, f"{prefix}.source_checksums")
                for checksum in _sequence(row["source_checksums"], f"{prefix}.source_checksums")
            ]
            if len(run_ids) != len(checksums) or not run_ids:
                raise ValueError(f"{prefix} source IDs and checksums must be nonempty and aligned")
            pairs = list(zip(run_ids, checksums, strict=True))
            for run_id, checksum in pairs:
                previous_checksum = checksum_by_run.setdefault(run_id, checksum)
                if previous_checksum != checksum:
                    raise ValueError(f"{prefix} source checksum conflicts with another row")
            if len({run_id for run_id in run_ids}) != len(run_ids):
                raise ValueError(f"{prefix} source IDs must be unique")
            if context is not None:
                for run_id, checksum in pairs:
                    if (run_id, checksum) not in known_sources:
                        raise ValueError(f"{prefix} source identity differs from the comparison")
                selected = [source_by_run[run_id] for run_id in run_ids]
                actual_source_kind = _source_kind(selected)
                if source_kind != actual_source_kind:
                    raise ValueError(f"{prefix}.source_kind conflicts with source identity")
                if family == "router" and actual_source_kind != "router":
                    raise ValueError(f"{prefix} router family cannot attribute a base-model source")
                if family != "router" and actual_source_kind != "base":
                    raise ValueError(f"{prefix} router source cannot be used by a base family")
            if len(run_ids) == 1:
                if row.get("run_id") != run_ids[0] or row.get("checksum") != checksums[0]:
                    raise ValueError(f"{prefix}.run_id or checksum differs from source identity")
            elif "run_id" in row or "checksum" in row:
                raise ValueError(f"{prefix} multi-source row must use source ID lists")
            plotted_values = _mapping(row["plotted_values"], f"{prefix}.plotted_values")
            if not plotted_values:
                raise ValueError(f"{prefix}.plotted_values must not be empty")
            for key, value in plotted_values.items():
                _number(value, f"{prefix}.plotted_values.{key}")
            _number(row["value"], f"{prefix}.value")
            for key in (
                "ci_high",
                "ci_low",
                "p_value",
                "p_value_holm",
                "replicates",
                "seed",
            ):
                if key in row:
                    _number(row[key], f"{prefix}.{key}")
            for key in (
                "contrast",
                "language",
                "metric",
                "route",
                "route_reason",
                "routing_scope",
                "unit",
            ):
                if key in row:
                    _text(row[key], f"{prefix}.{key}")
            if "route_reason_count" in row:
                _nonnegative_integer(row["route_reason_count"], f"{prefix}.route_reason_count")
            if "route_reasons" in row:
                route_reasons = [
                    _text(reason, f"{prefix}.route_reasons")
                    for reason in _sequence(row["route_reasons"], f"{prefix}.route_reasons")
                ]
                if not route_reasons or len(set(route_reasons)) != len(route_reasons):
                    raise ValueError(f"{prefix}.route_reasons must be nonempty and unique")
            if "route_reason_counts" in row:
                reason_counts = _mapping(
                    row["route_reason_counts"], f"{prefix}.route_reason_counts"
                )
                if not reason_counts:
                    raise ValueError(f"{prefix}.route_reason_counts must not be empty")
                for reason, count in reason_counts.items():
                    _text(reason, f"{prefix}.route_reason_counts")
                    _nonnegative_integer(count, f"{prefix}.route_reason_counts.{reason}")
            if family in {"calibration", "risk_coverage"}:
                if "primitive" not in row:
                    raise ValueError(f"{prefix}.primitive is required")
                _probability_primitive(row["primitive"], family, index)
            elif "primitive" in row:
                primitive = _text(row["primitive"], f"{prefix}.primitive")
                if primitive not in _PRIMITIVES:
                    raise ValueError(f"{prefix}.primitive is invalid")
            canonical_json(row)


def _assert_type_safe_equal(expected: object, actual: object, path: str) -> None:
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise TypeError(f"{path} differs in JSON type")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unexpected " + ", ".join(extra))
            raise ValueError(f"{path} differs from comparison: {'; '.join(details)}")
        for key in sorted(expected):
            _assert_type_safe_equal(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            raise TypeError(f"{path} differs in JSON type")
        if len(expected) != len(actual):
            raise ValueError(f"{path} differs in list length")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_type_safe_equal(left, right, f"{path}[{index}]")
        return
    if type(expected) is not type(actual):
        raise ValueError(
            f"{path} differs from comparison: expected {type(expected).__name__}, "
            f"got {type(actual).__name__}"
        )
    if expected != actual:
        raise ValueError(f"{path} differs from comparison")


def validate_figure_data(data: Mapping[str, Any], comparison: Mapping[str, Any]) -> None:
    value = _mapping(data, "figure data")
    context = _context(_mapping(comparison, "comparison"))
    _validate_rows(value, context)
    expected = build_figure_data(comparison)
    _assert_type_safe_equal(expected, value, "figure_data")


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_components(path: Path, name: str) -> None:
    absolute = _absolute_path(path)
    lexical_parts: list[str] = []
    root = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part == "..":
            if lexical_parts:
                lexical_parts.pop()
            continue
        lexical_parts.append(part)
        current = root.joinpath(*lexical_parts)
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"{name} could not be inspected") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} path contains a symlink component")


def _ensure_directory(path: Path, name: str) -> None:
    _reject_symlink_components(path, name)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{name} must be a real directory")
        return
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a real directory")


def _preflight_output(path: Path) -> None:
    _reject_symlink_components(path, "figure data output")
    if path.is_symlink():
        raise ValueError("figure data output must not be a symlink")
    if path.exists():
        raise FileExistsError(f"figure data artifact already exists: {path.name}")


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (Mapping, list, tuple)):
        encoded: bytes = canonical_json(value)
        return encoded.decode("utf-8")
    return str(value)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return f"empty_reason\n{EMPTY_FAMILY_REASON}\n".encode()
    extra = sorted({key for row in rows for key in row} - set(_BASE_CSV_COLUMNS))
    fieldnames = [*_BASE_CSV_COLUMNS, *extra]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def write_figure_data(data: Mapping[str, Any], output_root: Path) -> Path:
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    value = _mapping(data, "figure data")
    _validate_rows(value, None)
    sorted_value = {
        family: _sort_rows(cast(list[dict[str, Any]], value[family])) for family in FIGURE_FAMILIES
    }
    data_root = output_root / "data"
    _ensure_directory(output_root, "figure output root")
    _ensure_directory(data_root, "figure data directory")
    targets = {
        "figure-source.json": output_root / "figure-source.json",
        **{f"{family}.csv": data_root / f"{family}.csv" for family in FIGURE_FAMILIES},
    }
    for target in targets.values():
        _preflight_output(target)
    for family in FIGURE_FAMILIES:
        path = data_root / f"{family}.csv"
        with path.open("xb") as file:
            file.write(_csv_bytes(sorted_value[family]))
    source_path = output_root / "figure-source.json"
    write_json_exclusive(source_path, sorted_value)
    return source_path
