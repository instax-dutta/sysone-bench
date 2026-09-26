from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import stat
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, NoReturn, cast
from uuid import uuid4

from benchmark.canonical import canonical_json, digest_bytes, digest_file, digest_value
from benchmark.contracts import validate_answers
from benchmark.manifest import manifest_digest, validate_manifest
from benchmark.storage import (
    create_run_directory,
    write_checksums,
    write_json_exclusive,
    write_jsonl_exclusive,
)
from benchmark.timing import measure_call
from benchmark.usage import UsageCounter
from runners.base import BaseRunner

SEED = 42
ARTIFACT_SCHEMA_VERSION = 2
METRIC_SCHEMA_VERSION = 2
MODEL_PHASES = ("warmup", "benchmark", "speed")
SPLITS = ("calibration", "evaluation")
REQUIRED_RUN_ARTIFACTS = (
    "metadata.json",
    "predictions.jsonl",
    "summary.json",
    "usage.json",
    "checksums.sha256",
)
_RUN_RESPONSE_FIELDS = frozenset(
    {"answers", "_usage", "_routing", "_raw_model", "_response_digest"}
)
_MODEL_IDENTITY_FIELDS = (
    "runner",
    "model",
    "revision",
    "serving",
    "device",
    "adapter_version",
)
_SENSITIVE_KEY_PARTS = (
    "apikey",
    "authorization",
    "accesstoken",
    "refreshtoken",
    "access",
    "auth",
    "token",
    "credential",
    "password",
    "secret",
    "cookie",
)
RunnerFactory = Callable[[str], BaseRunner]


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _loads(value: str, source: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{source} contains malformed JSON at line {error.lineno}, column {error.colno}"
        ) from None
    except ValueError as error:
        raise ValueError(f"{source} contains malformed JSON: {error}") from None


def _read_json_object(path: Path, source: str) -> dict[str, Any]:
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not valid UTF-8: {error}") from None
    value = _loads(contents, source)
    if not isinstance(value, dict):
        raise TypeError(f"{source} must contain a JSON object")
    return cast(dict[str, Any], value)


def _jsonl_objects_from_text(contents: str, source: str) -> list[dict[str, Any]]:
    lines = contents.splitlines()
    if not lines:
        raise ValueError(f"{source} must contain at least one record")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{source} line {line_number} must be a JSON object")
        value = _loads(line, f"{source} line {line_number}")
        if not isinstance(value, dict):
            raise TypeError(f"{source} line {line_number} must be a JSON object")
        rows.append(cast(dict[str, Any], value))
    return rows


def _read_jsonl_objects(path: Path, source: str) -> list[dict[str, Any]]:
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not valid UTF-8: {error}") from None
    return _jsonl_objects_from_text(contents, source)


def _manifest_objects_from_bytes(contents: bytes) -> list[dict[str, Any]]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"manifest is not valid UTF-8 JSON: {error}") from None
    if text.lstrip().startswith("["):
        value = _loads(text, "manifest")
        if not isinstance(value, list):
            raise TypeError("manifest must contain a JSON list of records")
        records: list[dict[str, Any]] = []
        for index, record in enumerate(value):
            if not isinstance(record, dict):
                raise TypeError(f"manifest item {index} must be a JSON object")
            records.append(cast(dict[str, Any], record))
        return records
    return _jsonl_objects_from_text(text, "manifest")


def _sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _numeric_token_counter(key: object, value: object) -> bool:
    return (
        isinstance(key, str)
        and key.casefold().endswith("_tokens")
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _redact(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                deepcopy(child)
                if _numeric_token_counter(key, child)
                else "[REDACTED]"
                if _sensitive_key(key)
                else _redact(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(child) for child in value]
    return deepcopy(value)


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return {
        str(key): deepcopy(child) for key, child in cast(Mapping[object, object], value).items()
    }


def _string_key_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result: dict[str, Any] = {}
    for key, child in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = deepcopy(child)
    return result


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _normalized_path(path: Path) -> Path:
    return Path(os.path.normpath(_absolute_path(path)))


def _casefolded_path(path: Path) -> str:
    return str(_normalized_path(path)).casefold()


def _paths_overlap(left: Path, right: Path) -> bool:
    left_path = Path(_casefolded_path(left))
    right_path = Path(_casefolded_path(right))
    return (
        left_path == right_path
        or left_path.is_relative_to(right_path)
        or right_path.is_relative_to(left_path)
    )


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _artifact_integer(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must equal {expected}")
    return expected


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _optional_finite_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    try:
        return _finite_number(value, name)
    except TypeError as error:
        raise ValueError(f"{name} must be a finite number") from error


def _json_values_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_json_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(_json_values_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _validated_latency_summary(value: object, name: str) -> dict[str, float]:
    latency = _json_mapping(value, name)
    fields = ("mean_seconds", "p50_seconds", "p95_seconds")
    if set(latency) != set(fields):
        raise ValueError(f"{name} fields are invalid")
    result: dict[str, float] = {}
    for field in fields:
        try:
            number = _finite_number(latency.get(field), f"{name} {field}")
        except TypeError as error:
            raise ValueError(f"{name} {field} must be a finite number") from error
        if number < 0.0:
            raise ValueError(f"{name} {field} must be non-negative")
        result[field] = number
    return result


def _digest_text(value: object, name: str) -> str:
    text = _nonempty_text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a full lowercase SHA-256 digest")
    return text


def _expected_value(question: Mapping[str, Any], expected: object, qid: str) -> None:
    question_type = question.get("type")
    if question_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or not isinstance(expected, str):
            raise ValueError(f"{qid}: expected choice must be a legal label")
        if expected not in criteria:
            raise ValueError(f"{qid}: expected choice must be a legal label")
        return
    if question_type == "noul":
        if isinstance(expected, bool) or not isinstance(expected, int) or expected not in (0, 1):
            raise ValueError(f"{qid}: expected noul value must be integer 0 or 1")
        return
    if question_type == "score":
        max_score = question.get("max_score")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(expected)
            or isinstance(max_score, bool)
            or not isinstance(max_score, (int, float))
            or not math.isfinite(max_score)
            or not 0 <= expected <= max_score
        ):
            raise ValueError(f"{qid}: expected score must be a finite numeric level in range")
        return
    raise ValueError(f"{qid}: unsupported question type")


def _ordered_questions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    questions = record.get("questions")
    if isinstance(questions, (str, bytes, bytearray)) or not isinstance(questions, Sequence):
        raise TypeError("manifest questions must be an ordered sequence")
    result: list[Mapping[str, Any]] = []
    for index, question in enumerate(questions):
        if not isinstance(question, Mapping):
            raise TypeError(f"manifest question {index} must be an object")
        result.append(cast(Mapping[str, Any], question))
    return result


def _validate_expected_values(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        expected = record.get("expected")
        if not isinstance(expected, Mapping):
            raise TypeError("manifest expected must be an object")
        expected_map = _json_mapping(expected, "manifest expected")
        for question in _ordered_questions(record):
            qid = _nonempty_text(question.get("qid"), "question qid")
            _expected_value(question, expected_map.get(qid), qid)


def _validate_manifest_identity(
    records: Sequence[Mapping[str, Any]], *, require_zero_based: bool = True
) -> list[str]:
    suite_order: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    closed_suites: set[str] = set()
    dataset_versions: set[str] = set()
    for record in records:
        suite_id = _nonempty_text(record.get("suite_id"), "suite_id")
        dataset_version = _nonempty_text(record.get("dataset_version"), "dataset_version")
        dataset_versions.add(dataset_version)
        if suite_id not in grouped:
            grouped[suite_id] = []
            suite_order.append(suite_id)
        elif suite_id in closed_suites:
            raise ValueError(f"manifest suite order identity is interleaved at {suite_id!r}")
        grouped[suite_id].append(record)
        previous = grouped[suite_id][-2] if len(grouped[suite_id]) > 1 else None
        if previous is not None:
            previous_order = _nonnegative_integer(previous.get("order_index"), "order_index")
            current_order = _nonnegative_integer(record.get("order_index"), "order_index")
            if current_order <= previous_order:
                raise ValueError(f"manifest order_index must increase within suite {suite_id!r}")
        if len(grouped[suite_id]) == 1 and len(suite_order) > 1:
            closed_suites.add(suite_order[-2])
    if len(dataset_versions) != 1:
        raise ValueError("manifest dataset_version must be identical across records")
    for suite_id in suite_order:
        orders = [
            _nonnegative_integer(record.get("order_index"), "order_index")
            for record in grouped[suite_id]
        ]
        if require_zero_based and orders != list(range(len(orders))):
            raise ValueError(f"manifest order_index must be contiguous for suite {suite_id!r}")
    return suite_order


def _verify_manifest_checksum(manifest_bytes: bytes, checksum_bytes: bytes) -> str:
    digest: str = digest_bytes(manifest_bytes)
    expected = f"{digest}  manifest.jsonl\n".encode("ascii")
    if checksum_bytes != expected:
        raise ValueError("manifest checksum does not exactly match manifest.jsonl")
    return digest


def _validated_manifest(
    manifest_path: Path,
    checksum_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    _reject_symlink_components(manifest_path, "manifest path")
    if checksum_path is not None:
        _reject_symlink_components(checksum_path, "manifest checksum path")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise OSError("manifest is unavailable") from error
    file_digest = digest_bytes(manifest_bytes)
    records = _manifest_objects_from_bytes(manifest_bytes)
    counts: dict[str, int] = {}
    for record in records:
        suite_id = _nonempty_text(record.get("suite_id"), "suite_id")
        counts[suite_id] = counts.get(suite_id, 0) + 1
    summary = validate_manifest(records, counts)
    suite_order = _validate_manifest_identity(records)
    _validate_expected_values(records)
    if summary.digest != manifest_digest(records):
        raise ValueError("manifest canonical digest changed during validation")
    if checksum_path is not None:
        try:
            checksum_bytes = checksum_path.read_bytes()
        except OSError as error:
            raise OSError("manifest checksum is unavailable") from error
        checksum_digest = _verify_manifest_checksum(manifest_bytes, checksum_bytes)
        if checksum_digest != file_digest:
            raise ValueError("manifest checksum digest mismatch")
    else:
        checksum_digest = None
    dataset_version = _nonempty_text(records[0].get("dataset_version"), "dataset_version")
    manifest = {
        "checksum_digest": checksum_digest,
        "checksum_verified": checksum_path is not None,
        "dataset_version": dataset_version,
        "digest": summary.digest,
        "file_digest": file_digest,
        "suite_order": suite_order,
    }
    return records, manifest, suite_order


class DeterministicFakeRunner(BaseRunner):
    name = "fake"

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            question_type = question.get("type")
            if question_type == "noul":
                answers[qid] = {"type": "noul", "noul": 0.75}
            elif question_type == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, Mapping) or not criteria:
                    raise ValueError(f"{qid}: choice criteria are required")
                labels = tuple(criteria)
                label = labels[0]
                answers[qid] = {
                    "type": "choice",
                    "choice": label,
                    "probabilities": {
                        candidate: 1.0 if candidate == label else 0.0 for candidate in labels
                    },
                    "confidence": 1.0,
                }
            elif question_type == "score":
                answers[qid] = {"type": "score", "score": 0.0, "confidence": 1.0}
            else:
                raise ValueError(f"{qid}: unsupported fake question type")
        return {
            "answers": answers,
            "_usage": {"input_tokens": 2, "output_tokens": 1},
            "_raw_model": {"api_key": "fake-secret", "status": "ok"},
        }

    def info(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "model": self.name,
            "revision": "deterministic",
            "serving": "in-process",
            "device": "cpu",
            "adapter_version": "1",
        }


def default_runner_factory(name: str) -> BaseRunner:
    if name == "fake":
        return DeterministicFakeRunner()
    if name == "laya":
        from runners.laya_runner import LayaRunner

        return LayaRunner()
    if name in {"qwen", "laya-router", "router"}:
        if name == "qwen":
            from runners.qwen_runner import QwenRunner

            return QwenRunner()
        from runners.router_runner import RouterRunner

        return RouterRunner()
    if name == "jev":
        from runners.jev_runner import JevRunner

        return JevRunner()
    raise ValueError(f"unknown model: {name}")


def _runner_metadata(runner: BaseRunner) -> dict[str, Any]:
    raw = runner.info()
    metadata = _json_mapping(_redact(raw), "runner metadata")
    missing = [field for field in _MODEL_IDENTITY_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"runner metadata is missing required field(s): {', '.join(missing)}")
    for field in _MODEL_IDENTITY_FIELDS:
        _nonempty_text(metadata[field], f"runner metadata {field}")
    canonical_json(metadata)
    return metadata


def _stable_model_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): deepcopy(value)
        for key, value in metadata.items()
        if not (isinstance(key, str) and key.startswith("route_"))
    }


def _question_mapping(questions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for question in questions:
        qid = _nonempty_text(question.get("qid"), "question qid")
        result[qid] = _json_mapping(question, f"question {qid}")
    return result


def _confidence(answers: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for qid, raw_answer in answers.items():
        answer = _json_mapping(raw_answer, f"answer {qid}")
        answer_type = answer.get("type")
        if answer_type == "noul":
            probability = _finite_number(answer.get("noul"), f"{qid} noul")
            result[qid] = max(probability, 1.0 - probability)
        elif answer_type in {"choice", "score"}:
            value = answer.get("confidence")
            result[qid] = None if value is None else _finite_number(value, f"{qid} confidence")
        else:
            raise ValueError(f"{qid}: unsupported answer type")
    return result


def _validated_response(
    questions: Sequence[Mapping[str, Any]],
    response: object,
    counter: UsageCounter,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(response, Mapping):
        raise TypeError("runner response must be a mapping")
    result = _json_mapping(response, "runner response")
    unknown = set(result) - _RUN_RESPONSE_FIELDS
    if unknown:
        raise ValueError(f"runner response has unsupported field(s): {', '.join(sorted(unknown))}")
    answers = _string_key_mapping(result.get("answers"), "runner answer IDs")
    validate_answers(questions, answers)
    usage = _json_mapping(result.get("_usage", {}), "runner usage")
    redacted_usage = _json_mapping(_redact(usage), "runner usage")
    counter.record("benchmark", redacted_usage)
    confidence = _confidence(answers)
    return answers, redacted_usage, confidence


def _answer_is_correct(
    question: Mapping[str, Any], answer: Mapping[str, Any], expected: object
) -> bool:
    question_type = question.get("type")
    if question_type == "choice":
        return answer.get("choice") == expected
    if question_type == "noul":
        probability = _finite_number(answer.get("noul"), "noul")
        return int(probability >= 0.5) == expected
    if question_type == "score":
        score = _finite_number(answer.get("score"), "score")
        return math.floor(score + 0.5) == expected
    raise ValueError("unsupported question type")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    latencies = [cast(float, row["latency_seconds"]) for row in rows]
    if not latencies:
        return {"mean_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0}
    return {
        "mean_seconds": float(statistics.fmean(latencies)),
        "p50_seconds": float(statistics.median(latencies)),
        "p95_seconds": _percentile(latencies, 0.95),
    }


def _score_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[int, float | None, dict[str, Any]]:
    decisions = 0
    correct = 0
    per_suite_cases: dict[str, int] = {}
    per_suite_decisions: dict[str, int] = {}
    per_suite_correct: dict[str, int] = {}
    for row in rows:
        suite_id = cast(str, row["suite_id"])
        questions = cast(Sequence[Mapping[str, Any]], row["questions"])
        question_by_id = {cast(str, question["qid"]): question for question in questions}
        expected = cast(Mapping[str, Any], row["expected"])
        answers = cast(Mapping[str, Any], row["answers"])
        per_suite_cases[suite_id] = per_suite_cases.get(suite_id, 0) + 1
        for qid in cast(Sequence[str], row["question_ids"]):
            decisions += 1
            per_suite_decisions[suite_id] = per_suite_decisions.get(suite_id, 0) + 1
            if _answer_is_correct(
                question_by_id[qid], _json_mapping(answers[qid], "answer"), expected[qid]
            ):
                correct += 1
                per_suite_correct[suite_id] = per_suite_correct.get(suite_id, 0) + 1
    accuracy = None if decisions == 0 else correct / decisions
    suites: dict[str, Any] = {}
    for suite_id, case_count in per_suite_cases.items():
        suite_decisions = per_suite_decisions[suite_id]
        suites[suite_id] = {
            "accuracy": (
                None
                if suite_decisions == 0
                else per_suite_correct.get(suite_id, 0) / suite_decisions
            ),
            "cases": case_count,
            "decisions": suite_decisions,
            "latency": _latency_summary([row for row in rows if row["suite_id"] == suite_id]),
        }
    return decisions, accuracy, suites


def run_suite_v2(
    runner: BaseRunner,
    suite_id: str,
    records: Sequence[Mapping[str, Any]],
    counter: UsageCounter,
) -> dict[str, Any]:
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("suite_id must be a non-empty string")
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence")
    if not records:
        raise ValueError("records must not be empty")
    if not isinstance(counter, UsageCounter):
        raise TypeError("counter must be a UsageCounter")
    materialized = list(records)
    validate_manifest(materialized, {suite_id: len(materialized)})
    _validate_manifest_identity(materialized, require_zero_based=False)
    _validate_expected_values(materialized)
    splits = {_nonempty_text(record.get("split"), "split") for record in materialized}
    if len(splits) != 1:
        raise ValueError("run_suite_v2 records must contain exactly one split")
    split = next(iter(splits))
    if split not in SPLITS:
        raise ValueError("split must be calibration or evaluation")
    metadata = _stable_model_identity(_runner_metadata(runner))
    predictions: list[dict[str, Any]] = []
    for record in materialized:
        if record.get("suite_id") != suite_id:
            raise ValueError(f"record suite_id must equal {suite_id!r}")
        questions = _ordered_questions(record)
        question_map = _question_mapping(questions)
        state = _json_mapping(record.get("state"), "manifest state")
        state_digest = digest_value(state)
        response, latency = measure_call(
            perf_counter,
            partial(runner.predict, state, question_map, phase="benchmark"),
        )
        answers, usage, confidence = _validated_response(questions, response, counter)
        row: dict[str, Any] = {
            "_usage": usage,
            "answers": answers,
            "case_id": _nonempty_text(record.get("case_id"), "case_id"),
            "confidence": confidence,
            "expected": _json_mapping(record.get("expected"), "manifest expected"),
            "execution_phase": "benchmark",
            "latency_seconds": float(latency),
            "order_index": _nonnegative_integer(record.get("order_index"), "order_index"),
            "phase": split,
            "question_ids": [cast(str, question["qid"]) for question in questions],
            "questions": [deepcopy(dict(question)) for question in questions],
            "runner": metadata,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "split": split,
            "state_digest": state_digest,
            "suite_id": suite_id,
        }
        response_map = _json_mapping(response, "runner response")
        for field in ("_routing", "_raw_model", "_response_digest"):
            if field in response_map:
                row[field] = deepcopy(_redact(response_map[field]))
        canonical_json(row)
        predictions.append(row)
    decisions, accuracy, _ = _score_rows(predictions)
    return {
        "accuracy": accuracy,
        "cases": len(predictions),
        "decisions": decisions,
        "phase": split,
        "predictions": predictions,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "suite_id": suite_id,
    }


def _validate_runner_names(runner_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(runner_names, (str, bytes, bytearray)) or not isinstance(runner_names, Sequence):
        raise TypeError("runner_names must be an ordered sequence")
    names = tuple(runner_names)
    if not names:
        raise ValueError("runner_names must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("runner names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("runner names must be unique")
    return names


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", value)
    if not safe or safe in {".", ".."}:
        raise ValueError("run ID must be a safe path component")
    return safe[:128]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_hex() -> str:
    return uuid4().hex


def _generated_run_id(name: str) -> str:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    suffix = f"{timestamp}-{_uuid_hex()[:12]}"
    prefix = f"run-{_safe_component(name)}-"
    return f"{prefix[: max(1, 128 - len(suffix) - 1)]}{suffix}"


def _validate_run_id(run_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError("run ID must be a safe path component")
    return run_id


def _has_owned_run_marker(path: Path, run_id: str) -> bool:
    marker = path / ".sysone-owner"
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        return marker.read_text(encoding="utf-8") == f"{run_id}\n"
    except (OSError, UnicodeError):
        return False


def _is_existing_run_directory(path: Path, run_id: str) -> bool:
    return path.exists() and (path.name == run_id or _has_owned_run_marker(path, run_id))


def _target_paths(
    names: Sequence[str],
    output_root: Path,
    run_id: str | None,
) -> list[tuple[str, str, Path]]:
    selected_run_id = _validate_run_id(run_id) if run_id is not None else None
    targets: list[tuple[str, str, Path]] = []
    for name in names:
        identifier = selected_run_id or _generated_run_id(name)
        if selected_run_id is None:
            target = output_root / identifier
        elif len(names) == 1 and _is_existing_run_directory(output_root, selected_run_id):
            target = output_root
        elif len(names) == 1:
            target = output_root / identifier
        else:
            identifier = f"{selected_run_id}-{_safe_component(name)}"
            target = output_root / identifier
        targets.append((name, identifier, target))
    return targets


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
            raise ValueError(f"{name} path could not be inspected") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} path contains a symlink component")


def _reject_source_tree_symlinks(root: Path, name: str) -> None:
    try:
        children = tuple(root.iterdir())
    except OSError as error:
        raise ValueError(f"{name} could not be inspected") from error
    for child in children:
        try:
            mode = child.lstat().st_mode
        except OSError as error:
            raise ValueError(f"{name} could not be inspected") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} contains a symlink")
        if stat.S_ISDIR(mode):
            _reject_source_tree_symlinks(child, name)


def _validate_unique_targets(targets: Sequence[Path]) -> None:
    seen: set[str] = set()
    for target in targets:
        normalized = _casefolded_path(target)
        if normalized in seen:
            raise ValueError("v2 target path collision after normalization")
        seen.add(normalized)


def _preflight_target(path: Path, run_id: str) -> None:
    if path.is_symlink():
        raise ValueError("v2 output path must not be a symlink")
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError("v2 output path must be a directory")
    for name in REQUIRED_RUN_ARTIFACTS:
        artifact = path / name
        if artifact.exists() or artifact.is_symlink():
            raise FileExistsError(f"v2 run artifact already exists: {name}")
    if path.name != run_id or not _has_owned_run_marker(path, run_id):
        raise FileExistsError("v2 run directory already exists")


def _claim_target(path: Path, run_id: str) -> None:
    _reject_symlink_components(path, "v2 run target")
    _preflight_target(path, run_id)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    create_run_directory(path.parent, path.name)


def _zero_usage() -> dict[str, dict[str, int]]:
    return {phase: {"input_tokens": 0, "output_tokens": 0, "calls": 0} for phase in MODEL_PHASES}


def _total_usage(usages: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    total = _zero_usage()
    for usage in usages:
        for phase in MODEL_PHASES:
            phase_usage = usage[phase]
            for field in ("input_tokens", "output_tokens", "calls"):
                total[phase][field] += cast(int, phase_usage[field])
    return total


def _grouped_records(
    records: Sequence[Mapping[str, Any]], suite_order: Sequence[str]
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        suite_id: {split: [] for split in SPLITS} for suite_id in suite_order
    }
    for record in records:
        suite_id = cast(str, record["suite_id"])
        split = cast(str, record["split"])
        grouped[suite_id][split].append(record)
    return grouped


def _suite_metadata(
    records: Sequence[Mapping[str, Any]], suite_order: Sequence[str]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for suite_id in suite_order:
        suite_records = [record for record in records if record["suite_id"] == suite_id]
        result.append(
            {
                "case_ids": [record["case_id"] for record in suite_records],
                "order_indices": [record["order_index"] for record in suite_records],
                "splits": {
                    split: [
                        record["case_id"] for record in suite_records if record["split"] == split
                    ]
                    for split in SPLITS
                },
                "suite_id": suite_id,
            }
        )
    return result


def _validate_payloads(payloads: Sequence[Mapping[str, Any]], rows: Sequence[Any]) -> None:
    for payload in payloads:
        canonical_json(payload)
    for row in rows:
        canonical_json(row)


def run_all_v2(
    runner_names: Sequence[str],
    manifest_path: Path,
    output_root: Path,
    *,
    runner_factory: RunnerFactory | None = None,
    manifest_checksum_path: Path | None = None,
    run_id: str | None = None,
    progress: Callable[[str, str, str, int], None] | None = None,
) -> list[Path]:
    names = _validate_runner_names(runner_names)
    if not isinstance(manifest_path, Path) or not isinstance(output_root, Path):
        raise TypeError("manifest_path and output_root must be Path values")
    _reject_symlink_components(output_root, "v2 output root")
    records, manifest, suite_order = _validated_manifest(manifest_path, manifest_checksum_path)
    canonical_digest = cast(str, manifest["digest"])
    targets = _target_paths(names, output_root, run_id)
    _validate_unique_targets([target for _, _, target in targets])
    for _, identifier, target in targets:
        _reject_symlink_components(target, "v2 run target")
        _preflight_target(target, identifier)
    factory = default_runner_factory if runner_factory is None else runner_factory
    prepared: list[
        tuple[
            str,
            Path,
            dict[str, Any],
            list[dict[str, Any]],
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []
    grouped = _grouped_records(records, suite_order)
    for name, identifier, target in targets:
        random.seed(SEED)
        runner = factory(name)
        if not isinstance(runner, BaseRunner):
            raise TypeError("runner_factory must return a BaseRunner")
        counters = {split: UsageCounter() for split in SPLITS}
        predictions: list[dict[str, Any]] = []
        evaluation_suites: dict[str, Any] = {}
        for suite_id in suite_order:
            for split in SPLITS:
                suite_records = grouped[suite_id][split]
                if not suite_records:
                    continue
                result = run_suite_v2(
                    runner,
                    suite_id,
                    suite_records,
                    counters[split],
                )
                suite_rows = cast(list[dict[str, Any]], result["predictions"])
                predictions.extend(suite_rows)
                if progress is not None:
                    progress(
                        name,
                        suite_id,
                        split,
                        cast(int, result["decisions"]),
                    )
                if split == "evaluation":
                    evaluation_suites[suite_id] = {
                        "accuracy": result["accuracy"],
                        "cases": result["cases"],
                        "decisions": result["decisions"],
                        "latency": _latency_summary(suite_rows),
                    }
        decisions, accuracy, scored_suites = _score_rows(
            [row for row in predictions if row["split"] == "evaluation"]
        )
        if not _json_values_equal(scored_suites, evaluation_suites):
            raise ValueError("evaluation suite metrics could not be reproduced")
        metadata = {
            "dataset_version": manifest["dataset_version"],
            "manifest": manifest,
            "manifest_digest": canonical_digest,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "model": _runner_metadata(runner),
            "requested_runner": name,
            "run_id": identifier,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "seed": SEED,
            "suites": _suite_metadata(records, suite_order),
            "timing": {"clock": "time.perf_counter", "unit": "seconds"},
        }
        summary = {
            "accuracy": accuracy,
            "dataset_version": manifest["dataset_version"],
            "cases": len([row for row in predictions if row["split"] == "evaluation"]),
            "decisions": decisions,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "seed": SEED,
            "split": "evaluation",
            "suites": evaluation_suites,
        }
        split_usage = {split: counters[split].as_dict() for split in SPLITS}
        usage = {
            "calibration": split_usage["calibration"],
            "evaluation": split_usage["evaluation"],
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "totals": _total_usage(list(split_usage.values())),
        }
        _validate_payloads((metadata, summary, usage), predictions)
        prepared.append((identifier, target, metadata, predictions, summary, usage))
    written: list[Path] = []
    for identifier, target, metadata, predictions, summary, usage in prepared:
        _claim_target(target, identifier)
        write_json_exclusive(target / "metadata.json", metadata)
        write_jsonl_exclusive(target / "predictions.jsonl", predictions)
        write_json_exclusive(target / "summary.json", summary)
        write_json_exclusive(target / "usage.json", usage)
        if not _has_owned_run_marker(target, identifier):
            write_checksums(target)
        written.append(target)
    return written


def _validate_metadata(
    metadata: Mapping[str, Any], source: str, *, expected_run_id: str
) -> dict[str, Any]:
    value = _json_mapping(metadata, source)
    _artifact_integer(
        value.get("schema_version"),
        ARTIFACT_SCHEMA_VERSION,
        f"{source} artifact schema version",
    )
    _digest_text(value.get("manifest_digest"), f"{source} manifest digest")
    _artifact_integer(
        value.get("metric_schema_version"),
        METRIC_SCHEMA_VERSION,
        f"{source} metric schema version",
    )
    _artifact_integer(value.get("seed"), SEED, f"{source} seed")
    run_id = _nonempty_text(value.get("run_id"), f"{source} run ID")
    if run_id != expected_run_id:
        raise ValueError(f"{source} run ID must equal the run directory basename")
    model = _json_mapping(value.get("model"), f"{source} model identity")
    for field in _MODEL_IDENTITY_FIELDS:
        _nonempty_text(model.get(field), f"{source} model identity {field}")
    manifest = _json_mapping(value.get("manifest"), f"{source} manifest")
    dataset_version = _nonempty_text(manifest.get("dataset_version"), f"{source} dataset version")
    if value.get("dataset_version") != dataset_version:
        raise ValueError(f"{source} dataset version is not bound to manifest metadata")
    if manifest.get("digest") != value.get("manifest_digest"):
        raise ValueError(f"{source} manifest identity is inconsistent")
    _digest_text(manifest.get("file_digest"), f"{source} manifest file digest")
    checksum_verified = manifest.get("checksum_verified")
    checksum_digest = manifest.get("checksum_digest")
    if not isinstance(checksum_verified, bool):
        raise TypeError(f"{source} manifest checksum state is invalid")
    if checksum_verified:
        if _digest_text(checksum_digest, f"{source} manifest checksum digest") != manifest.get(
            "file_digest"
        ):
            raise ValueError(f"{source} manifest checksum identity is inconsistent")
    elif checksum_digest is not None:
        raise ValueError(f"{source} unsealed manifest has a checksum digest")
    suites = value.get("suites")
    if not isinstance(suites, list):
        raise TypeError(f"{source} suite metadata must be a list")
    canonical_json(value)
    return value


def _parse_checksum_line(line: str, source: str) -> tuple[str, str]:
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match is None:
        raise ValueError(f"{source} contains a malformed checksum line")
    relative = PurePosixPath(match.group(2))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() == "checksums.sha256"
    ):
        raise ValueError(f"{source} contains an unsafe checksum path")
    return match.group(1), relative.as_posix()


def _validate_artifact_checksums(run_path: Path) -> Path:
    if run_path.is_symlink() or not run_path.is_dir():
        raise ValueError("v2 run path must be a real directory")
    checksum_path = run_path / "checksums.sha256"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("v2 run checksum file is missing")
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("v2 run checksum file is unreadable") from error
    if not lines:
        raise ValueError("v2 run checksum file is empty")
    entries: dict[str, str] = {}
    ordered_paths: list[str] = []
    for line in lines:
        digest, relative = _parse_checksum_line(line, "v2 run checksum")
        if relative in entries:
            raise ValueError("v2 run checksum contains a duplicate path")
        entries[relative] = digest
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        raise ValueError("v2 run checksum paths are not sorted")
    all_paths = list(run_path.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise ValueError("v2 run directory must not contain symlinks")
    actual_files = {
        path.relative_to(run_path).as_posix(): path
        for path in all_paths
        if path.is_file() and path != checksum_path
    }
    if set(entries) != set(actual_files):
        raise ValueError("v2 run checksum file inventory mismatch")
    missing = {
        "metadata.json",
        "predictions.jsonl",
        "summary.json",
        "usage.json",
    } - set(entries)
    if missing:
        raise ValueError("v2 run checksum is missing required artifacts")
    for relative, path in actual_files.items():
        if digest_file(path) != entries[relative]:
            raise ValueError(f"v2 run checksum mismatch: {relative}")
    return checksum_path


def _validate_prediction_row(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    value = _json_mapping(row, source)
    _artifact_integer(
        value.get("schema_version"),
        ARTIFACT_SCHEMA_VERSION,
        f"{source} artifact schema version",
    )
    case_id = _nonempty_text(value.get("case_id"), f"{source} case ID")
    suite_id = _nonempty_text(value.get("suite_id"), f"{source} suite ID")
    split = _nonempty_text(value.get("split"), f"{source} split")
    if split not in SPLITS or value.get("phase") != split:
        raise ValueError(f"{source} split identity is invalid")
    if value.get("execution_phase") != "benchmark":
        raise ValueError(f"{source} execution phase identity is invalid")
    _nonnegative_integer(value.get("order_index"), f"{source} order index")
    questions = value.get("questions")
    if not isinstance(questions, list):
        raise TypeError(f"{source} questions must be an ordered list")
    typed_questions = [cast(Mapping[str, Any], question) for question in questions]
    answers = _json_mapping(value.get("answers"), f"{source} answers")
    validate_answers(typed_questions, answers)
    question_ids = value.get("question_ids")
    if question_ids != [question.get("qid") for question in typed_questions]:
        raise ValueError(f"{source} question definition order is invalid")
    expected = _json_mapping(value.get("expected"), f"{source} expected")
    if set(expected) != set(cast(list[str], question_ids)):
        raise ValueError(f"{source} expected question IDs differ")
    _validate_expected_values(
        [
            {
                "questions": typed_questions,
                "expected": expected,
            }
        ]
    )
    confidence = _json_mapping(value.get("confidence"), f"{source} confidence")
    if set(confidence) != set(cast(list[str], question_ids)):
        raise ValueError(f"{source} confidence question IDs differ")
    if not _json_values_equal(confidence, _confidence(answers)):
        raise ValueError(f"{source} confidence differs from normalized answers")
    _json_mapping(value.get("_usage"), f"{source} usage")
    latency = _finite_number(value.get("latency_seconds"), f"{source} latency")
    if latency < 0.0:
        raise ValueError(f"{source} latency must be non-negative")
    _digest_text(value.get("state_digest"), f"{source} state digest")
    row_identity = _json_mapping(value.get("runner"), f"{source} runner identity")
    expected_identity = _stable_model_identity(
        _json_mapping(metadata.get("model"), f"{source} model identity")
    )
    if not _json_values_equal(row_identity, expected_identity):
        raise ValueError(f"{source} model identity differs from run metadata")
    canonical_json(value)
    return {
        "case_id": case_id,
        "expected": digest_value(expected),
        "order_index": value["order_index"],
        "question_identity": digest_value(typed_questions),
        "question_ids": tuple(cast(list[str], question_ids)),
        "split": split,
        "state_digest": value["state_digest"],
        "suite_id": suite_id,
        "values": value,
    }


def _suite_identity(metadata: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    suites = cast(list[Any], metadata["suites"])
    result: list[dict[str, Any]] = []
    for raw_suite in suites:
        if not isinstance(raw_suite, Mapping):
            raise TypeError(f"{source} suite metadata entry is invalid")
        suite = _json_mapping(raw_suite, f"{source} suite metadata")
        suite_id = _nonempty_text(suite.get("suite_id"), f"{source} suite ID")
        case_ids = suite.get("case_ids")
        order_indices = suite.get("order_indices")
        if not isinstance(case_ids, list) or not isinstance(order_indices, list):
            raise TypeError(f"{source} suite case identity is invalid")
        if len(case_ids) != len(order_indices):
            raise ValueError(f"{source} suite case identity is invalid")
        for case_id in case_ids:
            _nonempty_text(case_id, f"{source} suite case ID")
        for order_index in order_indices:
            _nonnegative_integer(order_index, f"{source} suite order index")
        splits = _json_mapping(suite.get("splits"), f"{source} suite splits")
        case_id_list = cast(list[str], case_ids)
        if len(set(case_id_list)) != len(case_id_list):
            raise ValueError(f"{source} suite case IDs must be unique")
        split_case_ids: list[str] = []
        for split in SPLITS:
            values = splits.get(split)
            if not isinstance(values, list) or any(
                not isinstance(case_id, str) or not case_id for case_id in values
            ):
                raise ValueError(f"{source} suite split identity is invalid")
            split_values = cast(list[str], values)
            if len(set(split_values)) != len(split_values):
                raise ValueError(f"{source} suite split case IDs must be unique")
            split_case_ids.extend(split_values)
        if len(split_case_ids) != len(set(split_case_ids)) or set(split_case_ids) != set(
            case_id_list
        ):
            raise ValueError(f"{source} suite split membership is not an exact partition")
        result.append(
            {
                "case_ids": case_id_list,
                "order_indices": cast(list[int], order_indices),
                "splits": splits,
                "suite_id": suite_id,
            }
        )
    manifest_suite_order = cast(dict[str, Any], metadata["manifest"]).get("suite_order")
    if not isinstance(manifest_suite_order, list) or any(
        not isinstance(suite_id, str) or not suite_id for suite_id in manifest_suite_order
    ):
        raise ValueError(f"{source} manifest suite order is invalid")
    if [suite["suite_id"] for suite in result] != manifest_suite_order:
        raise ValueError(f"{source} suite IDs or order differ")
    return result


def _validate_rows_against_metadata(
    rows: Sequence[Mapping[str, Any]],
    suites: Sequence[Mapping[str, Any]],
    source: str,
) -> None:
    expected: list[tuple[str, str, int, str]] = []
    for suite in suites:
        case_ids = cast(list[str], suite["case_ids"])
        order_indices = cast(list[int], suite["order_indices"])
        order_by_case = dict(zip(case_ids, order_indices, strict=True))
        splits = cast(dict[str, Any], suite["splits"])
        for split in SPLITS:
            expected.extend(
                (cast(str, suite["suite_id"]), case_id, order_by_case[case_id], split)
                for case_id in cast(list[str], splits[split])
            )
    actual = [
        (
            cast(str, row["suite_id"]),
            cast(str, row["case_id"]),
            cast(int, row["order_index"]),
            cast(str, row["split"]),
        )
        for row in rows
    ]
    if len(actual) != len(set(actual)):
        raise ValueError(f"{source} prediction rows contain duplicate case identities")
    if actual != expected:
        raise ValueError(f"{source} prediction row order does not match metadata suite order")


def _compare_identity(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> None:
    if len(left) != len(right):
        raise ValueError("suite IDs or case counts differ")
    for left_suite, right_suite in zip(left, right):
        if left_suite["suite_id"] != right_suite["suite_id"]:
            raise ValueError("suite IDs differ")
        if left_suite["case_ids"] != right_suite["case_ids"]:
            raise ValueError("case IDs or order differ")
        if left_suite["order_indices"] != right_suite["order_indices"]:
            raise ValueError("case order differs")
        if left_suite["splits"] != right_suite["splits"]:
            raise ValueError("split identity differs")


def _compare_prediction_identity(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> None:
    if len(left) != len(right):
        raise ValueError("case IDs or order differ")
    for left_row, right_row in zip(left, right):
        if (left_row["suite_id"], left_row["case_id"]) != (
            right_row["suite_id"],
            right_row["case_id"],
        ):
            raise ValueError("case IDs or order differ")
        if left_row["order_index"] != right_row["order_index"]:
            raise ValueError("case order differs")
        if left_row["question_identity"] != right_row["question_identity"]:
            raise ValueError("question definition or order differs")
        if left_row["question_ids"] != right_row["question_ids"]:
            raise ValueError("question definition or order differs")
        if left_row["split"] != right_row["split"]:
            raise ValueError("split identity differs")
        if left_row["state_digest"] != right_row["state_digest"]:
            raise ValueError("state identity differs")
        if left_row["expected"] != right_row["expected"]:
            raise ValueError("expected answer identity differs")


def _validated_phase_usage(value: object, source: str) -> dict[str, dict[str, int]]:
    usage = _json_mapping(value, source)
    if set(usage) != set(MODEL_PHASES):
        raise ValueError(f"{source} phase set is invalid")
    result: dict[str, dict[str, int]] = {}
    for phase in MODEL_PHASES:
        phase_usage = _json_mapping(usage.get(phase), f"{source} {phase} usage")
        if set(phase_usage) != {"input_tokens", "output_tokens", "calls"}:
            raise ValueError(f"{source} {phase} usage fields are invalid")
        result[phase] = {
            "input_tokens": _nonnegative_integer(
                phase_usage.get("input_tokens"), f"{source} {phase} input tokens"
            ),
            "output_tokens": _nonnegative_integer(
                phase_usage.get("output_tokens"), f"{source} {phase} output tokens"
            ),
            "calls": _nonnegative_integer(phase_usage.get("calls"), f"{source} {phase} calls"),
        }
    return result


def _validate_usage(
    usage: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source: str,
) -> dict[str, Any]:
    value = _json_mapping(usage, source)
    _artifact_integer(
        value.get("schema_version"),
        ARTIFACT_SCHEMA_VERSION,
        f"{source} usage artifact schema version",
    )
    split_usage = {
        split: _validated_phase_usage(value.get(split), f"{source} {split} usage")
        for split in SPLITS
    }
    totals = _validated_phase_usage(value.get("totals"), f"{source} total usage")
    expected_totals = _total_usage([split_usage["calibration"], split_usage["evaluation"]])
    if totals != expected_totals:
        raise ValueError(f"{source} usage totals are inconsistent")
    for split in SPLITS:
        for phase in ("warmup", "speed"):
            if split_usage[split][phase] != _zero_usage()[phase]:
                raise ValueError(f"{source} is benchmark-only; {split} {phase} usage must be zero")
        expected_calls = sum(1 for row in rows if row["split"] == split)
        if split_usage[split]["benchmark"]["calls"] != expected_calls:
            raise ValueError(f"{source} {split} usage call count is inconsistent")
        expected_input = 0
        expected_output = 0
        for row in rows:
            if row["split"] != split:
                continue
            row_usage = _json_mapping(row["values"].get("_usage"), f"{source} row usage")
            if "input_tokens" in row_usage:
                expected_input += _nonnegative_integer(
                    row_usage["input_tokens"], f"{source} row input tokens"
                )
            if "output_tokens" in row_usage:
                expected_output += _nonnegative_integer(
                    row_usage["output_tokens"], f"{source} row output tokens"
                )
        if split_usage[split]["benchmark"]["input_tokens"] != expected_input:
            raise ValueError(f"{source} {split} input token total is inconsistent")
        if split_usage[split]["benchmark"]["output_tokens"] != expected_output:
            raise ValueError(f"{source} {split} output token total is inconsistent")
    canonical_json(value)
    return value


def _validate_summary(
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source: str,
) -> dict[str, Any]:
    value = _json_mapping(summary, source)
    _artifact_integer(
        value.get("schema_version"),
        ARTIFACT_SCHEMA_VERSION,
        f"{source} artifact schema version",
    )
    _artifact_integer(
        value.get("metric_schema_version"),
        METRIC_SCHEMA_VERSION,
        f"{source} metric schema version",
    )
    if value.get("dataset_version") != metadata.get("dataset_version"):
        raise ValueError(f"{source} dataset version differs from run metadata")
    _artifact_integer(value.get("seed"), SEED, f"{source} seed")
    if value.get("split") != "evaluation":
        raise ValueError(f"{source} headline split must be evaluation")
    evaluation_rows = [
        cast(Mapping[str, Any], row["values"]) for row in rows if row["split"] == "evaluation"
    ]
    decisions, expected_accuracy, expected_suites = _score_rows(evaluation_rows)
    cases = _nonnegative_integer(value.get("cases"), f"{source} cases")
    decision_count = _nonnegative_integer(value.get("decisions"), f"{source} decisions")
    accuracy = _optional_finite_number(value.get("accuracy"), f"{source} accuracy")
    if not _json_values_equal(cases, len(evaluation_rows)) or not _json_values_equal(
        decision_count, decisions
    ):
        raise ValueError(f"{source} decision counts are inconsistent")
    if not _json_values_equal(accuracy, expected_accuracy):
        raise ValueError(f"{source} metrics are inconsistent with predictions")
    suites_value = _json_mapping(value.get("suites"), f"{source} suites")
    if set(suites_value) != set(expected_suites):
        raise ValueError(f"{source} suite metrics are inconsistent with predictions")
    for suite_id, expected_suite in expected_suites.items():
        suite = _json_mapping(suites_value.get(suite_id), f"{source} {suite_id} suite metrics")
        normalized_suite = dict(suite)
        normalized_suite["cases"] = _nonnegative_integer(
            suite.get("cases"), f"{source} {suite_id} cases"
        )
        normalized_suite["decisions"] = _nonnegative_integer(
            suite.get("decisions"), f"{source} {suite_id} decisions"
        )
        normalized_suite["accuracy"] = _optional_finite_number(
            suite.get("accuracy"), f"{source} {suite_id} accuracy"
        )
        normalized_suite["latency"] = _validated_latency_summary(
            suite.get("latency"), f"{source} {suite_id} latency"
        )
        if not _json_values_equal(normalized_suite, expected_suite):
            raise ValueError(f"{source} metrics are inconsistent with predictions")
    canonical_json(value)
    return value


def _comparison_id(run_a: Mapping[str, Any], run_b: Mapping[str, Any]) -> str:
    first = _safe_component(cast(str, run_a["run_id"]))[:48]
    second = _safe_component(cast(str, run_b["run_id"]))[:48]
    return f"compare-{first}-vs-{second}-{digest_value([run_a['run_id'], run_b['run_id']])[:12]}"


def compare_v2(path_a: Path, path_b: Path, output_root: Path) -> Path:
    if (
        not isinstance(path_a, Path)
        or not isinstance(path_b, Path)
        or not isinstance(output_root, Path)
    ):
        raise TypeError("comparison paths must be Path values")
    _reject_symlink_components(path_a, "run A path")
    _reject_symlink_components(path_b, "run B path")
    _reject_symlink_components(output_root, "comparison output root")
    for run_path in (path_a, path_b):
        if _paths_overlap(output_root, run_path):
            raise ValueError(
                "comparison output root must not equal, contain, or be contained by a run path"
            )
    _reject_source_tree_symlinks(path_a, "run A source tree")
    _reject_source_tree_symlinks(path_b, "run B source tree")
    raw_metadata_a = _read_json_object(path_a / "metadata.json", "run A metadata")
    raw_metadata_b = _read_json_object(path_b / "metadata.json", "run B metadata")
    raw_digest_a = raw_metadata_a.get("manifest_digest")
    raw_digest_b = raw_metadata_b.get("manifest_digest")
    if (
        isinstance(raw_digest_a, str)
        and isinstance(raw_digest_b, str)
        and raw_digest_a != raw_digest_b
    ):
        raise ValueError("manifest digests differ")
    metadata_a = _validate_metadata(
        raw_metadata_a,
        "run A",
        expected_run_id=_normalized_path(path_a).name,
    )
    metadata_b = _validate_metadata(
        raw_metadata_b,
        "run B",
        expected_run_id=_normalized_path(path_b).name,
    )
    if metadata_a["manifest_digest"] != metadata_b["manifest_digest"]:
        raise ValueError("manifest digests differ")
    manifest_a = cast(dict[str, Any], metadata_a["manifest"])
    manifest_b = cast(dict[str, Any], metadata_b["manifest"])
    if manifest_a.get("dataset_version") != manifest_b.get("dataset_version") or metadata_a.get(
        "dataset_version"
    ) != metadata_b.get("dataset_version"):
        raise ValueError("manifest dataset versions differ")
    if (
        manifest_a.get("file_digest") != manifest_b.get("file_digest")
        or manifest_a.get("checksum_verified") != manifest_b.get("checksum_verified")
        or manifest_a.get("checksum_digest") != manifest_b.get("checksum_digest")
    ):
        raise ValueError("manifest checksum identity differs")
    if metadata_a["metric_schema_version"] != metadata_b["metric_schema_version"]:
        raise ValueError("metric schema versions differ")
    suites_a = _suite_identity(metadata_a, "run A")
    suites_b = _suite_identity(metadata_b, "run B")
    _compare_identity(suites_a, suites_b)
    checksum_a = _validate_artifact_checksums(path_a)
    checksum_b = _validate_artifact_checksums(path_b)
    raw_rows_a = _read_jsonl_objects(path_a / "predictions.jsonl", "run A predictions")
    raw_rows_b = _read_jsonl_objects(path_b / "predictions.jsonl", "run B predictions")
    rows_a = [
        _validate_prediction_row(row, metadata_a, f"run A prediction {index}")
        for index, row in enumerate(raw_rows_a)
    ]
    rows_b = [
        _validate_prediction_row(row, metadata_b, f"run B prediction {index}")
        for index, row in enumerate(raw_rows_b)
    ]
    _validate_rows_against_metadata(rows_a, suites_a, "run A")
    _validate_rows_against_metadata(rows_b, suites_b, "run B")
    _compare_prediction_identity(rows_a, rows_b)
    summary_a = _validate_summary(
        _read_json_object(path_a / "summary.json", "run A summary"),
        metadata_a,
        rows_a,
        "run A",
    )
    summary_b = _validate_summary(
        _read_json_object(path_b / "summary.json", "run B summary"),
        metadata_b,
        rows_b,
        "run B",
    )
    _validate_usage(
        _read_json_object(path_a / "usage.json", "run A usage"),
        rows_a,
        "run A",
    )
    _validate_usage(
        _read_json_object(path_b / "usage.json", "run B usage"),
        rows_b,
        "run B",
    )
    suite_ids = [suite["suite_id"] for suite in suites_a]
    suite_comparison: dict[str, Any] = {}
    for suite_id in suite_ids:
        left = cast(dict[str, Any], summary_a["suites"])[suite_id]
        right = cast(dict[str, Any], summary_b["suites"])[suite_id]
        left_accuracy = left.get("accuracy")
        right_accuracy = right.get("accuracy")
        delta = (
            None
            if left_accuracy is None or right_accuracy is None
            else cast(float, right_accuracy) - cast(float, left_accuracy)
        )
        suite_comparison[suite_id] = {
            "a": left,
            "accuracy_delta_b_minus_a": delta,
            "b": right,
        }
    comparison = {
        "dataset_version": metadata_a["dataset_version"],
        "manifest": {
            "checksum_digest": manifest_a.get("checksum_digest"),
            "checksum_verified": manifest_a.get("checksum_verified"),
            "digest": metadata_a["manifest_digest"],
        },
        "metric_schema_version": metadata_a["metric_schema_version"],
        "models": {
            "a": metadata_a["model"],
            "b": metadata_b["model"],
        },
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "seed": metadata_a["seed"],
        "sources": {
            "a": {
                "checksums_sha256": digest_file(checksum_a),
                "path": str(path_a),
                "run_id": metadata_a["run_id"],
            },
            "b": {
                "checksums_sha256": digest_file(checksum_b),
                "path": str(path_b),
                "run_id": metadata_b["run_id"],
            },
        },
        "suites": suite_comparison,
    }
    canonical_json(comparison)
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise ValueError("comparison output root must be a real directory")
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = create_run_directory(output_root, _comparison_id(metadata_a, metadata_b))
    write_json_exclusive(comparison_path / "comparison.json", comparison)
    write_checksums(comparison_path)
    return comparison_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark.orchestrator")
    parser.add_argument("--models", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-checksum", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def _required_options(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(
        option
        for option, value in (
            ("--models", args.models),
            ("--output-root", args.output_root),
            ("--manifest", args.manifest),
            ("--manifest-checksum", args.manifest_checksum),
            ("--run-id", args.run_id),
        )
        if not isinstance(value, str) or not value
    )


def _print_run_progress(model: str, suite_id: str, split: str, decisions: int) -> None:
    print(f"[{model}] {suite_id}/{split}: {decisions} decisions")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        return 1
    missing = _required_options(args)
    if missing:
        print(f"missing required options: {', '.join(missing)}", file=sys.stderr)
        return 2
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if not models:
        print("run failed: --models must contain at least one model", file=sys.stderr)
        return 2
    try:
        paths = run_all_v2(
            models,
            Path(args.manifest),
            Path(args.output_root),
            manifest_checksum_path=Path(args.manifest_checksum),
            run_id=args.run_id,
            progress=_print_run_progress,
        )
    except RuntimeError:
        print("run failed: runner execution failed", file=sys.stderr)
        return 1
    except (ImportError, OSError, TypeError, ValueError) as error:
        print(f"run failed: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"run artifacts: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
