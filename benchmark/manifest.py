from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from benchmark.canonical import digest_value
from benchmark.contracts import _validate_question

_REQUIRED_RECORD_FIELDS = (
    "schema_version",
    "dataset_version",
    "suite_id",
    "case_id",
    "order_index",
    "split",
    "state",
    "questions",
    "expected",
    "provenance_id",
    "label_provenance",
)
_ALLOWED_RECORD_FIELDS = frozenset(_REQUIRED_RECORD_FIELDS)
_ALLOWED_SPLITS = frozenset({"calibration", "evaluation"})


@dataclass
class ManifestSummary:
    total_cases: int
    total_decisions: int
    by_suite: dict[str, int]
    digest: str


def _format_keys(keys: Iterable[object]) -> str:
    ordered = sorted(keys, key=lambda key: (type(key).__name__, repr(key)))
    return ", ".join(repr(key) for key in ordered)


def _value_error(message: str) -> ValueError:
    return ValueError(message)


def _validate_json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _value_error(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise _value_error(f"{name} object keys must be strings")
            _validate_json_value(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{name}[{index}]")
        return
    raise _value_error(f"{name} must contain only JSON values")


def _require_nonempty_text(record: Mapping[str, Any], field: str, index: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value:
        raise _value_error(f"record {index}: {field} must be a non-empty string")
    return value


def _json_integer(value: object, field: str, index: int, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _value_error(f"record {index}: {field} must be an integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise _value_error(f"record {index}: {field} must be an integer")
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise _value_error(f"record {index}: {field} must be a non-negative integer")
    return integer


def _validate_expected_value(
    question: Mapping[str, Any], expected: object, qid: str, index: int
) -> None:
    question_type = question.get("type")
    if question_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or not isinstance(expected, str):
            raise _value_error(f"record {index}: {qid} expected choice must be a legal label")
        if expected not in criteria:
            raise _value_error(f"record {index}: {qid} expected choice must be a legal label")
        return
    if question_type == "noul":
        if isinstance(expected, bool) or not isinstance(expected, int) or expected not in (0, 1):
            raise _value_error(f"record {index}: {qid} expected noul value must be integer 0 or 1")
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
            raise _value_error(
                f"record {index}: {qid} expected score must be a finite numeric level in range"
            )
        return
    raise _value_error(f"record {index}: {qid} expected value has an unsupported question type")


def _validate_record(record: Mapping[str, Any], index: int) -> tuple[str, int, int]:
    missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise _value_error(f"record {index}: missing required field(s): {_format_keys(missing)}")

    unknown = [field for field in record if field not in _ALLOWED_RECORD_FIELDS]
    if unknown:
        raise _value_error(f"record {index}: additional record field(s): {_format_keys(unknown)}")

    schema_version = _json_integer(record["schema_version"], "schema_version", index)
    if schema_version != 2:
        raise _value_error(f"record {index}: schema_version must equal 2")

    _require_nonempty_text(record, "dataset_version", index)
    suite_id = _require_nonempty_text(record, "suite_id", index)
    _require_nonempty_text(record, "case_id", index)
    _require_nonempty_text(record, "provenance_id", index)
    _require_nonempty_text(record, "label_provenance", index)

    order_index = _json_integer(record["order_index"], "order_index", index, minimum=0)

    split = record["split"]
    if not isinstance(split, str) or split not in _ALLOWED_SPLITS:
        raise _value_error(f"record {index}: split must be calibration or evaluation")

    state = record["state"]
    if not isinstance(state, Mapping):
        raise _value_error(f"record {index}: state must be an object")
    _validate_json_value(state, f"record {index}.state")

    questions = record["questions"]
    if isinstance(questions, (str, bytes, bytearray)) or not isinstance(questions, Sequence):
        raise _value_error(f"record {index}: questions must be an ordered sequence")

    question_ids: list[str] = []
    question_objects: list[Mapping[str, Any]] = []
    seen_question_ids: set[str] = set()
    for position, question in enumerate(questions):
        if not isinstance(question, Mapping):
            raise _value_error(f"record {index}: question {position} must be an object")
        question_object = cast(Mapping[str, Any], question)
        try:
            qid, _ = _validate_question(question_object)
        except ValueError as error:
            raise _value_error(f"record {index}: question {position}: {error}") from None
        if qid in seen_question_ids:
            raise _value_error(
                f"record {index}: question {position}: duplicate question id {qid!r}"
            )
        seen_question_ids.add(qid)
        question_ids.append(qid)
        question_objects.append(question_object)
        _validate_json_value(question_object, f"record {index}.questions[{position}]")

    expected = record["expected"]
    if not isinstance(expected, Mapping):
        raise _value_error(f"record {index}: expected must be an object")
    expected_keys = list(expected.keys())
    if any(not isinstance(key, str) or not key for key in expected_keys):
        raise _value_error(f"record {index}: expected IDs must be non-empty strings")
    expected_id_set = set(expected_keys)
    question_id_set = set(question_ids)
    if expected_id_set != question_id_set:
        missing_ids = sorted(question_id_set - expected_id_set)
        extra_ids = sorted(expected_id_set - question_id_set)
        raise _value_error(
            f"record {index}: expected IDs differ: missing={missing_ids}, extra={extra_ids}"
        )
    _validate_json_value(expected, f"record {index}.expected")
    for question in question_objects:
        qid = cast(str, question["qid"])
        _validate_expected_value(question, expected[qid], qid, index)

    _validate_json_value(record, f"record {index}")
    return suite_id, order_index, len(expected_keys)


def _validate_expected_counts(expected_counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(expected_counts, Mapping):
        raise _value_error("expected_counts must be an object")

    normalized: dict[str, int] = {}
    for suite_id, count in expected_counts.items():
        if not isinstance(suite_id, str) or not suite_id:
            raise _value_error("expected_counts suite IDs must be non-empty strings")
        if type(count) is not int or count < 0:
            raise _value_error(f"expected_counts count for {suite_id!r} must be non-negative")
        normalized[suite_id] = count
    return normalized


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _value_error(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise _value_error(f"non-standard JSON constant {value!r}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise _value_error(f"manifest is not valid UTF-8 JSON: {error}") from None

    try:
        value: object = json.loads(
            contents,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise _value_error(
            f"manifest contains malformed JSON at line {error.lineno}, column {error.colno}"
        ) from None
    except ValueError as error:
        raise _value_error(f"manifest contains malformed JSON: {error}") from None

    if not isinstance(value, list):
        raise _value_error("manifest must contain a JSON list of records")

    records: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise _value_error(f"manifest item {index} must be a JSON object")
        records.append(record)
    return records


def validate_manifest(
    records: Sequence[Mapping[str, Any]], expected_counts: Mapping[str, int]
) -> ManifestSummary:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise _value_error("manifest records must be a sequence")
    if not isinstance(expected_counts, Mapping):
        raise _value_error("expected_counts must be an object")

    expected_by_suite = _validate_expected_counts(expected_counts)
    by_suite: dict[str, int] = {}
    total_decisions = 0
    seen_case_ids: set[str] = set()
    seen_order_pairs: set[tuple[str, int]] = set()

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be an object")
        suite_id, order_index, decision_count = _validate_record(record, index)
        total_decisions += decision_count
        case_id = cast(str, record["case_id"])
        if case_id in seen_case_ids:
            raise _value_error(f"record {index}: duplicate case ID {case_id!r}")
        seen_case_ids.add(case_id)
        order_pair = (suite_id, order_index)
        if order_pair in seen_order_pairs:
            raise _value_error(
                f"record {index}: duplicate suite_id/order_index pair ({suite_id!r}, {order_index})"
            )
        seen_order_pairs.add(order_pair)
        by_suite[suite_id] = by_suite.get(suite_id, 0) + 1

    expected_suites = set(expected_by_suite)
    actual_suites = set(by_suite)
    if expected_suites != actual_suites:
        missing_suites = sorted(expected_suites - actual_suites)
        extra_suites = sorted(actual_suites - expected_suites)
        raise _value_error(f"suite counts differ: missing={missing_suites}, extra={extra_suites}")

    for suite_id in sorted(actual_suites):
        expected_count = expected_by_suite[suite_id]
        actual_count = by_suite[suite_id]
        if actual_count != expected_count:
            raise _value_error(
                f"suite {suite_id!r} count mismatch: expected {expected_count}, got {actual_count}"
            )

    try:
        digest = manifest_digest(records)
    except (TypeError, ValueError) as error:
        raise _value_error(f"manifest cannot be canonicalized: {error}") from None

    return ManifestSummary(
        total_cases=len(records),
        total_decisions=total_decisions,
        by_suite=by_suite,
        digest=digest,
    )


def manifest_digest(records: Sequence[Mapping[str, Any]]) -> str:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise _value_error("manifest records must be a sequence")
    try:
        digest: str = digest_value(records)
        return digest
    except (TypeError, ValueError) as error:
        raise _value_error(f"manifest cannot be canonicalized: {error}") from None
