from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn, cast

from benchmark.canonical import canonical_json
from benchmark.manifest import manifest_digest, validate_manifest

from .validate import validate_dataset

PACKET_SEED = 42
SEED = PACKET_SEED
EXPORT_SCHEMA_VERSION = 2
REVIEWER_CODES = frozenset({"A", "B"})
ADJUDICATED_LABEL_PROVENANCE = "adjudicated-v2"
MANIFEST_NAME = "manifest.jsonl"
PROVENANCE_NAME = "provenance.json"
SEALED_NAME = "manifest.sha256"
_SEAL_LOCK_NAME = ".manifest.seal.lock"
_REQUIRED_SEAL_INPUTS = (
    Path(MANIFEST_NAME),
    Path(PROVENANCE_NAME),
    Path("labels/reviewer_a.json"),
    Path("labels/reviewer_b.json"),
    Path("labels/adjudication.json"),
)
_ANSWER_TYPES = frozenset({"choice", "noul", "score"})
_MULTILINGUAL_LANGUAGES = frozenset({"hi", "es", "fr", "de", "ar"})
_BLOCKED_STATE_KEYS = frozenset(
    {
        "answer",
        "answers",
        "calibration",
        "canonical",
        "case_id",
        "checksum",
        "class",
        "citation",
        "dataset",
        "dataset_id",
        "dataset_version",
        "evaluation",
        "expected",
        "ground_truth",
        "groundtruth",
        "label",
        "label_provenance",
        "label_provenance_id",
        "label_text",
        "label_value",
        "labels",
        "license",
        "local_modification",
        "manifest_digest",
        "manifest_id",
        "model",
        "model_id",
        "model_name",
        "model_output",
        "model_response",
        "model_result",
        "order_index",
        "original_row_id",
        "prediction",
        "predictions",
        "prior_result",
        "prior_results",
        "provenance",
        "provenance_id",
        "record_id",
        "result",
        "results",
        "revision",
        "row_id",
        "schema",
        "schema_version",
        "source",
        "source_id",
        "source_label",
        "source_metadata",
        "source_row",
        "split",
        "suite",
        "suite_id",
        "target",
        "target_label",
        "weak",
        "wrapper",
    }
)
_BLOCKED_STATE_PREFIXES = (
    "answer_",
    "calibration_",
    "checksum_",
    "dataset_",
    "evaluation_",
    "expected_",
    "ground_truth",
    "label_",
    "label_provenance",
    "label_text",
    "label_value",
    "license_",
    "local_modification",
    "model_",
    "prediction",
    "prior_result",
    "provenance_",
    "result_",
    "revision_",
    "row_",
    "run_",
    "schema_",
    "source_",
    "split_",
    "citation_",
    "suite_",
    "target_",
    "class_",
    "weak_",
)
_BLOCKED_STATE_FRAGMENTS = (
    "adjudicat",
    "annotat",
    "answerkey",
    "class",
    "classlabel",
    "final",
    "gold",
    "groundtruth",
    "human",
    "model",
    "provenance",
    "reference",
    "result",
    "split",
    "supervis",
    "truth",
    "weak",
    "weaklabel",
)
_BLOCKED_STATE_SEQUENCES = (
    ("answer", "key"),
    ("class", "label"),
    ("weak", "label"),
)


@dataclass(frozen=True)
class _Question:
    qid: str
    question_type: str
    instructions: str
    criteria: Mapping[str, str] | Sequence[str] | None = None
    max_score: int | float | None = None

    def as_packet_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "qid": self.qid,
            "type": self.question_type,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            value["criteria"] = self.criteria
        if self.max_score is not None:
            value["max_score"] = self.max_score
        return value


@dataclass(frozen=True)
class _Observation:
    case_id: str
    suite_id: str
    qid: str
    question_type: str
    reviewer_a: str | int | float
    reviewer_b: str | int | float
    language: str | None
    max_score: int | float | None


_PACKET_STYLE = """
:root {
  color-scheme: dark;
  font-family: system-ui, sans-serif;
  --background: #0b1220;
  --surface: #111827;
  --surface-raised: #172033;
  --border: #334155;
  --text: #e5e7eb;
  --muted: #a8b3c4;
  --accent: #60a5fa;
  --accent-strong: #2563eb;
  --focus: #fbbf24;
}
html { background: var(--background); }
body { margin: 0 auto; max-width: 60rem; padding: 1rem; color: var(--text); background: var(--background); }
header, footer { position: sticky; bottom: 0; padding: .75rem; background: var(--surface); border-top: 1px solid var(--border); }
button { margin: .25rem; padding: .5rem .8rem; color: var(--text); background: var(--surface-raised); border: 1px solid var(--border); border-radius: .35rem; }
button:hover { background: var(--accent-strong); border-color: var(--accent); }
button:focus-visible, input:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }
section, fieldset { background: var(--surface); border: 1px solid var(--border); border-radius: .35rem; }
section { margin: 1rem 0; padding: 1rem; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; color: var(--text); background: var(--surface-raised); padding: .75rem; border-radius: .25rem; }
fieldset { margin: 1rem 0; }
label { display: block; margin: .35rem 0; }
label:hover { color: var(--accent); }
label span, small { color: var(--muted); }
input { margin-right: .4rem; accent-color: var(--accent); }
"""

_PACKET_SCRIPT = """
(function () {
  const data = JSON.parse(document.getElementById("packet-data").textContent);
  const storageKey = "blind-review-v2-" + data.reviewer_code;
  const sections = Array.from(document.querySelectorAll("[data-case-id]"));
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const download = document.getElementById("download");
  const progress = document.getElementById("progress");
  let current = 0;

  function controlsFor(section) {
    return Array.from(section.querySelectorAll("input[data-qid]"));
  }

  function readLabels() {
    const labels = {};
    sections.forEach(function (section) {
      const caseId = section.getAttribute("data-case-id");
      const values = {};
      controlsFor(section).forEach(function (control) {
        const qid = control.getAttribute("data-qid");
        if (control.type === "radio" && !control.checked) {
          return;
        }
        if (control.type === "number" || control.getAttribute("data-value-kind") === "number") {
          if (control.value.trim() === "") {
            return;
          }
          values[qid] = Number(control.value);
        } else {
          values[qid] = control.value;
        }
      });
      labels[caseId] = values;
    });
    return labels;
  }

  function readReviewerLabels() {
    const reviewerLabels = {};
    data.records.forEach(function (record) {
      reviewerLabels[record.case_id] = {
        A: record.reviewer_labels.A,
        B: record.reviewer_labels.B
      };
    });
    return reviewerLabels;
  }

  function readReasons() {
    const reasons = {};
    document.querySelectorAll("[data-reason-qid]").forEach(function (input) {
      const section = input.closest("[data-case-id]");
      if (!section) {
        return;
      }
      const caseId = section.getAttribute("data-case-id");
      if (!reasons[caseId]) {
        reasons[caseId] = {};
      }
      reasons[caseId][input.getAttribute("data-reason-qid")] = input.value;
    });
    return reasons;
  }

  function clearDraft() {
    try {
      localStorage.removeItem(storageKey);
    } catch (error) {
    }
  }

  function saveDraft() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(readLabels()));
    } catch (error) {
      return error;
    }
    return null;
  }

  function applyDraft(draft) {
    sections.forEach(function (section) {
      const values = draft[section.getAttribute("data-case-id")];
      if (!values) {
        return;
      }
      controlsFor(section).forEach(function (control) {
        const value = values[control.getAttribute("data-qid")];
        if (value === undefined) {
          return;
        }
        if (control.type === "radio") {
          control.checked = control.value === String(value);
        } else {
          control.value = String(value);
        }
      });
    });
  }

  function isEditingTarget(target) {
    return Boolean(
      target && target.closest && target.closest("input, textarea, select, [contenteditable='true']")
    ) || Boolean(target && target.isContentEditable);
  }

  function show(index) {
    current = Math.max(0, Math.min(index, sections.length - 1));
    sections.forEach(function (section, position) {
      section.hidden = position !== current;
    });
    progress.textContent = "Case " + String(current + 1) + " of " + String(sections.length);
    previous.disabled = current === 0;
    next.disabled = current === sections.length - 1;
  }

  function downloadLabels() {
    const isAdjudication = data.packet_type === "adjudication";
    const payload = isAdjudication
      ? {
          schema_version: 2,
          adjudication_code: "ADJUDICATOR",
          labels: readLabels(),
          reviewer_labels: readReviewerLabels(),
          reasons: readReasons()
        }
      : {schema_version: 2, reviewer_code: data.reviewer_code, labels: readLabels()};
    let objectUrl = null;
    let link = null;
    try {
      const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
      objectUrl = URL.createObjectURL(blob);
      link = document.createElement("a");
      link.href = objectUrl;
      link.download = isAdjudication
        ? "adjudication.json"
        : "reviewer-" + data.reviewer_code.toLowerCase() + "-labels.json";
      document.body.appendChild(link);
      link.click();
      clearDraft();
    } finally {
      if (link) {
        link.remove();
      }
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    }
  }

  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      applyDraft(JSON.parse(saved));
    }
  } catch (error) {
  }

  document.addEventListener("change", saveDraft);
  previous.addEventListener("click", function () { show(current - 1); });
  next.addEventListener("click", function () { show(current + 1); });
  download.addEventListener("click", downloadLabels);
  document.addEventListener("keydown", function (event) {
    if (isEditingTarget(event.target)) {
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      show(current - 1);
    }
    if (event.key === "ArrowRight" || event.key === "PageDown") {
      event.preventDefault();
      show(current + 1);
    }
  });
  if (sections.length === 0) {
    progress.textContent = "No cases";
  } else {
    show(0);
  }
}());
"""


def _value_error(message: str) -> ValueError:
    return ValueError(message)


def _format_keys(keys: Iterable[object]) -> str:
    return ", ".join(
        repr(key) for key in sorted(keys, key=lambda key: (type(key).__name__, repr(key)))
    )


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _question_from_mapping(question: Mapping[str, Any], context: str) -> _Question:
    if not isinstance(question, Mapping):
        raise _value_error(f"{context} must be a mapping")
    qid = question.get("qid")
    if not isinstance(qid, str) or not qid:
        raise _value_error(f"{context} qid must be a non-empty string")
    question_type = question.get("type")
    if not isinstance(question_type, str) or question_type not in _ANSWER_TYPES:
        raise _value_error(f"{qid}: question type is invalid")
    instructions = question.get("instructions", question.get("text", ""))
    if not isinstance(instructions, str):
        raise _value_error(f"{qid}: instructions must be a string")
    if question_type == "choice":
        raw_criteria = question.get("criteria")
        if not isinstance(raw_criteria, Mapping) or not raw_criteria:
            raise _value_error(f"{qid}: choice criteria must be a non-empty mapping")
        choice_criteria: dict[str, str] = {}
        for label, description in raw_criteria.items():
            if not isinstance(label, str) or not label:
                raise _value_error(f"{qid}: choice labels must be non-empty strings")
            if not isinstance(description, str):
                raise _value_error(f"{qid}: choice descriptions must be strings")
            choice_criteria[label] = description
        return _Question(qid, question_type, instructions, choice_criteria)
    if question_type == "noul":
        if "criteria" in question:
            raw_criteria = question["criteria"]
            if not isinstance(raw_criteria, Mapping) or not raw_criteria:
                raise _value_error(f"{qid}: criteria must be a non-empty mapping")
            noul_criteria: dict[str, str] = {}
            for label, description in raw_criteria.items():
                if not isinstance(label, str) or not label:
                    raise _value_error(f"{qid}: criteria labels must be non-empty strings")
                if not isinstance(description, str):
                    raise _value_error(f"{qid}: criteria descriptions must be strings")
                noul_criteria[label] = description
            return _Question(qid, question_type, instructions, noul_criteria)
        return _Question(qid, question_type, instructions)
    max_score = question.get("max_score")
    if not _is_finite_number(max_score) or cast(float, max_score) < 0:
        raise _value_error(f"{qid}: max_score must be a finite nonnegative number")
    score_criteria: Sequence[str] | None = None
    if "criteria" in question:
        raw_criteria = question["criteria"]
        if isinstance(raw_criteria, (str, bytes, bytearray)) or not isinstance(
            raw_criteria, Sequence
        ):
            raise _value_error(f"{qid}: score criteria must be an ordered sequence")
        criteria_values: list[str] = []
        for label in raw_criteria:
            if not isinstance(label, str) or not label:
                raise _value_error(f"{qid}: score criteria must contain non-empty strings")
            criteria_values.append(label)
        if not criteria_values or len(set(criteria_values)) != len(criteria_values):
            raise _value_error(f"{qid}: score criteria must contain unique labels")
        score_criteria = criteria_values
    return _Question(qid, question_type, instructions, score_criteria, cast(float, max_score))


def _record_case_id(record: Mapping[str, Any], index: int) -> str:
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise _value_error(f"record {index} case_id must be a non-empty string")
    return case_id


def _questions_from_records(
    records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, tuple[_Question, ...]]:
    if records is None:
        return {}
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence of mappings")
    result: dict[str, tuple[_Question, ...]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
        case_id = _record_case_id(record, index)
        if case_id in result:
            raise _value_error(f"duplicate case ID {case_id!r}")
        raw_questions = record.get("questions")
        if isinstance(raw_questions, (str, bytes, bytearray)) or not isinstance(
            raw_questions, Sequence
        ):
            raise _value_error(f"{case_id}: questions must be an ordered sequence")
        questions: list[_Question] = []
        seen: set[str] = set()
        for position, question in enumerate(raw_questions):
            normalized = _question_from_mapping(question, f"{case_id} question {position}")
            if normalized.qid in seen:
                raise _value_error(f"{case_id}: duplicate question ID {normalized.qid!r}")
            seen.add(normalized.qid)
            questions.append(normalized)
        result[case_id] = tuple(questions)
    return result


def _questions_from_optional_input(
    questions: Any,
    expected_case_ids: set[str],
) -> dict[str, tuple[_Question, ...]]:
    if questions is None:
        return {}
    if isinstance(questions, Mapping) and "qid" in questions and "type" in questions:
        single_question = (_question_from_mapping(questions, "question 0"),)
        return {case_id: single_question for case_id in expected_case_ids}
    if isinstance(questions, Mapping):
        values = list(questions.values())
        if values and all(
            isinstance(value, Mapping) and "qid" in value and "type" in value for value in values
        ):
            return {
                case_id: tuple(
                    _question_from_mapping(value, f"question {index}")
                    for index, value in enumerate(values)
                )
                for case_id in expected_case_ids
            }
        result: dict[str, tuple[_Question, ...]] = {}
        for case_id, value in questions.items():
            if (
                not isinstance(case_id, str)
                or not isinstance(value, Sequence)
                or isinstance(value, (str, bytes, bytearray))
            ):
                raise _value_error("questions must map case IDs to ordered question sequences")
            normalized: list[_Question] = []
            seen: set[str] = set()
            for index, question in enumerate(value):
                current = _question_from_mapping(question, f"{case_id} question {index}")
                if current.qid in seen:
                    raise _value_error(f"{case_id}: duplicate question ID {current.qid!r}")
                seen.add(current.qid)
                normalized.append(current)
            result[case_id] = tuple(normalized)
        return result
    if isinstance(questions, (str, bytes, bytearray)) or not isinstance(questions, Sequence):
        raise TypeError("questions must be an ordered sequence or mapping")
    values = list(questions)
    if values and all(
        isinstance(value, Mapping) and "case_id" in value and "questions" in value
        for value in values
    ):
        return _questions_from_records(cast(Sequence[Mapping[str, Any]], values))
    shared_questions = tuple(
        _question_from_mapping(value, f"question {index}") for index, value in enumerate(values)
    )
    shared_seen: set[str] = set()
    for question in shared_questions:
        if question.qid in shared_seen:
            raise _value_error(f"duplicate question ID {question.qid!r}")
        shared_seen.add(question.qid)
    return {case_id: shared_questions for case_id in expected_case_ids}


def _looks_like_question_input(value: Any) -> bool:
    if isinstance(value, Mapping):
        values = list(value.values())
        return bool(values) and all(
            isinstance(item, Mapping) and "qid" in item and "type" in item for item in values
        )
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return False
    values = list(value)
    return bool(values) and all(
        isinstance(item, Mapping) and "qid" in item and "type" in item for item in values
    )


def _validate_top_level_export(
    payload: Mapping[str, Any], expected_case_ids: set[str]
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise _value_error("reviewer export must be a mapping")
    required = {"schema_version", "reviewer_code", "labels"}
    actual = set(payload)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={_format_keys(missing)}")
        if extra:
            details.append(f"extra={_format_keys(extra)}")
        raise _value_error("reviewer export fields are invalid: " + "; ".join(details))
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != EXPORT_SCHEMA_VERSION
    ):
        raise _value_error("schema_version must equal 2")
    reviewer_code = payload["reviewer_code"]
    if not isinstance(reviewer_code, str) or reviewer_code not in REVIEWER_CODES:
        raise _value_error("reviewer_code must be A or B")
    if not isinstance(expected_case_ids, set):
        try:
            expected_case_ids = set(expected_case_ids)
        except TypeError as error:
            raise TypeError("expected_case_ids must be an iterable of strings") from error
    if any(not isinstance(case_id, str) or not case_id for case_id in expected_case_ids):
        raise _value_error("expected case IDs must be non-empty strings")
    labels = payload["labels"]
    if not isinstance(labels, Mapping):
        raise _value_error("labels must be a mapping")
    actual_case_ids = set(labels)
    missing_cases = expected_case_ids - actual_case_ids
    extra_cases = actual_case_ids - expected_case_ids
    if missing_cases or extra_cases:
        details = []
        if missing_cases:
            details.append(f"missing={_format_keys(missing_cases)}")
        if extra_cases:
            details.append(f"extra={_format_keys(extra_cases)}")
        raise _value_error("case IDs differ: " + "; ".join(details))
    return cast(Mapping[str, Any], labels)


def _label_sort_key(value: str | float) -> tuple[str, Any]:
    if isinstance(value, str):
        return ("str", value)
    return ("num", value)


def _label_key(question_type: str, value: str | float) -> tuple[str, str | float]:
    if question_type in {"noul", "score"} and _is_finite_number(value):
        return (question_type, float(int(value)))
    if question_type == "mixed":
        if isinstance(value, str):
            return ("string", value)
        if _is_finite_number(value):
            return ("number", float(value))
    return (question_type, value)


def _validate_direct_label(question: _Question, value: object, case_id: str) -> str | int | float:
    if question.question_type == "choice":
        if type(value) is not str:
            raise _value_error(
                f"{case_id}/{question.qid}: choice answer must be a legal option string"
            )
        if not isinstance(question.criteria, Mapping) or value not in question.criteria:
            raise _value_error(f"{case_id}/{question.qid}: illegal choice label")
        return value
    if question.question_type == "noul":
        if type(value) is not int or value not in (0, 1):
            raise _value_error(f"{case_id}/{question.qid}: noul answer must be integer 0 or 1")
        return value
    if type(value) is not int:
        raise _value_error(f"{case_id}/{question.qid}: score answer must be an integer")
    max_score = question.max_score
    if max_score is None or value < 0 or value > max_score:
        raise _value_error(f"{case_id}/{question.qid}: score answer is outside the declared range")
    return value


def validate_reviewer_export(
    payload: Mapping[str, Any],
    expected_case_ids: set[str],
    records: Sequence[Mapping[str, Any]] | None = None,
    questions: Any = None,
) -> None:
    labels = _validate_top_level_export(payload, expected_case_ids)
    if records is not None and questions is None and _looks_like_question_input(records):
        questions = records
        records = None
    if isinstance(records, Mapping):
        records = [cast(Mapping[str, Any], records)]
    question_map = _questions_from_records(records)
    override_map = _questions_from_optional_input(questions, set(expected_case_ids))
    question_map.update(override_map)
    if labels and set(question_map) != set(labels):
        raise _value_error("question schema required for every labeled case")
    for case_id, case_labels in labels.items():
        if not isinstance(case_labels, Mapping):
            raise _value_error(f"{case_id}: labels must be a mapping")
        if case_id not in question_map:
            continue
        definitions = question_map[case_id]
        expected_qids = {question.qid for question in definitions}
        actual_qids = set(case_labels)
        missing_qids = expected_qids - actual_qids
        extra_qids = actual_qids - expected_qids
        if missing_qids or extra_qids:
            details: list[str] = []
            if missing_qids:
                details.append(f"missing={_format_keys(missing_qids)}")
            if extra_qids:
                details.append(f"extra={_format_keys(extra_qids)}")
            raise _value_error(f"{case_id}: question IDs differ: " + "; ".join(details))
        for question in definitions:
            _validate_direct_label(question, case_labels[question.qid], case_id)
    if records is not None and set(question_map) != set(expected_case_ids):
        raise _value_error("records do not match expected case IDs")


def _ordered_case_ids(case_ids: set[str]) -> list[str]:
    return sorted(
        case_ids,
        key=lambda case_id: (
            hashlib.sha256(f"{PACKET_SEED}:{case_id}".encode()).digest(),
            case_id,
        ),
    )


def _safe_json(value: Any) -> str:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        content.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _state_key_token(key: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^A-Za-z0-9]+", "_", separated).strip("_").lower()


def _is_blocked_state_key(key: str) -> bool:
    token = _state_key_token(key)
    if token in _BLOCKED_STATE_KEYS or token.startswith(_BLOCKED_STATE_PREFIXES):
        return True
    parts = tuple(token.split("_"))
    if any(part.startswith(fragment) for part in parts for fragment in _BLOCKED_STATE_FRAGMENTS):
        return True
    return any(pair in _BLOCKED_STATE_SEQUENCES for pair in pairwise(parts))


def _sanitize_state(value: object) -> object:
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise _value_error("state object keys must be strings")
            if _is_blocked_state_key(key):
                continue
            sanitized[key] = _sanitize_state(child)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_state(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_state(child) for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise _value_error("state must contain only JSON values")


def _state_text(state: object) -> str:
    try:
        return json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise _value_error(f"state cannot be serialized: {error}") from None


def _packet_record(
    record: Mapping[str, Any], index: int
) -> tuple[str, dict[str, Any], tuple[_Question, ...]]:
    case_id = _record_case_id(record, index)
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise _value_error(f"{case_id}: state must be a mapping")
    sanitized_state = _sanitize_state(state)
    if not isinstance(sanitized_state, dict):
        raise _value_error(f"{case_id}: state must be a mapping")
    _state_text(sanitized_state)
    raw_questions = record.get("questions")
    if isinstance(raw_questions, (str, bytes, bytearray)) or not isinstance(
        raw_questions, Sequence
    ):
        raise _value_error(f"{case_id}: questions must be an ordered sequence")
    questions: list[_Question] = []
    seen: set[str] = set()
    for position, question in enumerate(raw_questions):
        normalized = _question_from_mapping(question, f"{case_id} question {position}")
        if normalized.qid in seen:
            raise _value_error(f"{case_id}: duplicate question ID {normalized.qid!r}")
        seen.add(normalized.qid)
        questions.append(normalized)
    return (
        case_id,
        {
            "case_id": case_id,
            "state": sanitized_state,
            "questions": [q.as_packet_value() for q in questions],
        },
        tuple(questions),
    )


def _packet_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence of mappings")
    entries: list[tuple[str, dict[str, Any], tuple[_Question, ...]]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
        entry = _packet_record(record, index)
        if entry[0] in seen:
            raise _value_error(f"duplicate case ID {entry[0]!r}")
        seen.add(entry[0])
        entries.append(entry)
    by_case = {case_id: value for case_id, value, _ in entries}
    return [by_case[case_id] for case_id in _ordered_case_ids(seen)]


def _question_controls(case_id: str, question: _Question, suffix: str = "") -> str:
    name = html.escape(f"{case_id}:{question.qid}{suffix}", quote=True)
    if question.question_type == "choice" and isinstance(question.criteria, Mapping):
        values = [
            (
                label,
                f'<label><input type="radio" name="{name}" data-qid="{html.escape(question.qid, quote=True)}" value="{html.escape(label, quote=True)}"><span>{html.escape(label)}</span> - {html.escape(description)}</label>',
            )
            for label, description in question.criteria.items()
        ]
        return "".join(value for _, value in values)
    if question.question_type == "noul":
        return "".join(
            f'<label><input type="radio" name="{name}" data-qid="{html.escape(question.qid, quote=True)}" data-value-kind="number" value="{value}">{value}</label>'
            for value in ("0", "1")
        )
    maximum = question.max_score if question.max_score is not None else 0
    levels = ""
    if isinstance(question.criteria, Sequence) and not isinstance(
        question.criteria, (str, bytes, bytearray)
    ):
        levels = (
            "<small>Levels: "
            + ", ".join(html.escape(str(value)) for value in question.criteria)
            + "</small>"
        )
    return (
        f"<small>Allowed integer range: 0 to {html.escape(str(maximum))}</small>{levels}"
        f'<label><input type="number" name="{name}" data-qid="{html.escape(question.qid, quote=True)}" '
        f'min="0" max="{html.escape(str(maximum), quote=True)}" step="1"></label>'
    )


def _render_packet_body(
    records: Sequence[Mapping[str, Any]],
    reviewer_code: str,
) -> str:
    sections: list[str] = [
        "<header><h1>Blind review</h1>",
        "<p>Answer each question using only the state and wording shown. Select one response for every question. You may revise answers before downloading.</p>",
        "<p>Use the buttons or arrow keys to move between cases.</p>",
        '<button type="button" id="previous">Previous</button>',
        '<button type="button" id="next">Next</button>',
        '<button type="button" id="download">Download labels</button>',
        '<p id="progress" aria-live="polite"></p>',
        "</header>",
    ]
    for record in records:
        case_id = cast(str, record["case_id"])
        state = cast(Mapping[str, Any], record["state"])
        questions = cast(Sequence[Mapping[str, Any]], record["questions"])
        question_html: list[str] = []
        for question_value in questions:
            question = _question_from_mapping(question_value, f"{case_id} question")
            question_html.append(
                f"<fieldset><legend>{html.escape(question.qid)}</legend>"
                f"<p>{html.escape(question.instructions)}</p>"
                f"{_question_controls(case_id, question)}</fieldset>"
            )
        sections.append(
            f'<section data-case-id="{html.escape(case_id, quote=True)}">'
            f"<h2>{html.escape(case_id)}</h2>"
            f"<pre>{html.escape(_state_text(state))}</pre>"
            f"{''.join(question_html)}</section>"
        )
    sections.append(
        "<footer><small>Drafts remain in this browser until you download them.</small></footer>"
    )
    return "".join(sections)


def _write_html_packet(
    records: Sequence[Mapping[str, Any]],
    output: Path,
    reviewer_code: str,
    *,
    packet_data: Mapping[str, Any] | None = None,
    body: str | None = None,
) -> None:
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = packet_data or {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "reviewer_code": reviewer_code,
        "records": list(records),
    }
    document_body = body if body is not None else _render_packet_body(records, reviewer_code)
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Reviewer {html.escape(reviewer_code)}</title><style>{_PACKET_STYLE}</style></head>"
        f"<body>{document_body}"
        f'<script id="packet-data" type="application/json">{_safe_json(data)}</script>'
        f"<script>{_PACKET_SCRIPT}</script></body></html>"
    )
    output.write_text(document, encoding="utf-8")


def write_reviewer_packet(
    records: Sequence[Mapping[str, Any]], output: Path, reviewer_code: str
) -> None:
    if reviewer_code not in REVIEWER_CODES:
        raise _value_error("reviewer_code must be A or B")
    packet_records = _packet_records(records)
    _write_html_packet(packet_records, output, reviewer_code)


def _cohen_kappa_from_keys(
    pairs: Sequence[tuple[tuple[str, Any], tuple[str, Any]]],
) -> float:
    if not pairs:
        return 0.0
    categories = sorted({left for left, _ in pairs} | {right for _, right in pairs})
    row_counts = Counter(left for left, _ in pairs)
    column_counts = Counter(right for _, right in pairs)
    total = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / total
    expected = sum(row_counts[category] * column_counts[category] for category in categories) / (
        total * total
    )
    denominator = 1.0 - expected
    if abs(denominator) <= 1e-15:
        return 0.0
    return (observed - expected) / denominator


def _quadratic_weighted_kappa(
    pairs: Sequence[tuple[int, int]], max_score: float | None = None
) -> float:
    if not pairs:
        return 0.0
    highest = int(max_score) if max_score is not None else max(max(pair) for pair in pairs)
    levels = list(range(highest + 1))
    if len(levels) < 2:
        return 0.0
    scale = (len(levels) - 1) ** 2
    count = len(pairs)
    observed = sum((left - right) ** 2 for left, right in pairs) / (count * scale)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[left] * right_counts[right] * (left - right) ** 2
        for left in levels
        for right in levels
    ) / (count * count * scale)
    if abs(expected) <= 1e-15:
        return 0.0
    return 1.0 - observed / expected


def _confusion_matrix(
    pairs: Sequence[tuple[str | int | float, str | int | float]],
) -> dict[str, Any]:
    labels = sorted({value for pair in pairs for value in pair}, key=_label_sort_key)
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for left, right in pairs:
        matrix[positions[left]][positions[right]] += 1
    return {
        "labels": labels,
        "matrix": matrix,
        "row_label": "reviewer_a",
        "column_label": "reviewer_b",
    }


def _question_summaries(
    observations: Sequence[_Observation],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[_Observation]]] = {}
    for observation in observations:
        grouped.setdefault(observation.suite_id, {}).setdefault(observation.qid, []).append(
            observation
        )
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for suite_id in sorted(grouped):
        suite_summaries: dict[str, dict[str, Any]] = {}
        for qid in sorted(grouped[suite_id]):
            values = grouped[suite_id][qid]
            pairs = [(observation.reviewer_a, observation.reviewer_b) for observation in values]
            score_pairs = [
                (int(observation.reviewer_a), int(observation.reviewer_b))
                for observation in values
                if observation.question_type == "score"
            ]
            question_type = values[0].question_type
            typed_pairs = [
                (
                    _label_key(observation.question_type, observation.reviewer_a),
                    _label_key(observation.question_type, observation.reviewer_b),
                )
                for observation in values
            ]
            agreements = sum(1 for left, right in typed_pairs if left == right)
            suite_summaries[qid] = {
                "question_type": question_type,
                "count": len(values),
                "agreements": agreements,
                "raw_agreement": agreements / len(values),
                "nominal_cohens_kappa": _cohen_kappa_from_keys(typed_pairs),
                "quadratic_weighted_kappa": _quadratic_weighted_kappa(
                    score_pairs, values[0].max_score
                ),
                "confusion_matrix": _confusion_matrix(pairs),
                "disagreements": len(values) - agreements,
            }
        summaries[suite_id] = suite_summaries
    return summaries


def _language_summaries(
    observations: Sequence[_Observation],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[_Observation]]] = {}
    for observation in observations:
        if observation.language is not None:
            grouped.setdefault(observation.suite_id, {}).setdefault(
                observation.language, []
            ).append(observation)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for suite_id in sorted(grouped):
        suite_result: dict[str, dict[str, Any]] = {}
        for language in sorted(grouped[suite_id]):
            values = grouped[suite_id][language]
            typed_pairs = [
                (
                    _label_key(observation.question_type, observation.reviewer_a),
                    _label_key(observation.question_type, observation.reviewer_b),
                )
                for observation in values
            ]
            agreements = sum(1 for left, right in typed_pairs if left == right)
            suite_result[language] = {
                "count": len(values),
                "agreements": agreements,
                "raw_agreement": agreements / len(values),
                "nominal_cohens_kappa": _cohen_kappa_from_keys(typed_pairs),
            }
        result[suite_id] = suite_result
    return result


def _observations(
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[_Observation]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence of mappings")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
    expected_ids = {_record_case_id(record, index) for index, record in enumerate(records)}
    if len(expected_ids) != len(records):
        raise _value_error("records must have unique case IDs")
    validate_reviewer_export(reviewer_a, expected_ids, records)
    validate_reviewer_export(reviewer_b, expected_ids, records)
    if reviewer_a["reviewer_code"] != "A" or reviewer_b["reviewer_code"] != "B":
        raise _value_error("reviewer exports must have reviewer codes A and B")
    a_labels = cast(Mapping[str, Any], reviewer_a["labels"])
    b_labels = cast(Mapping[str, Any], reviewer_b["labels"])
    record_map: dict[str, Mapping[str, Any]] = {}
    question_map = _questions_from_records(records)
    for record in records:
        record_map[_record_case_id(record, 0)] = record
    result: list[_Observation] = []
    for case_id in sorted(record_map):
        record = record_map[case_id]
        state = record.get("state")
        suite_id = record.get("suite_id")
        if not isinstance(suite_id, str) or not suite_id:
            raise _value_error(f"{case_id}: suite_id must be a non-empty string")
        if not isinstance(state, Mapping):
            raise _value_error(f"{case_id}: state must be a mapping")
        if suite_id == "multilingual_intent":
            language = state.get("language")
            if not isinstance(language, str) or language not in _MULTILINGUAL_LANGUAGES:
                raise _value_error(
                    f"{case_id}: multilingual state.language must be a valid language"
                )
        else:
            language = None
        for question in question_map[case_id]:
            left = cast(str | int | float, a_labels[case_id][question.qid])
            right = cast(str | int | float, b_labels[case_id][question.qid])
            result.append(
                _Observation(
                    case_id,
                    suite_id,
                    question.qid,
                    question.question_type,
                    left,
                    right,
                    language,
                    question.max_score,
                )
            )
    return result


def agreement_report(
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observations = _observations(reviewer_a, reviewer_b, records)
    disagreements = [
        {
            "case_id": observation.case_id,
            "qid": observation.qid,
            "question_type": observation.question_type,
            "reviewer_a": observation.reviewer_a,
            "reviewer_b": observation.reviewer_b,
        }
        for observation in observations
        if _label_key(observation.question_type, observation.reviewer_a)
        != _label_key(observation.question_type, observation.reviewer_b)
    ]
    agreements = len(observations) - len(disagreements)
    raw_agreement = agreements / len(observations) if observations else 0.0
    score_pairs = [
        (int(observation.reviewer_a), int(observation.reviewer_b))
        for observation in observations
        if observation.question_type == "score"
    ]
    score_max = max(
        (
            observation.max_score
            for observation in observations
            if observation.question_type == "score" and observation.max_score is not None
        ),
        default=None,
    )
    per_question = _question_summaries(observations)
    per_language = _language_summaries(observations)
    confusion = {
        suite_id: {qid: summary["confusion_matrix"] for qid, summary in suite_summaries.items()}
        for suite_id, suite_summaries in per_question.items()
    }
    typed_pairs = [
        (
            _label_key(observation.question_type, observation.reviewer_a),
            _label_key(observation.question_type, observation.reviewer_b),
        )
        for observation in observations
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "total_cases": len({observation.case_id for observation in observations}),
        "total_questions": len(observations),
        "total_agreements": agreements,
        "disagreement_count": len(disagreements),
        "unresolved_disagreement_count": len(disagreements),
        "raw_agreement": raw_agreement,
        "nominal_cohens_kappa": _cohen_kappa_from_keys(typed_pairs),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(score_pairs, score_max),
        "per_question": per_question,
        "per_question_summaries": per_question,
        "confusion_matrices": confusion,
        "per_class_confusion_matrices": confusion,
        "per_language": per_language,
        "per_language_agreement": per_language,
        "disagreements": disagreements,
    }
    return report


def _adjudication_record(
    record: Mapping[str, Any],
    index: int,
    labels_a: Mapping[str, Any],
    labels_b: Mapping[str, Any],
    disagreements: set[tuple[str, str]],
) -> dict[str, Any] | None:
    case_id = _record_case_id(record, index)
    if not any(case == case_id for case, _ in disagreements):
        return None
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise _value_error(f"{case_id}: state must be a mapping")
    sanitized_state = _sanitize_state(state)
    if not isinstance(sanitized_state, dict):
        raise _value_error(f"{case_id}: state must be a mapping")
    _state_text(sanitized_state)
    raw_questions = record.get("questions")
    if isinstance(raw_questions, (str, bytes, bytearray)) or not isinstance(
        raw_questions, Sequence
    ):
        raise _value_error(f"{case_id}: questions must be an ordered sequence")
    safe_questions: list[dict[str, Any]] = []
    for position, question in enumerate(raw_questions):
        normalized = _question_from_mapping(question, f"{case_id} question {position}")
        if (case_id, normalized.qid) not in disagreements:
            continue
        safe_questions.append(normalized.as_packet_value())
    return {
        "case_id": case_id,
        "state": sanitized_state,
        "questions": safe_questions,
        "reviewer_labels": {
            "A": {
                qid: labels_a[case_id][qid]
                for qid in labels_a[case_id]
                if (case_id, qid) in disagreements
            },
            "B": {
                qid: labels_b[case_id][qid]
                for qid in labels_b[case_id]
                if (case_id, qid) in disagreements
            },
        },
    }


def _render_adjudication_body(records: Sequence[Mapping[str, Any]]) -> str:
    sections = [
        "<header><h1>Adjudication</h1>",
        "<p>Review only the cases below. Each case includes both independent responses. Choose the final response for every listed question.</p>",
        '<button type="button" id="previous">Previous</button>',
        '<button type="button" id="next">Next</button>',
        '<button type="button" id="download">Download final labels</button>',
        '<p id="progress" aria-live="polite"></p>',
        "</header>",
    ]
    for record in records:
        case_id = cast(str, record["case_id"])
        state = cast(Mapping[str, Any], record["state"])
        questions = cast(Sequence[Mapping[str, Any]], record["questions"])
        rendered: list[str] = [
            f'<section data-case-id="{html.escape(case_id, quote=True)}">',
            f"<h2>{html.escape(case_id)}</h2>",
            f"<pre>{html.escape(_state_text(state))}</pre>",
        ]
        labels = cast(Mapping[str, Any], record["reviewer_labels"])
        for question_value in questions:
            question = _question_from_mapping(question_value, f"{case_id} question")
            qid = question.qid
            rendered.append(
                f"<fieldset><legend>{html.escape(qid)}</legend>"
                f"<p>{html.escape(question.instructions)}</p>"
                f"<p>Reviewer A: {html.escape(str(labels['A'][qid]))}</p>"
                f"<p>Reviewer B: {html.escape(str(labels['B'][qid]))}</p>"
                f"{_question_controls(case_id, question, ':final')}"
                f'<label>Reason: <input type="text" name="{html.escape(case_id + ":" + qid + ":reason", quote=True)}" data-reason-qid="{html.escape(qid, quote=True)}" value="" required placeholder="Explain the final decision"></label>'
                "</fieldset>"
            )
        rendered.append("</section>")
        sections.append("".join(rendered))
    sections.append("<footer><small>Only disagreements are listed in this packet.</small></footer>")
    return "".join(sections)


def write_adjudication_packet(
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence of mappings")
    expected_ids = {
        _record_case_id(record, index)
        for index, record in enumerate(records)
        if isinstance(record, Mapping)
    }
    validate_reviewer_export(reviewer_a, expected_ids, records)
    validate_reviewer_export(reviewer_b, expected_ids, records)
    if reviewer_a["reviewer_code"] != "A" or reviewer_b["reviewer_code"] != "B":
        raise _value_error("reviewer exports must have reviewer codes A and B")
    report = agreement_report(reviewer_a, reviewer_b, records)
    disagreement_keys = {
        (str(entry["case_id"]), str(entry["qid"]))
        for entry in cast(list[dict[str, Any]], report["disagreements"])
    }
    entries: list[tuple[str, dict[str, Any]]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
        entry = _adjudication_record(
            record,
            index,
            cast(Mapping[str, Any], reviewer_a["labels"]),
            cast(Mapping[str, Any], reviewer_b["labels"]),
            disagreement_keys,
        )
        if entry is not None:
            entries.append((entry["case_id"], entry))
    by_case = {case_id: entry for case_id, entry in entries}
    ordered = [by_case[case_id] for case_id in _ordered_case_ids(set(by_case))]
    packet_data = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "reviewer_code": "ADJUDICATION",
        "packet_type": "adjudication",
        "records": ordered,
    }
    _write_html_packet(
        ordered,
        output,
        "ADJUDICATION",
        packet_data=packet_data,
        body=_render_adjudication_body(ordered),
    )


def _object_from_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _value_error("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise _value_error(f"non-standard JSON constant {value!r}")


def _loads_strict_json(contents: str, context: str) -> Any:
    try:
        return json.loads(
            contents,
            object_pairs_hook=_object_from_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise _value_error(
            f"{context} contains malformed JSON at line {error.lineno}, column {error.colno}"
        ) from None
    except ValueError as error:
        raise _value_error(f"{context} contains malformed JSON: {error}") from None


def _read_strict_json(path: Path, description: str) -> Any:
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise _value_error(f"{description} is not valid UTF-8 JSON: {error}") from None
    except OSError as error:
        raise _value_error(f"unable to read {description}: {error}") from None
    return _loads_strict_json(contents, description)


def _read_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise _value_error(f"manifest is not valid UTF-8 JSONL: {error}") from None
    except OSError as error:
        raise _value_error(f"unable to read manifest: {error}") from None
    lines = contents.splitlines()
    if not lines:
        raise _value_error("manifest.jsonl must contain at least one record")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise _value_error(f"manifest.jsonl line {line_number} must contain a JSON object")
        value = _loads_strict_json(line, f"manifest.jsonl line {line_number}")
        if not isinstance(value, dict):
            raise _value_error(f"manifest.jsonl line {line_number} must be a JSON object")
        records.append(value)
    return records


def _same_direct_label(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _validate_adjudication_envelope(adjudication: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "adjudication_code",
        "labels",
        "reviewer_labels",
        "reasons",
    }
    actual = set(adjudication)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={_format_keys(missing)}")
        if extra:
            details.append(f"extra={_format_keys(extra)}")
        raise _value_error("adjudication fields are invalid: " + "; ".join(details))
    if type(adjudication["schema_version"]) is not int or adjudication["schema_version"] != 2:
        raise _value_error("adjudication schema_version must equal 2")
    code = adjudication["adjudication_code"]
    if not isinstance(code, str) or not code.strip():
        raise _value_error("adjudication_code must be a nonempty string")


def _validate_disagreement_mapping(
    value: object,
    expected_by_case: Mapping[str, set[str]],
    description: str,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise _value_error(f"adjudication {description} must be a mapping")
    expected_cases = set(expected_by_case)
    actual_cases = set(value)
    missing_cases = expected_cases - actual_cases
    extra_cases = actual_cases - expected_cases
    if missing_cases or extra_cases:
        details: list[str] = []
        if missing_cases:
            details.append(f"missing={_format_keys(missing_cases)}")
        if extra_cases:
            details.append(f"extra={_format_keys(extra_cases)}")
        raise _value_error(
            f"adjudication disagreement cases differ in {description}: " + "; ".join(details)
        )
    result: dict[str, Mapping[str, Any]] = {}
    for case_id, expected_qids in expected_by_case.items():
        case_labels = value[case_id]
        if not isinstance(case_labels, Mapping):
            raise _value_error(f"adjudication {description} for {case_id!r} must be a mapping")
        actual_qids = set(case_labels)
        missing_qids = expected_qids - actual_qids
        extra_qids = actual_qids - expected_qids
        if missing_qids or extra_qids:
            details = []
            if missing_qids:
                details.append(f"missing={_format_keys(missing_qids)}")
            if extra_qids:
                details.append(f"extra={_format_keys(extra_qids)}")
            raise _value_error(
                f"adjudication disagreement question IDs differ in {description} for "
                f"{case_id!r}: " + "; ".join(details)
            )
        result[case_id] = cast(Mapping[str, Any], case_labels)
    return result


def merge_adjudication(
    records: Sequence[Mapping[str, Any]],
    reviewer_a: Mapping[str, Any],
    reviewer_b: Mapping[str, Any],
    adjudication: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence of mappings")
    copied_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
        copied_records.append(deepcopy(dict(record)))
    report = agreement_report(reviewer_a, reviewer_b, copied_records)
    if reviewer_a["reviewer_code"] != "A" or reviewer_b["reviewer_code"] != "B":
        raise _value_error("reviewer exports must have reviewer codes A and B")
    _validate_adjudication_envelope(adjudication)
    disagreements = cast(list[dict[str, Any]], report["disagreements"])
    expected_by_case: dict[str, set[str]] = {}
    for disagreement in disagreements:
        case_id = cast(str, disagreement["case_id"])
        qid = cast(str, disagreement["qid"])
        expected_by_case.setdefault(case_id, set()).add(qid)
    final_labels = _validate_disagreement_mapping(
        adjudication["labels"], expected_by_case, "labels"
    )
    reasons = _validate_disagreement_mapping(adjudication["reasons"], expected_by_case, "reasons")
    reviewer_labels = adjudication["reviewer_labels"]
    if not isinstance(reviewer_labels, Mapping) or set(reviewer_labels) != set(REVIEWER_CODES):
        raise _value_error("adjudication reviewer_labels must contain exactly A and B")
    preserved_a = _validate_disagreement_mapping(
        reviewer_labels["A"], expected_by_case, "reviewer_labels A"
    )
    preserved_b = _validate_disagreement_mapping(
        reviewer_labels["B"], expected_by_case, "reviewer_labels B"
    )
    questions_by_case = _questions_from_records(copied_records)
    labels_a = cast(Mapping[str, Any], reviewer_a["labels"])
    labels_b = cast(Mapping[str, Any], reviewer_b["labels"])
    for case_id, qids in expected_by_case.items():
        question_map = {question.qid: question for question in questions_by_case[case_id]}
        for qid in qids:
            final_value = final_labels[case_id][qid]
            _validate_direct_label(question_map[qid], final_value, case_id)
            reason = reasons[case_id][qid]
            if not isinstance(reason, str) or not reason.strip():
                raise _value_error(
                    f"{case_id}/{qid}: adjudication reason must be a nonempty string"
                )
            original_a = labels_a[case_id][qid]
            original_b = labels_b[case_id][qid]
            if not _same_direct_label(original_a, preserved_a[case_id][qid]):
                raise _value_error(
                    f"{case_id}/{qid}: adjudication reviewer A label does not match the export"
                )
            if not _same_direct_label(original_b, preserved_b[case_id][qid]):
                raise _value_error(
                    f"{case_id}/{qid}: adjudication reviewer B label does not match the export"
                )
    suite_counts: Counter[str] = Counter()
    for index, record in enumerate(copied_records):
        suite_id = record.get("suite_id")
        if not isinstance(suite_id, str) or not suite_id:
            raise _value_error(f"record {index}: suite_id must be a non-empty string")
        suite_counts[suite_id] += 1
        expected = record.get("expected")
        if not isinstance(expected, dict):
            raise _value_error(f"record {index}: expected must be a mapping")
        questions = questions_by_case[cast(str, record["case_id"])]
        expected_qids = {question.qid for question in questions}
        if set(expected) != expected_qids:
            raise _value_error(f"record {index}: expected IDs differ from question IDs")
        for question in questions:
            original_a = labels_a[cast(str, record["case_id"])][question.qid]
            original_b = labels_b[cast(str, record["case_id"])][question.qid]
            final_value = (
                original_a
                if _same_direct_label(original_a, original_b)
                else final_labels[cast(str, record["case_id"])][question.qid]
            )
            expected[question.qid] = deepcopy(final_value)
        record["label_provenance"] = ADJUDICATED_LABEL_PROVENANCE
    merged = copied_records
    validate_manifest(merged, dict(suite_counts))
    return merged


def _manifest_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(record).decode("utf-8") + "\n" for record in records).encode(
        "utf-8"
    )


def _stage_bytes(path: Path, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _restore_bytes(path: Path, content: bytes) -> None:
    temporary_path = _stage_bytes(path, content)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def _exclusive_seal_lock(output_root: Path) -> Iterator[None]:
    lock_path = output_root / _SEAL_LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"dataset seal already in progress for {output_root}") from error
    try:
        yield
    finally:
        try:
            os.close(descriptor)
        finally:
            lock_path.unlink(missing_ok=True)


def _refuse_existing_seal(output_root: Path, sealed_path: Path) -> None:
    if os.path.lexists(sealed_path):
        raise FileExistsError(f"refusing to overwrite sealed manifest in {output_root}")


def _require_provisional_draft(provenance: Mapping[str, Any], output_root: Path) -> None:
    if "provisional" not in provenance:
        raise _value_error("provenance.json must include provisional=true")
    provisional = provenance["provisional"]
    if type(provisional) is bool and provisional is False:
        raise FileExistsError(f"refusing to overwrite sealed manifest in {output_root}")
    if type(provisional) is not bool or provisional is not True:
        raise _value_error("provenance.json provisional must be boolean true")


def _write_checksum_exclusive(sealed_path: Path, content: bytes, output_root: Path) -> None:
    try:
        checksum_file = sealed_path.open("xb")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite sealed manifest in {output_root}") from error
    try:
        with checksum_file:
            checksum_file.write(content)
            checksum_file.flush()
            os.fsync(checksum_file.fileno())
    except BaseException:
        sealed_path.unlink(missing_ok=True)
        raise


def _seal_manifest_locked(output_root: Path) -> str:
    missing = [
        relative for relative in _REQUIRED_SEAL_INPUTS if not (output_root / relative).is_file()
    ]
    if missing:
        names = ", ".join(relative.as_posix() for relative in missing)
        raise _value_error(f"seal requires reviewer and dataset artifacts; missing: {names}")
    manifest_path = output_root / MANIFEST_NAME
    provenance_path = output_root / PROVENANCE_NAME
    sealed_path = output_root / SEALED_NAME
    _refuse_existing_seal(output_root, sealed_path)
    manifest_before = manifest_path.read_bytes()
    provenance_before = provenance_path.read_bytes()
    records = _read_manifest_jsonl(manifest_path)
    provenance_value = _read_strict_json(provenance_path, "provenance.json")
    if not isinstance(provenance_value, Mapping):
        raise _value_error("provenance.json must contain a JSON object")
    _require_provisional_draft(cast(Mapping[str, Any], provenance_value), output_root)
    reviewer_a = _read_strict_json(output_root / "labels" / "reviewer_a.json", "reviewer_a.json")
    reviewer_b = _read_strict_json(output_root / "labels" / "reviewer_b.json", "reviewer_b.json")
    adjudication = _read_strict_json(
        output_root / "labels" / "adjudication.json", "adjudication.json"
    )
    if not all(isinstance(payload, Mapping) for payload in (reviewer_a, reviewer_b, adjudication)):
        raise _value_error("reviewer and adjudication exports must contain JSON objects")
    final_records = merge_adjudication(
        records,
        cast(Mapping[str, Any], reviewer_a),
        cast(Mapping[str, Any], reviewer_b),
        cast(Mapping[str, Any], adjudication),
    )
    summary = validate_dataset(
        final_records,
        cast(Mapping[str, Any], provenance_value),
        registry=cast(Mapping[str, Any], provenance_value),
    )
    digest = cast(str, manifest_digest(final_records))
    if summary.get("digest") != digest:
        raise _value_error("manifest digest changed during dataset validation")
    finalized_provenance = deepcopy(dict(cast(Mapping[str, Any], provenance_value)))
    finalized_provenance["provisional"] = False
    manifest_content = _manifest_jsonl_bytes(final_records)
    provenance_content = canonical_json(finalized_provenance) + b"\n"
    checksum_content = f"{digest}  {MANIFEST_NAME}\n".encode()
    manifest_temporary: Path | None = None
    provenance_temporary: Path | None = None
    manifest_replaced = False
    provenance_replaced = False
    checksum_created = False
    try:
        manifest_temporary = _stage_bytes(manifest_path, manifest_content)
        provenance_temporary = _stage_bytes(provenance_path, provenance_content)
        _refuse_existing_seal(output_root, sealed_path)
        _write_checksum_exclusive(sealed_path, checksum_content, output_root)
        checksum_created = True
        os.replace(manifest_temporary, manifest_path)
        manifest_temporary = None
        manifest_replaced = True
        os.replace(provenance_temporary, provenance_path)
        provenance_temporary = None
        provenance_replaced = True
    except BaseException as error:
        rollback_error: BaseException | None = None
        if checksum_created:
            try:
                sealed_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                rollback_error = cleanup_error
        if manifest_replaced:
            try:
                _restore_bytes(manifest_path, manifest_before)
            except OSError as restore_error:
                rollback_error = restore_error
        if provenance_replaced:
            try:
                _restore_bytes(provenance_path, provenance_before)
            except OSError as restore_error:
                rollback_error = restore_error
        if rollback_error is not None:
            raise RuntimeError("unable to roll back failed dataset seal") from error
        raise
    finally:
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        if provenance_temporary is not None:
            provenance_temporary.unlink(missing_ok=True)
    return digest


def seal_manifest(output_root: Path) -> str:
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if not output_root.is_dir():
        raise _value_error(
            "seal requires reviewer and dataset artifacts; output root is not a directory"
        )
    with _exclusive_seal_lock(output_root):
        return _seal_manifest_locked(output_root)


_CLI_PATH_OPTIONS = (
    ("--records", "records"),
    ("--output", "output"),
    ("--reviewer-a", "reviewer_a"),
    ("--reviewer-b", "reviewer_b"),
    ("--adjudication", "adjudication"),
    ("--report", "report"),
)
_CLI_MODE_OPTIONS = {
    "packet": frozenset({"--records", "--output"}),
    "agreement": frozenset({"--records", "--output", "--reviewer-a", "--reviewer-b", "--report"}),
    "seal": frozenset({"--output", "--records", "--reviewer-a", "--reviewer-b", "--adjudication"}),
}
_CLI_SAFE_OPTIONS = tuple(option for option, _ in _CLI_PATH_OPTIONS) + (
    "--agreement",
    "--seal",
)
_CLI_SAFE_TERMS = (
    "collision",
    "required",
    "not valid",
    "existing file",
    "existing directory",
    "duplicate JSON object key",
    "path resolution failed",
    "reviewer",
    "manifest.jsonl",
    "mode",
)


class _CliArgumentError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliArgumentError


def _safe_cli_error(error: BaseException) -> str:
    if isinstance(error, RuntimeError):
        return "error: path resolution failed"
    try:
        message = str(error).casefold()
    except BaseException as conversion_error:
        if isinstance(conversion_error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        return "error: CLI operation failed"
    parts = [option for option in _CLI_SAFE_OPTIONS if option.casefold() in message]
    parts.extend(mode for mode in _CLI_MODE_OPTIONS if mode in message)
    parts.extend(term for term in _CLI_SAFE_TERMS if term.casefold() in message)
    if not parts:
        return "error: CLI operation failed"
    return "error: " + " ".join(parts)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="python -m datasets.v2.labeling",
        description="Generate blind v2 reviewer packets, agreement artifacts, and dataset seals.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--agreement",
        action="store_true",
        help="write an agreement report and disagreement-only adjudication packet",
    )
    modes.add_argument(
        "--seal",
        action="store_true",
        help="seal a manifest after validating human reviewer and adjudication artifacts",
    )
    parser.add_argument("--records", type=Path, help="manifest.jsonl input path")
    parser.add_argument("--output", type=Path, help="packet directory or adjudication HTML path")
    parser.add_argument("--reviewer-a", type=Path, help="Reviewer A export path")
    parser.add_argument("--reviewer-b", type=Path, help="Reviewer B export path")
    parser.add_argument("--adjudication", type=Path, help="adjudication export path for sealing")
    parser.add_argument("--report", type=Path, help="agreement JSON report path")
    return parser


def _validate_cli_mode_options(arguments: argparse.Namespace, mode: str) -> None:
    allowed = _CLI_MODE_OPTIONS[mode]
    for option, destination in _CLI_PATH_OPTIONS:
        if getattr(arguments, destination) is not None and option not in allowed:
            raise _value_error(f"{option} is not valid for {mode} mode")


@dataclass(frozen=True)
class _PathIdentity:
    canonical: str
    file_id: tuple[int, int] | None


def _path_identity(path: Path) -> _PathIdentity:
    try:
        resolved_path = path.resolve()
    except RuntimeError:
        raise RuntimeError("path resolution failed") from None
    canonical = os.path.normcase(os.path.realpath(str(resolved_path)))
    if sys.platform == "darwin":
        canonical = canonical.casefold()
    try:
        stat_result = path.stat()
    except OSError:
        return _PathIdentity(canonical, None)
    return _PathIdentity(canonical, (stat_result.st_dev, stat_result.st_ino))


def _path_identities_collide(left: _PathIdentity, right: _PathIdentity) -> bool:
    if left.canonical == right.canonical:
        return True
    return left.file_id is not None and left.file_id == right.file_id


def _validate_resolved_cli_paths(
    outputs: Sequence[tuple[str, Path]],
    inputs: Sequence[tuple[str, Path]],
) -> None:
    output_identities = tuple((name, _path_identity(path)) for name, path in outputs)
    input_identities = tuple((name, path.name, _path_identity(path)) for name, path in inputs)
    for output_name, output_identity in output_identities:
        for input_name, input_filename, input_identity in input_identities:
            if _path_identities_collide(output_identity, input_identity):
                raise _value_error(
                    f"{output_name} path collision with {input_name} ({input_filename})"
                )
    for index, (left_name, left_identity) in enumerate(output_identities):
        for right_name, right_identity in output_identities[index + 1 :]:
            if _path_identities_collide(left_identity, right_identity):
                raise _value_error(f"{left_name} path collision with {right_name}")


def _required_cli_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise _value_error(f"{name} is required")
    return value


def _read_cli_mapping(path: Path, description: str) -> Mapping[str, Any]:
    value = _read_strict_json(path, description)
    if not isinstance(value, Mapping):
        raise _value_error(f"{description} must contain a JSON object")
    return cast(Mapping[str, Any], value)


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload) + b"\n")


def _reject_existing_cli_directory(path: Path, option: str) -> None:
    if os.path.lexists(path) and path.is_dir():
        raise _value_error(f"{option} must not be an existing directory")


def _preflight_cli_parent(parent: Path) -> None:
    current = parent
    while True:
        try:
            exists = os.path.lexists(current)
        except OSError:
            raise _value_error("output parent is obstructed") from None
        if exists:
            try:
                is_directory = current.is_dir()
            except OSError:
                raise _value_error("output parent is obstructed") from None
            if not is_directory:
                raise _value_error("output parent is obstructed")
            return
        next_parent = current.parent
        if next_parent == current:
            return
        current = next_parent


def _preflight_cli_target(path: Path, option: str) -> None:
    if os.path.lexists(path):
        try:
            is_directory = path.is_dir()
            is_file = path.is_file()
        except OSError:
            raise _value_error("output path is obstructed") from None
        if is_directory:
            _reject_existing_cli_directory(path, option)
        if not is_file:
            raise _value_error(f"{option} must be a regular file or absent")
    _preflight_cli_parent(path.parent)


def _preflight_cli_targets(targets: Sequence[tuple[str, Path]]) -> None:
    for option, path in targets:
        _preflight_cli_target(path, option)


@dataclass(frozen=True)
class _CliStagedOutput:
    target: Path
    temporary: Path


def _new_cli_temporary(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_cli_output(target: Path, writer: Callable[[Path], None]) -> _CliStagedOutput:
    temporary = _new_cli_temporary(target)
    try:
        writer(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _CliStagedOutput(target, temporary)


def _capture_cli_targets(targets: Sequence[Path]) -> dict[Path, bytes | None]:
    snapshots: dict[Path, bytes | None] = {}
    for target in targets:
        if not os.path.lexists(target):
            snapshots[target] = None
            continue
        try:
            snapshots[target] = target.read_bytes()
        except OSError:
            raise _value_error("unable to snapshot CLI output") from None
    return snapshots


def _write_cli_temporary_bytes(target: Path, content: bytes) -> Path:
    temporary = _new_cli_temporary(target)
    try:
        with temporary.open("wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore_cli_target(target: Path, content: bytes | None) -> None:
    if content is None:
        target.unlink(missing_ok=True)
        return
    temporary = _write_cli_temporary_bytes(target, content)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _cleanup_cli_outputs(outputs: Sequence[_CliStagedOutput]) -> None:
    for output in outputs:
        output.temporary.unlink(missing_ok=True)


def _commit_cli_outputs(
    outputs: Sequence[_CliStagedOutput], snapshots: Mapping[Path, bytes | None]
) -> None:
    attempted: list[_CliStagedOutput] = []
    try:
        for output in outputs:
            attempted.append(output)
            os.replace(output.temporary, output.target)
    except BaseException as error:
        rollback_error: OSError | None = None
        for output in reversed(attempted):
            try:
                _restore_cli_target(output.target, snapshots[output.target])
            except OSError as restore_error:
                rollback_error = restore_error
        if rollback_error is not None:
            raise RuntimeError("unable to roll back CLI output transaction") from error
        raise


def _agreement_records_path(arguments: argparse.Namespace) -> Path:
    records = arguments.records
    if records is not None:
        return cast(Path, records)
    candidates: list[Path] = []
    output = arguments.output
    report = arguments.report
    if output is not None:
        output_path = cast(Path, output)
        candidates.extend(
            [output_path.parent.parent / MANIFEST_NAME, output_path.parent / MANIFEST_NAME]
        )
    if report is not None:
        report_path = cast(Path, report)
        candidates.append(report_path.parent.parent / MANIFEST_NAME)
    candidates.extend([Path.cwd() / MANIFEST_NAME, Path.cwd() / "datasets" / "v2" / MANIFEST_NAME])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise _value_error(
        "agreement mode requires --records or a manifest.jsonl discoverable beside --output"
    )


def _run_packet_cli(arguments: argparse.Namespace) -> int:
    records_path = _required_cli_path(arguments.records, "--records")
    output = _required_cli_path(arguments.output, "--output")
    if _path_identities_collide(_path_identity(output), _path_identity(records_path)):
        raise _value_error(f"--output path collision with --records ({records_path.name})")
    if os.path.lexists(output) and not output.is_dir():
        raise _value_error("--output must be a directory and cannot be an existing file")
    reviewer_a = output / "reviewer_a.html"
    reviewer_b = output / "reviewer_b.html"
    _validate_resolved_cli_paths(
        (("--output", reviewer_a), ("--output", reviewer_b)),
        (("--records", records_path),),
    )
    _preflight_cli_targets((("--output", reviewer_a), ("--output", reviewer_b)))
    records = _read_manifest_jsonl(records_path)
    snapshots = _capture_cli_targets((reviewer_a, reviewer_b))
    staged: list[_CliStagedOutput] = []
    try:
        staged.append(
            _stage_cli_output(
                reviewer_a, lambda temporary: write_reviewer_packet(records, temporary, "A")
            )
        )
        staged.append(
            _stage_cli_output(
                reviewer_b, lambda temporary: write_reviewer_packet(records, temporary, "B")
            )
        )
        _commit_cli_outputs(staged, snapshots)
    finally:
        _cleanup_cli_outputs(staged)
    return 0


def _run_agreement_cli(arguments: argparse.Namespace) -> int:
    records_path = _agreement_records_path(arguments)
    reviewer_a_path = _required_cli_path(arguments.reviewer_a, "--reviewer-a")
    reviewer_b_path = _required_cli_path(arguments.reviewer_b, "--reviewer-b")
    output = _required_cli_path(arguments.output, "--output")
    report_path = _required_cli_path(arguments.report, "--report")
    _validate_resolved_cli_paths(
        (("--output", output), ("--report", report_path)),
        (
            ("--records", records_path),
            ("--reviewer-a", reviewer_a_path),
            ("--reviewer-b", reviewer_b_path),
        ),
    )
    _preflight_cli_targets((("--output", output), ("--report", report_path)))
    records = _read_manifest_jsonl(records_path)
    reviewer_a = _read_cli_mapping(reviewer_a_path, "reviewer A export")
    reviewer_b = _read_cli_mapping(reviewer_b_path, "reviewer B export")
    report = agreement_report(reviewer_a, reviewer_b, records)
    snapshots = _capture_cli_targets((output, report_path))
    staged: list[_CliStagedOutput] = []
    try:
        staged.append(
            _stage_cli_output(
                output,
                lambda temporary: write_adjudication_packet(
                    reviewer_a, reviewer_b, records, temporary
                ),
            )
        )
        staged.append(
            _stage_cli_output(
                report_path, lambda temporary: _write_canonical_json(temporary, report)
            )
        )
        _commit_cli_outputs(staged, snapshots)
    finally:
        _cleanup_cli_outputs(staged)
    return 0


def _run_seal_cli(arguments: argparse.Namespace) -> int:
    output = _required_cli_path(arguments.output, "--output")
    records = _required_cli_path(arguments.records, "--records")
    expected_records = output / MANIFEST_NAME
    if not _path_identities_collide(_path_identity(records), _path_identity(expected_records)):
        raise _value_error("--records must point to <output>/manifest.jsonl")
    expected_paths = {
        "--reviewer-a": output / "labels" / "reviewer_a.json",
        "--reviewer-b": output / "labels" / "reviewer_b.json",
        "--adjudication": output / "labels" / "adjudication.json",
    }
    for name, expected in expected_paths.items():
        supplied = _required_cli_path(getattr(arguments, name[2:].replace("-", "_")), name)
        if not _path_identities_collide(_path_identity(supplied), _path_identity(expected)):
            raise _value_error(f"{name} must point to {expected}")
    digest = seal_manifest(output)
    sys.stdout.write(f"{digest}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = _build_cli_parser()
        arguments = parser.parse_args(argv)
        if arguments.agreement:
            mode = "agreement"
        elif arguments.seal:
            mode = "seal"
        else:
            mode = "packet"
        _validate_cli_mode_options(arguments, mode)
        if mode == "agreement":
            return _run_agreement_cli(arguments)
        if mode == "seal":
            return _run_seal_cli(arguments)
        return _run_packet_cli(arguments)
    except _CliArgumentError:
        sys.stderr.write("error: CLI operation failed\n")
        return 1
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        sys.stderr.write(f"{_safe_cli_error(error)}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
