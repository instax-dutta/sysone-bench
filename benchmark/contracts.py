from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

_ANSWER_TYPES = frozenset({"choice", "noul", "score"})
_BASE_QUESTION_FIELDS = frozenset({"qid", "type", "instructions"})
_CHOICE_ANSWER_FIELDS = frozenset({"type", "choice", "probabilities", "confidence"})
_NOUL_ANSWER_FIELDS = frozenset({"type", "noul"})
_SCORE_ANSWER_FIELDS = frozenset({"type", "score", "confidence"})


def _value_error(message: str) -> ValueError:
    return ValueError(message)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if not _is_number(value):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _format_keys(keys: Iterable[object]) -> str:
    ordered = sorted(keys, key=lambda key: (type(key).__name__, repr(key)))
    return ", ".join(repr(key) for key in ordered)


def _question_id(question: Mapping[str, Any]) -> str | None:
    value = question.get("qid")
    if isinstance(value, str) and value:
        return value
    return None


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _value_error(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _validate_text(value: object, qid: str, field: str) -> None:
    if not isinstance(value, str):
        raise _value_error(f"{qid}: {field} must be a string")


def _validate_criteria_mapping(criteria: object, qid: str) -> tuple[str, ...]:
    if not isinstance(criteria, Mapping):
        raise _value_error(f"{qid}: criteria must be a mapping")
    labels: list[str] = []
    for label, description in criteria.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{qid}: criteria labels must be non-empty strings")
        _validate_text(description, qid, f"criteria[{label!r}]")
        labels.append(label)
    if not labels:
        raise ValueError(f"{qid}: criteria must contain at least one label")
    return tuple(labels)


def _validate_score_criteria(criteria: object, qid: str) -> None:
    if isinstance(criteria, (str, bytes, bytearray)) or not isinstance(criteria, Sequence):
        raise _value_error(f"{qid}: criteria must be an ordered sequence")
    if not criteria:
        raise ValueError(f"{qid}: criteria must contain at least one level")
    labels: list[str] = []
    for level in criteria:
        if not isinstance(level, str) or not level:
            raise ValueError(f"{qid}: criteria levels must be non-empty strings")
        labels.append(level)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{qid}: criteria levels must be unique")


def _validate_question_fields(question: Mapping[str, Any], qid: str, question_type: str) -> None:
    if "instructions" in question:
        _validate_text(question["instructions"], qid, "instructions")

    allowed_fields = set(_BASE_QUESTION_FIELDS)
    if question_type == "choice":
        allowed_fields.add("criteria")
        _validate_criteria_mapping(question.get("criteria"), qid)
    elif question_type == "noul":
        if "criteria" in question:
            _validate_criteria_mapping(question["criteria"], qid)
        else:
            allowed_fields.add("criteria")
    else:
        allowed_fields.add("criteria")
        allowed_fields.add("max_score")
        if "criteria" in question:
            _validate_score_criteria(question["criteria"], qid)
        if "max_score" not in question:
            raise ValueError(f"{qid}: missing score range max_score")
        max_score = question["max_score"]
        if not _is_finite_number(max_score):
            raise ValueError(f"{qid}: max_score must be a finite number")
        if max_score < 0:
            raise ValueError(f"{qid}: max_score must be nonnegative")

    unknown_fields = set(question) - allowed_fields
    if unknown_fields:
        raise ValueError(f"{qid}: unsupported question field(s): {_format_keys(unknown_fields)}")


def _validate_question(question: Mapping[str, Any]) -> tuple[str, str]:
    qid = _question_id(question)
    if qid is None:
        raise ValueError("answer set: question qid must be a non-empty string")
    question_type = question.get("type")
    if not isinstance(question_type, str):
        raise _value_error(f"{qid}: question type must be a string")
    if question_type not in _ANSWER_TYPES:
        raise ValueError(f"{qid}: unsupported answer type {question_type!r}")
    _validate_question_fields(question, qid, question_type)
    return qid, question_type


def _probability(value: object) -> float:
    if not _is_number(value) or not _is_finite_number(value):
        raise ValueError("probability must be finite and in [0, 1]")
    number = cast(int | float, value)
    if not 0 <= number <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    return float(number)


def _validate_probability(value: object, qid: str, field: str) -> float:
    try:
        return _probability(value)
    except ValueError as error:
        raise ValueError(f"{qid}: {field} {error}") from None


def _validate_number(value: object, qid: str, field: str) -> int | float:
    if not _is_finite_number(value):
        raise ValueError(f"{qid}: {field} must be a finite number")
    return cast(int | float, value)


def _validate_answer_fields(answer: Mapping[str, Any], qid: str, answer_type: str) -> None:
    required: frozenset[str]
    if answer_type == "choice":
        required = _CHOICE_ANSWER_FIELDS
    elif answer_type == "noul":
        required = _NOUL_ANSWER_FIELDS
    else:
        required = _SCORE_ANSWER_FIELDS
    keys = set(answer)
    unknown = keys - _SCORE_ANSWER_FIELDS if answer_type == "score" else keys - required
    if unknown:
        raise ValueError(f"{qid}: unsupported answer field(s): {_format_keys(unknown)}")
    missing = required - keys
    if missing:
        raise ValueError(f"{qid}: missing answer field(s): {_format_keys(missing)}")


def _validate_choice_answer(
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    qid: str,
) -> None:
    labels = _validate_criteria_mapping(question.get("criteria"), qid)
    choice = answer["choice"]
    if not isinstance(choice, str) or choice not in labels:
        raise ValueError(f"{qid}: illegal choice label")
    probabilities = answer["probabilities"]
    if not isinstance(probabilities, Mapping):
        raise _value_error(f"{qid}: probabilities must be a mapping")
    if set(probabilities) != set(labels):
        raise ValueError(f"{qid}: probability labels do not match choices")
    values = [
        _validate_probability(probabilities[label], qid, f"probabilities[{label!r}]")
        for label in labels
    ]
    if abs(math.fsum(values) - 1.0) > 1e-6:
        raise ValueError(f"{qid}: probabilities must sum to one")
    _validate_probability(answer["confidence"], qid, "confidence")


def _validate_noul_answer(answer: Mapping[str, Any], qid: str) -> None:
    _validate_probability(answer["noul"], qid, "noul probability")


def _validate_score_answer(
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    qid: str,
) -> None:
    value = _validate_number(answer["score"], qid, "score")
    max_score = _validate_number(question["max_score"], qid, "max_score")
    if not 0 <= value <= max_score:
        raise ValueError(f"{qid}: score is outside the declared range")
    if "confidence" in answer and answer["confidence"] is not None:
        _validate_probability(answer["confidence"], qid, "confidence")


def validate_answer(question: Mapping[str, Any], answer: Mapping[str, Any]) -> None:
    question_object = _require_mapping(question, "question")
    qid, question_type = _validate_question(question_object)
    try:
        answer_object = _require_mapping(answer, "answer")
    except ValueError as error:
        raise ValueError(f"{qid}: {error}") from None
    answer_type = answer_object.get("type")
    if not isinstance(answer_type, str):
        raise _value_error(f"{qid}: missing answer type")
    if answer_type not in _ANSWER_TYPES:
        raise ValueError(f"{qid}: unsupported answer type {answer_type!r}")
    if answer_type != question_type:
        raise ValueError(f"{qid}: answer type does not match question type")
    _validate_answer_fields(answer_object, qid, answer_type)
    if answer_type == "choice":
        _validate_choice_answer(question_object, answer_object, qid)
    elif answer_type == "noul":
        _validate_noul_answer(answer_object, qid)
    else:
        _validate_score_answer(question_object, answer_object, qid)


def validate_answers(questions: Sequence[Mapping[str, Any]], answers: Mapping[str, Any]) -> None:
    if isinstance(questions, (str, bytes, bytearray)) or not isinstance(questions, Sequence):
        raise _value_error("questions must be a sequence")
    answers_object = _require_mapping(answers, "answer set")
    validated_questions: list[tuple[Mapping[str, Any], str]] = []
    seen: set[str] = set()
    for question in questions:
        question_object = _require_mapping(question, "question")
        qid, _ = _validate_question(question_object)
        if qid in seen:
            raise ValueError(f"{qid}: duplicate question id")
        seen.add(qid)
        validated_questions.append((question_object, qid))

    actual = set(answers_object)
    missing = seen - actual
    extra = actual - seen
    if missing or extra:
        details: list[str] = []
        if missing:
            missing_text = ", ".join(f"{qid}: missing answer" for qid in sorted(missing))
            details.append(missing_text)
        if extra:
            if seen:
                expected_text = ", ".join(sorted(seen))
                details.append(f"{expected_text}: extra answer(s): {_format_keys(extra)}")
            else:
                details.append(f"answer set: extra answer(s): {_format_keys(extra)}")
        raise ValueError("question IDs differ: " + "; ".join(details))

    for question_object, qid in validated_questions:
        validate_answer(question_object, answers_object[qid])
