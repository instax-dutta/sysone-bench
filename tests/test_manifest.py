import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.canonical import digest_value
from benchmark.manifest import load_manifest, manifest_digest, validate_manifest


def noul_question(qid: str) -> dict[str, Any]:
    return {"qid": qid, "type": "noul", "instructions": f"Answer {qid}."}


def choice_question(qid: str) -> dict[str, Any]:
    return {
        "qid": qid,
        "type": "choice",
        "instructions": f"Choose {qid}.",
        "criteria": {"left": "left answer", "right": "right answer"},
    }


def score_question(qid: str) -> dict[str, Any]:
    return {
        "qid": qid,
        "type": "score",
        "instructions": f"Score {qid}.",
        "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
        "max_score": 4,
    }


def valid_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 2,
        "dataset_version": "2.0.0",
        "suite_id": "triage",
        "case_id": "triage-0001",
        "order_index": 0,
        "split": "evaluation",
        "state": {"text": "example"},
        "questions": [noul_question("a")],
        "expected": {"a": 1},
        "provenance_id": "source-1",
        "label_provenance": "adjudicated-v2",
    }
    record.update(overrides)
    return record


def test_valid_manifest_returns_exact_summary() -> None:
    records = [
        valid_record(),
        valid_record(
            suite_id="guardrails",
            case_id="guardrails-0001",
            order_index=0,
            questions=[noul_question("risk"), noul_question("safe")],
            expected={"risk": 0, "safe": 1},
        ),
        valid_record(
            suite_id="triage",
            case_id="triage-0002",
            order_index=1,
            split="calibration",
        ),
    ]

    summary = validate_manifest(records, {"triage": 2, "guardrails": 1})

    assert summary.total_cases == 3
    assert summary.total_decisions == 4
    assert summary.by_suite == {"triage": 2, "guardrails": 1}
    assert summary.digest == digest_value(records)
    assert len(summary.digest) == 64


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_required_record_fields(field: str) -> None:
    record = valid_record()
    del record[field]

    with pytest.raises(ValueError, match=field):
        validate_manifest([record], {"triage": 1})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("schema_version", float("nan"), "schema_version"),
        ("dataset_version", "", "dataset_version"),
        ("suite_id", "", "suite_id"),
        ("case_id", "", "case_id"),
        ("order_index", -1, "order_index"),
        ("order_index", True, "order_index"),
        ("order_index", float("inf"), "order_index"),
        ("split", "validation", "split"),
        ("state", [], "state"),
        ("provenance_id", "", "provenance_id"),
        ("label_provenance", "", "label_provenance"),
    ],
)
def test_record_fields_are_strict(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_manifest([valid_record(**{field: value})], {"triage": 1})


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2.0), ("order_index", 0.0)],
)
def test_integral_json_numbers_are_accepted_without_normalizing(field: str, value: float) -> None:
    record = valid_record(**{field: value})
    original = copy.deepcopy(record)

    summary = validate_manifest([record], {"triage": 1})

    assert summary.total_cases == 1
    assert record == original
    assert type(record[field]) is type(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2.5), ("order_index", 0.5)],
)
def test_fractional_manifest_numbers_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        validate_manifest([valid_record(**{field: value})], {"triage": 1})


def test_integral_numeric_order_keys_share_uniqueness_checks() -> None:
    records = [valid_record(order_index=0.0), valid_record(case_id="triage-0002", order_index=0)]

    with pytest.raises(ValueError, match="order_index"):
        validate_manifest(records, {"triage": 2})


def test_expected_counts_reject_integral_floats() -> None:
    with pytest.raises(ValueError, match="count"):
        validate_manifest([valid_record()], {"triage": 1.0})


def test_empty_criteria_keys_are_rejected() -> None:
    questions = [
        {"qid": "a", "type": "choice", "criteria": {"": "description"}},
        {"qid": "a", "type": "noul", "criteria": {"": "description"}},
    ]

    for question in questions:
        with pytest.raises(ValueError, match="criteria|qid"):
            validate_manifest([valid_record(questions=[question])], {"triage": 1})


def test_unknown_record_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="additional"):
        validate_manifest([valid_record(extra=True)], {"triage": 1})


def test_questions_must_be_an_ordered_sequence_of_objects() -> None:
    with pytest.raises(ValueError, match="questions"):
        validate_manifest([valid_record(questions="a")], {"triage": 1})

    with pytest.raises(ValueError, match="question"):
        validate_manifest([valid_record(questions=[1])], {"triage": 1})


@pytest.mark.parametrize(
    "question",
    [
        {},
        {"qid": "", "type": "noul"},
        {"qid": "a"},
        {"qid": "a", "type": "unknown"},
        {"qid": "a", "type": "noul", "unexpected": True},
    ],
)
def test_malformed_questions_are_rejected(question: object) -> None:
    with pytest.raises(ValueError, match="question|qid|type"):
        validate_manifest([valid_record(questions=[question])], {"triage": 1})


def test_duplicate_question_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate question"):
        validate_manifest(
            [valid_record(questions=[noul_question("a"), noul_question("a")])],
            {"triage": 1},
        )


def test_expected_ids_must_match_question_ids() -> None:
    with pytest.raises(ValueError, match="expected.*IDs|IDs.*expected"):
        validate_manifest([valid_record(expected={"b": 1})], {"triage": 1})

    with pytest.raises(ValueError, match="expected.*IDs|IDs.*expected"):
        validate_manifest([valid_record(expected={"a": 1, "b": 2})], {"triage": 1})


@pytest.mark.parametrize("expected", [0, 4, 2.0, 1.5])
def test_score_expected_accepts_finite_numeric_levels(expected: float) -> None:
    record = valid_record(questions=[score_question("score")], expected={"score": expected})

    summary = validate_manifest([record], {"triage": 1})

    assert summary.total_decisions == 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (choice_question("choice"), "missing"),
        (choice_question("choice"), 1),
        (noul_question("noul"), True),
        (noul_question("noul"), 1.0),
        (noul_question("noul"), "1"),
        (score_question("score"), True),
        (score_question("score"), "2"),
        (score_question("score"), -0.1),
        (score_question("score"), 4.1),
        (score_question("score"), float("nan")),
        (score_question("score"), float("inf")),
    ],
)
def test_expected_values_match_question_type_and_range(
    question: dict[str, Any], expected: object
) -> None:
    record = valid_record(questions=[question], expected={question["qid"]: expected})

    with pytest.raises(ValueError, match="expected"):
        validate_manifest([record], {"triage": 1})


def test_duplicate_case_ids_are_rejected_across_suites() -> None:
    records = [
        valid_record(),
        valid_record(suite_id="other", order_index=0),
    ]

    with pytest.raises(ValueError, match="duplicate case ID"):
        validate_manifest(records, {"triage": 1, "other": 1})


def test_duplicate_suite_order_pairs_are_rejected() -> None:
    records = [
        valid_record(),
        valid_record(case_id="triage-0002"),
    ]

    with pytest.raises(ValueError, match="order_index"):
        validate_manifest(records, {"triage": 2})


def test_suite_counts_must_match_exactly() -> None:
    with pytest.raises(ValueError, match="triage"):
        validate_manifest([valid_record()], {"triage": 2})

    with pytest.raises(ValueError, match="other"):
        validate_manifest([valid_record()], {"other": 1})

    with pytest.raises(ValueError, match="other"):
        validate_manifest([valid_record()], {"triage": 1, "other": 1})


def test_expected_counts_require_nonnegative_integer_values() -> None:
    with pytest.raises(ValueError, match="count"):
        validate_manifest([valid_record()], {"triage": -1})

    with pytest.raises(ValueError, match="count"):
        validate_manifest([valid_record()], {"triage": True})


def test_question_order_changes_full_digest() -> None:
    first = valid_record(
        questions=[noul_question("a"), noul_question("b")],
        expected={"a": 1, "b": 0},
    )
    second = valid_record(
        questions=[noul_question("b"), noul_question("a")],
        expected={"a": 1, "b": 0},
    )

    first_digest = manifest_digest([first])
    second_digest = manifest_digest([second])

    assert first_digest != second_digest
    assert first_digest == digest_value([first])
    assert len(first_digest) == 64


def test_record_order_changes_full_digest() -> None:
    first = valid_record()
    second = valid_record(case_id="triage-0002", order_index=1)

    assert manifest_digest([first, second]) != manifest_digest([second, first])


def test_validation_does_not_mutate_records() -> None:
    records = [valid_record()]
    original = copy.deepcopy(records)

    validate_manifest(records, {"triage": 1})

    assert records == original


def test_load_manifest_returns_records_without_normalizing(tmp_path: Path) -> None:
    records = [valid_record()]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    loaded = load_manifest(path)

    assert loaded == records
    assert isinstance(loaded, list)
    assert all(isinstance(record, dict) for record in loaded)


def test_load_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        load_manifest(path)


@pytest.mark.parametrize("contents", ["{}", "null", "[]\n[]", "[1]", "[null]"])
def test_load_manifest_requires_a_list_of_objects(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_manifest(path)


def test_manifest_schema_matches_python_record_contract() -> None:
    schema_path = Path("schemas/manifest-v2.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    record_schema = schema["items"]

    assert schema["type"] == "array"
    assert set(record_schema["required"]) == {
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
    }
    assert record_schema["additionalProperties"] is False
    properties = record_schema["properties"]
    assert properties["split"]["enum"] == ["calibration", "evaluation"]
    assert properties["schema_version"] == {"const": 2, "type": "integer"}
    assert properties["order_index"] == {"minimum": 0, "type": "integer"}

    questions_schema = properties["questions"]
    assert questions_schema["type"] == "array"
    assert questions_schema["uniqueItems"] is True
    question_schema = questions_schema["items"]
    criteria_object_forms = [
        question_schema["properties"]["criteria"]["oneOf"][0],
        question_schema["allOf"][0]["then"]["properties"]["criteria"],
        question_schema["allOf"][1]["then"]["properties"]["criteria"],
    ]
    for criteria_schema in criteria_object_forms:
        assert criteria_schema["propertyNames"] == {"minLength": 1, "type": "string"}
        assert criteria_schema["minProperties"] == 1

    description = schema["description"]
    for constraint in ("case_id", "suite_id", "order_index", "question", "expected"):
        assert constraint in description
    assert "authoritative" in description.lower()
