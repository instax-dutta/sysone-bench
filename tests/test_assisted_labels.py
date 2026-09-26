from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fixtures import load_test_records
from test_dataset_seal import _synthetic_provenance, _write_json, _write_manifest

from benchmark.manifest import manifest_digest, validate_manifest
from datasets.v2 import assisted_labels
from datasets.v2.assisted_labels import (
    ASSISTED_LABEL_PROVENANCE,
    ASSISTED_REVIEW_PROTOCOL_ID,
    ASSISTED_REVIEWED_QUESTIONS,
    apply_assisted_review,
    finalize_assisted_dataset,
)

ASSISTANT_MODEL = "test-assistant-model"
ASSISTED_EXPORT_NAME = "assistant_confirmed.json"
SEALED_NAME = "manifest.sha256"
MANIFEST_NAME = "manifest.jsonl"
PROVENANCE_NAME = "provenance.json"
PROVISIONAL_MANIFEST_NAME = "manifest.provisional.jsonl"
PROVISIONAL_PROVENANCE_NAME = "provenance.provisional.json"


def _assisted_records() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = [
        {
            "qid": "intent",
            "type": "choice",
            "instructions": "Choose the synthetic intent.",
            "criteria": {"alpha": "Alpha", "beta": "Beta"},
        },
        {"qid": "flag", "type": "noul", "instructions": "Choose the synthetic flag."},
        {
            "qid": "intensity",
            "type": "score",
            "instructions": "Choose the synthetic score.",
            "criteria": ["zero", "one", "two"],
            "max_score": 2,
        },
    ]
    return [
        {
            "schema_version": 2,
            "dataset_version": "2.0.0",
            "suite_id": "multilingual_intent",
            "case_id": case_id,
            "order_index": index,
            "split": split,
            "state": {"text": f"synthetic assisted state {index}", "language": language},
            "questions": copy.deepcopy(questions),
            "expected": {"intent": "alpha", "flag": 0, "intensity": 1},
            "provenance_id": "synthetic-source",
            "label_provenance": "pending-human-review",
        }
        for index, (case_id, split, language) in enumerate(
            (("assisted-alpha", "evaluation", "hi"), ("assisted-beta", "calibration", "es"))
        )
    ]


def _assistant_payload(
    records: Sequence[Mapping[str, Any]],
    model: str = ASSISTANT_MODEL,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "cerebras-assistant",
        "official_labels": False,
        "model": model,
        "entries": [
            {
                "case_id": record["case_id"],
                "qid": question["qid"],
                "answer": record["expected"][question["qid"]],
                "question_type": question["type"],
                "model": model,
            }
            for record in records
            for question in record["questions"]
        ],
    }


def _apply(
    records: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return apply_assisted_review(records, payload)


def test_apply_replaces_expected_values_and_sets_protocol_provenance() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["answer"] = "beta"
    payload["entries"][1]["answer"] = 1
    payload["entries"][2]["answer"] = 2

    applied = _apply(records, payload)

    assert applied[0]["expected"] == {"intent": "beta", "flag": 1, "intensity": 2}
    assert applied[1]["expected"] == {"intent": "alpha", "flag": 0, "intensity": 1}
    assert all(record["label_provenance"] == ASSISTED_LABEL_PROVENANCE for record in applied)


def test_apply_preserves_order_state_questions_splits_and_provenance() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)

    applied = _apply(records, payload)

    assert [record["case_id"] for record in applied] == [record["case_id"] for record in records]
    for applied_record, record in zip(applied, records, strict=True):
        for field in (
            "suite_id",
            "order_index",
            "split",
            "provenance_id",
            "schema_version",
            "dataset_version",
        ):
            assert applied_record[field] == record[field]
        assert applied_record["state"] == record["state"]
        assert applied_record["questions"] == record["questions"]


def test_apply_never_mutates_records_or_payload() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["answer"] = "beta"
    records_before = copy.deepcopy(records)
    payload_before = copy.deepcopy(payload)

    _apply(records, payload)

    assert records == records_before
    assert payload == payload_before


def test_apply_returns_a_deep_copy() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][3]["answer"] = "beta"

    applied = _apply(records, payload)

    assert applied[0] is not records[0]
    assert applied[0]["state"] is not records[0]["state"]
    assert applied[0]["questions"] is not records[0]["questions"]
    assert applied[0]["expected"] is not records[0]["expected"]
    assert applied[1]["expected"]["intent"] == "beta"


def test_apply_result_passes_manifest_validation() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["answer"] = "beta"

    applied = _apply(records, payload)

    summary = validate_manifest(applied, {"multilingual_intent": 2})
    assert summary.total_cases == 2
    assert summary.total_decisions == 6
    assert summary.digest == manifest_digest(applied)


def test_apply_rejects_records_that_are_not_a_sequence() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)

    for invalid in ("records", b"records", {"case_id": "x"}, 3):
        with pytest.raises(TypeError):
            apply_assisted_review(invalid, payload)  # type: ignore[arg-type]


def test_apply_rejects_payload_that_is_not_a_mapping() -> None:
    records = _assisted_records()

    for invalid in ([], "payload", 3, None):
        with pytest.raises(TypeError):
            apply_assisted_review(records, invalid)  # type: ignore[arg-type]


def test_apply_rejects_records_that_are_not_mappings() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    records[0] = "not-a-record"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must be a mapping"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize(
    "field", ["schema_version", "source", "official_labels", "model", "entries"]
)
def test_export_requires_exact_top_level_keys(field: str) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    del payload[field]

    with pytest.raises(ValueError, match="export fields are invalid: missing"):
        apply_assisted_review(records, payload)


def test_export_rejects_unknown_top_level_key() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["assistant_reasoning"] = "forbidden"

    with pytest.raises(ValueError, match="export fields are invalid: extra"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", [0, 2, "1", None, True])
def test_export_requires_exact_schema_version(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["schema_version"] = value

    with pytest.raises(ValueError, match="schema_version must equal 1"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize(
    "value",
    ["cerebras-assistant-v1", "Cerebras-Assistant", "opencode", "", None, 3],
)
def test_export_requires_exact_source(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["source"] = value

    with pytest.raises(ValueError, match="source must be"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", [0, 1, "false", "False", None, [], {}, "true"])
def test_export_requires_exact_official_labels_false(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["official_labels"] = value

    with pytest.raises(ValueError, match="official_labels must be exactly false"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", ["", "   ", None, 3, [], {}])
def test_export_requires_nonempty_model(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["model"] = value

    with pytest.raises(ValueError, match="model must be a nonempty string"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", [{}, "entries", 3, None])
def test_export_requires_a_list_of_entries(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"] = value

    with pytest.raises(ValueError, match="entries must be a list"):
        apply_assisted_review(records, payload)


def test_export_rejects_empty_entries() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"] = []

    with pytest.raises(ValueError, match="missing a manifest question"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("field", ["case_id", "qid", "answer", "question_type", "model"])
def test_export_entry_requires_exact_keys(field: str) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    del payload["entries"][0][field]

    with pytest.raises(ValueError, match="entry 0 fields are invalid: missing"):
        apply_assisted_review(records, payload)


def test_export_entry_rejects_unknown_key() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["confidence"] = 0.5

    with pytest.raises(ValueError, match="entry 0 fields are invalid: extra"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", ["entry", 3, None, ["assisted-alpha", "intent"]])
def test_export_entry_must_be_an_object(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0] = value

    with pytest.raises(ValueError, match="entry 0 must be an object"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize(
    ("case_id", "qid"),
    [
        ("assisted-alpha", "missing_question"),
        ("missing-case", "intent"),
    ],
)
def test_export_entry_must_match_a_manifest_question(case_id: str, qid: str) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["case_id"] = case_id
    payload["entries"][0]["qid"] = qid

    with pytest.raises(ValueError, match="does not match a manifest question"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("field", ["case_id", "qid"])
@pytest.mark.parametrize("value", ["", "   ", None, 3])
def test_export_entry_requires_nonempty_identity(field: str, value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0][field] = value

    with pytest.raises(ValueError, match=f"{field} must be a nonempty string"):
        apply_assisted_review(records, payload)


def test_export_rejects_duplicate_entries() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"].append(copy.deepcopy(payload["entries"][0]))

    with pytest.raises(ValueError, match="duplicates an earlier entry"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", ["noul", "score", "Choice", "nOul"])
def test_export_entry_question_type_must_match_manifest(value: str) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["question_type"] = value

    with pytest.raises(ValueError, match="question_type must match"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", ["", "   ", None, 3])
def test_export_entry_requires_question_type_text(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["question_type"] = value

    with pytest.raises(ValueError, match="question_type must be a nonempty string"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("value", ["", "   ", None, 3])
def test_export_entry_requires_nonempty_model(value: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["model"] = value

    with pytest.raises(ValueError, match="model must be a nonempty string"):
        apply_assisted_review(records, payload)


def test_apply_rejects_incomplete_question_coverage() -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"] = payload["entries"][:-1]

    with pytest.raises(ValueError, match="missing a manifest question"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("answer", ["gamma", "", 3, None, True])
def test_apply_rejects_illegal_choice_answers(answer: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][0]["answer"] = answer

    with pytest.raises(ValueError, match="choice"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("answer", [True, 2, -1, "1", 1.0, None])
def test_apply_rejects_illegal_noul_answers(answer: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][1]["answer"] = answer

    with pytest.raises(ValueError, match="noul"):
        apply_assisted_review(records, payload)


@pytest.mark.parametrize("answer", [3, -1, 2.5, "two", True, None])
def test_apply_rejects_illegal_score_answers(answer: object) -> None:
    records = _assisted_records()
    payload = _assistant_payload(records)
    payload["entries"][2]["answer"] = answer

    with pytest.raises(ValueError, match="score"):
        apply_assisted_review(records, payload)


def _assistant_export_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _seed_assisted_root(
    tmp_path: Path, name: str = "dataset", model: str = ASSISTANT_MODEL
) -> tuple[list[dict[str, Any]], Path, Path]:
    records = load_test_records()
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    _write_manifest(root / MANIFEST_NAME, records)
    _write_json(root / PROVENANCE_NAME, _synthetic_provenance(records))
    payload = _assistant_payload(records, model)
    for entry in payload["entries"]:
        if entry["case_id"] == "triage-0000" and entry["qid"] == "intent":
            entry["answer"] = "technical_help"
    assistant = tmp_path / f"{name}-downloads" / ASSISTED_EXPORT_NAME
    assistant.parent.mkdir(parents=True, exist_ok=True)
    assistant.write_bytes(_assistant_export_bytes(payload))
    return records, root, assistant


def _digest_validator(records: Sequence[Mapping[str, Any]], sources: Any = None, **_: Any) -> Any:
    del sources
    return {"digest": manifest_digest(records)}


def test_finalize_writes_transparent_protocol_artifacts_and_exact_checksum(tmp_path: Path) -> None:
    records, root, assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()
    provenance_before = json.loads((root / PROVENANCE_NAME).read_text(encoding="utf-8"))
    assistant_before = assistant.read_bytes()

    digest = finalize_assisted_dataset(root, assistant)

    final_records = [
        json.loads(line) for line in (root / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    ]
    raw_digest = hashlib.sha256((root / MANIFEST_NAME).read_bytes()).hexdigest()
    assert digest == raw_digest
    assert len(digest) == 64
    assert (root / SEALED_NAME).read_text(encoding="utf-8") == f"{raw_digest}  {MANIFEST_NAME}\n"
    assert (root / PROVISIONAL_MANIFEST_NAME).read_bytes() == manifest_before
    assert (root / PROVISIONAL_PROVENANCE_NAME).read_bytes() != b""
    assert (
        json.loads((root / PROVISIONAL_PROVENANCE_NAME).read_text(encoding="utf-8"))
        == provenance_before
    )
    assert (root / "labels" / ASSISTED_EXPORT_NAME).read_bytes() == assistant_before
    assert (root / MANIFEST_NAME).read_bytes() != manifest_before
    assert [record["case_id"] for record in final_records] == [
        record["case_id"] for record in records
    ]
    assert final_records[0]["expected"]["intent"] == "technical_help"
    assert all(record["label_provenance"] == ASSISTED_LABEL_PROVENANCE for record in final_records)
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))
    assert not list((root / "labels").glob(".assistant_confirmed.json.*"))


def test_finalize_records_the_approved_single_review_protocol(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)

    finalize_assisted_dataset(root, assistant)

    finalized = json.loads((root / PROVENANCE_NAME).read_text(encoding="utf-8"))
    review = finalized["label_review"]
    assert finalized["provisional"] is False
    assert (
        finalized["records"]
        == json.loads((root / PROVISIONAL_PROVENANCE_NAME).read_text(encoding="utf-8"))["records"]
    )
    assert review["protocol"] == ASSISTED_REVIEW_PROTOCOL_ID
    assert review["reviewer_count"] == 1
    assert review["independent_human_review"] is False
    assert review["adjudication"] is False
    assert review["assistant_model"] == ASSISTANT_MODEL
    assert review["assistant_export_sha256"] == hashlib.sha256(assistant.read_bytes()).hexdigest()
    assert review["questions_reviewed"] == ASSISTED_REVIEWED_QUESTIONS
    assert review["entries_applied"] == ASSISTED_REVIEWED_QUESTIONS
    assert len(review["method"]) < 240
    assert "corrected" in review["method"]
    assert "one human" in review["method"].lower()


def test_finalize_never_fabricates_reviewer_or_adjudication_files(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)

    finalize_assisted_dataset(root, assistant)

    for name in ("reviewer_a.json", "reviewer_b.json", "adjudication.json"):
        assert not (root / "labels" / name).exists()
    assert sorted(path.name for path in (root / "labels").iterdir()) == [ASSISTED_EXPORT_NAME]


def test_finalize_validates_with_the_finalized_provenance_as_sources_and_registry(
    tmp_path: Path,
) -> None:
    from datasets.v2.validate import validate_dataset

    _records, root, assistant = _seed_assisted_root(tmp_path)
    observed: dict[str, Any] = {}

    def capture(records: Sequence[Mapping[str, Any]], sources: Any, **kwargs: Any) -> Any:
        observed["sources"] = sources
        observed["registry"] = kwargs.get("registry")
        return validate_dataset(records, sources, **kwargs)

    digest = finalize_assisted_dataset(root, assistant, validator=capture)

    final_records = [
        json.loads(line) for line in (root / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert observed["sources"] is observed["registry"]
    assert observed["sources"]["provisional"] is False
    assert observed["sources"]["label_review"]["protocol"] == ASSISTED_REVIEW_PROTOCOL_ID
    assert (
        observed["sources"]["records"]
        == json.loads((root / PROVISIONAL_PROVENANCE_NAME).read_text(encoding="utf-8"))["records"]
    )
    raw_digest = hashlib.sha256((root / MANIFEST_NAME).read_bytes()).hexdigest()
    assert digest == raw_digest
    assert len(final_records) == 1190


def test_finalize_refuses_existing_checksum_and_preserves_every_artifact(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    checksum = root / SEALED_NAME
    checksum.write_text("occupied  manifest.jsonl\n", encoding="utf-8")
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="sealed"):
        finalize_assisted_dataset(root, assistant)

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("value", [0, "true", None, 1, [], {}])
def test_finalize_requires_provisional_boolean_true(tmp_path: Path, value: object) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    provenance_path = root / PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["provisional"] = value
    _write_json(provenance_path, provenance)
    before = provenance_path.read_bytes()

    with pytest.raises(ValueError, match="provisional.*boolean true"):
        finalize_assisted_dataset(root, assistant)

    assert provenance_path.read_bytes() == before
    assert not (root / SEALED_NAME).exists()
    assert not (root / PROVISIONAL_MANIFEST_NAME).exists()


def test_finalize_requires_the_provisional_key(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    provenance_path = root / PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["provisional"]
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="provisional.*true"):
        finalize_assisted_dataset(root, assistant)

    assert not (root / SEALED_NAME).exists()


def test_finalize_refuses_a_finalized_sidecar(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    provenance_path = root / PROVENANCE_NAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["provisional"] = False
    _write_json(provenance_path, provenance)

    with pytest.raises(FileExistsError, match="sealed"):
        finalize_assisted_dataset(root, assistant)

    assert not (root / SEALED_NAME).exists()


@pytest.mark.parametrize("backup", [PROVISIONAL_MANIFEST_NAME, PROVISIONAL_PROVENANCE_NAME])
def test_finalize_refuses_existing_backups(tmp_path: Path, backup: str) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    (root / backup).write_bytes(b"occupied\n")

    with pytest.raises(FileExistsError, match=backup):
        finalize_assisted_dataset(root, assistant)

    assert (root / backup).read_bytes() == b"occupied\n"
    assert not (root / SEALED_NAME).exists()
    assert not (root / "labels").exists()


@pytest.mark.parametrize("required", [MANIFEST_NAME, PROVENANCE_NAME])
def test_finalize_requires_regular_dataset_inputs(tmp_path: Path, required: str) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    (root / required).unlink()

    with pytest.raises(ValueError, match="requires|not valid"):
        finalize_assisted_dataset(root, assistant)

    assert not (root / SEALED_NAME).exists()


def test_finalize_requires_the_assistant_export(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    assistant.unlink()

    with pytest.raises(ValueError, match="requires|not valid"):
        finalize_assisted_dataset(root, assistant)

    assert not (root / SEALED_NAME).exists()
    assert not (root / "labels").exists()


def test_finalize_requires_a_directory_output_root(tmp_path: Path) -> None:
    _records, _root, assistant = _seed_assisted_root(tmp_path)

    with pytest.raises(ValueError, match="existing directory"):
        finalize_assisted_dataset(tmp_path / "absent", assistant)


@pytest.mark.parametrize("required", [MANIFEST_NAME, PROVENANCE_NAME, ASSISTED_EXPORT_NAME])
def test_finalize_refuses_symlinked_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: str
) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    target = root / required if required != ASSISTED_EXPORT_NAME else assistant
    relocated = target.with_name(f"relocated-{required}")
    os.replace(target, relocated)
    target.symlink_to(relocated)

    with pytest.raises(ValueError, match="not valid.*symbolic link"):
        finalize_assisted_dataset(root, assistant)

    assert relocated.read_bytes()
    assert not (root / SEALED_NAME).exists()
    assert not (root / PROVISIONAL_MANIFEST_NAME).exists()
    del monkeypatch


def test_finalize_rejects_non_path_arguments(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)

    with pytest.raises(TypeError):
        finalize_assisted_dataset(str(root), assistant)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        finalize_assisted_dataset(root, str(assistant))  # type: ignore[arg-type]


def test_finalize_rolls_back_and_removes_only_owned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()
    provenance_before = (root / PROVENANCE_NAME).read_bytes()
    assistant_before = assistant.read_bytes()
    real_replace = os.replace
    failed = False

    def fail_provenance_replace(source: str, destination: str) -> None:
        nonlocal failed
        if Path(destination).name == PROVENANCE_NAME and not failed:
            failed = True
            raise OSError("synthetic provenance replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(assisted_labels.os, "replace", fail_provenance_replace)

    with pytest.raises(OSError, match="synthetic provenance replacement failure"):
        finalize_assisted_dataset(root, assistant)

    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert (root / PROVENANCE_NAME).read_bytes() == provenance_before
    assert assistant.read_bytes() == assistant_before
    assert not (root / SEALED_NAME).exists()
    assert not (root / PROVISIONAL_MANIFEST_NAME).exists()
    assert not (root / PROVISIONAL_PROVENANCE_NAME).exists()
    assert not (root / "labels").exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_finalize_restores_a_preexisting_assisted_label_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    copy_path = labels_dir / ASSISTED_EXPORT_NAME
    copy_path.write_bytes(b"earlier export\n")
    real_replace = os.replace
    failed = False

    def fail_provenance_replace(source: str, destination: str) -> None:
        nonlocal failed
        if Path(destination).name == PROVENANCE_NAME and not failed:
            failed = True
            raise OSError("synthetic provenance replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(assisted_labels.os, "replace", fail_provenance_replace)

    with pytest.raises(OSError, match="synthetic provenance replacement failure"):
        finalize_assisted_dataset(root, assistant)

    assert copy_path.read_bytes() == b"earlier export\n"
    assert not (root / SEALED_NAME).exists()
    assert not (root / PROVISIONAL_MANIFEST_NAME).exists()


def test_finalize_rolls_back_when_the_checksum_collides_after_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()
    provenance_before = (root / PROVENANCE_NAME).read_bytes()
    real_stage = assisted_labels._stage_bytes

    def stage_then_collide(path: Path, content: bytes) -> Path:
        staged = real_stage(path, content)
        if path.name == PROVENANCE_NAME:
            (root / SEALED_NAME).write_bytes(b"concurrent  manifest.jsonl\n")
        return staged

    def unexpected_replace(source: str, destination: str) -> None:
        del source, destination
        raise AssertionError("replacement occurred before checksum collision")

    monkeypatch.setattr(assisted_labels, "_stage_bytes", stage_then_collide)
    monkeypatch.setattr(assisted_labels.os, "replace", unexpected_replace)

    with pytest.raises(FileExistsError, match="sealed"):
        finalize_assisted_dataset(root, assistant)

    assert (root / SEALED_NAME).read_bytes() == b"concurrent  manifest.jsonl\n"
    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert (root / PROVENANCE_NAME).read_bytes() == provenance_before
    assert not (root / PROVISIONAL_MANIFEST_NAME).exists()
    assert not (root / "labels").exists()


def test_finalize_rejects_a_validator_digest_mismatch(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()
    provenance_before = (root / PROVENANCE_NAME).read_bytes()

    with pytest.raises(ValueError, match="digest"):
        finalize_assisted_dataset(
            root, assistant, validator=lambda records, sources, **kwargs: {"digest": "b" * 64}
        )

    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert (root / PROVENANCE_NAME).read_bytes() == provenance_before
    assert not (root / SEALED_NAME).exists()
    assert not (root / PROVISIONAL_MANIFEST_NAME).exists()
    assert not (root / "labels").exists()


def test_finalize_rejects_a_non_mapping_validator_summary(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)

    with pytest.raises(TypeError):
        finalize_assisted_dataset(
            root, assistant, validator=lambda records, sources, **kwargs: ["digest"]
        )

    assert not (root / SEALED_NAME).exists()
    assert not (root / "labels").exists()


def test_finalize_requires_exactly_the_approved_question_count(tmp_path: Path) -> None:
    records = _assisted_records()
    root = tmp_path / "small"
    root.mkdir(parents=True)
    _write_manifest(root / MANIFEST_NAME, records)
    _write_json(root / PROVENANCE_NAME, {"provisional": True})
    assistant = tmp_path / "small" / ASSISTED_EXPORT_NAME
    assistant.write_bytes(_assistant_export_bytes(_assistant_payload(records)))

    with pytest.raises(ValueError, match="1550"):
        finalize_assisted_dataset(root, assistant, validator=_digest_validator)

    assert not (root / SEALED_NAME).exists()
    assert not (root / "labels").exists()


def test_finalize_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    _records, first_root, first_assistant = _seed_assisted_root(tmp_path, "first")
    _records, second_root, second_assistant = _seed_assisted_root(tmp_path, "second")

    first = finalize_assisted_dataset(first_root, first_assistant, validator=_digest_validator)
    second = finalize_assisted_dataset(second_root, second_assistant, validator=_digest_validator)

    assert first == second
    assert (first_root / MANIFEST_NAME).read_bytes() == (second_root / MANIFEST_NAME).read_bytes()
    assert (first_root / SEALED_NAME).read_bytes() == (second_root / SEALED_NAME).read_bytes()
    assert (first_root / PROVENANCE_NAME).read_bytes() == (
        second_root / PROVENANCE_NAME
    ).read_bytes()


def test_finalize_digest_changes_when_confirmed_answers_change(tmp_path: Path) -> None:
    _records, first_root, first_assistant = _seed_assisted_root(tmp_path, "first")
    _records, second_root, second_assistant = _seed_assisted_root(tmp_path, "second")
    payload = json.loads(second_assistant.read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        if entry["case_id"] == "triage-0000" and entry["qid"] == "intent":
            entry["answer"] = "information"
    second_assistant.write_bytes(_assistant_export_bytes(payload))

    first = finalize_assisted_dataset(first_root, first_assistant, validator=_digest_validator)
    second = finalize_assisted_dataset(second_root, second_assistant, validator=_digest_validator)

    assert first != second


def test_finalize_uses_the_exact_export_bytes_for_its_checksum(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    payload = json.loads(assistant.read_text(encoding="utf-8"))
    assistant.write_bytes((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    before = hashlib.sha256(assistant.read_bytes()).hexdigest()

    finalize_assisted_dataset(root, assistant, validator=_digest_validator)

    finalized = json.loads((root / PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert finalized["label_review"]["assistant_export_sha256"] == before


def test_assisted_labels_module_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "datasets.v2.assisted_labels", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--output" in result.stdout
    assert "--assistant" in result.stdout


def test_assisted_cli_prints_only_the_full_digest(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)

    result = _run_cli("--output", str(root), "--assistant", str(assistant))

    assert result.returncode == 0
    assert result.stderr == ""
    assert len(result.stdout.strip()) == 64
    checksum = (root / SEALED_NAME).read_text(encoding="utf-8")
    assert result.stdout == f"{checksum.split()[0]}\n"
    assert "\n" not in result.stdout.strip()


@pytest.mark.parametrize(
    ("provided", "expected_terms"),
    [
        (["--output"], ["--assistant", "required"]),
        (["--assistant"], ["--output", "required"]),
        ([], ["--output", "required"]),
    ],
)
def test_assisted_cli_requires_output_and_assistant(
    tmp_path: Path, provided: list[str], expected_terms: list[str]
) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)

    result = _run_cli(*_complete_arguments(provided, root, assistant))

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    for term in expected_terms:
        assert term in result.stderr
    assert not (root / SEALED_NAME).exists()
    assert not (root / "labels").exists()


def _complete_arguments(provided: Sequence[str], root: Path, assistant: Path) -> list[str]:
    values = {"--output": root, "--assistant": assistant}
    arguments: list[str] = []
    for option in provided:
        arguments.extend([option, str(values[option])])
    return arguments


def test_assisted_cli_rejects_unknown_flags_and_values(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()
    secret_flag = "--unknown-secret-flag"
    secret_value = "unknown-secret-value"

    result = _run_cli(
        "--output",
        str(root),
        "--assistant",
        str(assistant),
        secret_flag,
        secret_value,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    assert secret_flag not in result.stderr
    assert secret_value not in result.stderr
    assert "Traceback" not in result.stderr
    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert not (root / SEALED_NAME).exists()


def test_assisted_cli_redacts_secret_like_input_paths(tmp_path: Path) -> None:
    secret_path = tmp_path / "path-secret-marker.jsonl"
    secret_path.write_text("secret-secret-value\n", encoding="utf-8")
    _records, root, assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()

    result = _run_cli("--output", str(root), "--assistant", str(secret_path))

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    assert "path-secret-marker" not in result.stderr
    assert "secret-secret-value" not in result.stderr
    assert "Traceback" not in result.stderr
    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert not (root / SEALED_NAME).exists()
    del assistant


def test_assisted_cli_redacts_unknown_assistant_export_keys(tmp_path: Path) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    payload = json.loads(assistant.read_text(encoding="utf-8"))
    payload["secret_export_key"] = "secret_export_value"
    assistant.write_bytes(_assistant_export_bytes(payload))
    manifest_before = (root / MANIFEST_NAME).read_bytes()

    result = _run_cli("--output", str(root), "--assistant", str(assistant))

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    assert "secret_export_key" not in result.stderr
    assert "secret_export_value" not in result.stderr
    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert not (root / SEALED_NAME).exists()


def test_assisted_cli_rejects_output_assistant_collision(tmp_path: Path) -> None:
    _records, root, _assistant = _seed_assisted_root(tmp_path)
    manifest_before = (root / MANIFEST_NAME).read_bytes()

    result = _run_cli("--output", str(root), "--assistant", str(root))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "collision" in result.stderr
    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert not (root / SEALED_NAME).exists()


def test_assisted_cli_rejects_output_colliding_with_the_export_by_identity(
    tmp_path: Path,
) -> None:
    _records, root, assistant = _seed_assisted_root(tmp_path)
    hard_link = tmp_path / "export-alias.json"
    os.link(assistant, hard_link)
    manifest_before = (root / MANIFEST_NAME).read_bytes()

    result = _run_cli("--output", str(hard_link), "--assistant", str(assistant))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "collision" in result.stderr
    assert (root / MANIFEST_NAME).read_bytes() == manifest_before
    assert not (root / SEALED_NAME).exists()


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "datasets.v2.assisted_labels", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
