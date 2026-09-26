from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Self

import pytest
from fixtures import load_sources, load_test_provenance, load_test_records

from benchmark.canonical import canonical_json
from benchmark.manifest import manifest_digest, validate_manifest
from datasets.v2 import labeling
from datasets.v2.labeling import merge_adjudication, seal_manifest

PUBLIC_SUITE_IDS = frozenset({"agnews", "emotion", "banking77_12", "mnli", "sst5"})
ADJUDICATED_LABEL_PROVENANCE = "adjudicated-v2"
SEAL_LOCK_NAME = ".manifest.seal.lock"


def _merge_records() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": 2,
            "dataset_version": "2.0.0",
            "suite_id": "multilingual_intent",
            "case_id": "merge-case",
            "order_index": 0,
            "split": "evaluation",
            "state": {"text": "synthetic merge state", "language": "hi"},
            "questions": [
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
            ],
            "expected": {"intent": "alpha", "flag": 0, "intensity": 1},
            "provenance_id": "synthetic-source",
            "label_provenance": "pending-human-review",
        }
    ]


def _reviewer_export(code: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "reviewer_code": code,
        "labels": {str(record["case_id"]): copy.deepcopy(record["expected"]) for record in records},
    }


def _adjudication(
    records: list[dict[str, Any]],
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
) -> dict[str, Any]:
    case_id = records[0]["case_id"]
    a_value = reviewer_a["labels"][case_id]["intent"]
    b_value = reviewer_b["labels"][case_id]["intent"]
    if a_value == b_value:
        return {
            "schema_version": 2,
            "adjudication_code": "ADJUDICATOR",
            "labels": {},
            "reviewer_labels": {"A": {}, "B": {}},
            "reasons": {},
        }
    return {
        "schema_version": 2,
        "adjudication_code": "ADJUDICATOR",
        "labels": {case_id: {"intent": a_value}},
        "reviewer_labels": {
            "A": {case_id: {"intent": a_value}},
            "B": {case_id: {"intent": b_value}},
        },
        "reasons": {case_id: {"intent": "Synthetic reason reviewed the question wording."}},
    }


def _merge_exports(
    disagree: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = _merge_records()
    reviewer_a = _reviewer_export("A", records)
    reviewer_b = _reviewer_export("B", records)
    if disagree:
        reviewer_b["labels"]["merge-case"]["intent"] = "beta"
    return records, reviewer_a, reviewer_b, _adjudication(records, reviewer_a, reviewer_b)


@pytest.mark.parametrize("reviewer", ["A", "B"])
@pytest.mark.parametrize("coverage", ["missing", "extra"])
def test_merge_rejects_incomplete_or_extra_reviewer_cases(reviewer: str, coverage: str) -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    payload = reviewer_a if reviewer == "A" else reviewer_b
    if coverage == "missing":
        del payload["labels"]["merge-case"]
    else:
        payload["labels"]["extra-case"] = {"intent": "alpha"}

    with pytest.raises(ValueError, match="case IDs differ"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


def test_merge_rejects_mismatched_reviewer_codes() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    reviewer_a["reviewer_code"] = "B"
    reviewer_b["reviewer_code"] = "A"

    with pytest.raises(ValueError, match="reviewer codes A and B"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


@pytest.mark.parametrize("container", ["labels", "reviewer_labels", "reasons"])
@pytest.mark.parametrize("coverage", ["missing", "extra"])
def test_merge_requires_exact_disagreement_coverage(container: str, coverage: str) -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    if container == "labels":
        target = adjudication["labels"]["merge-case"]
    elif container == "reviewer_labels":
        target = adjudication["reviewer_labels"]["A"]["merge-case"]
    else:
        target = adjudication["reasons"]["merge-case"]
    if coverage == "missing":
        del target["intent"]
    else:
        target["flag"] = 0 if container != "reasons" else "Synthetic extra reason."

    with pytest.raises(ValueError, match="disagreement"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


def test_merge_rejects_preserved_reviewer_label_mismatch() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    adjudication["reviewer_labels"]["B"]["merge-case"]["intent"] = "alpha"

    with pytest.raises(ValueError, match="reviewer B label"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


@pytest.mark.parametrize(
    ("qid", "value", "message"),
    [
        ("intent", "gamma", "choice"),
        ("flag", True, "noul"),
        ("intensity", 3, "score"),
    ],
)
def test_merge_rejects_invalid_final_labels(qid: str, value: Any, message: str) -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    if qid != "intent":
        reviewer_value = 1 if qid == "flag" else 2
        reviewer_b["labels"]["merge-case"][qid] = reviewer_value
        adjudication["reviewer_labels"]["A"]["merge-case"][qid] = reviewer_a["labels"][
            "merge-case"
        ][qid]
        adjudication["reviewer_labels"]["B"]["merge-case"][qid] = reviewer_value
        adjudication["reasons"]["merge-case"][qid] = "Synthetic target disagreement."
    adjudication["labels"]["merge-case"][qid] = value

    with pytest.raises(ValueError, match=message):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


@pytest.mark.parametrize("reason", ["", "   ", None, 3])
def test_merge_rejects_empty_or_nonstring_final_reasons(reason: object) -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    adjudication["reasons"]["merge-case"]["intent"] = reason

    with pytest.raises(ValueError, match="reason"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


def test_merge_rejects_untouched_adjudication_reason() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    adjudication["reasons"]["merge-case"]["intent"] = ""

    with pytest.raises(ValueError, match="nonempty string"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


def test_merge_rejects_core_invalid_records_without_mutating_inputs() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports(disagree=False)
    records[0]["unexpected"] = "core-invalid"
    original = copy.deepcopy(records)

    with pytest.raises(ValueError, match="additional record field"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)

    assert records == original


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1},
        {"adjudication_code": ""},
        {"adjudication_code": 3},
    ],
)
def test_merge_rejects_invalid_adjudication_envelope(payload: dict[str, Any]) -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    adjudication.update(payload)

    with pytest.raises(ValueError, match="adjudication"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


def test_merge_rejects_extra_adjudication_fields() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    adjudication["model_output"] = "forbidden"

    with pytest.raises(ValueError, match="adjudication fields"):
        merge_adjudication(records, reviewer_a, reviewer_b, adjudication)


def test_merge_replaces_labels_and_returns_core_valid_records_without_mutating_inputs() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports()
    originals = tuple(
        copy.deepcopy(value) for value in (records, reviewer_a, reviewer_b, adjudication)
    )

    merged = merge_adjudication(records, reviewer_a, reviewer_b, adjudication)

    assert (records, reviewer_a, reviewer_b, adjudication) == originals
    assert merged == [
        {
            **records[0],
            "expected": {"intent": "alpha", "flag": 0, "intensity": 1},
            "label_provenance": ADJUDICATED_LABEL_PROVENANCE,
        }
    ]
    assert merged[0] is not records[0]
    assert merged[0]["state"] is not records[0]["state"]
    assert merged[0]["questions"] is not records[0]["questions"]
    validate_manifest(merged, {"multilingual_intent": 1})


def test_merge_accepts_empty_adjudication_when_reviewers_agree() -> None:
    records, reviewer_a, reviewer_b, adjudication = _merge_exports(disagree=False)

    merged = merge_adjudication(records, reviewer_a, reviewer_b, adjudication)

    assert merged[0]["expected"] == {"intent": "alpha", "flag": 0, "intensity": 1}
    assert merged[0]["label_provenance"] == ADJUDICATED_LABEL_PROVENANCE


def _synthetic_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    registry: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["suite_id"] not in PUBLIC_SUITE_IDS:
            continue
        entry = copy.deepcopy(load_test_provenance(record["suite_id"], record["order_index"]))
        entry["provenance_id"] = entry["source_id"]
        source_row = {"state": copy.deepcopy(record["state"]), "label": entry["source_label"]}
        entry["source_row"] = source_row
        entry["checksum"] = hashlib.sha256(canonical_json(source_row)).hexdigest()
        registry[record["case_id"]] = entry
    return {
        "schema_version": 1,
        "dataset_version": "2.0.0",
        "provisional": True,
        "records": registry,
        "sources": load_sources(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(canonical_json(record).decode("utf-8") + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def _seed_seal_root(root: Path, reverse_first_questions: bool = False) -> list[dict[str, Any]]:
    records = load_test_records()
    if reverse_first_questions:
        records[0]["questions"] = list(reversed(records[0]["questions"]))
    reviewer_a = _reviewer_export("A", records)
    reviewer_b = _reviewer_export("B", records)
    reviewer_b["labels"]["triage-0000"]["intent"] = "technical_help"
    adjudication = {
        "schema_version": 2,
        "adjudication_code": "SYNTHETIC_ADJUDICATOR",
        "labels": {"triage-0000": {"intent": "information"}},
        "reviewer_labels": {
            "A": {"triage-0000": {"intent": "refund"}},
            "B": {"triage-0000": {"intent": "technical_help"}},
        },
        "reasons": {
            "triage-0000": {
                "intent": "Synthetic adjudication selected the explicit refund request."
            }
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    _write_manifest(root / "manifest.jsonl", records)
    _write_json(root / "provenance.json", _synthetic_provenance(records))
    _write_json(root / "labels" / "reviewer_a.json", reviewer_a)
    _write_json(root / "labels" / "reviewer_b.json", reviewer_b)
    _write_json(root / "labels" / "adjudication.json", adjudication)
    return records


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("manifest.jsonl"),
        Path("provenance.json"),
        Path("labels/reviewer_a.json"),
        Path("labels/reviewer_b.json"),
        Path("labels/adjudication.json"),
    ],
)
def test_seal_requires_every_input_artifact(tmp_path: Path, relative_path: Path) -> None:
    for required in (
        Path("manifest.jsonl"),
        Path("provenance.json"),
        Path("labels/reviewer_a.json"),
        Path("labels/reviewer_b.json"),
        Path("labels/adjudication.json"),
    ):
        path = tmp_path / required
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / relative_path).unlink()

    with pytest.raises(ValueError, match=relative_path.name):
        seal_manifest(tmp_path)


def test_seal_writes_final_manifest_provenance_and_exact_checksum_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "seal"
    records = _seed_seal_root(root)
    input_manifest = (root / "manifest.jsonl").read_bytes()
    input_provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    lock_path = root / SEAL_LOCK_NAME
    lock_observations: list[bool] = []
    real_merge = labeling.merge_adjudication

    def merge_under_lock(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        lock_observations.append(lock_path.exists())
        return real_merge(*args, **kwargs)

    monkeypatch.setattr(labeling, "merge_adjudication", merge_under_lock)

    digest = seal_manifest(root)

    assert lock_observations == [True]
    assert not lock_path.exists()
    final_records = _read_manifest(root / "manifest.jsonl")
    assert final_records != records
    assert len(final_records) == 1190
    assert final_records[0]["expected"]["intent"] == "information"
    assert all(
        record["label_provenance"] == ADJUDICATED_LABEL_PROVENANCE for record in final_records
    )
    assert [record["case_id"] for record in final_records] == [
        record["case_id"] for record in records
    ]
    assert digest == manifest_digest(final_records)
    assert (root / "manifest.sha256").read_text(encoding="utf-8") == (f"{digest}  manifest.jsonl\n")
    finalized = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    assert finalized["provisional"] is False
    assert finalized["records"] == input_provenance["records"]
    assert finalized["sources"] == input_provenance["sources"]
    assert input_manifest != (root / "manifest.jsonl").read_bytes()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_seal_digest_is_stable_for_identical_synthetic_exports(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _seed_seal_root(first_root)
    _seed_seal_root(second_root)

    first = seal_manifest(first_root)
    second = seal_manifest(second_root)

    assert first == second
    assert (first_root / "manifest.jsonl").read_bytes() == (
        second_root / "manifest.jsonl"
    ).read_bytes()
    assert (first_root / "manifest.sha256").read_bytes() == (
        second_root / "manifest.sha256"
    ).read_bytes()


def test_seal_digest_changes_when_question_order_changes(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _seed_seal_root(first_root)
    _seed_seal_root(second_root, reverse_first_questions=True)

    first = seal_manifest(first_root)
    second = seal_manifest(second_root)

    assert first != second


def test_seal_validation_failure_preserves_draft_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "invalid"
    _seed_seal_root(root)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()
    reviewer_b_path = root / "labels" / "reviewer_b.json"
    reviewer_b = json.loads(reviewer_b_path.read_text(encoding="utf-8"))
    reviewer_b["labels"]["triage-0000"]["refund_requested"] = True
    _write_json(reviewer_b_path, reviewer_b)

    with pytest.raises(ValueError, match="noul"):
        seal_manifest(root)

    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))
    assert not (root / SEAL_LOCK_NAME).exists()


def test_seal_runs_complete_provenance_validation_and_preserves_drafts(tmp_path: Path) -> None:
    root = tmp_path / "provenance-failure"
    _seed_seal_root(root)
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["records"]["agnews-0000"]
    _write_json(provenance_path, provenance)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = provenance_path.read_bytes()

    with pytest.raises(ValueError, match="provenance|registry"):
        seal_manifest(root)

    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert provenance_path.read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not (root / SEAL_LOCK_NAME).exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_seal_runs_split_total_validation_and_preserves_drafts(tmp_path: Path) -> None:
    root = tmp_path / "split-total-failure"
    records = _seed_seal_root(root)
    records[0]["split"] = "evaluation"
    _write_manifest(root / "manifest.jsonl", records)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()

    with pytest.raises(ValueError, match="calibration count|split case totals"):
        seal_manifest(root)

    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not (root / SEAL_LOCK_NAME).exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_seal_runs_duplicate_gate_and_preserves_drafts(tmp_path: Path) -> None:
    root = tmp_path / "duplicate-state-failure"
    records = _seed_seal_root(root)
    records[1]["state"] = copy.deepcopy(records[0]["state"])
    _write_manifest(root / "manifest.jsonl", records)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()

    with pytest.raises(ValueError, match="normalized state duplicate"):
        seal_manifest(root)

    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not (root / SEAL_LOCK_NAME).exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_seal_runs_approved_suite_count_gate_for_extra_core_valid_case(tmp_path: Path) -> None:
    root = tmp_path / "extra-suite-count-failure"
    records = _seed_seal_root(root)
    extra = copy.deepcopy(records[0])
    extra.update(
        {
            "suite_id": "synthetic_extra",
            "case_id": "synthetic-extra-0000",
            "order_index": 0,
            "split": "evaluation",
            "state": {"text": "synthetic extra suite"},
            "provenance_id": "synthetic-extra-source",
        }
    )
    records.append(extra)
    expected_counts: dict[str, int] = {}
    for record in records:
        suite_id = str(record["suite_id"])
        expected_counts[suite_id] = expected_counts.get(suite_id, 0) + 1
    validate_manifest(records, expected_counts)
    _write_manifest(root / "manifest.jsonl", records)
    for reviewer_name in ("reviewer_a.json", "reviewer_b.json"):
        reviewer_path = root / "labels" / reviewer_name
        reviewer = json.loads(reviewer_path.read_text(encoding="utf-8"))
        reviewer["labels"][extra["case_id"]] = copy.deepcopy(extra["expected"])
        _write_json(reviewer_path, reviewer)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()

    with pytest.raises(ValueError, match="suite set|approved inventory|case count"):
        seal_manifest(root)

    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not (root / SEAL_LOCK_NAME).exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


@pytest.mark.parametrize("value", [0, "false", None, 1, []], ids=repr)
def test_seal_requires_provisional_to_be_exact_boolean_true(tmp_path: Path, value: object) -> None:
    root = tmp_path / f"provisional-{type(value).__name__}-{value!s}"
    _seed_seal_root(root)
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["provisional"] = value
    _write_json(provenance_path, provenance)
    provenance_before = provenance_path.read_bytes()

    with pytest.raises(ValueError, match="provisional.*boolean true"):
        seal_manifest(root)

    assert provenance_path.read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not (root / SEAL_LOCK_NAME).exists()


def test_seal_requires_provisional_key(tmp_path: Path) -> None:
    root = tmp_path / "provisional-missing"
    _seed_seal_root(root)
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["provisional"]
    _write_json(provenance_path, provenance)
    provenance_before = provenance_path.read_bytes()

    with pytest.raises(ValueError, match="provisional.*true"):
        seal_manifest(root)

    assert provenance_path.read_bytes() == provenance_before
    assert not (root / SEAL_LOCK_NAME).exists()


def test_seal_rolls_back_atomic_replacements_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "replace-failure"
    _seed_seal_root(root)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()
    real_replace = os.replace
    failed = False

    def fail_provenance_replace(source: str, destination: str) -> None:
        nonlocal failed
        if Path(destination).name == "provenance.json" and not failed:
            failed = True
            raise OSError("synthetic provenance replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(labeling.os, "replace", fail_provenance_replace)

    with pytest.raises(OSError, match="synthetic provenance replacement failure"):
        seal_manifest(root)

    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert not (root / "manifest.sha256").exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))
    assert not (root / SEAL_LOCK_NAME).exists()


def test_seal_checksum_collision_after_staging_blocks_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "staging-collision"
    _seed_seal_root(root)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()
    lock_path = root / SEAL_LOCK_NAME
    checksum = root / "manifest.sha256"
    collision_bytes = b"concurrent  manifest.jsonl\n"
    lock_observations: list[bool] = []
    real_stage = labeling._stage_bytes

    def stage_then_collide(path: Path, content: bytes) -> Path:
        staged = real_stage(path, content)
        if path.name == "provenance.json":
            lock_observations.append(lock_path.exists())
            checksum.write_bytes(collision_bytes)
        return staged

    def unexpected_replace(source: str, destination: str) -> None:
        del source, destination
        raise AssertionError("draft replacement occurred before checksum collision")

    monkeypatch.setattr(labeling, "_stage_bytes", stage_then_collide)
    monkeypatch.setattr(labeling.os, "replace", unexpected_replace)

    with pytest.raises(FileExistsError, match="sealed"):
        seal_manifest(root)

    assert lock_observations == [True]
    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert checksum.read_bytes() == collision_bytes
    assert not lock_path.exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_seal_checksum_write_failure_removes_owned_resources_and_preserves_drafts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checksum-write-failure"
    _seed_seal_root(root)
    manifest_before = (root / "manifest.jsonl").read_bytes()
    provenance_before = (root / "provenance.json").read_bytes()
    lock_path = root / SEAL_LOCK_NAME
    checksum = root / "manifest.sha256"
    lock_observations: list[bool] = []
    replacements: list[str] = []
    real_open = Path.open

    class FailingChecksum:
        def __init__(self) -> None:
            self.file: Any = None

        def __enter__(self) -> Self:
            lock_observations.append(lock_path.exists())
            self.file = real_open(checksum, "xb")
            self.file.write(b"partial")
            self.file.flush()
            return self

        def write(self, content: bytes) -> int:
            del content
            raise OSError("synthetic checksum write failure")

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback
            self.file.close()

    def failing_checksum_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if path == checksum:
            return FailingChecksum()
        return real_open(path, mode, *args, **kwargs)

    def unexpected_replace(source: str, destination: str) -> None:
        del source
        replacements.append(destination)
        raise AssertionError("draft replacement occurred before checksum write")

    monkeypatch.setattr(Path, "open", failing_checksum_open)
    monkeypatch.setattr(labeling.os, "replace", unexpected_replace)

    with pytest.raises(OSError, match="synthetic checksum write failure"):
        seal_manifest(root)

    assert lock_observations == [True]
    assert replacements == []
    assert (root / "manifest.jsonl").read_bytes() == manifest_before
    assert (root / "provenance.json").read_bytes() == provenance_before
    assert not checksum.exists()
    assert not lock_path.exists()
    assert not list(root.glob(".manifest.jsonl.*"))
    assert not list(root.glob(".provenance.json.*"))


def test_seal_refuses_concurrent_lock_without_removing_lock_owner_resources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lock-collision"
    _seed_seal_root(root)
    lock = root / SEAL_LOCK_NAME
    lock.write_bytes(b"held by another seal\n")
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="in progress"):
        seal_manifest(root)

    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert lock.read_bytes() == b"held by another seal\n"


def test_seal_refuses_existing_checksum_and_preserves_every_artifact(tmp_path: Path) -> None:
    root = tmp_path / "collision"
    _seed_seal_root(root)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    checksum = root / "manifest.sha256"
    checksum.write_text("occupied  manifest.jsonl\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="sealed"):
        seal_manifest(root)

    assert checksum.read_bytes() == b"occupied  manifest.jsonl\n"
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert after == {**before, "manifest.sha256": b"occupied  manifest.jsonl\n"}


def test_seal_refuses_manifest_marked_final_without_checksum(tmp_path: Path) -> None:
    root = tmp_path / "finalized-without-seal"
    _seed_seal_root(root)
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["provisional"] = False
    _write_json(provenance_path, provenance)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}

    with pytest.raises(FileExistsError, match="sealed manifest"):
        seal_manifest(root)

    assert {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()} == before
    assert not (root / SEAL_LOCK_NAME).exists()
