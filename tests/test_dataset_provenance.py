from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any

import pytest
from fixtures import load_sources, load_test_provenance, load_test_records

from benchmark.canonical import canonical_json
from datasets.v2.validate import assign_splits, validate_dataset

PUBLIC_SUITE_IDS = {"agnews", "emotion", "banking77_12", "mnli", "sst5"}


def fixture_records() -> list[dict[str, Any]]:
    return load_test_records()


def public_registry(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
    return registry


def assigned_fixture() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = assign_splits(fixture_records())
    return records, public_registry(records)


def test_split_counts_are_exact_disjoint_and_deterministic() -> None:
    records = fixture_records()
    original = copy.deepcopy(records)

    first = assign_splits(records)
    second = assign_splits(records)

    assert records == original
    assert first == second
    assert [record["case_id"] for record in first] == [record["case_id"] for record in records]
    for before, after in zip(records, first, strict=True):
        assert list(after) == list(before)
        assert list(after["questions"]) == list(before["questions"])
        assert list(after["expected"]) == list(before["expected"])

    expected_calibration = {
        "triage": 12,
        "guardrails": 12,
        "moderation": 12,
        "agnews": 40,
        "emotion": 48,
        "banking77_12": 24,
        "mnli": 30,
        "sst5": 30,
        "multilingual_intent": 30,
    }
    calibration_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    for suite_id, count in expected_calibration.items():
        suite_records = [record for record in first if record["suite_id"] == suite_id]
        assert Counter(record["split"] for record in suite_records) == {
            "calibration": count,
            "evaluation": len(suite_records) - count,
        }
        calibration_ids.update(
            record["case_id"] for record in suite_records if record["split"] == "calibration"
        )
        evaluation_ids.update(
            record["case_id"] for record in suite_records if record["split"] == "evaluation"
        )
    assert calibration_ids.isdisjoint(evaluation_ids)
    assert len(calibration_ids) == 238
    assert len(evaluation_ids) == 952


def test_split_assignment_balances_primary_strata() -> None:
    records = assign_splits(load_test_records())
    triage = [
        record
        for record in records
        if record["suite_id"] == "triage" and record["split"] == "calibration"
    ]
    moderation = [
        record
        for record in records
        if record["suite_id"] == "moderation" and record["split"] == "calibration"
    ]

    assert sorted(Counter(record["expected"]["intent"] for record in triage).values()) == [
        2,
        2,
        2,
        3,
        3,
    ]
    assert Counter(record["expected"]["toxic"] for record in moderation) == {0: 6, 1: 6}


def test_split_primary_strata_are_defined_for_every_suite() -> None:
    records = assign_splits(load_test_records())
    fields = {
        "triage": "intent",
        "guardrails": "jailbreak",
        "moderation": "toxic",
        "agnews": "topic",
        "emotion": "emotion",
        "banking77_12": "intent",
        "mnli": "relation",
        "sst5": "sentiment",
    }
    for suite_id, field in fields.items():
        assert all(
            field in record["expected"] for record in records if record["suite_id"] == suite_id
        )
    assert all(
        "language" in record["state"] and "intent" in record["expected"]
        for record in records
        if record["suite_id"] == "multilingual_intent"
    )


def test_split_assignment_preserves_multilingual_language_intent_strata() -> None:
    records = assign_splits(fixture_records())
    multilingual = [record for record in records if record["suite_id"] == "multilingual_intent"]
    counts = Counter(
        (record["state"]["language"], record["expected"]["intent"])
        for record in multilingual
        if record["split"] == "calibration"
    )
    assert len(counts) == 30
    assert set(counts.values()) == {1}


def test_public_case_requires_resolved_source_revision_from_external_registry() -> None:
    records, registry = assigned_fixture()
    registry["agnews-0000"]["revision"] = "main"

    with pytest.raises(ValueError, match="resolved revision"):
        validate_dataset(records, load_sources(), registry=registry)


@pytest.mark.parametrize(
    "revision",
    [
        "main",
        "latest",
        "HEAD",
        "refs/heads/main",
        "deadbeef",
        "A" * 40,
        "g" * 40,
        "a" * 39,
        "a" * 41,
    ],
)
def test_public_provenance_rejects_mutable_or_abbreviated_revisions(revision: str) -> None:
    records, registry = assigned_fixture()
    registry["agnews-0000"]["revision"] = revision

    with pytest.raises(ValueError, match="resolved revision"):
        validate_dataset(records, load_sources(), registry=registry)


@pytest.mark.parametrize("revision", ["HEAD", "deadbeef", "A" * 40])
def test_public_source_metadata_rejects_mutable_or_abbreviated_revisions(revision: str) -> None:
    records, registry = assigned_fixture()
    sources = load_sources()
    sources["agnews"] = {**sources["agnews"], "revision": revision}

    with pytest.raises(ValueError, match="resolved revision"):
        validate_dataset(records, sources, registry=registry)


def test_missing_public_provenance_field_fails() -> None:
    records, registry = assigned_fixture()
    del registry["emotion-0000"]["citation"]

    with pytest.raises(ValueError, match="citation"):
        validate_dataset(records, load_sources(), registry=registry)


def test_public_source_metadata_must_match_external_registry() -> None:
    records, registry = assigned_fixture()
    sources = load_sources()
    sources["mnli"] = {**sources["mnli"], "revision": "different-revision"}

    with pytest.raises(ValueError, match="revision"):
        validate_dataset(records, sources, registry=registry)


def test_checksum_mismatch_fails() -> None:
    records, registry = assigned_fixture()
    registry["sst5-0000"]["checksum"] = "0" * 64

    with pytest.raises(ValueError, match="checksum"):
        validate_dataset(records, load_sources(), registry=registry)


def test_curated_sidecar_revisions_remain_legacy_values() -> None:
    records, registry = assigned_fixture()
    sidecar_sources = {
        **load_sources(),
        "guardrails": {
            "canonical": "curated-guardrails",
            "wrapper": "datasets/cases.py",
            "revision": "legacy-curated",
            "split": "curated",
            "license": "not_declared",
            "citation": "repository legacy curated cases",
        },
    }

    summary = validate_dataset(records, {"records": registry, "sources": sidecar_sources})

    assert summary["total_cases"] == 1190
    assert summary["provenance"]["public_cases"] == 860


def test_two_argument_validation_accepts_a_sidecar_payload() -> None:
    records, registry = assigned_fixture()
    sidecar_payload = {"records": registry, "sources": load_sources()}

    summary = validate_dataset(records, sidecar_payload)

    assert summary["calibration_cases"] == 238
    assert summary["provenance"]["public_cases"] == 860


def test_missing_row_level_registry_fails_for_public_cases() -> None:
    records = assign_splits(fixture_records())

    with pytest.raises(ValueError, match="provenance|registry"):
        validate_dataset(records, load_sources(), registry={"records": {}})


def test_public_record_provenance_id_must_match_source_id() -> None:
    records, registry = assigned_fixture()
    records[180]["provenance_id"] = "wrong-source"

    with pytest.raises(ValueError, match="provenance ID"):
        validate_dataset(records, load_sources(), registry=registry)


def test_valid_fixture_summary_is_json_serializable() -> None:
    records, registry = assigned_fixture()

    summary = validate_dataset(records, load_sources(), registry=registry)

    assert summary["calibration_cases"] == 238
    assert summary["evaluation_cases"] == 952
    assert summary["calibration_decisions"] == 310
    assert summary["evaluation_decisions"] == 1240
    assert summary["total_cases"] == 1190
    assert summary["total_decisions"] == 1550
    assert summary["provenance"]["checksums_verified"] == 860
    assert isinstance(summary["near_duplicate_clusters"], list)
    assert summary["digest"]
    json.dumps(summary, ensure_ascii=False, sort_keys=True)
