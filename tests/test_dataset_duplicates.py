from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pytest
from fixtures import load_sources, load_test_provenance, load_test_records

from benchmark.canonical import canonical_json
from datasets.v2.validate import assign_splits, normalized_state, validate_dataset

PUBLIC_SUITE_IDS = {"agnews", "emotion", "banking77_12", "mnli", "sst5"}
LANGUAGES = ("hi", "es", "fr", "de", "ar")


def fixture_records() -> list[dict[str, Any]]:
    records = load_test_records()
    for record in records:
        if record["suite_id"] == "multilingual_intent":
            record["state"]["language"] = LANGUAGES[record["order_index"] // 30]
    return records


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


def test_normalized_state_is_compact_sorted_and_unicode_preserving() -> None:
    first = {"text": "café", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "text": "café"}

    normalized = normalized_state(first)

    assert normalized == normalized_state(second)
    assert normalized == '{"nested":{"a":1,"b":2},"text":"café"}'
    assert "\\u" not in normalized
    assert normalized_state({"text": " a "}) != normalized_state({"text": "a"})


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), {"bad": {1, 2}}, {"bad": object()}],
)
def test_normalized_state_rejects_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalized_state(value)


def test_exact_normalized_state_duplicates_are_rejected() -> None:
    records = assign_splits(fixture_records())
    records[1]["state"] = copy.deepcopy(records[0]["state"])
    registry = public_registry(records)

    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(records, load_sources(), registry=registry)


def test_near_duplicates_are_reported_without_deleting_cases() -> None:
    records = assign_splits(fixture_records())
    first_state = records[0]["state"]["text"]
    records[2]["state"]["text"] = f"{first_state}!"
    registry = public_registry(records)

    first = validate_dataset(records, load_sources(), registry=registry)
    second = validate_dataset(records, load_sources(), registry=registry)

    assert len(first["near_duplicate_clusters"]) >= 1
    assert first["near_duplicate_clusters"] == second["near_duplicate_clusters"]
    assert any(
        {records[0]["case_id"], records[2]["case_id"]}.issubset(set(cluster["members"]))
        for cluster in first["near_duplicate_clusters"]
    )
    assert len(records) == 1190


def test_duplicate_gate_does_not_mutate_records() -> None:
    records = assign_splits(fixture_records())
    original = copy.deepcopy(records)
    registry = public_registry(records)

    with pytest.raises(ValueError):
        records[1]["state"] = copy.deepcopy(records[0]["state"])
        validate_dataset(records, load_sources(), registry=registry)

    records[1]["state"] = original[1]["state"]
    assert records == original


def test_summary_duplicate_diagnostics_are_json_serializable() -> None:
    records = assign_splits(fixture_records())
    registry = public_registry(records)
    summary = validate_dataset(records, load_sources(), registry=registry)

    json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False)
    assert isinstance(summary["near_duplicate_clusters"], list)
    assert all(
        isinstance(cluster["cluster_id"], str)
        and isinstance(cluster["members"], list)
        and all(isinstance(member, str) for member in cluster["members"])
        for cluster in summary["near_duplicate_clusters"]
    )


def test_normalized_state_accepts_nested_json_sequences() -> None:
    value: Mapping[str, Any] = {"items": (1, 2)}

    assert normalized_state(value) == '{"items":[1,2]}'
