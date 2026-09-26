from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, cast

import pytest

from datasets.v2.sources import (
    SOURCES,
    SUITE_SOURCE_IDS,
    derive_seed,
    sample_rows,
    source_record,
)

LOCK_PATH = Path(__file__).parents[1] / "datasets" / "v2" / "sources.lock.json"

EXPECTED_SOURCES = {
    "agnews": {
        "canonical": "fancyzhx/ag_news",
        "wrapper": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "split": "test",
        "license": "unknown",
        "citation": "https://huggingface.co/datasets/fancyzhx/ag_news",
    },
    "emotion": {
        "canonical": "dair-ai/emotion",
        "wrapper": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "split": "test",
        "license": "other",
        "citation": "https://huggingface.co/datasets/dair-ai/emotion",
    },
    "banking77": {
        "canonical": "PolyAI/banking77",
        "wrapper": "mteb/banking77",
        "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2",
        "split": "test",
        "license": "mit",
        "citation": "https://huggingface.co/datasets/PolyAI/banking77",
    },
    "mnli": {
        "canonical": "nyu-mll/glue",
        "wrapper": "nyu-mll/glue",
        "config": "mnli",
        "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        "split": "validation_matched",
        "license": "other",
        "citation": "https://huggingface.co/datasets/nyu-mll/glue",
    },
    "sst5": {
        "canonical": "SetFit/sst5",
        "wrapper": "SetFit/sst5",
        "revision": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
        "split": "test",
        "license": "not_declared",
        "citation": "https://huggingface.co/datasets/SetFit/sst5",
    },
}


def test_suite_seed_is_stable_and_distinct() -> None:
    assert derive_seed("agnews") == derive_seed("agnews")
    assert derive_seed("agnews") != derive_seed("emotion")
    assert derive_seed("agnews", base_seed=7) != derive_seed("agnews", base_seed=8)


def test_suite_seed_uses_sha256_first_eight_bytes() -> None:
    expected_digest = hashlib.sha256(b"42:agnews").digest()

    assert derive_seed("agnews") == int.from_bytes(expected_digest[:8], "big")


def test_sampling_is_deterministic_and_preserves_rows() -> None:
    rows = [{"id": index, "payload": {"value": index}} for index in range(20)]
    original = copy.deepcopy(rows)

    first = sample_rows(rows, 5, "agnews")
    second = sample_rows(rows, 5, "agnews")

    assert first == second
    assert len(first) == 5
    assert len({row["id"] for row in first}) == 5
    assert all(row is rows[row["id"]] for row in first)
    assert rows == original


def test_sampling_uses_a_local_rng() -> None:
    random_state = random.getstate()
    rows = [{"id": index} for index in range(10)]

    sample_rows(rows, 3, "agnews")

    assert random.getstate() == random_state


@pytest.mark.parametrize(
    ("count", "error_type"),
    [(-1, ValueError), (11, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_sampling_rejects_invalid_counts(count: object, error_type: type[Exception]) -> None:
    rows = [{"id": index} for index in range(10)]

    with pytest.raises(error_type, match="sample count"):
        sample_rows(rows, cast(int, count), "agnews")


def test_registry_contains_exact_source_metadata() -> None:
    assert SOURCES == EXPECTED_SOURCES

    for source_id, expected in EXPECTED_SOURCES.items():
        assert source_record(source_id) == expected


def test_source_record_returns_an_isolated_copy() -> None:
    record = cast(dict[str, Any], source_record("mnli"))
    record["canonical"] = "mutated"
    record["config"] = "mutated"

    assert source_record("mnli") == EXPECTED_SOURCES["mnli"]


def test_unknown_source_ids_are_rejected_deterministically() -> None:
    with pytest.raises(KeyError, match="unknown source ID: missing"):
        source_record("missing")

    with pytest.raises(KeyError, match="unknown source ID: missing"):
        source_record("missing")


def test_lock_matches_registry_and_contains_resolved_metadata() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock == SOURCES
    for source_id, record in lock.items():
        assert set(record) >= {
            "canonical",
            "wrapper",
            "revision",
            "split",
            "license",
            "citation",
        }
        assert record["revision"] != "main"
        assert record["revision"]
        assert record["license"]
        assert record["citation"]
        if source_id == "mnli":
            assert record["config"] == "mnli"
        else:
            assert "config" not in record


def test_lock_preserves_missing_license_metadata_and_raw_sources() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    assert lock["banking77"]["canonical"] == "PolyAI/banking77"
    assert lock["banking77"]["wrapper"] == "mteb/banking77"
    assert lock["sst5"]["license"] == "not_declared"
    assert "main" not in json.dumps(lock)


def test_banking_suite_maps_to_bankings_source_id() -> None:
    assert SUITE_SOURCE_IDS == {
        "agnews": "agnews",
        "emotion": "emotion",
        "banking77_12": "banking77",
        "mnli": "mnli",
        "sst5": "sst5",
    }
