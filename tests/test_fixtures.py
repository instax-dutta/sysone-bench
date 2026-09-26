from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from fixtures import (
    CALIBRATION_CASES,
    SUITE_COUNTS,
    SUITE_SOURCE_IDS,
    fixture_checksum,
    load_sources,
    load_test_provenance,
    load_test_records,
    seal_test_manifest,
    write_test_packet,
)

from benchmark.manifest import validate_manifest


def test_fixture_inventory_matches_v2_manifest_contract() -> None:
    records = load_test_records()
    counts = Counter(record["suite_id"] for record in records)

    assert counts == SUITE_COUNTS
    assert len(records) == 1190
    assert sum(len(record["expected"]) for record in records) == 1550
    assert sum(record["split"] == "calibration" for record in records) == 238
    assert sum(record["split"] == "evaluation" for record in records) == 952
    assert (
        sum(len(record["expected"]) for record in records if record["split"] == "calibration")
        == 310
    )
    assert (
        sum(len(record["expected"]) for record in records if record["split"] == "evaluation")
        == 1240
    )

    summary = validate_manifest(records, SUITE_COUNTS)
    assert summary.total_cases == 1190
    assert summary.total_decisions == 1550
    assert all("provenance" not in record for record in records)
    assert all(
        record["provenance_id"] == SUITE_SOURCE_IDS[record["suite_id"]] for record in records
    )


def test_fixture_score_labels_are_numeric_levels() -> None:
    records = load_test_records()
    score_records = [record for record in records if record["questions"][0]["type"] == "score"]

    assert score_records
    assert all(
        isinstance(record["expected"]["sentiment"], (int, float))
        and not isinstance(record["expected"]["sentiment"], bool)
        for record in score_records
    )

    malformed = deepcopy(next(record for record in score_records if record["suite_id"] == "sst5"))
    malformed["expected"]["sentiment"] = "positive"
    malformed["case_id"] = "malformed-score"
    malformed["order_index"] = 150

    with pytest.raises(ValueError, match="expected"):
        validate_manifest([malformed], {"sst5": 1})


def test_fixture_splits_follow_approved_per_suite_counts() -> None:
    records = load_test_records()

    for suite_id, calibration_count in CALIBRATION_CASES.items():
        suite_records = [record for record in records if record["suite_id"] == suite_id]
        assert (
            sum(record["split"] == "calibration" for record in suite_records) == calibration_count
        )
        assert [record["order_index"] for record in suite_records] == list(
            range(SUITE_COUNTS[suite_id])
        )


def test_fixture_sources_use_registry_ids_and_bank_mapping() -> None:
    sources = load_sources()

    assert set(sources) == {"agnews", "emotion", "banking77", "mnli", "sst5"}
    assert SUITE_SOURCE_IDS["banking77_12"] == "banking77"
    assert sources["banking77"]["canonical"] == "PolyAI/banking77"
    assert sources["banking77"]["wrapper"] == "mteb/banking77"
    assert load_sources() == sources


def test_fixture_provenance_keeps_deterministic_checksum_metadata() -> None:
    checksum = fixture_checksum("agnews", 7)
    provenance = load_test_provenance("agnews", 7)

    assert checksum == provenance["checksum"]
    assert provenance["source_id"] == "agnews"
    assert provenance["row_id"] == 7
    assert provenance["canonical"] == "fancyzhx/ag_news"


def test_test_packet_is_synthetic_and_label_free(tmp_path: Path) -> None:
    path = write_test_packet(tmp_path)
    text = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "expected" not in text
    assert "source_label" not in text
    assert "Laya" not in text
    assert "Jev" not in text


def test_fixture_records_and_seal_digests_are_deterministic(tmp_path: Path) -> None:
    assert load_test_records() == load_test_records()
    assert seal_test_manifest(tmp_path, ["a", "b"]) != seal_test_manifest(tmp_path, ["b", "a"])
