from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

from benchmark.manifest import manifest_digest, validate_manifest
from datasets.v2.build_manifest import build_records, write_manifest
from datasets.v2.sources import load_sources

SUITE_COUNTS = {
    "triage": 60,
    "guardrails": 60,
    "moderation": 60,
    "agnews": 200,
    "emotion": 240,
    "banking77_12": 120,
    "mnli": 150,
    "sst5": 150,
    "multilingual_intent": 150,
}
SUITE_DECISIONS = {
    "triage": 4,
    "guardrails": 2,
    "moderation": 3,
    "agnews": 1,
    "emotion": 1,
    "banking77_12": 1,
    "mnli": 1,
    "sst5": 1,
    "multilingual_intent": 1,
}
PUBLIC_LABELS = {
    "agnews": (0, 1, 2, 3),
    "emotion": (0, 1, 2, 3, 4, 5),
    "banking77": (
        "card_arrival",
        "card_linking",
        "exchange_rate",
        "lost_or_stolen_card",
        "order_physical_card",
        "pin_blocked",
        "refund_not_showing_up",
        "request_refund",
        "terminate_account",
        "topping_up_by_card",
        "top_up_failed",
        "transaction_charged_twice",
    ),
    "mnli": (0, 1, 2),
    "sst5": (0, 1, 2, 3, 4),
}
PUBLIC_PER_LABEL = {
    "agnews": 50,
    "emotion": 40,
    "banking77": 10,
    "mnli": 50,
    "sst5": 30,
}


def synthetic_sources() -> dict[str, Any]:
    sources = load_sources()
    for source_id, labels in PUBLIC_LABELS.items():
        rows: list[dict[str, Any]] = []
        for label in labels:
            for index in range(PUBLIC_PER_LABEL[source_id]):
                row: dict[str, Any] = {
                    "id": f"{source_id}-{label}-{index}",
                    "text": f"{source_id} synthetic row {label} {index}",
                    "label": label,
                }
                if source_id == "banking77":
                    row["label_text"] = label
                if source_id == "mnli":
                    row["premise"] = f"premise {label} {index}"
                    row["hypothesis"] = f"hypothesis {label} {index}"
                rows.append(row)
        sources[source_id] = {"rows": rows}
    return sources


def records_for(tmp_path: Path) -> list[dict[str, Any]]:
    return build_records(tmp_path, synthetic_sources())


def test_v2_counts_match_the_approved_spec(tmp_path: Path) -> None:
    records = records_for(tmp_path)

    assert Counter(record["suite_id"] for record in records) == SUITE_COUNTS
    assert len(records) == 1190
    assert sum(len(record["expected"]) for record in records) == 1550
    assert sum(record["split"] == "calibration" for record in records) == 238
    assert sum(record["split"] == "evaluation" for record in records) == 952


def test_builder_accepts_an_injected_source_loader_without_network_access(
    tmp_path: Path,
) -> None:
    sources = synthetic_sources()
    agnews_rows = cast(list[dict[str, Any]], sources["agnews"]["rows"])
    calls: list[bool] = []

    def load_agnews() -> list[dict[str, Any]]:
        calls.append(True)
        return agnews_rows

    sources["agnews"] = {"loader": load_agnews}
    records = build_records(tmp_path, sources)

    assert calls == [True]
    assert len(records) == 1190


def test_agnews_zero_based_labels_map_to_all_topics(tmp_path: Path) -> None:
    sources = synthetic_sources()

    records = build_records(tmp_path, sources)

    agnews = [record for record in records if record["suite_id"] == "agnews"]
    assert Counter(record["expected"]["topic"] for record in agnews) == {
        "world": 50,
        "sports": 50,
        "business": 50,
        "scitech": 50,
    }


def test_builder_rejects_unresolved_source_revisions(tmp_path: Path) -> None:
    sources = synthetic_sources()
    sources["agnews"]["revision"] = "main"

    with pytest.raises(ValueError, match="revision"):
        build_records(tmp_path, sources)


def test_triage_guardrails_and_moderation_distributions(tmp_path: Path) -> None:
    records = records_for(tmp_path)
    by_suite = {
        suite_id: [record for record in records if record["suite_id"] == suite_id]
        for suite_id in ("triage", "guardrails", "moderation")
    }

    assert Counter(record["expected"]["intent"] for record in by_suite["triage"]) == {
        "refund": 12,
        "technical_help": 12,
        "billing_question": 12,
        "information": 12,
        "cancellation": 12,
    }
    for question_id in ("refund_requested", "churn_risk", "is_urgent"):
        assert Counter(record["expected"][question_id] for record in by_suite["triage"]) == {
            0: 30,
            1: 30,
        }
    for question_id in ("jailbreak", "prompt_injection"):
        assert Counter(record["expected"][question_id] for record in by_suite["guardrails"]) == {
            0: 30,
            1: 30,
        }
    for question_id in ("toxic", "threat", "spam"):
        assert Counter(record["expected"][question_id] for record in by_suite["moderation"]) == {
            0: 30,
            1: 30,
        }


def test_public_distributions_and_multilingual_grid(tmp_path: Path) -> None:
    records = records_for(tmp_path)

    agnews = [record for record in records if record["suite_id"] == "agnews"]
    assert Counter(record["expected"]["topic"] for record in agnews) == {
        "world": 50,
        "sports": 50,
        "business": 50,
        "scitech": 50,
    }
    emotion = [record for record in records if record["suite_id"] == "emotion"]
    assert Counter(record["expected"]["emotion"] for record in emotion) == {
        "sadness": 40,
        "joy": 40,
        "love": 40,
        "anger": 40,
        "fear": 40,
        "surprise": 40,
    }
    banking = [record for record in records if record["suite_id"] == "banking77_12"]
    assert Counter(record["expected"]["intent"] for record in banking) == {
        intent: 10 for intent in PUBLIC_LABELS["banking77"]
    }
    mnli = [record for record in records if record["suite_id"] == "mnli"]
    assert Counter(record["expected"]["relation"] for record in mnli) == {
        "entailment": 50,
        "neutral": 50,
        "contradiction": 50,
    }
    sst5 = [record for record in records if record["suite_id"] == "sst5"]
    assert Counter(record["expected"]["sentiment"] for record in sst5) == {
        0: 30,
        1: 30,
        2: 30,
        3: 30,
        4: 30,
    }
    multilingual = [record for record in records if record["suite_id"] == "multilingual_intent"]
    assert Counter(record["state"]["language"] for record in multilingual) == {
        language: 30 for language in ("hi", "es", "fr", "de", "ar")
    }
    assert Counter(
        (record["state"]["language"], record["expected"]["intent"]) for record in multilingual
    ) == {
        (language, intent): 5
        for language in ("hi", "es", "fr", "de", "ar")
        for intent in (
            "refund",
            "technical_help",
            "billing_question",
            "information",
            "cancellation",
            "other",
        )
    }


def test_records_are_core_valid_and_keep_ordered_questions_and_ids(tmp_path: Path) -> None:
    records = records_for(tmp_path)

    summary = validate_manifest(records, SUITE_COUNTS)
    assert summary.total_cases == 1190
    assert summary.total_decisions == 1550
    assert all("provenance" not in record for record in records)
    for suite_id, suite_count in SUITE_COUNTS.items():
        suite_records = [record for record in records if record["suite_id"] == suite_id]
        assert [record["case_id"] for record in suite_records] == [
            f"{suite_id}-{index:04d}" for index in range(suite_count)
        ]
        assert [record["order_index"] for record in suite_records] == list(range(suite_count))
        assert all(
            record["label_provenance"].endswith("pending-adjudication") for record in suite_records
        )
    assert [question["qid"] for question in records[0]["questions"]] == [
        "intent",
        "refund_requested",
        "churn_risk",
        "is_urgent",
    ]
    assert list(records[0]["expected"]) == ["intent", "refund_requested", "churn_risk", "is_urgent"]


def test_build_is_deterministic_and_writes_provenance_sidecar(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    sources = synthetic_sources()
    original_sources = copy.deepcopy(sources)

    first = build_records(first_root, sources)
    second = build_records(second_root, synthetic_sources())

    assert first == second
    assert sources == original_sources
    first_sidecar = json.loads((first_root / "provenance.json").read_text(encoding="utf-8"))
    second_sidecar = json.loads((second_root / "provenance.json").read_text(encoding="utf-8"))
    assert first_sidecar == second_sidecar
    assert first_sidecar["provisional"] is True
    entry = first_sidecar["records"]["agnews-0000"]
    assert {
        "source_id",
        "canonical",
        "wrapper",
        "revision",
        "split",
        "row_id",
        "source_label",
        "local_modification",
        "license",
        "citation",
        "checksum",
    } <= set(entry)
    assert entry["revision"] != "main"
    assert entry["source_label"] == entry["source_row"]["label"]
    assert entry["license"] and entry["citation"] and entry["checksum"]
    assert (
        entry["checksum"]
        == hashlib.sha256(
            json.dumps(
                entry["source_row"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def test_write_manifest_is_canonical_atomic_and_digest_stable(tmp_path: Path) -> None:
    records = records_for(tmp_path / "records")
    first_root = tmp_path / "first-output"
    second_root = tmp_path / "second-output"

    first_digest = write_manifest(records, first_root)
    second_digest = write_manifest(copy.deepcopy(records), second_root)

    assert first_digest == second_digest == manifest_digest(records)
    first_bytes = (first_root / "manifest.jsonl").read_bytes()
    assert first_bytes == (second_root / "manifest.jsonl").read_bytes()
    assert first_bytes.endswith(b"\n")
    assert b"\n  " not in first_bytes
    assert first_bytes.splitlines()[0].startswith(b'{"case_id":')
    assert json.loads(first_bytes.splitlines()[0]) == records[0]

    changed = copy.deepcopy(records)
    changed[0]["questions"] = list(reversed(changed[0]["questions"]))
    changed_root = tmp_path / "changed-output"
    changed_digest = write_manifest(changed, changed_root)
    assert changed_digest != first_digest


def test_write_manifest_validation_failure_preserves_draft_bytes(tmp_path: Path) -> None:
    records = records_for(tmp_path / "records")
    invalid = copy.deepcopy(records)
    invalid[0]["case_id"] = ""
    output = tmp_path / "draft-output"
    output.mkdir()
    manifest = output / "manifest.jsonl"
    manifest.write_bytes(b"draft\n")

    with pytest.raises(ValueError, match="case_id"):
        write_manifest(invalid, output)

    assert manifest.read_bytes() == b"draft\n"
    assert not list(output.glob(".manifest.jsonl.*"))


def test_write_manifest_rejects_sealed_artifact_and_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    records = records_for(tmp_path / "records")
    output = tmp_path / "sealed-output"
    output.mkdir()
    sealed = output / "manifest.sha256"
    sealed.write_text("sealed\n", encoding="utf-8")
    manifest = output / "manifest.jsonl"
    manifest.write_bytes(b"original\n")

    with pytest.raises(FileExistsError, match="sealed"):
        write_manifest(records, output)

    assert sealed.read_text(encoding="utf-8") == "sealed\n"
    assert manifest.read_bytes() == b"original\n"


def test_cli_keeps_external_datasets_package_and_uses_locked_load_arguments(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "external"
    fake_package = fake_root / "datasets"
    fake_package.mkdir(parents=True)
    rows_path = fake_root / "rows.json"
    rows_path.write_text(json.dumps(_fake_loader_rows()), encoding="utf-8")
    (fake_package / "__init__.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"rows = json.loads(Path({str(rows_path)!r}).read_text())\n"
        "def load_dataset(path, name=None, split=None, revision=None):\n"
        "    key = path + '|' + (name or '') + '|' + (split or '') + '|' + (revision or '')\n"
        "    return rows[key]\n",
        encoding="utf-8",
    )
    output = tmp_path / "cli-output"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_root)
    env["PYTHONNOUSERSITE"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "tools" / "build_dataset_v2.py"),
        "--output",
        str(output),
    ]

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / "manifest.jsonl").exists()
    assert (output / "provenance.json").exists()
    first_line = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"provenance"' not in first_line


def _fake_loader_rows() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for source_id, metadata in load_sources().items():
        config = metadata.get("config", "")
        key = f"{metadata['wrapper']}|{config}|{metadata['split']}|{metadata['revision']}"
        result[key] = synthetic_sources()[source_id]["rows"]
    return result
