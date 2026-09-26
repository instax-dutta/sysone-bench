from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_dataset_seal import _seed_seal_root

from datasets.v2 import labeling
from datasets.v2.labeling import (
    agreement_report,
    validate_reviewer_export,
    write_adjudication_packet,
    write_reviewer_packet,
)


def _records() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": 2,
            "dataset_version": "2.0.0",
            "suite_id": "multilingual_intent",
            "case_id": "case-alpha",
            "order_index": 0,
            "split": "calibration",
            "state": {
                "text": "first state",
                "language": "hi",
                "unsafe": "</script><script>alert('state')</script>",
            },
            "questions": [
                {
                    "qid": "intent",
                    "type": "choice",
                    "instructions": "Choose the intent.",
                    "criteria": {"alpha": "Alpha option", "beta": "Beta option"},
                },
                {
                    "qid": "flag",
                    "type": "noul",
                    "instructions": "Answer the flag.",
                },
                {
                    "qid": "intensity",
                    "type": "score",
                    "instructions": "Choose an integer level.",
                    "criteria": ["zero", "one", "two", "three", "four"],
                    "max_score": 4,
                },
            ],
            "expected": {
                "intent": "alpha",
                "flag": 0,
                "intensity": 0,
            },
            "source_label": "source-label-secret",
            "provenance_id": "source-id-secret",
            "label_provenance": "prior-label-secret",
            "model_name": "Laya",
            "prior_results": "Jev prior result",
        },
        {
            "schema_version": 2,
            "dataset_version": "2.0.0",
            "suite_id": "multilingual_intent",
            "case_id": "case-beta",
            "order_index": 1,
            "split": "evaluation",
            "state": {"text": "second state", "language": "es"},
            "questions": [
                {
                    "qid": "intent",
                    "type": "choice",
                    "instructions": "Choose the intent.",
                    "criteria": {"alpha": "Alpha option", "beta": "Beta option"},
                },
                {
                    "qid": "flag",
                    "type": "noul",
                    "instructions": "Answer the flag.",
                },
                {
                    "qid": "intensity",
                    "type": "score",
                    "instructions": "Choose an integer level.",
                    "criteria": ["zero", "one", "two", "three", "four"],
                    "max_score": 4,
                },
            ],
            "expected": {
                "intent": "beta",
                "flag": 1,
                "intensity": 4,
            },
            "source_label": "source-label-secret",
            "provenance_id": "source-id-secret",
            "label_provenance": "prior-label-secret",
            "model_name": "Laya",
            "prior_results": "Jev prior result",
        },
    ]


def _export(
    records: list[dict[str, Any]],
    reviewer_code: str,
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {"schema_version": 2, "reviewer_code": reviewer_code, "labels": labels}


def _valid_labels(records: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
    return {
        record["case_id"]: {
            "intent": values.get(record["case_id"], {}).get("intent", "alpha"),
            "flag": values.get(record["case_id"], {}).get("flag", 0),
            "intensity": values.get(record["case_id"], {}).get("intensity", 0),
        }
        for record in records
    }


def _case_ids(path: Path) -> list[str]:
    return re.findall(r'data-case-id="([^"]+)"', path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "datasets.v2.labeling", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def _embedded_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="packet-data" type="application/json">(.*?)</script>',
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_labeling_module_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "datasets.v2.labeling", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "--agreement" in result.stdout
    assert "--seal" in result.stdout


def test_packet_cli_writes_two_blind_packets_in_identical_seeded_order(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    manifest = dataset / "manifest.jsonl"
    output = dataset / "labeling"
    _write_manifest(manifest, _records())

    result = _run_cli("--records", str(manifest), "--output", str(output))

    assert result.returncode == 0
    assert result.stderr == ""
    reviewer_a = output / "reviewer_a.html"
    reviewer_b = output / "reviewer_b.html"
    assert reviewer_a.is_file()
    assert reviewer_b.is_file()
    assert _case_ids(reviewer_a) == _case_ids(reviewer_b)
    assert not list(output.glob("*.json"))
    for path in (reviewer_a, reviewer_b):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("expected", "source_label", "Laya", "Jev", "calibration", "evaluation"):
            assert forbidden not in text


def test_reviewer_packet_uses_dark_theme_tokens(tmp_path: Path) -> None:
    output = tmp_path / "reviewer_a.html"

    write_reviewer_packet(_records(), output, "A")

    text = output.read_text(encoding="utf-8")
    assert "color-scheme: dark" in text
    assert "--background: #0b1220" in text
    assert "--surface: #111827" in text
    assert "--text: #e5e7eb" in text
    assert "background: var(--background)" in text
    assert "background: #ffffff" not in text
    assert "button:focus-visible" in text


def test_agreement_cli_writes_deterministic_report_and_disagreement_packet(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    manifest = dataset / "manifest.jsonl"
    records = _records()
    _write_manifest(manifest, records)
    reviewer_a = dataset / "labels" / "reviewer_a.json"
    reviewer_b = dataset / "labels" / "reviewer_b.json"
    _write_json(reviewer_a, _export(records, "A", _valid_labels(records, {})))
    _write_json(
        reviewer_b,
        _export(records, "B", _valid_labels(records, {"case-beta": {"intent": "beta"}})),
    )
    output = dataset / "labeling" / "adjudication.html"
    report = dataset / "labels" / "agreement.json"
    second_output = dataset / "labeling" / "adjudication-2.html"
    second_report = dataset / "labels" / "agreement-2.json"

    result = _run_cli(
        "--agreement",
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(output),
        "--report",
        str(report),
    )
    second_result = _run_cli(
        "--agreement",
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(second_output),
        "--report",
        str(second_report),
    )

    assert result.returncode == 0
    assert second_result.returncode == 0
    assert result.stderr == ""
    assert report.read_bytes() == second_report.read_bytes()
    assert output.read_bytes() == second_output.read_bytes()
    assert _case_ids(output) == ["case-beta"]
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert report_payload["disagreement_count"] == 1
    assert "expected" not in report.read_text(encoding="utf-8")
    for forbidden in ("expected", "source_label", "Laya", "Jev", "case-alpha"):
        assert forbidden not in output.read_text(encoding="utf-8")


def test_seal_cli_prints_full_digest_from_temporary_handoff(tmp_path: Path) -> None:
    root = tmp_path / "seal"
    _seed_seal_root(root)
    manifest = root / "manifest.jsonl"
    reviewer_a = root / "labels" / "reviewer_a.json"
    reviewer_b = root / "labels" / "reviewer_b.json"
    adjudication = root / "labels" / "adjudication.json"

    result = _run_cli(
        "--seal",
        "--records",
        str(manifest),
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--adjudication",
        str(adjudication),
        "--output",
        str(root),
    )

    assert result.returncode == 0
    assert result.stderr == ""
    checksum = (root / "manifest.sha256").read_text(encoding="utf-8")
    assert result.stdout.strip() == checksum.split()[0]
    assert len(result.stdout.strip()) == 64


def test_seal_cli_fails_closed_when_human_artifact_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "missing-seal"
    _seed_seal_root(root)
    (root / "labels" / "reviewer_b.json").unlink()

    result = _run_cli(
        "--seal",
        "--records",
        str(root / "manifest.jsonl"),
        "--reviewer-a",
        str(root / "labels" / "reviewer_a.json"),
        "--reviewer-b",
        str(root / "labels" / "reviewer_b.json"),
        "--adjudication",
        str(root / "labels" / "adjudication.json"),
        "--output",
        str(root),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "reviewer" in result.stderr.lower()
    assert not (root / "manifest.sha256").exists()


@pytest.mark.parametrize(
    ("mode", "incompatible_flag"),
    [
        ("packet", "--report"),
        ("packet", "--reviewer-a"),
        ("packet", "--reviewer-b"),
        ("packet", "--adjudication"),
        ("agreement", "--adjudication"),
        ("seal", "--report"),
    ],
)
def test_cli_rejects_flags_not_valid_for_mode(
    tmp_path: Path, mode: str, incompatible_flag: str
) -> None:
    manifest = tmp_path / "dataset" / "manifest.jsonl"
    reviewer_a = tmp_path / "dataset" / "labels" / "reviewer_a.json"
    reviewer_b = tmp_path / "dataset" / "labels" / "reviewer_b.json"
    records = _records()
    _write_manifest(manifest, records)
    _write_json(reviewer_a, _export(records, "A", _valid_labels(records, {})))
    _write_json(reviewer_b, _export(records, "B", _valid_labels(records, {})))
    packet_output = tmp_path / "packet-output"
    agreement_output = tmp_path / "agreement.html"
    agreement_report = tmp_path / "agreement.json"
    seal_root = tmp_path / "seal"
    if mode == "packet":
        arguments = ["--records", str(manifest), "--output", str(packet_output)]
        protected = (manifest,)
        generated = (packet_output,)
    elif mode == "agreement":
        arguments = [
            "--agreement",
            "--records",
            str(manifest),
            "--reviewer-a",
            str(reviewer_a),
            "--reviewer-b",
            str(reviewer_b),
            "--output",
            str(agreement_output),
            "--report",
            str(agreement_report),
        ]
        protected = (manifest, reviewer_a, reviewer_b)
        generated = (agreement_output, agreement_report)
    else:
        _seed_seal_root(seal_root)
        arguments = [
            "--seal",
            "--records",
            str(seal_root / "manifest.jsonl"),
            "--reviewer-a",
            str(seal_root / "labels" / "reviewer_a.json"),
            "--reviewer-b",
            str(seal_root / "labels" / "reviewer_b.json"),
            "--adjudication",
            str(seal_root / "labels" / "adjudication.json"),
            "--output",
            str(seal_root),
        ]
        protected = (
            seal_root / "manifest.jsonl",
            seal_root / "provenance.json",
            seal_root / "labels" / "reviewer_a.json",
            seal_root / "labels" / "reviewer_b.json",
            seal_root / "labels" / "adjudication.json",
        )
        generated = (seal_root / "manifest.sha256",)
    before = {path: path.read_bytes() for path in protected}

    result = _run_cli(*arguments, incompatible_flag, str(tmp_path / "unused.json"))

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    assert incompatible_flag in result.stderr
    assert mode in result.stderr
    assert "not valid" in result.stderr
    assert {path: path.read_bytes() for path in protected} == before
    assert not any(path.exists() for path in generated)


def test_packet_cli_rejects_existing_file_output_without_changing_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "occupied-output"
    _write_manifest(manifest, _records())
    output.write_bytes(b"do not replace")
    before = {path: path.read_bytes() for path in (manifest, output)}

    result = _run_cli("--records", str(manifest), "--output", str(output))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "existing file" in result.stderr
    assert {path: path.read_bytes() for path in (manifest, output)} == before
    assert not (tmp_path / "reviewer_a.html").exists()
    assert not (tmp_path / "reviewer_b.html").exists()


def test_packet_cli_rejects_output_equal_to_records_without_changing_input(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, _records())
    before = manifest.read_bytes()

    result = _run_cli("--records", str(manifest), "--output", str(manifest))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "--records" in result.stderr
    assert "collision" in result.stderr
    assert manifest.read_bytes() == before


def test_packet_cli_rejects_resolved_packet_collision_with_records(tmp_path: Path) -> None:
    output = tmp_path / "packets"
    manifest = output / "reviewer_a.html"
    _write_manifest(manifest, _records())
    before = manifest.read_bytes()

    result = _run_cli("--records", str(manifest), "--output", str(output))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "--records" in result.stderr
    assert "collision" in result.stderr
    assert manifest.read_bytes() == before
    assert not (output / "reviewer_b.html").exists()


@pytest.mark.parametrize("output_flag", ["--output", "--report"])
@pytest.mark.parametrize("input_flag", ["--records", "--reviewer-a", "--reviewer-b"])
def test_agreement_cli_rejects_resolved_output_input_collisions(
    tmp_path: Path, output_flag: str, input_flag: str
) -> None:
    manifest = tmp_path / "dataset" / "manifest.jsonl"
    reviewer_a = tmp_path / "dataset" / "labels" / "reviewer_a.json"
    reviewer_b = tmp_path / "dataset" / "labels" / "reviewer_b.json"
    output = tmp_path / "dataset" / "adjudication.html"
    report = tmp_path / "dataset" / "agreement.json"
    records = _records()
    _write_manifest(manifest, records)
    _write_json(reviewer_a, _export(records, "A", _valid_labels(records, {})))
    _write_json(reviewer_b, _export(records, "B", _valid_labels(records, {})))
    inputs = {"--records": manifest, "--reviewer-a": reviewer_a, "--reviewer-b": reviewer_b}
    (inputs[input_flag].parent / "resolved").mkdir()
    collision = inputs[input_flag].parent / "resolved" / ".." / inputs[input_flag].name
    destinations = {"--output": output, "--report": report}
    destinations[output_flag] = collision
    before = {path: path.read_bytes() for path in inputs.values()}

    result = _run_cli(
        "--agreement",
        "--records",
        str(manifest),
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(destinations["--output"]),
        "--report",
        str(destinations["--report"]),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert output_flag in result.stderr
    assert input_flag in result.stderr
    assert "collision" in result.stderr
    assert {path: path.read_bytes() for path in inputs.values()} == before
    assert not output.exists()
    assert not report.exists()


def test_agreement_cli_rejects_output_collision_with_discovered_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    manifest = dataset / "manifest.jsonl"
    reviewer_a = dataset / "labels" / "reviewer_a.json"
    reviewer_b = dataset / "labels" / "reviewer_b.json"
    records = _records()
    _write_manifest(manifest, records)
    _write_json(reviewer_a, _export(records, "A", _valid_labels(records, {})))
    _write_json(reviewer_b, _export(records, "B", _valid_labels(records, {})))
    before = {path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b)}

    result = _run_cli(
        "--agreement",
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(manifest),
        "--report",
        str(dataset / "agreement.json"),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "manifest.jsonl" in result.stderr
    assert "collision" in result.stderr
    assert {path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b)} == before
    assert not (dataset / "agreement.json").exists()


def test_agreement_cli_rejects_resolved_output_report_collision(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset" / "manifest.jsonl"
    reviewer_a = tmp_path / "dataset" / "labels" / "reviewer_a.json"
    reviewer_b = tmp_path / "dataset" / "labels" / "reviewer_b.json"
    shared_output = tmp_path / "dataset" / "shared-output"
    records = _records()
    _write_manifest(manifest, records)
    _write_json(reviewer_a, _export(records, "A", _valid_labels(records, {})))
    _write_json(reviewer_b, _export(records, "B", _valid_labels(records, {})))
    (shared_output.parent / "resolved").mkdir()
    shared_output.write_bytes(b"do not replace")
    before = {path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b, shared_output)}

    result = _run_cli(
        "--agreement",
        "--records",
        str(manifest),
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(shared_output),
        "--report",
        str(shared_output.parent / "resolved" / ".." / shared_output.name),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "--report" in result.stderr
    assert "collision" in result.stderr
    assert {path: path.read_bytes() for path in before} == before


def test_seal_cli_path_collision_rejection_preserves_all_inputs(tmp_path: Path) -> None:
    root = tmp_path / "seal"
    _seed_seal_root(root)
    manifest = root / "manifest.jsonl"
    reviewer_a = root / "labels" / "reviewer_a.json"
    reviewer_b = root / "labels" / "reviewer_b.json"
    adjudication = root / "labels" / "adjudication.json"
    provenance = root / "provenance.json"
    before = {
        path: path.read_bytes()
        for path in (manifest, reviewer_a, reviewer_b, adjudication, provenance)
    }

    result = _run_cli(
        "--seal",
        "--records",
        str(reviewer_a),
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--adjudication",
        str(adjudication),
        "--output",
        str(root),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--records" in result.stderr
    assert "manifest.jsonl" in result.stderr
    assert {path: path.read_bytes() for path in before} == before
    assert not (root / "manifest.sha256").exists()


def test_packet_excludes_ground_truth_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "reviewer.html"
    write_reviewer_packet(_records(), path, "A")

    text = path.read_text(encoding="utf-8")

    for forbidden in (
        "expected",
        "source_label",
        "source-id-secret",
        "Laya",
        "Jev",
        "prior-label-secret",
        "prior_results",
        "calibration",
        "evaluation",
        "provenance",
        "label_provenance",
    ):
        assert forbidden not in text
    data = _embedded_data(path)
    assert set(data) == {"schema_version", "reviewer_code", "records"}
    assert set(data["records"][0]) == {"case_id", "state", "questions"}
    assert set(data["records"][0]["questions"][0]) == {
        "qid",
        "type",
        "instructions",
        "criteria",
    }


def test_reviewer_packet_recursively_sanitizes_state_metadata(tmp_path: Path) -> None:
    records = _records()
    records[0]["state"] = {
        "text": "keep this user content",
        "language": "hi",
        "nested": {
            "expected": "secret",
            "source_label": "secret",
            "source_id": "secret",
            "provenance": "secret",
            "provenance_id": "secret",
            "label_provenance": "secret",
            "model_name": "secret",
            "prior_results": "secret",
            "split": "calibration",
            "calibration": "secret",
            "evaluation": "secret",
            "revision": "secret",
            "checksum": "secret",
            "license": "secret",
            "citation": "secret",
            "row_id": "secret",
            "local_modification": "secret",
            "user_value": "keep",
            "items": [{"expected": "secret", "user_value": "keep"}],
        },
    }
    path = tmp_path / "reviewer.html"
    write_reviewer_packet(records, path, "A")
    text = path.read_text(encoding="utf-8")
    data = _embedded_data(path)
    packet_record = next(record for record in data["records"] if record["case_id"] == "case-alpha")

    assert packet_record["state"] == {
        "text": "keep this user content",
        "language": "hi",
        "nested": {"user_value": "keep", "items": [{"user_value": "keep"}]},
    }
    assert "secret" not in text
    assert "expected" not in text
    assert "source_label" not in text
    assert "provenance" not in text


def test_reviewer_packet_recursively_sanitizes_expanded_label_key_families(
    tmp_path: Path,
) -> None:
    records = _records()
    legitimate_state = {
        "message": "keep message",
        "prompt": "keep prompt",
        "post": "keep post",
        "text": "keep text",
        "language": "hi",
        "premise": "keep premise",
        "hypothesis": "keep hypothesis",
    }
    records[0]["state"] = {
        **legitimate_state,
        "nested": {
            **legitimate_state,
            "gold_label": "secret",
            "GOLD-LABEL": "secret",
            "internal_gold_value": "secret",
            "contains_truth": "secret",
            "referenceAnswer": "secret",
            "FINAL_DECISION": "secret",
            "adjudicatedValue": "secret",
            "humanAnnotation": "secret",
            "annotator_id": "secret",
            "officialAnswerKey": "secret",
            "classLabel": "secret",
            "weakLabel": "secret",
            "isSupervised": "secret",
            "supervisionStatus": "secret",
            "groundTruthHash": "secret",
            "provenanceDetails": "secret",
            "modelResponse": "secret",
            "resultScore": "secret",
            "splitName": "secret",
            "deeper": [{"gold_label": "secret", "prompt": "keep deep prompt"}],
        },
    }
    path = tmp_path / "reviewer.html"

    write_reviewer_packet(records, path, "A")

    data = _embedded_data(path)
    packet_record = next(record for record in data["records"] if record["case_id"] == "case-alpha")
    assert packet_record["state"] == {
        **legitimate_state,
        "nested": {
            **legitimate_state,
            "deeper": [{"prompt": "keep deep prompt"}],
        },
    }
    assert "secret" not in path.read_text(encoding="utf-8")


def test_packet_order_is_seeded_and_identical_for_reviewers(tmp_path: Path) -> None:
    records = _records()
    records.append({**records[0], "case_id": "case-gamma", "order_index": 2})
    records.reverse()

    reviewer_a = tmp_path / "a.html"
    reviewer_b = tmp_path / "b.html"
    write_reviewer_packet(records, reviewer_a, "A")
    write_reviewer_packet(list(reversed(records)), reviewer_b, "B")

    assert _case_ids(reviewer_a) == _case_ids(reviewer_b)
    assert len(_case_ids(reviewer_a)) == 3
    assert _case_ids(reviewer_a) == sorted(
        _case_ids(reviewer_a),
        key=lambda case_id: hashlib.sha256(f"42:{case_id}".encode()).digest(),
    )


def test_packet_is_self_contained_and_script_safety_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "reviewer.html"
    write_reviewer_packet(_records(), path, "A")
    text = path.read_text(encoding="utf-8")

    assert "<script" in text
    assert "</script><script>alert" not in text
    assert "localStorage" in text
    assert "Blob" in text
    assert "download" in text
    assert "ArrowLeft" in text
    assert "ArrowRight" in text
    assert "src=" not in text
    assert "href=" not in text
    assert "&lt;/script&gt;" in text


def test_noul_controls_are_marked_for_numeric_export(tmp_path: Path) -> None:
    path = tmp_path / "reviewer.html"
    write_reviewer_packet(_records(), path, "A")
    text = path.read_text(encoding="utf-8")

    assert 'data-value-kind="number"' in text
    assert "Number(control.value)" in text


def test_export_schema_coverage_and_semantic_validation() -> None:
    records = _records()
    expected_ids = {record["case_id"] for record in records}
    valid = _export(records, "A", _valid_labels(records, {}))
    with pytest.raises(ValueError, match="question schema required"):
        validate_reviewer_export(valid, expected_ids)
    validate_reviewer_export({"schema_version": 2, "reviewer_code": "A", "labels": {}}, set())
    validate_reviewer_export(valid, expected_ids, records)
    validate_reviewer_export(valid, expected_ids, records[0]["questions"])

    missing_case = {"schema_version": 2, "reviewer_code": "A", "labels": {"case-alpha": {}}}
    with pytest.raises(ValueError, match="case"):
        validate_reviewer_export(missing_case, expected_ids)

    extra_case = _export(
        records,
        "A",
        {**_valid_labels(records, {}), "case-extra": {"intent": "alpha"}},
    )
    with pytest.raises(ValueError, match="case"):
        validate_reviewer_export(extra_case, expected_ids)

    missing_question = _export(
        records,
        "A",
        {
            records[0]["case_id"]: {"intent": "alpha", "flag": 0},
        },
    )
    with pytest.raises(ValueError, match="question"):
        validate_reviewer_export(missing_question, {records[0]["case_id"]}, [records[0]])

    extra_question = _export(
        records,
        "A",
        {records[0]["case_id"]: {"intent": "alpha", "flag": 0, "intensity": 0, "extra": 1}},
    )
    with pytest.raises(ValueError, match="question"):
        validate_reviewer_export(extra_question, {records[0]["case_id"]}, [records[0]])


def test_schema_is_required_for_nonempty_labels_without_question_definitions() -> None:
    payload = {
        "schema_version": 2,
        "reviewer_code": "A",
        "labels": {"case-alpha": {}},
    }

    with pytest.raises(ValueError, match="question schema required"):
        validate_reviewer_export(payload, {"case-alpha"})

    arbitrary = {
        "schema_version": 2,
        "reviewer_code": "A",
        "labels": {"case-alpha": {"anything": "unvalidated"}},
    }
    with pytest.raises(ValueError, match="question schema required"):
        validate_reviewer_export(arbitrary, {"case-alpha"})


@pytest.mark.parametrize(
    ("qid", "value", "message"),
    [
        ("intent", "gamma", "choice"),
        ("flag", True, "noul"),
        ("flag", 0.0, "noul"),
        ("flag", 1.0, "noul"),
        ("flag", 0.5, "noul"),
        ("intensity", True, "score"),
        ("intensity", 1.0, "score"),
        ("intensity", 5, "score"),
    ],
)
def test_export_rejects_illegal_direct_labels(qid: str, value: Any, message: str) -> None:
    record = _records()[0]
    labels = _valid_labels([record], {record["case_id"]: {qid: value}})
    payload = _export([record], "A", labels)

    with pytest.raises(ValueError, match=message):
        validate_reviewer_export(payload, {record["case_id"]}, [record])


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1, "reviewer_code": "A", "labels": {}},
        {"schema_version": 2, "reviewer_code": "C", "labels": {}},
        {"schema_version": True, "reviewer_code": "A", "labels": {}},
        {"schema_version": 2, "reviewer_code": "A", "labels": {}, "extra": True},
    ],
)
def test_export_rejects_schema_failures(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_reviewer_export(payload, set())


def test_agreement_report_contains_metrics_confusion_and_disagreements() -> None:
    records = _records()
    records.extend(
        [
            {**records[0], "case_id": "case-gamma", "order_index": 2},
            {**records[0], "case_id": "case-delta", "order_index": 3},
        ]
    )
    labels_a = _valid_labels(
        records,
        {
            "case-alpha": {"intent": "alpha", "flag": 0, "intensity": 0},
            "case-beta": {"intent": "beta", "flag": 1, "intensity": 1},
            "case-gamma": {"intent": "beta", "flag": 1, "intensity": 2},
            "case-delta": {"intent": "alpha", "flag": 0, "intensity": 3},
        },
    )
    labels_b = _valid_labels(
        records,
        {
            "case-alpha": {"intent": "alpha", "flag": 0, "intensity": 0},
            "case-beta": {"intent": "beta", "flag": 1, "intensity": 1},
            "case-gamma": {"intent": "beta", "flag": 1, "intensity": 2},
            "case-delta": {"intent": "beta", "flag": 1, "intensity": 2},
        },
    )
    reviewer_a = _export(records, "A", labels_a)
    reviewer_b = _export(records, "B", labels_b)

    report = agreement_report(reviewer_a, reviewer_b, records)

    assert report["raw_agreement"] == 9 / 12
    assert report["nominal_cohens_kappa"] == 0.7096774193548387
    assert report["quadratic_weighted_kappa"] == 0.875
    assert report["per_question"]["multilingual_intent"]["intent"]["confusion_matrix"][
        "matrix"
    ] == [[1, 1], [0, 2]]
    assert report["per_question"]["multilingual_intent"]["flag"]["raw_agreement"] == 3 / 4
    assert report["per_language"]["multilingual_intent"]["hi"]["count"] == 9
    assert report["per_language"]["multilingual_intent"]["es"]["count"] == 3
    assert report["disagreements"] == [
        {
            "case_id": "case-delta",
            "qid": "intent",
            "question_type": "choice",
            "reviewer_a": "alpha",
            "reviewer_b": "beta",
        },
        {
            "case_id": "case-delta",
            "qid": "flag",
            "question_type": "noul",
            "reviewer_a": 0,
            "reviewer_b": 1,
        },
        {
            "case_id": "case-delta",
            "qid": "intensity",
            "question_type": "score",
            "reviewer_a": 3,
            "reviewer_b": 2,
        },
    ]
    assert "expected" not in json.dumps(report)
    assert "source" not in json.dumps(report)


def test_agreement_summaries_are_scoped_by_suite() -> None:
    first = _records()[0]
    second = copy.deepcopy(first)
    second["case_id"] = "other-0001"
    second["suite_id"] = "other"
    second["state"] = {"text": "other suite"}
    records = [first, second]
    labels_a = _valid_labels(records, {})
    labels_b = _valid_labels(records, {"other-0001": {"intent": "beta"}})

    report = agreement_report(
        _export(records, "A", labels_a),
        _export(records, "B", labels_b),
        records,
    )

    assert set(report["per_question"]) == {"multilingual_intent", "other"}
    assert report["per_question"]["multilingual_intent"]["intent"]["count"] == 1
    assert report["per_question"]["other"]["intent"]["count"] == 1
    assert set(report["confusion_matrices"]) == {"multilingual_intent", "other"}
    assert report["confusion_matrices"]["other"]["intent"]["matrix"] == [[0, 1], [0, 0]]


def test_agreement_handles_degenerate_kappa_without_crashing() -> None:
    records = _records()[:1]
    labels = _valid_labels(records, {})
    report = agreement_report(
        _export(records, "A", labels),
        _export(records, "B", labels),
        records,
    )

    assert report["per_question"]["multilingual_intent"]["intent"]["nominal_cohens_kappa"] == 0.0
    assert report["per_question"]["multilingual_intent"]["flag"]["nominal_cohens_kappa"] == 0.0
    assert (
        report["per_question"]["multilingual_intent"]["intensity"]["quadratic_weighted_kappa"]
        == 0.0
    )


@pytest.mark.parametrize("language", [None, "", 3])
def test_agreement_rejects_invalid_multilingual_language(language: object) -> None:
    record = copy.deepcopy(_records()[0])
    if language is None:
        record["state"].pop("language")
    else:
        record["state"]["language"] = language
    records = [record]
    labels = _valid_labels(records, {})

    with pytest.raises(ValueError, match="language"):
        agreement_report(
            _export(records, "A", labels),
            _export(records, "B", labels),
            records,
        )


def test_adjudication_packet_contains_only_disagreements_and_both_labels(tmp_path: Path) -> None:
    records = _records()
    labels_a = _valid_labels(records, {})
    labels_b = _valid_labels(records, {"case-beta": {"intent": "beta"}})
    path = tmp_path / "adjudication.html"

    write_adjudication_packet(
        _export(records, "A", labels_a),
        _export(records, "B", labels_b),
        records,
        path,
    )
    text = path.read_text(encoding="utf-8")

    assert _case_ids(path) == ["case-beta"]
    assert "Reviewer A" in text
    assert "Reviewer B" in text
    assert "alpha" in text
    assert "beta" in text
    assert "reviewer_labels: readReviewerLabels()" in text
    assert "reasons: readReasons()" in text
    assert "data-reason-qid" in text
    assert 'value="" required placeholder="Explain the final decision"' in text
    assert "Reviewer disagreement requires adjudication" not in text
    assert "input[data-qid]" in text
    assert "localStorage.removeItem(storageKey)" in text
    assert "input, textarea, select, [contenteditable='true']" in text
    assert text.index("link.click();") < text.index("clearDraft();")
    for forbidden in ("expected", "source_label", "source-id-secret", "Laya", "Jev", "case-alpha"):
        assert forbidden not in text


def test_adjudication_packet_recursively_sanitizes_state_metadata(tmp_path: Path) -> None:
    records = _records()
    records[1]["state"] = {
        "text": "keep this user content",
        "language": "es",
        "nested": {
            "expected": "secret",
            "provenance_id": "secret",
            "model_result": "secret",
            "user_value": "keep",
        },
    }
    labels_a = _valid_labels(records, {})
    labels_b = _valid_labels(records, {"case-beta": {"intent": "beta"}})
    path = tmp_path / "adjudication.html"
    write_adjudication_packet(
        _export(records, "A", labels_a),
        _export(records, "B", labels_b),
        records,
        path,
    )
    data = _embedded_data(path)
    packet_record = next(record for record in data["records"] if record["case_id"] == "case-beta")
    text = path.read_text(encoding="utf-8")

    assert packet_record["state"] == {
        "text": "keep this user content",
        "language": "es",
        "nested": {"user_value": "keep"},
    }
    assert packet_record["reviewer_labels"] == {
        "A": {"intent": "alpha"},
        "B": {"intent": "beta"},
    }
    assert "secret" not in text
    assert "expected" not in text
    assert "provenance" not in text


def _seed_agreement_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset"
    records = _records()
    manifest = dataset / "manifest.jsonl"
    reviewer_a = dataset / "labels" / "reviewer_a.json"
    reviewer_b = dataset / "labels" / "reviewer_b.json"
    _write_manifest(manifest, records)
    _write_json(reviewer_a, _export(records, "A", _valid_labels(records, {})))
    _write_json(reviewer_b, _export(records, "B", _valid_labels(records, {})))
    return manifest, reviewer_a, reviewer_b


def test_packet_cli_rejects_existing_reviewer_b_directory_before_writes(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset" / "manifest.jsonl"
    output = tmp_path / "dataset" / "labeling"
    _write_manifest(manifest, _records())
    output.mkdir()
    reviewer_b = output / "reviewer_b.html"
    reviewer_b.mkdir()
    sentinel = reviewer_b / "sentinel.txt"
    sentinel.write_bytes(b"keep sentinel")
    before = {path: path.read_bytes() for path in (manifest, sentinel)}

    result = _run_cli("--records", str(manifest), "--output", str(output))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "existing directory" in result.stderr
    assert {path: path.read_bytes() for path in before} == before
    assert not (output / "reviewer_a.html").exists()
    assert reviewer_b.is_dir()
    assert not list(output.glob(".*"))


def test_agreement_cli_rolls_back_html_when_report_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, reviewer_a, reviewer_b = _seed_agreement_inputs(tmp_path)
    output = tmp_path / "agreement.html"
    report = tmp_path / "agreement.json"
    output.write_bytes(b"existing html")
    report.write_bytes(b"existing report")
    before = {
        path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b, output, report)
    }

    def fail_report_write(path: Path, payload: dict[str, Any]) -> None:
        del payload
        path.write_bytes(b"partial report")
        raise OSError("synthetic report write failure")

    monkeypatch.setattr(labeling, "_write_canonical_json", fail_report_write)
    result = labeling.main(
        [
            "--agreement",
            "--records",
            str(manifest),
            "--reviewer-a",
            str(reviewer_a),
            "--reviewer-b",
            str(reviewer_b),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert {path: path.read_bytes() for path in before} == before
    assert not list(tmp_path.glob(".*"))


def test_agreement_cli_rolls_back_report_when_html_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, reviewer_a, reviewer_b = _seed_agreement_inputs(tmp_path)
    output = tmp_path / "agreement.html"
    report = tmp_path / "agreement.json"
    output.write_bytes(b"existing html")
    report.write_bytes(b"existing report")
    before = {
        path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b, output, report)
    }

    def fail_html_write(
        reviewer_a_payload: Any, reviewer_b_payload: Any, records: Any, output_path: Path
    ) -> None:
        del reviewer_a_payload, reviewer_b_payload, records
        output_path.write_bytes(b"partial html")
        raise OSError("synthetic html write failure")

    monkeypatch.setattr(labeling, "write_adjudication_packet", fail_html_write)
    result = labeling.main(
        [
            "--agreement",
            "--records",
            str(manifest),
            "--reviewer-a",
            str(reviewer_a),
            "--reviewer-b",
            str(reviewer_b),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert {path: path.read_bytes() for path in before} == before
    assert not list(tmp_path.glob(".*"))


def test_packet_cli_rejects_case_only_input_alias_without_overwriting_input(
    tmp_path: Path,
) -> None:
    physical_output = tmp_path / "Out"
    physical_output.mkdir()
    alias_output = tmp_path / "OUT"
    if not alias_output.is_dir():
        pytest.skip("filesystem is case-sensitive")
    records = alias_output / "REVIEWER_A.HTML"
    _write_manifest(records, _records())
    before = records.read_bytes()

    result = _run_cli("--records", str(records), "--output", str(alias_output))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "collision" in result.stderr
    assert records.read_bytes() == before
    assert not (physical_output / "reviewer_b.html").exists()


def test_packet_cli_rejects_hard_link_input_alias_without_overwriting_input(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    records = output / "input.jsonl"
    output_file = output / "reviewer_a.html"
    _write_manifest(records, _records())
    os.link(records, output_file)
    before = records.read_bytes()

    result = _run_cli("--records", str(records), "--output", str(output))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "--output" in result.stderr
    assert "collision" in result.stderr
    assert records.read_bytes() == before
    assert output_file.read_bytes() == before


@pytest.mark.parametrize("blocked_option", ["--output", "--report"])
def test_agreement_cli_rejects_existing_artifact_directory_before_writes(
    tmp_path: Path, blocked_option: str
) -> None:
    manifest, reviewer_a, reviewer_b = _seed_agreement_inputs(tmp_path)
    output = tmp_path / "agreement.html"
    report = tmp_path / "agreement.json"
    blocked_path = output if blocked_option == "--output" else report
    blocked_path.mkdir()
    sentinel = blocked_path / "sentinel.txt"
    sentinel.write_bytes(b"keep sentinel")
    before = {path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b, sentinel)}

    result = _run_cli(
        "--agreement",
        "--records",
        str(manifest),
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(output),
        "--report",
        str(report),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert blocked_option in result.stderr
    assert "existing directory" in result.stderr
    assert {path: path.read_bytes() for path in before} == before
    if blocked_option == "--output":
        assert not report.exists()
    else:
        assert not output.exists()


def test_agreement_cli_does_not_write_report_when_html_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, reviewer_a, reviewer_b = _seed_agreement_inputs(tmp_path)
    output = tmp_path / "agreement.html"
    report = tmp_path / "agreement.json"

    def fail_html_write(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("synthetic html write failure")

    monkeypatch.setattr(labeling, "write_adjudication_packet", fail_html_write)
    result = labeling.main(
        [
            "--agreement",
            "--records",
            str(manifest),
            "--reviewer-a",
            str(reviewer_a),
            "--reviewer-b",
            str(reviewer_b),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert not report.exists()
    assert not output.exists()


def test_cli_redacts_path_resolution_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "packets"
    _write_manifest(manifest, _records())
    secret = "runtime-secret-path-value"

    def fail_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        del path, args, kwargs
        raise RuntimeError(secret)

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    result = labeling.main(["--records", str(manifest), "--output", str(output)])
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
    assert secret not in captured.err
    assert not output.exists()


def test_duplicate_json_key_error_is_generic_and_redacted(tmp_path: Path) -> None:
    secret_key = "duplicate-secret-key-value"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        f'{{"{secret_key}": 1, "{secret_key}": 2}}\n',
        encoding="utf-8",
    )

    result = _run_cli("--records", str(manifest), "--output", str(tmp_path / "packets"))

    assert result.returncode != 0
    assert result.stdout == ""
    assert "duplicate JSON object key" in result.stderr
    assert secret_key not in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_redacts_extra_reviewer_json_key_and_preserves_inputs(tmp_path: Path) -> None:
    manifest, reviewer_a, reviewer_b = _seed_agreement_inputs(tmp_path)
    secret_key = "extra-reviewer-secret-key"
    secret_value = "extra-reviewer-secret-value"
    reviewer_payload = json.loads(reviewer_a.read_text(encoding="utf-8"))
    reviewer_payload[secret_key] = secret_value
    _write_json(reviewer_a, reviewer_payload)
    output = tmp_path / "agreement.html"
    report = tmp_path / "agreement.json"
    before = {path: path.read_bytes() for path in (manifest, reviewer_a, reviewer_b)}

    result = _run_cli(
        "--agreement",
        "--records",
        str(manifest),
        "--reviewer-a",
        str(reviewer_a),
        "--reviewer-b",
        str(reviewer_b),
        "--output",
        str(output),
        "--report",
        str(report),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    assert secret_key not in result.stderr
    assert secret_value not in result.stderr
    assert "Traceback" not in result.stderr
    assert {path: path.read_bytes() for path in before} == before
    assert not output.exists()
    assert not report.exists()


def test_cli_redacts_unknown_flag_and_value_without_creating_output(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "packets"
    _write_manifest(manifest, _records())
    before = manifest.read_bytes()
    secret_flag = "--unknown-secret-flag"
    secret_value = "unknown-secret-value"

    result = _run_cli(
        "--records",
        str(manifest),
        "--output",
        str(output),
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
    assert manifest.read_bytes() == before
    assert not output.exists()


def test_cli_redacts_secret_like_input_path_without_creating_output(tmp_path: Path) -> None:
    secret_path = tmp_path / "path-secret-marker.jsonl"
    output = tmp_path / "packets"
    manifest = tmp_path / "existing-manifest.jsonl"
    _write_manifest(manifest, _records())
    before = manifest.read_bytes()

    result = _run_cli("--records", str(secret_path), "--output", str(output))

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.splitlines()) == 1
    assert "path-secret-marker" not in result.stderr
    assert "Traceback" not in result.stderr
    assert manifest.read_bytes() == before
    assert not secret_path.exists()
    assert not output.exists()


def test_packet_blocks_bare_class_and_weak_families_and_preserves_content_keys(
    tmp_path: Path,
) -> None:
    records = _records()
    legitimate = {
        "message": "keep message",
        "prompt": "keep prompt",
        "post": "keep post",
        "text": "keep text",
        "language": "hi",
        "premise": "keep premise",
        "hypothesis": "keep hypothesis",
    }
    records[0]["state"] = {
        **legitimate,
        "nested": {
            **legitimate,
            "class": "blocked",
            "CLASS": "blocked",
            "class_value": "blocked",
            "classLabel": "blocked",
            "weak": "blocked",
            "WEAK": "blocked",
            "weak_score": "blocked",
            "weakLabel": "blocked",
        },
    }
    path = tmp_path / "reviewer.html"

    write_reviewer_packet(records, path, "A")

    data = _embedded_data(path)
    packet_record = next(record for record in data["records"] if record["case_id"] == "case-alpha")
    assert packet_record["state"] == {**legitimate, "nested": legitimate}
    text = path.read_text(encoding="utf-8")
    assert "blocked" not in text
    for key in legitimate:
        assert key in text
