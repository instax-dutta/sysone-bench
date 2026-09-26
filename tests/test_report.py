from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from benchmark import figure_data, report

SEED = 42


def _question(qid: str, question_type: str) -> dict[str, Any]:
    if question_type == "choice":
        return {
            "qid": qid,
            "type": "choice",
            "instructions": "pick one",
            "criteria": {"alpha": "a", "beta": "b"},
        }
    if question_type == "noul":
        return {"qid": qid, "type": "noul", "instructions": "yes or no"}
    return {
        "qid": qid,
        "type": "score",
        "instructions": "how much",
        "criteria": ["low", "high"],
        "max_score": 1,
    }


def _manifest_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suite_index, (suite_id, question_type) in enumerate(
        (("triage", "choice"), ("guardrails", "noul"), ("sst5", "score"))
    ):
        for case_index in range(4):
            case_id = f"{suite_id}-{case_index:04d}"
            if question_type == "choice":
                expected: Any = "alpha" if case_index % 2 == 0 else "beta"
            elif question_type == "noul":
                expected = case_index % 2
            else:
                expected = case_index % 2
            records.append(
                {
                    "case_id": case_id,
                    "suite_id": suite_id,
                    "split": "calibration" if case_index == 0 else "evaluation",
                    "order_index": suite_index * 10 + case_index,
                    "state": {"message": f"state {case_id}"},
                    "questions": [_question("q", question_type)],
                    "expected": {"q": expected},
                }
            )
    return records


def _write_manifest(path: Path) -> dict[str, Mapping[str, Any]]:
    records = _manifest_records()
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return {record["case_id"]: record for record in records}


def _answer_for(question_type: str, correct: bool) -> dict[str, Any]:
    if question_type == "choice":
        probabilities = {"alpha": 0.7, "beta": 0.3}
        if not correct:
            probabilities = {"alpha": 0.2, "beta": 0.8}
        return {
            "type": "choice",
            "choice": "alpha" if correct else "beta",
            "probabilities": probabilities,
            "confidence": 0.7,
        }
    if question_type == "noul":
        return {"type": "noul", "noul": 0.9 if correct else 0.1}
    return {"type": "score", "score": 1.0 if correct else 0.0, "confidence": 0.6}


def _write_run(
    root: Path,
    label: str,
    manifest: Mapping[str, Mapping[str, Any]],
    *,
    correct_fraction: float,
    manifest_digest_value: str = "d" * 64,
) -> Path:
    run_dir = root / f"run-{label}"
    run_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    ordered = sorted(manifest.items(), key=lambda item: (item[1]["suite_id"], item[0]))
    for index, (case_id, record) in enumerate(ordered):
        question_type = cast(str, record["questions"][0]["type"])
        expected = record["expected"]["q"]
        correct = (index % 10) < correct_fraction * 10
        answer = _answer_for(question_type, correct)
        # Keep the recomputed correctness consistent with the sealed expectation.
        if question_type == "choice" and expected != answer["choice"]:
            correct = False
            answer = _answer_for(question_type, False)
        rows.append(
            {
                "case_id": case_id,
                "suite_id": record["suite_id"],
                "split": record["split"],
                "phase": record["split"],
                "execution_phase": "benchmark",
                "order_index": record["order_index"],
                "question_ids": ["q"],
                "questions": record["questions"],
                "expected": record["expected"],
                "answers": {"q": answer},
                "confidence": {"q": 0.7},
                "latency_seconds": 0.1 + index / 1000.0,
                "state_digest": "e" * 64,
                "runner": {"runner": label, "model": f"model-{label}"},
            }
        )
    (run_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    evaluation = [row for row in rows if row["phase"] == "evaluation"]
    for suite_id in sorted({cast(str, row["suite_id"]) for row in evaluation}):
        suite_rows = [row for row in evaluation if row["suite_id"] == suite_id]
        correct = sum(
            1
            for row in suite_rows
            if report.metrics.decision_correct("choice" if suite_id == "triage" else (
                "noul" if suite_id == "guardrails" else "score"
            ), row["expected"]["q"], row["answers"]["q"])
        )
        summaries[suite_id] = {
            "accuracy": correct / len(suite_rows),
            "cases": len(suite_rows),
            "decisions": len(suite_rows),
        }
    total_correct = sum(
        report.metrics.decision_correct(
            cast(str, row["questions"][0]["type"]),
            row["expected"]["q"],
            row["answers"]["q"],
        )
        for row in evaluation
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "accuracy": total_correct / len(evaluation),
                "cases": len(evaluation),
                "decisions": len(evaluation),
                "split": "evaluation",
                "suites": summaries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "manifest_digest": manifest_digest_value,
                "dataset_version": "2.0.0",
                "model": {
                    "runner": label,
                    "model": f"model-{label}",
                    "revision": "rev",
                    "serving": "local",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    import hashlib

    entries = []
    for name in ("metadata.json", "predictions.jsonl", "summary.json"):
        entries.append(
            f"{hashlib.sha256((run_dir / name).read_bytes()).hexdigest()}  {name}\n"
        )
    (run_dir / "checksums.sha256").write_text("".join(entries), encoding="utf-8")
    return run_dir


@pytest.fixture(autouse=True)
def small_replicates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "REPLICATES", 200)


def _three_runs(
    tmp_path: Path, manifest: Mapping[str, Mapping[str, Any]]
) -> dict[str, report.RunFacts]:
    runs = {
        "laya": _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5),
        "jev": _write_run(tmp_path / "jev", "jev", manifest, correct_fraction=0.9),
        "qwen-pcd": _write_run(tmp_path / "qwen", "qwen-pcd", manifest, correct_fraction=0.3),
    }
    return {label: report.load_run(label, path) for label, path in runs.items()}


def test_load_run_verifies_artifact_checksums(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    run_dir = _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5)

    run = report.load_run("laya", run_dir)
    assert run.run_id == run_dir.name
    assert run.checksum

    (run_dir / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fails its checksum"):
        report.load_run("laya", run_dir)


def test_load_run_rejects_a_run_id_that_disagrees_with_its_directory(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    run_dir = _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["run_id"] = "somewhere-else"
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _refresh_checksums(run_dir)

    with pytest.raises(ValueError, match="does not match its directory"):
        report.load_run("laya", run_dir)


def test_load_run_rejects_unknown_labels(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    run_dir = _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5)

    with pytest.raises(ValueError, match="unknown model label"):
        report.load_run("not-a-model", run_dir)


def _refresh_checksums(run_dir: Path) -> None:
    import hashlib

    entries = []
    for name in ("metadata.json", "predictions.jsonl", "summary.json"):
        entries.append(
            f"{hashlib.sha256((run_dir / name).read_bytes()).hexdigest()}  {name}\n"
        )
    (run_dir / "checksums.sha256").write_text("".join(entries), encoding="utf-8")


def test_suite_table_reproduces_the_sealed_summary(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    runs = _three_runs(tmp_path, manifest)

    for run in runs.values():
        correctness = report.decision_correctness(run, manifest)
        table = report.suite_table(run, correctness)
        report.verify_against_sealed_summary(run, table)
        assert set(table) == {"guardrails", "sst5", "triage"}
        assert all(entry["cases"] == 3 for entry in table.values())


def test_verify_fails_when_the_sealed_summary_disagrees(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    run = report.load_run("laya", _write_run(tmp_path, "laya", manifest, correct_fraction=0.5))
    summary = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
    summary["suites"]["triage"]["accuracy"] = 0.99
    (run.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _refresh_checksums(run.run_dir)
    tampered = report.load_run("laya", run.run_dir)

    correctness = report.decision_correctness(tampered, manifest)
    table = report.suite_table(tampered, correctness)
    with pytest.raises(ValueError, match="disagrees with its sealed summary"):
        report.verify_against_sealed_summary(tampered, table)


def test_paired_contrast_is_paired_and_adjusted(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    runs = _three_runs(tmp_path, manifest)
    correctness = {label: report.decision_correctness(run, manifest) for label, run in runs.items()}

    contrast = report.paired_contrast(
        runs["laya"], runs["jev"], correctness["laya"], correctness["jev"], manifest
    )

    assert set(contrast) == {"guardrails", "sst5", "triage"}
    for entry in contrast.values():
        assert entry["ci_low"] <= entry["delta"] <= entry["ci_high"]
        assert 0.0 <= entry["p_value"] <= 1.0
        assert entry["p_value_holm"] >= entry["p_value"]
    holm = [entry["p_value_holm"] for entry in contrast.values()]
    assert holm == sorted(holm, reverse=True) or len(set(holm)) == 1


def test_paired_contrast_rejects_incomplete_correctness(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    runs = _three_runs(tmp_path, manifest)
    correctness = {label: report.decision_correctness(run, manifest) for label, run in runs.items()}
    dropped = dict(correctness["laya"])
    # The paired contrast only walks evaluation decisions, so drop one of those.
    dropped.pop(next(key for key in sorted(dropped) if key[0].endswith("-0001")))

    with pytest.raises(ValueError, match="correctness is missing decision"):
        report.paired_contrast(runs["laya"], runs["jev"], dropped, correctness["jev"], manifest)


def test_build_comparison_emits_the_three_model_document(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    runs = _three_runs(tmp_path, manifest)
    correctness = {label: report.decision_correctness(run, manifest) for label, run in runs.items()}
    tables = {label: report.suite_table(run, correctness[label]) for label, run in runs.items()}

    comparison = report.build_comparison(runs, manifest, correctness, tables)

    assert set(comparison["runs"]) == {"laya", "jev", "qwen-pcd"}
    assert comparison["dataset"]["version"] == "2.0.0"
    for entry in comparison["suites"].values():
        assert set(entry) >= {"laya", "jev", "qwen-pcd", "delta", "ci_low", "ci_high", "p_value"}
    data = figure_data.build_figure_data(comparison)
    assert {row["model"] for row in data["accuracy"]} == {"laya", "jev", "qwen-pcd"}
    assert {row["contrast"] for row in data["paired_effects"]} == {"jev_minus_laya"}


def test_build_comparison_refuses_mixed_manifests(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    runs = _three_runs(tmp_path, manifest)
    other = _write_run(
        tmp_path / "other", "other-label-placeholder", manifest, correct_fraction=0.5,
        manifest_digest_value="e" * 64,
    )
    del other
    runs["laya"] = report.load_run(
        "laya",
        _write_run(
            tmp_path / "laya2", "laya", manifest, correct_fraction=0.5, manifest_digest_value="e" * 64
        ),
    )
    correctness = {label: report.decision_correctness(run, manifest) for label, run in runs.items()}
    tables = {label: report.suite_table(run, correctness[label]) for label, run in runs.items()}

    with pytest.raises(ValueError, match="different manifests"):
        report.build_comparison(runs, manifest, correctness, tables)


def test_write_report_emits_the_comparison_and_figure_source(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = _write_manifest(manifest_path)
    runs = _three_runs(tmp_path, manifest)
    output_root = tmp_path / "out"

    written = report.write_report(runs, manifest_path, output_root)

    assert written["comparison"].is_file()
    assert written["figure_source"].is_file()
    assert (output_root / "figures" / "data" / "accuracy.csv").is_file()
    document = json.loads(written["comparison"].read_text(encoding="utf-8"))
    assert document["dataset"]["manifest_digest"] == "d" * 64

    with pytest.raises(FileExistsError):
        report.write_report(runs, manifest_path, output_root)


def test_efficiency_and_multilingual_blocks_are_per_model(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.jsonl")
    runs = _three_runs(tmp_path, manifest)
    correctness = {label: report.decision_correctness(run, manifest) for label, run in runs.items()}
    tables = {label: report.suite_table(run, correctness[label]) for label, run in runs.items()}

    comparison = report.build_comparison(runs, manifest, correctness, tables)

    assert set(comparison["efficiency"]) == {"laya", "jev", "qwen-pcd"}
    assert all("p50_ms" in entry for entry in comparison["efficiency"].values())
    calibration_models = {record["model"] for record in comparison["calibration"]}
    assert calibration_models == {"laya", "jev", "qwen-pcd"}


def test_report_cli_writes_a_report_and_prints_only_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = _write_manifest(manifest_path)
    runs = {
        "laya": _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5),
        "jev": _write_run(tmp_path / "jev", "jev", manifest, correct_fraction=0.9),
        "qwen": _write_run(tmp_path / "qwen", "qwen-pcd", manifest, correct_fraction=0.3),
    }
    output_root = tmp_path / "report"

    code = report.main(
        [
            "--laya", str(runs["laya"]),
            "--jev", str(runs["jev"]),
            "--qwen", str(runs["qwen"]),
            "--manifest", str(manifest_path),
            "--output-root", str(output_root),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.count("\n") == 2
    assert (output_root / "report.json").is_file()
    assert (output_root / "figures" / "figure-source.json").is_file()


def test_report_cli_refuses_an_existing_output_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = _write_manifest(manifest_path)
    runs = {
        "laya": _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5),
        "jev": _write_run(tmp_path / "jev", "jev", manifest, correct_fraction=0.9),
        "qwen": _write_run(tmp_path / "qwen", "qwen-pcd", manifest, correct_fraction=0.3),
    }
    output_root = tmp_path / "report"
    output_root.mkdir()
    (output_root / "occupied").write_text("x", encoding="utf-8")

    code = report.main(
        [
            "--laya", str(runs["laya"]),
            "--jev", str(runs["jev"]),
            "--qwen", str(runs["qwen"]),
            "--manifest", str(manifest_path),
            "--output-root", str(output_root),
        ]
    )

    assert code == 1
    assert "already exists" in capsys.readouterr().err


def test_report_cli_reports_a_bad_run_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest = _write_manifest(manifest_path)
    runs = {
        "laya": _write_run(tmp_path / "laya", "laya", manifest, correct_fraction=0.5),
        "jev": _write_run(tmp_path / "jev", "jev", manifest, correct_fraction=0.9),
        "qwen": _write_run(tmp_path / "qwen", "qwen-pcd", manifest, correct_fraction=0.3),
    }
    (runs["qwen"] / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "report"

    code = report.main(
        [
            "--laya", str(runs["laya"]),
            "--jev", str(runs["jev"]),
            "--qwen", str(runs["qwen"]),
            "--manifest", str(manifest_path),
            "--output-root", str(output_root),
        ]
    )

    assert code == 1
    assert "fails its checksum" in capsys.readouterr().err
    assert not output_root.exists()


def test_report_cli_requires_every_run_flag() -> None:
    # argparse exits 2 for a usage error; the CLI turns that into a return code, like compare.py.
    assert report.main(["--laya", "x"]) == 2
