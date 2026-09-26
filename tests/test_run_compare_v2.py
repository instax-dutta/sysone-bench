from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fakes import write_fake_run

import compare
import run
from benchmark import orchestrator
from benchmark.canonical import canonical_json, digest_value
from benchmark.orchestrator import compare_v2, run_all_v2, run_suite_v2
from benchmark.storage import write_checksums
from benchmark.usage import UsageCounter
from runners.base import BaseRunner


def parametrize(
    argnames: str | tuple[str, ...], argvalues: Sequence[object]
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    return cast(
        Callable[[Callable[..., None]], Callable[..., None]],
        pytest.mark.parametrize(argnames, argvalues),
    )


class NamedFakeRunner(BaseRunner):
    def __init__(self, name: str) -> None:
        self.name = name

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        answers: dict[str, Any] = {}
        for qid, question in questions.items():
            question_type = question["type"]
            if question_type == "noul":
                answers[qid] = {"type": "noul", "noul": 0.75}
            elif question_type == "choice":
                labels = tuple(question["criteria"])
                label = labels[0]
                answers[qid] = {
                    "type": "choice",
                    "choice": label,
                    "probabilities": {
                        candidate: 1.0 if candidate == label else 0.0 for candidate in labels
                    },
                    "confidence": 1.0,
                }
            else:
                answers[qid] = {"type": "score", "score": 0, "confidence": 1.0}
        return {
            "answers": answers,
            "_usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "access_tokens": 3,
                "cached_tokens": 4,
                "token": "credential",
                "auth_token": "credential",
                "access_tokens_secret": "credential",
            },
            "_raw_model": {
                "api_key": "must-not-persist",
                "access_tokens": "credential",
                "credential": "secret",
                "status": "ok",
            },
        }


class MutatingStateRunner(NamedFakeRunner):
    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        cast(dict[str, Any], state)["text"] = "mutated"
        return super().predict(state, questions, phase=phase)


class IntegerKeyRunner(NamedFakeRunner):
    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        del state, questions
        return {
            "answers": {1: {"type": "noul", "noul": 1.0}},
            "_usage": {},
        }


class DynamicMetadataRunner(NamedFakeRunner):
    def __init__(self, name: str, multiplier: int = 1) -> None:
        super().__init__(name)
        self.multiplier = multiplier
        self.calls = 0

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        response = super().predict(state, questions, phase=phase)
        self.calls += self.multiplier
        return response

    def info(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "model": "dynamic-router",
            "revision": "router-revision",
            "serving": "fake-router",
            "device": "cpu",
            "adapter_version": "1",
            "checkpoint_identities": {"english": {"model": "laya", "revision": "english-revision"}},
            "identities": {"english": {"model": "laya", "revision": "english-revision"}},
            "pcd_source_revision": "pcd-revision",
            "pcd_implementation_digest": "a" * 64,
            "preloaded": ["english"],
            "route_counts": {"english": self.calls},
            "route_counts_by_phase": {
                "warmup": {},
                "benchmark": {"english": self.calls},
                "speed": {},
            },
            "route_reasons": {"english": f"reason-{self.calls}"},
            "route_reason_collections": {"english": [f"reason-{self.calls}"]},
            "route_reason_counts": {"english": {f"reason-{self.calls}": self.calls}},
        }


class SeededSequenceRunner(NamedFakeRunner):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.sequence: list[float] = []

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        response = super().predict(state, questions, phase=phase)
        self.sequence.append(random.random())
        response["_raw_model"] = {"sequence": list(self.sequence)}
        return response


class MaxLoadedFakeRunner(NamedFakeRunner):
    def info(self) -> dict[str, Any]:
        metadata = cast(dict[str, Any], super().info())
        metadata["max_loaded"] = 1
        return metadata


class MissingAnswerRunner(NamedFakeRunner):
    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        response = super().predict(state, questions, phase=phase)
        del response["answers"]["q"]
        return response


def noul_record(
    index: int,
    *,
    split: str = "evaluation",
    suite_id: str = "fixture",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_version": "2.0.0",
        "suite_id": suite_id,
        "case_id": f"{suite_id}-{index:04d}",
        "order_index": index,
        "split": split,
        "state": {"text": str(index)},
        "questions": [{"qid": "q", "type": "noul", "instructions": "fixture question"}],
        "expected": {"q": index % 2},
        "provenance_id": "fixture",
        "label_provenance": "fixture",
    }


def score_record(index: int, *, split: str = "evaluation") -> dict[str, Any]:
    record = noul_record(index, split=split)
    record["questions"] = [
        {
            "qid": "q",
            "type": "score",
            "instructions": "fixture score question",
            "max_score": 1,
        }
    ]
    record["expected"] = {"q": 0}
    return record


def write_manifest(
    tmp_path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    name: str = "manifest.jsonl",
    checksum: bool = True,
) -> tuple[Path, Path | None]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / name
    if name.endswith(".jsonl"):
        contents = b"".join(canonical_json(record) + b"\n" for record in records)
    else:
        contents = canonical_json(list(records)) + b"\n"
    manifest_path.write_bytes(contents)
    if not checksum:
        return manifest_path, None
    digest = hashlib.sha256(contents).hexdigest()
    checksum_path = tmp_path / "manifest.sha256"
    checksum_path.write_text(f"{digest}  manifest.jsonl\n", encoding="utf-8")
    return manifest_path, checksum_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def refresh_checksums(run_path: Path) -> None:
    (run_path / "checksums.sha256").unlink()
    write_checksums(run_path)


def tree_inventory(path: Path) -> dict[str, bytes | None]:
    inventory: dict[str, bytes | None] = {}
    for child in path.rglob("*"):
        relative = child.relative_to(path).as_posix()
        inventory[relative] = None if child.is_dir() else child.read_bytes()
    return inventory


def test_fake_run_writes_all_exclusive_artifacts_and_evaluation_summary(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(
        tmp_path,
        [noul_record(index) for index in range(3)],
    )
    output_root = tmp_path / "results"

    paths = run_all_v2(
        ["fake"],
        manifest_path,
        output_root,
        manifest_checksum_path=checksum_path,
    )

    assert len(paths) == 1
    run_path = paths[0]
    assert {
        "metadata.json",
        "predictions.jsonl",
        "summary.json",
        "usage.json",
        "checksums.sha256",
    } == {path.name for path in run_path.iterdir()}
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    usage = json.loads((run_path / "usage.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_path / "predictions.jsonl")
    assert metadata["seed"] == 42
    assert metadata["manifest_digest"] == metadata["manifest"]["digest"]
    assert metadata["manifest"]["checksum_verified"] is True
    assert summary["split"] == "evaluation"
    assert summary["cases"] == 3
    assert summary["decisions"] == 3
    assert usage["evaluation"]["benchmark"]["calls"] == 3
    assert [row["case_id"] for row in rows] == [f"fixture-{index:04d}" for index in range(3)]
    assert "must-not-persist" not in (run_path / "predictions.jsonl").read_text(encoding="utf-8")


def test_case_row_preserves_ordered_questions_full_answers_confidence_and_metadata(
    tmp_path: Path,
) -> None:
    questions = [
        {"qid": "z", "type": "noul", "instructions": "first question"},
        {
            "qid": "a",
            "type": "choice",
            "instructions": "second question",
            "criteria": {"left": "left answer", "right": "right answer"},
        },
    ]
    record = noul_record(0)
    record["questions"] = questions
    record["expected"] = {"z": 1, "a": "left"}
    manifest_path, checksum_path = write_manifest(tmp_path, [record])

    run_path = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    row = read_jsonl(run_path / "predictions.jsonl")[0]

    assert row["question_ids"] == ["z", "a"]
    assert row["questions"] == questions
    assert row["expected"] == {"z": 1, "a": "left"}
    assert set(row["answers"]) == {"z", "a"}
    assert set(row["confidence"]) == {"z", "a"}
    assert row["phase"] == "evaluation"
    assert row["split"] == "evaluation"
    assert row["latency_seconds"] >= 0.0
    assert row["runner"]["model"] == "fake-a"
    assert row["_usage"] == {
        "input_tokens": 2,
        "output_tokens": 1,
        "access_tokens": 3,
        "cached_tokens": 4,
        "token": "[REDACTED]",
        "auth_token": "[REDACTED]",
        "access_tokens_secret": "[REDACTED]",
    }
    assert row["_raw_model"] == {
        "api_key": "[REDACTED]",
        "access_tokens": "[REDACTED]",
        "credential": "[REDACTED]",
        "status": "ok",
    }


def test_calibration_is_retained_but_excluded_from_headline_metrics(tmp_path: Path) -> None:
    records = [
        noul_record(0, split="calibration"),
        noul_record(1, split="evaluation"),
        noul_record(2, split="evaluation"),
    ]
    manifest_path, checksum_path = write_manifest(tmp_path, records)

    run_path = run_all_v2(
        ["fake"],
        manifest_path,
        tmp_path / "results",
        manifest_checksum_path=checksum_path,
    )[0]
    rows = read_jsonl(run_path / "predictions.jsonl")
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    usage = json.loads((run_path / "usage.json").read_text(encoding="utf-8"))

    assert [row["phase"] for row in rows] == ["calibration", "evaluation", "evaluation"]
    assert summary["cases"] == 2
    assert summary["decisions"] == 2
    assert summary["suites"]["fixture"]["cases"] == 2
    assert usage["calibration"]["benchmark"]["calls"] == 1
    assert usage["evaluation"]["benchmark"]["calls"] == 2
    assert usage["totals"]["benchmark"]["calls"] == 3


def test_run_suite_uses_one_clock_read_before_and_after_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    values = iter([1.0, 1.125, 2.0, 2.5])

    def clock() -> float:
        events.append("clock")
        return next(values)

    monkeypatch.setattr(orchestrator, "perf_counter", clock)
    counter = UsageCounter()

    result = run_suite_v2(NamedFakeRunner("fake-a"), "fixture", [noul_record(0)], counter)

    assert result["predictions"][0]["latency_seconds"] == 0.125
    assert events == ["clock", "clock"]
    assert counter.as_dict()["benchmark"] == {
        "input_tokens": 2,
        "output_tokens": 1,
        "calls": 1,
    }


def test_adapter_answer_ids_are_validated_before_usage_is_recorded() -> None:
    counter = UsageCounter()

    with pytest.raises(ValueError, match="question IDs"):
        run_suite_v2(MissingAnswerRunner("bad"), "fixture", [noul_record(0)], counter)

    assert counter.as_dict()["benchmark"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "calls": 0,
    }


def test_state_digest_is_computed_before_mutating_runner_call() -> None:
    record = noul_record(0)
    original_state = {"text": "0"}

    result = run_suite_v2(
        MutatingStateRunner("mutating"),
        "fixture",
        [record],
        UsageCounter(),
    )

    row = result["predictions"][0]
    assert row["state_digest"] == digest_value(original_state)
    assert row["state_digest"] != digest_value({"text": "mutated"})
    assert record["state"] == original_state


def test_integer_answer_key_cannot_satisfy_string_question_id() -> None:
    record = noul_record(0)
    record["questions"] = [{"qid": "1", "type": "noul", "instructions": "fixture question"}]
    record["expected"] = {"1": 1}
    counter = UsageCounter()

    with pytest.raises(ValueError, match="question IDs|answer IDs"):
        run_suite_v2(IntegerKeyRunner("integer-key"), "fixture", [record], counter)

    assert counter.as_dict()["benchmark"]["calls"] == 0


def test_real_provisional_manifest_executes_with_offline_fake(tmp_path: Path) -> None:
    run_path = run_all_v2(
        ["fake"],
        Path("datasets/v2/manifest.jsonl"),
        tmp_path / "real-manifest-results",
    )[0]
    rows = read_jsonl(run_path / "predictions.jsonl")
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 1190
    assert summary["cases"] == 952
    assert summary["decisions"] == 1240
    assert {row["suite_id"] for row in rows} == {
        "triage",
        "guardrails",
        "moderation",
        "agnews",
        "emotion",
        "banking77_12",
        "mnli",
        "sst5",
        "multilingual_intent",
    }


def test_manifest_order_mismatch_fails_before_output_creation(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(
        tmp_path,
        [noul_record(1), noul_record(0)],
    )
    output_root = tmp_path / "results"

    with pytest.raises(ValueError, match="order"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            manifest_checksum_path=checksum_path,
        )

    assert not output_root.exists()


def test_checksum_mismatch_fails_before_runner_or_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    assert checksum_path is not None
    checksum_path.write_text(f"{'0' * 64}  manifest.jsonl\n", encoding="utf-8")
    output_root = tmp_path / "results"

    def forbidden_factory(name: str) -> BaseRunner:
        raise AssertionError(f"runner factory called for {name}")

    with pytest.raises(ValueError, match="manifest checksum"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            runner_factory=forbidden_factory,
            manifest_checksum_path=checksum_path,
        )

    assert not output_root.exists()


def test_empty_manifest_fails_before_output_creation(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [])
    output_root = tmp_path / "results"

    with pytest.raises(ValueError, match="manifest.*record"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            manifest_checksum_path=checksum_path,
        )

    assert not output_root.exists()


def test_existing_json_list_manifest_is_accepted(tmp_path: Path) -> None:
    manifest_path, _ = write_manifest(
        tmp_path,
        [noul_record(0)],
        name="manifest.json",
        checksum=False,
    )

    run_path = run_all_v2(["fake"], manifest_path, tmp_path / "results")[0]

    assert (run_path / "summary.json").is_file()


def test_compare_reports_manifest_mismatch_before_schema_validation(tmp_path: Path) -> None:
    first_run = write_fake_run(tmp_path, "one")
    second_run = write_fake_run(tmp_path, "two")
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="manifest"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_different_manifest_hashes_without_output(tmp_path: Path) -> None:
    first_manifest, first_checksum = write_manifest(
        tmp_path / "first",
        [noul_record(0)],
    )
    second_manifest, second_checksum = write_manifest(
        tmp_path / "second",
        [{**noul_record(0), "state": {"text": "different"}}],
    )
    first_root = tmp_path / "first-results"
    second_root = tmp_path / "second-results"
    first_run = run_all_v2(
        ["fake"],
        first_manifest,
        first_root,
        manifest_checksum_path=first_checksum,
    )[0]
    second_run = run_all_v2(
        ["fake"],
        second_manifest,
        second_root,
        manifest_checksum_path=second_checksum,
    )[0]
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="manifest"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_tampered_question_definition_without_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    predictions_path = second_run / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    rows[0]["questions"][0]["instructions"] = "tampered question"
    predictions_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="question definition"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_metric_schema_mismatch_without_reading_metrics(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    for run_path in (first_run, second_run):
        metadata_path = run_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["metric_schema_version"] = 99
        metadata_path.write_bytes(canonical_json(metadata) + b"\n")
        summary_path = run_path / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["metric_schema_version"] = 99
        summary_path.write_bytes(canonical_json(summary) + b"\n")
        refresh_checksums(run_path)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="metric schema"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_non_seed_42_metadata(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    metadata_path = second_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["seed"] = 41
    metadata_path.write_bytes(canonical_json(metadata) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="seed"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_dataset_version_not_bound_to_manifest_metadata(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    metadata_path = second_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_version"] = "wrong"
    metadata_path.write_bytes(canonical_json(metadata) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="dataset version"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_checksum_tampering_without_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    summary_path = second_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["decisions"] = 999
    summary_path.write_bytes(canonical_json(summary) + b"\n")
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="checksum"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_usage_schema_mismatch_without_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    usage_path = second_run / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    usage["evaluation"]["benchmark"]["calls"] = 999
    usage_path.write_bytes(canonical_json(usage) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="usage"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_dynamic_router_metadata_compares_by_stable_identity(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["router-a"],
        manifest_path,
        tmp_path / "router-a",
        runner_factory=lambda name: DynamicMetadataRunner(name, multiplier=1),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["router-b"],
        manifest_path,
        tmp_path / "router-b",
        runner_factory=lambda name: DynamicMetadataRunner(name, multiplier=2),
        manifest_checksum_path=checksum_path,
    )[0]
    first_row = read_jsonl(first_run / "predictions.jsonl")[0]
    first_metadata = json.loads((first_run / "metadata.json").read_text(encoding="utf-8"))

    assert first_row["runner"] == {
        "runner": "router-a",
        "model": "dynamic-router",
        "revision": "router-revision",
        "serving": "fake-router",
        "device": "cpu",
        "adapter_version": "1",
        "checkpoint_identities": {"english": {"model": "laya", "revision": "english-revision"}},
        "identities": {"english": {"model": "laya", "revision": "english-revision"}},
        "pcd_source_revision": "pcd-revision",
        "pcd_implementation_digest": "a" * 64,
        "preloaded": ["english"],
    }
    assert first_metadata["model"]["route_counts"] == {"english": 1}

    comparison_path = compare_v2(first_run, second_run, tmp_path / "comparisons")

    assert (comparison_path / "comparison.json").is_file()


def test_compare_binds_rows_to_metadata_even_when_both_runs_are_reordered(
    tmp_path: Path,
) -> None:
    records = [
        noul_record(0, split="calibration"),
        noul_record(1, split="evaluation"),
        noul_record(2, split="calibration"),
    ]
    manifest_path, checksum_path = write_manifest(tmp_path, records)
    run_paths = [
        run_all_v2(
            [name],
            manifest_path,
            tmp_path / name,
            runner_factory=lambda model_name: NamedFakeRunner(model_name),
            manifest_checksum_path=checksum_path,
        )[0]
        for name in ("fake-a", "fake-b")
    ]

    for run_path in run_paths:
        predictions_path = run_path / "predictions.jsonl"
        rows = read_jsonl(predictions_path)
        rows.reverse()
        predictions_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
        refresh_checksums(run_path)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="prediction.*metadata|row.*order"):
        compare_v2(run_paths[0], run_paths[1], output_root)

    assert not output_root.exists()


def test_compare_rejects_nonzero_benchmark_only_warmup_or_speed_usage(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    usage_path = second_run / "usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    usage["calibration"]["warmup"]["calls"] = 1
    usage["totals"]["warmup"]["calls"] = 1
    usage["evaluation"]["speed"] = {
        "input_tokens": 2,
        "output_tokens": 1,
        "calls": 1,
    }
    usage["totals"]["speed"] = dict(usage["evaluation"]["speed"])
    usage_path.write_bytes(canonical_json(usage) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="benchmark-only"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_successful_compare_writes_exclusive_comparison_directory(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    output_root = tmp_path / "comparisons"

    comparison_path = compare_v2(first_run, second_run, output_root)
    comparison = json.loads((comparison_path / "comparison.json").read_text(encoding="utf-8"))

    assert comparison_path.parent == output_root
    assert {path.name for path in comparison_path.iterdir()} == {
        "comparison.json",
        "checksums.sha256",
    }
    assert comparison["sources"]["a"]["run_id"] == first_run.name
    assert comparison["sources"]["b"]["run_id"] == second_run.name
    assert comparison["models"]["a"]["model"] == "fake-a"
    assert comparison["models"]["b"]["model"] == "fake-b"
    assert comparison["suites"]["fixture"]["accuracy_delta_b_minus_a"] == 0.0


def test_supplied_run_id_creates_named_child_when_output_root_is_absent(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "results"

    run_path = run_all_v2(
        ["fake"],
        manifest_path,
        output_root,
        manifest_checksum_path=checksum_path,
        run_id="fixture",
    )[0]

    assert run_path == output_root / "fixture"
    assert (
        json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))["run_id"] == "fixture"
    )


def test_existing_output_root_uses_run_id_child_without_run_directory_marker(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "results"
    output_root.mkdir()

    run_path = run_all_v2(
        ["fake"],
        manifest_path,
        output_root,
        manifest_checksum_path=checksum_path,
        run_id="fixture",
    )[0]

    assert run_path == output_root / "fixture"


def test_omitted_run_ids_use_distinct_utc_uuid_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    timestamps = iter(
        [
            datetime(2026, 9, 25, 1, 2, 3, tzinfo=UTC),
            datetime(2026, 9, 25, 1, 2, 4, tzinfo=UTC),
        ]
    )
    uuid_values = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr(orchestrator, "_utc_now", lambda: next(timestamps), raising=False)
    monkeypatch.setattr(orchestrator, "_uuid_hex", lambda: next(uuid_values), raising=False)
    output_root = tmp_path / "results"

    first = run_all_v2(
        ["fake"],
        manifest_path,
        output_root,
        manifest_checksum_path=checksum_path,
    )[0]
    second = run_all_v2(
        ["fake"],
        manifest_path,
        output_root,
        manifest_checksum_path=checksum_path,
    )[0]

    run_ids = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))["run_id"]
        for path in (first, second)
    ]
    assert run_ids == [
        "run-fake-20260925T010203Z-aaaaaaaaaaaa",
        "run-fake-20260925T010204Z-bbbbbbbbbbbb",
    ]
    assert first != second


def test_normalized_target_collision_fails_before_runner_or_output(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "results"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="target.*collision"):
        run_all_v2(
            ["a/b", "a-b"],
            manifest_path,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert factory_calls == []
    assert not output_root.exists()


def test_symlinked_output_parent_is_rejected_before_runner_or_output(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output_root = linked_parent / "results"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="symlink"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
        )

    assert factory_calls == []
    assert not (real_parent / "results").exists()


def test_symlinked_comparison_parent_is_rejected_before_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        compare_v2(first_run, second_run, linked_parent / "comparisons")

    assert not (real_parent / "comparisons").exists()


def test_run_directory_collision_preserves_existing_artifacts(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "results"
    run_path = run_all_v2(
        ["fake"],
        manifest_path,
        output_root,
        manifest_checksum_path=checksum_path,
        run_id="fixture",
    )[0]
    before = {path.name: path.read_bytes() for path in run_path.iterdir()}

    with pytest.raises(FileExistsError):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert {path.name: path.read_bytes() for path in run_path.iterdir()} == before


def test_module_cli_accepts_task5_contract_and_rejects_missing_sealed_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = orchestrator.main(["--models", "fake"])

    assert missing != 0
    assert "manifest" in capsys.readouterr().err.lower()

    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "fixture"
    output_root.mkdir()
    (output_root / ".sysone-owner").write_text("fixture\n", encoding="utf-8")

    result = orchestrator.main(
        [
            "--models",
            "fake",
            "--output-root",
            str(output_root),
            "--manifest",
            str(manifest_path),
            "--manifest-checksum",
            str(checksum_path),
            "--run-id",
            "fixture",
        ]
    )

    assert result == 0
    assert (output_root / "summary.json").is_file()
    assert not (output_root / "checksums.sha256").exists()
    write_checksums(output_root)
    assert (output_root / "checksums.sha256").is_file()
    assert "must-not-persist" not in capsys.readouterr().out


def test_root_run_and_compare_clis_use_v2_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    run_output = tmp_path / "run-output"
    run_output.mkdir()

    run_result = run.main(
        [
            "--models",
            "fake",
            "--output-root",
            str(run_output),
            "--manifest",
            str(manifest_path),
            "--manifest-checksum",
            str(checksum_path),
            "--run-id",
            "fixture",
        ]
    )
    run_path = run_output / "fixture"
    default_comparisons = tmp_path / "default-comparisons"
    monkeypatch.setattr(compare, "DEFAULT_OUTPUT_ROOT", default_comparisons, raising=False)
    default_compare_result = compare.main([str(run_path), str(run_path)])
    explicit_compare_result = compare.main(
        [
            str(run_path),
            str(run_path),
            "--output-root",
            str(tmp_path / "comparisons"),
        ]
    )

    progress = capsys.readouterr().out
    assert run_result == 0
    assert default_compare_result == 0
    assert explicit_compare_result == 0
    assert (run_path / "checksums.sha256").is_file()
    assert list(default_comparisons.iterdir())
    assert list((tmp_path / "comparisons").iterdir())
    assert "[fake] fixture/evaluation" in progress
    assert "fake-secret" not in progress


def test_root_run_models_only_fails_clearly_on_unsealed_default_manifest(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "datasets" / "v2"
    dataset.mkdir(parents=True)
    manifest = dataset / "manifest.jsonl"
    manifest.write_bytes(run.DEFAULT_MANIFEST.read_bytes())
    monkeypatch.setattr(run, "DEFAULT_MANIFEST", manifest, raising=False)
    monkeypatch.setattr(
        run, "DEFAULT_MANIFEST_CHECKSUM", dataset / "manifest.sha256", raising=False
    )

    result = run.main(["--models", "fake"])

    assert result != 0
    assert "sealed manifest checksum" in capsys.readouterr().err.lower()


def test_v2_import_does_not_require_model_sdk(tmp_path: Path) -> None:
    script = "import sys; sys.modules['laya'] = None; import benchmark.orchestrator, compare, run"

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_preexisting_normal_run_directory_with_sentinel_is_never_populated(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "results"
    target = output_root / "fixture"
    target.mkdir(parents=True)
    sentinel = target / "sentinel"
    sentinel.write_text("occupied", encoding="utf-8")
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(FileExistsError, match="run directory"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert factory_calls == []
    assert tree_inventory(target) == {"sentinel": b"occupied"}


def test_casefolded_target_collision_fails_before_runner_or_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "results"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="target.*collision"):
        run_all_v2(
            ["Runner", "runner"],
            manifest_path,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert factory_calls == []
    assert not output_root.exists()


def test_compare_rejects_output_root_inside_source_without_mutating_source(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    output_root = first_run / "inside"
    before = tree_inventory(first_run)

    with pytest.raises(ValueError, match="contain"):
        compare_v2(first_run, second_run, output_root)

    assert tree_inventory(first_run) == before
    assert not output_root.exists()


@parametrize("relation", ["equal", "ancestor"])
def test_compare_rejects_equal_or_ancestor_output_root(tmp_path: Path, relation: str) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    output_root = first_run if relation == "equal" else tmp_path
    inventory_root = first_run if relation == "equal" else tmp_path
    before = tree_inventory(inventory_root)

    with pytest.raises(ValueError, match="contain"):
        compare_v2(first_run, second_run, output_root)

    assert tree_inventory(inventory_root) == before


@parametrize(
    ("artifact", "field", "replacement", "message"),
    [
        ("metadata", "schema_version", 2.0, "schema"),
        ("metadata", "metric_schema_version", 2.0, "metric schema"),
        ("metadata", "seed", 42.0, "seed"),
        ("row", "schema_version", 2.0, "schema"),
        ("usage", "schema_version", 2.0, "schema"),
        ("summary", "schema_version", 2.0, "schema"),
        ("summary", "metric_schema_version", 2.0, "metric schema"),
        ("summary", "seed", 42.0, "seed"),
    ],
)
def test_compare_rejects_integral_float_artifact_identity(
    tmp_path: Path,
    artifact: str,
    field: str,
    replacement: float,
    message: str,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    if artifact == "metadata":
        path = second_run / "metadata.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value[field] = replacement
    elif artifact == "row":
        path = second_run / "predictions.jsonl"
        values = read_jsonl(path)
        values[0][field] = replacement
        path.write_bytes(b"".join(canonical_json(value) + b"\n" for value in values))
        refresh_checksums(second_run)
        value = None
    elif artifact == "usage":
        path = second_run / "usage.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value[field] = replacement
    else:
        path = second_run / "summary.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value[field] = replacement
    if value is not None:
        path.write_bytes(canonical_json(value) + b"\n")
        refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match=message):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_tampered_row_confidence(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    predictions_path = second_run / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    rows[0]["confidence"]["q"] = 0.0
    predictions_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="confidence"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


@parametrize(
    ("latency_token", "message"),
    [("-0.001", "latency.*non-negative"), ("1e999", "latency.*finite")],
)
def test_compare_rejects_invalid_row_latency(
    tmp_path: Path, latency_token: str, message: str
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    predictions_path = second_run / "predictions.jsonl"
    contents = predictions_path.read_text(encoding="utf-8")
    field = '"latency_seconds":'
    start = contents.index(field) + len(field)
    end = contents.index(",", start)
    predictions_path.write_text(
        contents[:start] + latency_token + contents[end:],
        encoding="utf-8",
    )
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match=message):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_renamed_run_directory_before_output(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    renamed = tmp_path / "renamed-run"
    first_run.rename(renamed)
    before = tree_inventory(renamed)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="run ID.*directory"):
        compare_v2(renamed, second_run, output_root)

    assert tree_inventory(renamed) == before
    assert not output_root.exists()


def test_each_runner_starts_from_independent_seed_42_sequence(tmp_path: Path) -> None:
    records = [noul_record(index) for index in range(3)]
    manifest_path, checksum_path = write_manifest(tmp_path, records)
    runners: dict[str, SeededSequenceRunner] = {}

    def factory(name: str) -> SeededSequenceRunner:
        runner = SeededSequenceRunner(name)
        runners[name] = runner
        return runner

    paths = run_all_v2(
        ["alpha", "beta"],
        manifest_path,
        tmp_path / "results",
        runner_factory=factory,
        manifest_checksum_path=checksum_path,
    )
    expected_rng = random.Random(42)
    expected = [expected_rng.random() for _ in records]
    sequences: dict[str, list[float]] = {}
    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        rows = read_jsonl(path / "predictions.jsonl")
        sequences[metadata["requested_runner"]] = rows[-1]["_raw_model"]["sequence"]

    assert sequences == {"alpha": expected, "beta": expected}
    assert list(runners) == ["alpha", "beta"]


@parametrize("linked", ["manifest", "checksum"])
def test_manifest_and_checksum_symlink_components_are_rejected(tmp_path: Path, linked: str) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    assert checksum_path is not None
    linked_path = tmp_path / f"linked-{linked}"
    if linked == "manifest":
        linked_path.symlink_to(manifest_path)
        supplied_manifest = linked_path
        supplied_checksum = checksum_path
    else:
        linked_path.symlink_to(checksum_path)
        supplied_manifest = manifest_path
        supplied_checksum = linked_path
    output_root = tmp_path / "results"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="symlink"):
        run_all_v2(
            ["fake"],
            supplied_manifest,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=supplied_checksum,
        )

    assert factory_calls == []
    assert not output_root.exists()


def test_manifest_and_checksum_bytes_are_each_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    assert checksum_path is not None
    original_read_bytes = Path.read_bytes
    read_counts = {manifest_path: 0, checksum_path: 0}

    def tracked_read_bytes(path: Path) -> bytes:
        if path in read_counts:
            read_counts[path] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)

    run_all_v2(
        ["fake"],
        manifest_path,
        tmp_path / "results",
        manifest_checksum_path=checksum_path,
    )

    assert read_counts == {manifest_path: 1, checksum_path: 1}


def test_manifest_replacement_after_parse_uses_original_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _ = write_manifest(
        tmp_path,
        [noul_record(index) for index in range(3)],
        checksum=False,
    )
    original_bytes = manifest_path.read_bytes()
    original_loads = orchestrator._loads

    def loads_then_replace(value: str, source: str) -> object:
        parsed = original_loads(value, source)
        if source == "manifest line 1":
            manifest_path.write_bytes(b" " + original_bytes)
        return parsed

    monkeypatch.setattr(orchestrator, "_loads", loads_then_replace)

    run_path = run_all_v2(["fake"], manifest_path, tmp_path / "results")[0]
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["manifest"]["file_digest"] == hashlib.sha256(original_bytes).hexdigest()
    assert manifest_path.read_bytes() == b" " + original_bytes


@parametrize("kind", ["manifest", "checksum", "output"])
def test_symlink_parent_dotdot_is_rejected_before_read_or_output(tmp_path: Path, kind: str) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    assert checksum_path is not None
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    lexical_manifest = linked_parent / ".." / "manifest.jsonl"
    lexical_checksum = linked_parent / ".." / "manifest.sha256"
    lexical_output = linked_parent / ".." / "results"
    supplied_manifest = lexical_manifest if kind == "manifest" else manifest_path
    supplied_checksum = lexical_checksum if kind == "checksum" else checksum_path
    supplied_output = lexical_output if kind == "output" else tmp_path / "results"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="symlink"):
        run_all_v2(
            ["fake"],
            supplied_manifest,
            supplied_output,
            runner_factory=factory,
            manifest_checksum_path=supplied_checksum,
            run_id="fixture",
        )

    assert factory_calls == []
    assert not (tmp_path / "results").exists()


def test_owned_marker_requires_existing_directory_basename_to_match_run_id(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    output_root = tmp_path / "different"
    output_root.mkdir()
    marker = output_root / ".sysone-owner"
    marker.write_text("fixture\n", encoding="utf-8")
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(FileExistsError, match="run directory"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert factory_calls == []
    assert tree_inventory(output_root) == {".sysone-owner": b"fixture\n"}


def test_run_rejects_symlink_after_missing_component_and_dotdot(tmp_path: Path) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (real_parent / "link").symlink_to(external, target_is_directory=True)
    output_root = real_parent / "missing" / ".." / "link" / "output"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="symlink"):
        run_all_v2(
            ["fake"],
            manifest_path,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert factory_calls == []
    assert not (real_parent / "missing").exists()
    assert not (external / "output").exists()


def test_compare_rejects_symlink_after_missing_component_and_dotdot(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    (real_parent / "link").symlink_to(first_run, target_is_directory=True)
    source_path = real_parent / "missing" / ".." / "link"
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="symlink"):
        compare_v2(source_path, second_run, output_root)

    assert not output_root.exists()


def test_manifest_rejects_symlink_after_missing_component_and_dotdot(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "manifest.jsonl").write_bytes(manifest_path.read_bytes())
    (real_parent / "link").symlink_to(external, target_is_directory=True)
    lexical_manifest = real_parent / "missing" / ".." / "link" / "manifest.jsonl"
    output_root = tmp_path / "results"
    factory_calls: list[str] = []

    def factory(name: str) -> BaseRunner:
        factory_calls.append(name)
        return NamedFakeRunner(name)

    with pytest.raises(ValueError, match="symlink"):
        run_all_v2(
            ["fake"],
            lexical_manifest,
            output_root,
            runner_factory=factory,
            manifest_checksum_path=checksum_path,
            run_id="fixture",
        )

    assert factory_calls == []
    assert not output_root.exists()


@parametrize("kind", ["metadata", "artifact", "directory"])
def test_compare_rejects_source_tree_symlink_before_metadata_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    external = tmp_path / "external"
    external.mkdir()
    if kind == "metadata":
        external_file = external / "metadata.json"
        external_file.write_text("external metadata", encoding="utf-8")
        (second_run / "metadata.json").unlink()
        (second_run / "metadata.json").symlink_to(external_file)
    elif kind == "artifact":
        external_file = external / "summary.json"
        external_file.write_text("external summary", encoding="utf-8")
        (second_run / "summary.json").unlink()
        (second_run / "summary.json").symlink_to(external_file)
    else:
        external_file = external / "payload.txt"
        external_file.write_text("external payload", encoding="utf-8")
        (second_run / "nested").symlink_to(external, target_is_directory=True)
    read_paths: list[Path] = []
    original_read_json_object = orchestrator._read_json_object

    def tracking_read_json_object(path: Path, source: str) -> dict[str, Any]:
        read_paths.append(path)
        return original_read_json_object(path, source)

    monkeypatch.setattr(orchestrator, "_read_json_object", tracking_read_json_object)
    output_root = tmp_path / "comparisons"
    external_before = tree_inventory(external)

    with pytest.raises(ValueError, match="symlink"):
        compare_v2(first_run, second_run, output_root)

    assert read_paths == []
    assert tree_inventory(external) == external_before
    assert not output_root.exists()


@parametrize(
    ("location", "field", "replacement"),
    [
        ("top", "cases", True),
        ("top", "decisions", True),
        ("top", "cases", 1.0),
        ("top", "decisions", 1.0),
        ("top", "accuracy", True),
        ("suite", "cases", True),
        ("suite", "decisions", True),
        ("suite", "cases", 1.0),
        ("suite", "decisions", 1.0),
        ("suite", "accuracy", True),
    ],
)
def test_compare_rejects_non_exact_derived_summary_types(
    tmp_path: Path,
    location: str,
    field: str,
    replacement: object,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    summary_path = second_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = summary if location == "top" else summary["suites"]["fixture"]
    target[field] = replacement
    summary_path.write_bytes(canonical_json(summary) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="cases|decisions|accuracy"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


@parametrize("expected", [0, 1])
def test_compare_accepts_integer_zero_or_one_accuracy(tmp_path: Path, expected: int) -> None:
    record = noul_record(0)
    record["expected"] = {"q": expected}
    manifest_path, checksum_path = write_manifest(tmp_path, [record])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    summary_path = second_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["accuracy"] = expected
    summary["suites"]["fixture"]["accuracy"] = expected
    summary_path.write_bytes(canonical_json(summary) + b"\n")
    refresh_checksums(second_run)

    comparison = compare_v2(first_run, second_run, tmp_path / "comparisons")

    assert (comparison / "comparison.json").is_file()


def test_compare_rejects_boolean_stable_identity_equal_to_numeric_metadata(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: MaxLoadedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: MaxLoadedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    first_rows = read_jsonl(first_run / "predictions.jsonl")
    second_rows = read_jsonl(second_run / "predictions.jsonl")
    assert first_rows[0]["runner"]["max_loaded"] == 1
    assert second_rows[0]["runner"]["max_loaded"] == 1
    second_rows[0]["runner"]["max_loaded"] = True
    predictions_path = second_run / "predictions.jsonl"
    predictions_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in second_rows))
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="model identity"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_boolean_confidence_equal_to_one(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [score_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    predictions_path = second_run / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    rows[0]["confidence"]["q"] = True
    predictions_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="confidence"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


@parametrize(
    ("field", "value", "message"),
    [
        ("execution_phase", None, "execution phase"),
        ("execution_phase", "warmup", "execution phase"),
        ("phase", None, "split"),
        ("phase", "calibration", "split"),
    ],
)
def test_compare_rejects_malformed_row_phase_identity(
    tmp_path: Path,
    field: str,
    value: str | None,
    message: str,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    predictions_path = second_run / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    if value is None:
        del rows[0][field]
    else:
        rows[0][field] = value
    predictions_path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match=message):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_boolean_suite_latency_equal_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [noul_record(0)])
    ticks = iter([0.0, 1.0, 0.0, 1.0])
    monkeypatch.setattr(orchestrator, "perf_counter", lambda: next(ticks))
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    summary_path = second_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["suites"]["fixture"]["latency"]["mean_seconds"] == 1.0
    summary["suites"]["fixture"]["latency"]["mean_seconds"] = True
    summary_path.write_bytes(canonical_json(summary) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="latency|metrics"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()


def test_compare_rejects_boolean_top_level_accuracy_equal_to_one(
    tmp_path: Path,
) -> None:
    manifest_path, checksum_path = write_manifest(tmp_path, [score_record(0)])
    first_run = run_all_v2(
        ["fake-a"],
        manifest_path,
        tmp_path / "results-a",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    second_run = run_all_v2(
        ["fake-b"],
        manifest_path,
        tmp_path / "results-b",
        runner_factory=lambda name: NamedFakeRunner(name),
        manifest_checksum_path=checksum_path,
    )[0]
    summary_path = second_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["accuracy"] == 1.0
    summary["accuracy"] = True
    summary_path.write_bytes(canonical_json(summary) + b"\n")
    refresh_checksums(second_run)
    output_root = tmp_path / "comparisons"

    with pytest.raises(ValueError, match="accuracy"):
        compare_v2(first_run, second_run, output_root)

    assert not output_root.exists()
