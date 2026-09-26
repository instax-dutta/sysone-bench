from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from benchmark.manifest import manifest_digest, validate_manifest

SUITE_COUNTS: dict[str, int] = {
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
SUITE_DECISIONS: dict[str, int] = {
    "triage": 4,
    "guardrails": 2,
    "moderation": 3,
}
CALIBRATION_CASES: dict[str, int] = {
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
SUITE_SOURCE_IDS: dict[str, str] = {suite_id: suite_id for suite_id in SUITE_COUNTS}
SUITE_SOURCE_IDS["banking77_12"] = "banking77"

TEST_REVISION = "deadbeef" * 5

_SOURCE_METADATA: dict[str, dict[str, str]] = {
    "agnews": {
        "canonical": "fancyzhx/ag_news",
        "wrapper": "fancyzhx/ag_news",
        "revision": TEST_REVISION,
        "split": "test",
        "license": "test-license",
        "citation": "test-citation",
    },
    "emotion": {
        "canonical": "dair-ai/emotion",
        "wrapper": "dair-ai/emotion",
        "revision": TEST_REVISION,
        "split": "test",
        "license": "test-license",
        "citation": "test-citation",
    },
    "banking77": {
        "canonical": "PolyAI/banking77",
        "wrapper": "mteb/banking77",
        "revision": TEST_REVISION,
        "split": "test",
        "license": "test-license",
        "citation": "test-citation",
    },
    "mnli": {
        "canonical": "nyu-mll/glue",
        "wrapper": "nyu-mll/glue",
        "revision": TEST_REVISION,
        "split": "validation_matched",
        "license": "test-license",
        "citation": "test-citation",
    },
    "sst5": {
        "canonical": "SetFit/sst5",
        "wrapper": "SetFit/sst5",
        "revision": TEST_REVISION,
        "split": "test",
        "license": "test-license",
        "citation": "test-citation",
    },
}


def load_sources() -> dict[str, object]:
    return {source_id: metadata.copy() for source_id, metadata in _SOURCE_METADATA.items()}


def fixture_checksum(suite_id: str, index: int) -> str:
    return hashlib.sha256(f"{suite_id}:{index}".encode()).hexdigest()


def load_test_provenance(suite_id: str, index: int) -> dict[str, object]:
    source_id = SUITE_SOURCE_IDS[suite_id]
    metadata = load_sources().get(source_id)
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "source_id": source_id,
        **metadata,
        "row_id": index,
        "source_label": "fixture",
        "local_modification": "synthetic fixture",
        "checksum": fixture_checksum(suite_id, index),
    }


def _fixture_case_payload(suite_id: str, index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if suite_id == "triage":
        intents: tuple[str, ...] = (
            "refund",
            "technical_help",
            "billing_question",
            "information",
            "cancellation",
        )
        intent = intents[index % len(intents)]
        questions: list[dict[str, Any]] = [
            {
                "qid": "intent",
                "type": "choice",
                "instructions": "fixture intent question",
                "criteria": {label: f"intent {label}" for label in intents},
            },
            {"qid": "refund_requested", "type": "noul", "instructions": "fixture refund question"},
            {"qid": "churn_risk", "type": "noul", "instructions": "fixture churn question"},
            {"qid": "is_urgent", "type": "noul", "instructions": "fixture urgency question"},
        ]
        return questions, {
            "intent": intent,
            "refund_requested": int(intent == "refund"),
            "churn_risk": int(intent == "cancellation"),
            "is_urgent": int(index % 3 == 0),
        }
    if suite_id == "guardrails":
        value = index % 2
        return [
            {"qid": "jailbreak", "type": "noul", "instructions": "fixture jailbreak question"},
            {
                "qid": "prompt_injection",
                "type": "noul",
                "instructions": "fixture prompt-injection question",
            },
        ], {"jailbreak": value, "prompt_injection": value}
    if suite_id == "moderation":
        return [
            {"qid": "toxic", "type": "noul", "instructions": "fixture toxicity question"},
            {"qid": "threat", "type": "noul", "instructions": "fixture threat question"},
            {"qid": "spam", "type": "noul", "instructions": "fixture spam question"},
        ], {
            "toxic": index % 2,
            "threat": int(index % 4 == 0),
            "spam": int(index % 5 == 0),
        }
    labels_by_suite: dict[str, tuple[str, ...]] = {
        "agnews": ("world", "sports", "business", "scitech"),
        "emotion": ("sadness", "joy", "love", "anger", "fear", "surprise"),
        "banking77_12": (
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
        "mnli": ("entailment", "neutral", "contradiction"),
    }
    if suite_id in labels_by_suite:
        labels = labels_by_suite[suite_id]
        question_ids = {
            "agnews": "topic",
            "emotion": "emotion",
            "banking77_12": "intent",
            "mnli": "relation",
        }
        question_id = question_ids[suite_id]
        return [
            {
                "qid": question_id,
                "type": "choice",
                "instructions": f"fixture {question_id} question",
                "criteria": {label: f"label {label}" for label in labels},
            }
        ], {question_id: labels[index % len(labels)]}
    if suite_id == "sst5":
        labels = ("very negative", "negative", "neutral", "positive", "very positive")
        return [
            {
                "qid": "sentiment",
                "type": "score",
                "instructions": "fixture sentiment question",
                "criteria": list(labels),
                "max_score": 4,
            }
        ], {"sentiment": index % len(labels)}
    if suite_id == "multilingual_intent":
        languages = ("hi", "es", "fr", "de", "ar")
        intents = (
            "refund",
            "technical_help",
            "billing_question",
            "information",
            "cancellation",
            "other",
        )
        intent = intents[(index % 30) // len(languages)]
        return [
            {
                "qid": "intent",
                "type": "choice",
                "instructions": "fixture multilingual intent question",
                "criteria": {label: f"intent {label}" for label in intents},
            }
        ], {"intent": intent}
    raise KeyError(f"unknown fixture suite: {suite_id}")


def load_test_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suite_id, count in SUITE_COUNTS.items():
        calibration_count = CALIBRATION_CASES[suite_id]
        for index in range(count):
            question_objects, expected = _fixture_case_payload(suite_id, index)
            state: dict[str, Any] = {"text": f"{suite_id} fixture {index}"}
            if suite_id == "multilingual_intent":
                state["language"] = ("hi", "es", "fr", "de", "ar")[(index // 30) % 5]
            records.append(
                {
                    "schema_version": 2,
                    "dataset_version": "2.0.0",
                    "suite_id": suite_id,
                    "case_id": f"{suite_id}-{index:04d}",
                    "order_index": index,
                    "split": "calibration" if index < calibration_count else "evaluation",
                    "state": state,
                    "questions": question_objects,
                    "expected": expected,
                    "provenance_id": SUITE_SOURCE_IDS[suite_id],
                    "label_provenance": "fixture-label",
                }
            )
    return records


def _write_fallback_packet(
    records: list[dict[str, object]], output: Path, reviewer_code: str
) -> None:
    packet_records = [
        {
            "case_id": record["case_id"],
            "state": record["state"],
            "questions": record["questions"],
        }
        for record in records
    ]
    packet_data = {
        "schema_version": 2,
        "reviewer_code": reviewer_code,
        "records": packet_records,
    }
    document = (
        '<!doctype html><html><head><meta charset="utf-8"></head><body>'
        f"<h1>Reviewer {reviewer_code}</h1>"
        f'<script id="fixture-data" type="application/json">{json.dumps(packet_data, ensure_ascii=False)}</script>'
        "</body></html>"
    )
    output.write_text(document, encoding="utf-8")


def write_test_packet(tmp_path: Path) -> Path:
    output = tmp_path / "packet.html"
    records = load_test_records()[:2]
    try:
        labeling_module = import_module("datasets.v2.labeling")
    except ModuleNotFoundError:
        _write_fallback_packet(records, output, "A")
    else:
        write_reviewer_packet = cast(Any, labeling_module).write_reviewer_packet
        write_reviewer_packet(records, output, "A")
    return output


def seal_test_manifest(tmp_path: Path, order: list[str]) -> str:
    del tmp_path
    record: dict[str, object] = {
        "schema_version": 2,
        "dataset_version": "2.0.0",
        "suite_id": "fixture",
        "case_id": "fixture-0001",
        "order_index": 0,
        "split": "evaluation",
        "state": {},
        "questions": [
            {"qid": qid, "type": "noul", "instructions": "fixture question"} for qid in order
        ],
        "expected": {qid: 1 for qid in order},
        "provenance_id": "fixture",
        "label_provenance": "fixture-label",
    }
    validate_manifest([record], {"fixture": 1})
    return manifest_digest([record])


def comparison_fixture() -> dict[str, object]:
    suites = [
        "triage",
        "guardrails",
        "moderation",
        "agnews",
        "emotion",
        "banking77_12",
        "mnli",
        "sst5",
        "multilingual_intent",
    ]
    return {
        "schema_version": 2,
        "dataset": {
            "version": "2.0.0",
            "manifest_digest": "a" * 64,
            "evaluation_cases": 952,
            "evaluation_decisions": 1240,
        },
        "runs": {
            "laya": {"run_id": "laya-test", "checksum": "b" * 64, "model": "fake-laya"},
            "jev": {"run_id": "jev-test", "checksum": "c" * 64, "model": "jev-1.13.0"},
            "qwen-pcd": {
                "run_id": "qwen-test",
                "checksum": "d" * 64,
                "model": "fake-qwen",
            },
        },
        "suites": {
            suite: {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            }
            for suite in suites
        },
        "calibration": {
            "choice": {"ece_raw": 0.1, "ece_fitted": 0.05},
            "noul": {"ece_raw": 0.08, "ece_fitted": 0.04},
        },
        "efficiency": {
            "laya": {"p50_ms": 600, "p95_ms": 900, "throughput": 1.2},
            "jev": {"p50_ms": 900, "p95_ms": 1200, "throughput": 0.8},
            "qwen-pcd": {"p50_ms": 1000, "p95_ms": 1800, "throughput": 0.6},
        },
        "multilingual": {
            "Hindi": {"laya": 0.3, "jev": 1.0, "qwen-pcd": 0.8},
            "Spanish": {"laya": 0.3, "jev": 1.0, "qwen-pcd": 0.8},
        },
        "router": {
            "english_delta": 0.0,
            "multilingual_delta": 0.4,
            "route_counts": {"english": 500, "multilingual": 100},
        },
    }


def figure_fixture() -> dict[str, object]:
    comparison = comparison_fixture()
    suite_values = cast(dict[str, dict[str, object]], comparison["suites"])
    run_values = cast(dict[str, dict[str, object]], comparison["runs"])
    dataset_values = cast(dict[str, object], comparison["dataset"])
    accuracy_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    uncertainty_method = "paired_state_cluster_bootstrap_20000_seed_42"
    for suite_id, suite in suite_values.items():
        for model in ("laya", "jev", "qwen-pcd"):
            run = run_values[model]
            accuracy_rows.append(
                {
                    "checksum": run["checksum"],
                    "dataset_version": dataset_values["version"],
                    "manifest_digest": dataset_values["manifest_digest"],
                    "metric": "accuracy",
                    "metric_definition": (
                        "Micro accuracy across reviewed decisions in the evaluation split"
                    ),
                    "model": model,
                    "plotted_values": {"value": suite[model]},
                    "run_id": run["run_id"],
                    "sample_count": 10,
                    "source_checksums": [run["checksum"]],
                    "source_kind": "base",
                    "source_run_ids": [run["run_id"]],
                    "split": "evaluation",
                    "suite_id": suite_id,
                    "uncertainty_method": uncertainty_method,
                    "unit": "fraction",
                    "value": suite[model],
                }
            )
        paired_rows.append(
            {
                "ci_high": suite["ci_high"],
                "ci_low": suite["ci_low"],
                "contrast": "jev_minus_laya",
                "dataset_version": dataset_values["version"],
                "manifest_digest": dataset_values["manifest_digest"],
                "metric": "accuracy_delta_b_minus_a",
                "metric_definition": "Jev evaluation accuracy minus Laya evaluation accuracy",
                "model": "jev_minus_laya",
                "p_value": suite["p_value"],
                "plotted_values": {
                    "ci_high": suite["ci_high"],
                    "ci_low": suite["ci_low"],
                    "estimate": suite["delta"],
                    "p_value": suite["p_value"],
                },
                "sample_count": 10,
                "source_checksums": [run_values["laya"]["checksum"], run_values["jev"]["checksum"]],
                "source_kind": "base",
                "source_run_ids": [run_values["laya"]["run_id"], run_values["jev"]["run_id"]],
                "split": "evaluation",
                "suite_id": suite_id,
                "uncertainty_method": uncertainty_method,
                "unit": "fraction",
                "value": suite["delta"],
            }
        )
    return {
        "accuracy": accuracy_rows,
        "paired_effects": paired_rows,
        "calibration": [],
        "risk_coverage": [],
        "efficiency": [],
        "multilingual": [],
        "router": [],
    }


def write_fake_release(
    tmp_path: Path, extra_file: str | None = None, content: str | None = None
) -> None:
    (tmp_path / "README.md").write_text("current", encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("1,240 evaluation decisions", encoding="utf-8")
    if extra_file is not None:
        (tmp_path / extra_file).write_text(content or "", encoding="utf-8")


def fake_comparison_path(tmp_path: Path) -> Path:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison_fixture()), encoding="utf-8")
    return path


def _assert_fixture_contract() -> None:
    records = load_test_records()
    summary = validate_manifest(records, SUITE_COUNTS)
    assert len(records) == 1190
    assert summary.total_decisions == 1550
    assert sum(record["split"] == "calibration" for record in records) == 238
    assert all("provenance" not in record for record in records)


_assert_fixture_contract()
