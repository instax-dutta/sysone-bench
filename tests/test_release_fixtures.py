from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import fixtures

_COMPARISON_FACTORY = "comparison_fixture"
_FIGURE_FACTORY = "figure_fixture"
_RELEASE_WRITER = "write_fake_release"
_COMPARISON_PATH_FACTORY = "fake_comparison_path"


def _fixture_factory(name: str) -> Callable[..., object]:
    factory = getattr(fixtures, name, None)
    assert callable(factory)
    return cast(Callable[..., object], factory)


def test_comparison_fixture_is_deterministic_and_complete() -> None:
    factory = cast(Callable[[], dict[str, object]], _fixture_factory(_COMPARISON_FACTORY))

    first = factory()
    second = factory()

    assert first == second
    assert first == {
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
            "triage": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "guardrails": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "moderation": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "agnews": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "emotion": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "banking77_12": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "mnli": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "sst5": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
            "multilingual_intent": {
                "laya": 0.80,
                "jev": 0.88,
                "qwen-pcd": 0.75,
                "decisions": 10,
                "delta": 0.08,
                "ci_low": 0.01,
                "ci_high": 0.15,
                "p_value": 0.03,
            },
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


def test_figure_fixture_builds_rows_from_the_comparison_contract() -> None:
    comparison = cast(Callable[[], dict[str, object]], _fixture_factory(_COMPARISON_FACTORY))()
    figure = cast(Callable[[], dict[str, object]], _fixture_factory(_FIGURE_FACTORY))()
    rows = cast(list[dict[str, object]], figure["accuracy"])
    paired_rows = cast(list[dict[str, object]], figure["paired_effects"])
    runs = cast(dict[str, dict[str, object]], comparison["runs"])

    assert list(figure) == [
        "accuracy",
        "paired_effects",
        "calibration",
        "risk_coverage",
        "efficiency",
        "multilingual",
        "router",
    ]
    assert len(rows) == 27
    assert len(paired_rows) == 9
    assert [row["model"] for row in rows] == [
        model for _suite_id in comparison["suites"] for model in ("laya", "jev", "qwen-pcd")
    ]
    assert rows[0] == {
        "checksum": "b" * 64,
        "dataset_version": "2.0.0",
        "manifest_digest": "a" * 64,
        "metric": "accuracy",
        "metric_definition": (
            "Micro accuracy across reviewed decisions in the evaluation split"
        ),
        "model": "laya",
        "plotted_values": {"value": 0.80},
        "run_id": "laya-test",
        "sample_count": 10,
        "source_checksums": ["b" * 64],
        "source_kind": "base",
        "source_run_ids": ["laya-test"],
        "split": "evaluation",
        "suite_id": "triage",
        "uncertainty_method": "paired_state_cluster_bootstrap_20000_seed_42",
        "unit": "fraction",
        "value": 0.80,
    }
    assert rows[-1]["suite_id"] == "multilingual_intent"
    assert rows[-1]["model"] == "qwen-pcd"
    assert rows[-1]["value"] == 0.75
    assert all(row["run_id"] == runs[str(row["model"])]["run_id"] for row in rows)
    assert paired_rows[0]["source_run_ids"] == ["laya-test", "jev-test"]
    assert paired_rows[0]["source_checksums"] == ["b" * 64, "c" * 64]
    assert "run_id" not in paired_rows[0]
    assert "checksum" not in paired_rows[0]


def test_write_fake_release_writes_deterministic_documents_and_optional_file(
    tmp_path: Path,
) -> None:
    writer = cast(Callable[..., None], _fixture_factory(_RELEASE_WRITER))

    writer(tmp_path, "notes.txt", "synthetic release notes")

    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "current"
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8") == "1,240 evaluation decisions"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "synthetic release notes"


def test_fake_comparison_path_round_trips_comparison_fixture(tmp_path: Path) -> None:
    factory = cast(Callable[[Path], Path], _fixture_factory(_COMPARISON_PATH_FACTORY))
    comparison = cast(Callable[[], dict[str, object]], _fixture_factory(_COMPARISON_FACTORY))()

    path = factory(tmp_path)

    assert path == tmp_path / "comparison.json"
    assert json.loads(path.read_text(encoding="utf-8")) == comparison
