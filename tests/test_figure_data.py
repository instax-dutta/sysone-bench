from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from fixtures import comparison_fixture, figure_fixture

from benchmark.canonical import canonical_json
from benchmark.figure_data import (
    EMPTY_FAMILY_REASON,
    FIGURE_FAMILIES,
    build_figure_data,
    validate_figure_data,
    write_figure_data,
)

_REQUIRED_ROW_FIELDS = {
    "dataset_version",
    "manifest_digest",
    "metric_definition",
    "model",
    "plotted_values",
    "sample_count",
    "source_checksums",
    "source_kind",
    "source_run_ids",
    "split",
    "suite_id",
    "uncertainty_method",
    "value",
}


def _actual_comparison() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_version": "2.0.0",
        "manifest": {
            "checksum_digest": "1" * 64,
            "checksum_verified": True,
            "digest": "2" * 64,
        },
        "metric_schema_version": 2,
        "models": {
            "a": {
                "adapter_version": "1",
                "device": "cpu",
                "model": "model-a",
                "revision": "a",
                "runner": "runner-a",
                "serving": "local",
            },
            "b": {
                "adapter_version": "1",
                "device": "cpu",
                "model": "model-b",
                "revision": "b",
                "runner": "runner-b",
                "serving": "local",
            },
        },
        "seed": 42,
        "sources": {
            "a": {
                "checksums_sha256": "3" * 64,
                "path": "/runs/a",
                "run_id": "run-a",
            },
            "b": {
                "checksums_sha256": "4" * 64,
                "path": "/runs/b",
                "run_id": "run-b",
            },
        },
        "suites": {
            "fixture": {
                "a": {
                    "accuracy": 0.75,
                    "cases": 4,
                    "decisions": 8,
                    "latency": {
                        "mean_seconds": 0.2,
                        "p50_seconds": 0.1,
                        "p95_seconds": 0.4,
                    },
                },
                "accuracy_delta_b_minus_a": 0.125,
                "b": {
                    "accuracy": 0.875,
                    "cases": 4,
                    "decisions": 8,
                    "latency": {
                        "mean_seconds": 0.3,
                        "p50_seconds": 0.2,
                        "p95_seconds": 0.5,
                    },
                },
            }
        },
    }


def test_build_figure_data_contains_every_suite_and_model_with_provenance() -> None:
    comparison = comparison_fixture()

    data = build_figure_data(comparison)

    assert list(data) == list(FIGURE_FAMILIES)
    assert len(data["accuracy"]) == 27
    assert [row["suite_id"] for row in data["accuracy"]] == sorted(
        row["suite_id"] for row in data["accuracy"]
    )
    assert {(row["suite_id"], row["model"]) for row in data["accuracy"]} == {
        (suite_id, model) for suite_id in comparison["suites"] for model in comparison["runs"]
    }
    for family, rows in data.items():
        for row in rows:
            assert _REQUIRED_ROW_FIELDS <= set(row)
            assert row["dataset_version"] == "2.0.0"
            assert row["manifest_digest"] == "a" * 64
            assert row["split"] == "evaluation"
            assert row["source_kind"] == ("router" if family == "router" else "base")
            assert row["metric_definition"]
            assert row["plotted_values"]
            assert row["uncertainty_method"] == "paired_state_cluster_bootstrap_20000_seed_42"
            assert isinstance(row["value"], (int, float))
            assert not isinstance(row["value"], bool)
            assert all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in row["plotted_values"].values()
            )
            if row["model"] in comparison["runs"]:
                run = comparison["runs"][row["model"]]
                assert row["source_run_ids"] == [run["run_id"]]
                assert row["source_checksums"] == [run["checksum"]]
                assert row["run_id"] == run["run_id"]
                assert row["checksum"] == run["checksum"]
            elif family == "paired_effects":
                assert row["source_run_ids"] == ["laya-test", "jev-test"]
                assert row["source_checksums"] == ["b" * 64, "c" * 64]
            elif row["model"] == "comparison":
                assert row["source_run_ids"] == ["laya-test", "jev-test", "qwen-test"]
                assert row["source_checksums"] == ["b" * 64, "c" * 64, "d" * 64]
            else:
                assert len(row["source_run_ids"]) == 1
                assert len(row["source_checksums"]) == 1


def test_build_figure_data_preserves_supported_synthetic_families_without_inference() -> None:
    data = build_figure_data(comparison_fixture())

    assert len(data["paired_effects"]) == 9
    assert len(data["calibration"]) == 4
    assert data["risk_coverage"] == []
    assert len(data["efficiency"]) == 9
    assert len(data["multilingual"]) == 6
    assert data["router"] == []
    paired = data["paired_effects"][0]
    assert paired["source_run_ids"] == ["laya-test", "jev-test"]
    assert paired["source_checksums"] == ["b" * 64, "c" * 64]
    assert paired["value"] == 0.08
    assert paired["ci_low"] == 0.01
    assert paired["ci_high"] == 0.15
    assert paired["p_value"] == 0.03
    assert "run_id" not in paired
    assert {row["primitive"] for row in data["calibration"]} == {"choice", "noul"}
    assert {row["metric"] for row in data["calibration"]} == {
        "ece_fitted",
        "ece_raw",
    }
    assert {row["metric"] for row in data["efficiency"]} == {
        "p50_ms",
        "p95_ms",
        "throughput",
    }
    assert {row["language"] for row in data["multilingual"]} == {"Hindi", "Spanish"}


def test_synthetic_laya_router_run_is_excluded_from_base_accuracy_and_preserves_telemetry() -> None:
    comparison = comparison_fixture()
    runs = cast(dict[str, dict[str, object]], comparison["runs"])
    runs["laya"] = {
        "run_id": "laya-test",
        "checksum": "b" * 64,
        "model": "laya-router",
        "runner": "laya-router",
        "serving": "local-router",
        "route_counts": {"english": 7},
    }
    cast(dict[str, object], comparison)["router"] = {"english_delta": 0.1}

    data = build_figure_data(comparison)

    assert all(row["model"] != "laya" for row in data["accuracy"])
    assert {row["metric"] for row in data["router"]} == {"english_delta", "route_count"}
    route_row = next(row for row in data["router"] if row["metric"] == "route_count")
    assert route_row["run_id"] == "laya-test"
    assert route_row["value"] == 7
    assert route_row["source_kind"] == "router"


def test_build_figure_data_supports_actual_pairwise_v2_schema() -> None:
    data = build_figure_data(_actual_comparison())

    assert [(row["model"], row["value"]) for row in data["accuracy"]] == [
        ("a", 0.75),
        ("b", 0.875),
    ]
    assert data["accuracy"][0]["source_run_ids"] == ["run-a"]
    assert data["accuracy"][0]["source_checksums"] == ["3" * 64]
    assert data["accuracy"][0]["sample_count"] == 8
    paired = data["paired_effects"]
    assert len(paired) == 1
    assert paired[0]["value"] == 0.125
    assert paired[0]["plotted_values"] == {"estimate": 0.125}
    assert paired[0]["unit"] == "fraction"
    assert "ci_low" not in paired[0]
    assert "p_value" not in paired[0]
    assert [(row["model"], row["metric"]) for row in data["efficiency"]] == [
        ("a", "p50_seconds"),
        ("a", "p95_seconds"),
        ("b", "p50_seconds"),
        ("b", "p95_seconds"),
    ]
    assert data["calibration"] == []
    assert data["risk_coverage"] == []
    assert data["multilingual"] == []
    assert data["router"] == []


def test_build_figure_data_preserves_attributed_risk_and_router_source_rows() -> None:
    comparison = comparison_fixture()
    runs = cast(dict[str, dict[str, object]], comparison["runs"])
    runs["router"] = {
        "run_id": "router-test",
        "checksum": "e" * 64,
        "model": "fake-router",
    }
    comparison["risk_coverage"] = [
        {
            "model": "laya",
            "primitive": "choice",
            "coverage": 1.0,
            "risk": 0.25,
        }
    ]

    data = build_figure_data(comparison)

    assert data["risk_coverage"] == [
        {
            "checksum": "b" * 64,
            "dataset_version": "2.0.0",
            "manifest_digest": "a" * 64,
            "metric": "risk",
            "metric_definition": "Selective risk at the stated coverage",
            "model": "laya",
            "plotted_values": {"coverage": 1.0, "risk": 0.25},
            "primitive": "choice",
            "run_id": "laya-test",
            "sample_count": None,
            "source_checksums": ["b" * 64],
            "source_kind": "base",
            "source_run_ids": ["laya-test"],
            "split": "evaluation",
            "suite_id": "comparison",
            "uncertainty_method": "paired_state_cluster_bootstrap_20000_seed_42",
            "unit": "fraction",
            "value": 0.25,
        }
    ]
    assert len(data["router"]) == 4
    assert {row["run_id"] for row in data["router"]} == {"router-test"}
    assert {row["metric"] for row in data["router"]} == {
        "english_delta",
        "multilingual_delta",
        "route_count",
    }


def test_build_figure_data_is_deterministic_and_does_not_mutate_comparison() -> None:
    comparison = comparison_fixture()
    original = deepcopy(comparison)
    reversed_comparison = {key: comparison[key] for key in reversed(tuple(comparison))}
    if isinstance(comparison["suites"], dict):
        reversed_comparison["suites"] = dict(reversed(tuple(comparison["suites"].items())))
    if isinstance(comparison["runs"], dict):
        reversed_comparison["runs"] = dict(reversed(tuple(comparison["runs"].items())))

    first = build_figure_data(comparison)
    second = build_figure_data(reversed_comparison)

    assert canonical_json(first) == canonical_json(second)
    assert comparison == original
    validate_figure_data(first, comparison)


def test_validate_figure_data_rejects_missing_or_type_changed_source_fields() -> None:
    comparison = comparison_fixture()
    missing = build_figure_data(comparison)
    missing["accuracy"][0].pop("run_id")
    with pytest.raises(ValueError, match=r"accuracy\[0\]\.run_id"):
        validate_figure_data(missing, comparison)

    wrong_type = build_figure_data(comparison)
    wrong_type["accuracy"][0]["sample_count"] = True
    with pytest.raises((TypeError, ValueError), match=r"accuracy\[0\]\.sample_count.*bool"):
        validate_figure_data(wrong_type, comparison)

    wrong_value = build_figure_data(comparison)
    wrong_value["accuracy"][0]["plotted_values"]["value"] = 0.5
    with pytest.raises(ValueError, match=r"accuracy\[0\]\.plotted_values\.value"):
        validate_figure_data(wrong_value, comparison)


def test_build_figure_data_rejects_incomplete_provenance() -> None:
    missing_checksum = comparison_fixture()
    del cast(dict[str, dict[str, object]], missing_checksum["runs"])["jev"]["checksum"]
    with pytest.raises(ValueError, match="checksum"):
        build_figure_data(missing_checksum)

    missing_manifest = comparison_fixture()
    cast(dict[str, dict[str, object]], missing_manifest["dataset"]).pop("manifest_digest")
    with pytest.raises(ValueError, match="manifest"):
        build_figure_data(missing_manifest)


def test_write_figure_data_writes_deterministic_exclusive_artifacts(tmp_path: Path) -> None:
    data = build_figure_data(comparison_fixture())
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    source_path = write_figure_data(data, first_root)
    second_source_path = write_figure_data(data, second_root)

    assert source_path == first_root / "figure-source.json"
    assert second_source_path.read_bytes() == source_path.read_bytes()
    assert source_path.read_bytes() == canonical_json(data) + b"\n"
    assert json.loads(source_path.read_text(encoding="utf-8")) == data
    assert {path.name for path in (first_root / "data").iterdir()} == {
        f"{family}.csv" for family in FIGURE_FAMILIES
    }
    for family in FIGURE_FAMILIES:
        content = (first_root / "data" / f"{family}.csv").read_bytes()
        assert content.endswith(b"\n")
        if not data[family]:
            assert content.decode("utf-8") == f"empty_reason\n{EMPTY_FAMILY_REASON}\n"
        else:
            assert "manifest_digest" in content.decode("utf-8").splitlines()[0]


def test_write_figure_data_never_overwrites_existing_artifacts(tmp_path: Path) -> None:
    data = build_figure_data(comparison_fixture())
    output_root = tmp_path / "graphs"
    data_root = output_root / "data"
    data_root.mkdir(parents=True)
    existing = data_root / "accuracy.csv"
    existing.write_bytes(b"original\n")
    source_path = output_root / "figure-source.json"
    source_path.write_bytes(b"original-json\n")

    with pytest.raises(FileExistsError, match="figure|accuracy"):
        write_figure_data(data, output_root)

    assert existing.read_bytes() == b"original\n"
    assert source_path.read_bytes() == b"original-json\n"
    assert {path.name for path in data_root.iterdir()} == {"accuracy.csv"}


def test_write_figure_data_rejects_symlink_targets(tmp_path: Path) -> None:
    data = build_figure_data(comparison_fixture())
    target = tmp_path / "target"
    target.mkdir()
    output_root = tmp_path / "graphs"
    output_root.mkdir()
    (output_root / "data").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_figure_data(data, output_root)

    linked_file_root = tmp_path / "file-graphs"
    linked_file_root.mkdir()
    (linked_file_root / "data").mkdir()
    (linked_file_root / "figure-source.json").symlink_to(target / "unused")
    with pytest.raises(ValueError, match="symlink"):
        write_figure_data(data, linked_file_root)


def test_write_figure_data_rejects_symlinked_output_root(tmp_path: Path) -> None:
    data = build_figure_data(comparison_fixture())
    target = tmp_path / "target"
    target.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        write_figure_data(data, linked_root)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_version",
        "manifest_digest",
        "metric_definition",
        "plotted_values",
        "source_checksums",
        "source_run_ids",
        "uncertainty_method",
    ],
)
def test_write_figure_data_rejects_stale_or_unprovenanced_rows_before_output(
    tmp_path: Path, field: str
) -> None:
    data = build_figure_data(comparison_fixture())
    cast(dict[str, Any], data["accuracy"][0]).pop(field, None)
    output_root = tmp_path / field

    with pytest.raises((TypeError, ValueError), match=field.replace("_", ".*")):
        write_figure_data(data, output_root)

    assert not output_root.exists()


def test_write_figure_data_rejects_misaligned_source_identity_before_output(
    tmp_path: Path,
) -> None:
    data = build_figure_data(comparison_fixture())
    cast(dict[str, Any], data["accuracy"][0])["source_checksums"] = ["f" * 64]
    output_root = tmp_path / "graphs"

    with pytest.raises(ValueError, match="source|checksum"):
        write_figure_data(data, output_root)

    assert not output_root.exists()


def test_write_figure_data_rejects_base_row_in_router_family(tmp_path: Path) -> None:
    data = build_figure_data(comparison_fixture())
    data["router"] = [cast(dict[str, Any], data["accuracy"][0])]
    output_root = tmp_path / "graphs"

    with pytest.raises((TypeError, ValueError), match="source_kind|router"):
        write_figure_data(data, output_root)

    assert not output_root.exists()


def test_generic_rows_reject_unknown_model_without_explicit_attribution() -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "coverage": 0.8,
            "metric": "risk",
            "model": "unregistered-panel",
            "primitive": "choice",
            "value": 0.2,
        }
    ]

    with pytest.raises(ValueError, match="attribution|source run"):
        build_figure_data(comparison)


def test_generic_rows_honor_explicit_single_and_aligned_list_identity() -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "checksum": "b" * 64,
            "coverage": 0.8,
            "metric": "risk",
            "model": "laya-panel",
            "primitive": "choice",
            "run_id": "laya-test",
            "source_checksums": ["b" * 64],
            "source_run_ids": ["laya-test"],
            "value": 0.2,
        }
    ]

    row = build_figure_data(comparison)["risk_coverage"][0]

    assert row["run_id"] == "laya-test"
    assert row["checksum"] == "b" * 64
    assert row["source_run_ids"] == ["laya-test"]
    assert row["source_checksums"] == ["b" * 64]


@pytest.mark.parametrize(
    "attribution",
    [
        {"run_id": "laya-test"},
        {"checksum": "b" * 64},
        {"run_id": "laya-test", "checksum": "f" * 64},
        {"source_run_ids": ["laya-test"]},
        {"source_checksums": ["b" * 64]},
        {
            "run_id": "laya-test",
            "checksum": "b" * 64,
            "source_run_ids": ["jev-test"],
            "source_checksums": ["c" * 64],
        },
        {
            "model": "laya",
            "run_id": "jev-test",
            "checksum": "c" * 64,
        },
    ],
)
def test_generic_rows_reject_partial_or_conflicting_attribution(
    attribution: dict[str, object],
) -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "coverage": 0.8,
            "metric": "risk",
            "model": "laya-panel",
            "primitive": "choice",
            "value": 0.2,
            **attribution,
        }
    ]

    with pytest.raises((TypeError, ValueError), match="attribution|source|run|checksum"):
        build_figure_data(comparison)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("value", True),
        ("value", "0.2"),
        ("suite_id", 1),
        ("metric", False),
        ("coverage", "0.8"),
    ],
)
def test_generic_rows_reject_metric_and_metadata_coercion(field: str, value: object) -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "coverage": 0.8,
            "metric": "risk",
            "model": "laya",
            "primitive": "choice",
            "run_id": "laya-test",
            "checksum": "b" * 64,
            "value": 0.2,
            field: value,
        }
    ]

    with pytest.raises((TypeError, ValueError), match=field.replace("_", ".*")):
        build_figure_data(comparison)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", None),
        ("dataset_version", "3.0.0"),
        ("manifest_digest", "f" * 64),
        ("uncertainty_method", "other"),
        ("metric_definition", None),
    ],
)
def test_generic_rows_reject_conflicting_or_null_explicit_metadata(
    field: str, value: object
) -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "checksum": "b" * 64,
            "coverage": 0.8,
            "metric": "risk",
            "model": "laya",
            "primitive": "choice",
            "run_id": "laya-test",
            "value": 0.2,
            field: value,
        }
    ]

    with pytest.raises((TypeError, ValueError), match=field.replace("_", ".*")):
        build_figure_data(comparison)


def test_structured_calibration_rejects_inner_primitive_conflict() -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["calibration"] = {
        "choice": {"ece_raw": 0.1, "primitive": "score"}
    }

    with pytest.raises(ValueError, match="primitive|score|choice|noul"):
        build_figure_data(comparison)


def test_risk_coverage_rejects_conflicting_value_and_risk() -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "checksum": "b" * 64,
            "coverage": 0.8,
            "metric": "risk",
            "model": "laya",
            "primitive": "choice",
            "risk": 0.3,
            "run_id": "laya-test",
            "value": 0.2,
        }
    ]

    with pytest.raises(ValueError, match="risk|value"):
        build_figure_data(comparison)


def test_structured_calibration_rejects_non_string_metadata() -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["calibration"] = {
        "choice": {"ece_raw": 0.1, "model": 1, "suite_id": 1}
    }

    with pytest.raises((TypeError, ValueError), match="model|suite_id"):
        build_figure_data(comparison)


@pytest.mark.parametrize("primitive", ["score", "unknown"])
def test_probability_families_reject_score_or_unknown_primitive(primitive: str) -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "checksum": "b" * 64,
            "coverage": 0.8,
            "metric": "risk",
            "model": "laya",
            "primitive": primitive,
            "run_id": "laya-test",
            "value": 0.2,
        }
    ]

    with pytest.raises(ValueError, match="primitive|score|choice|noul"):
        build_figure_data(comparison)


def test_actual_compare_v2_router_identity_and_route_reasons_are_preserved() -> None:
    comparison = _actual_comparison()
    cast(dict[str, object], comparison)["models"]["b"] = {
        "adapter_version": "1",
        "device": "cpu",
        "identities": {"english": {"model": "laya", "revision": "revision-a"}},
        "model": "laya-router",
        "revision": "english:revision-a",
        "route_reason_counts": {"english": {"latin-script": 4}},
        "route_reasons": {"english": "latin-script"},
        "runner": "laya-router",
        "serving": "local-router",
    }
    cast(dict[str, object], comparison)["router"] = {
        "checksum": "4" * 64,
        "english_delta": 0.0,
        "route_counts": {"english": 4},
        "run_id": "run-b",
    }

    data = build_figure_data(comparison)

    assert [(row["model"], row["value"]) for row in data["accuracy"]] == [("a", 0.75)]
    assert data["paired_effects"] == []
    assert {row["metric"] for row in data["router"]} == {"english_delta", "route_count"}
    route_row = next(row for row in data["router"] if row["metric"] == "route_count")
    assert route_row["run_id"] == "run-b"
    assert route_row["checksum"] == "4" * 64
    assert route_row["route"] == "english"
    assert route_row.get("route_reason") == "latin-script"
    assert route_row.get("route_reason_count") == 4
    assert route_row.get("route_reasons") == ["latin-script"]
    assert route_row.get("route_reason_counts") == {"latin-script": 4}


def test_router_metadata_identity_mismatch_is_rejected() -> None:
    comparison = _actual_comparison()
    cast(dict[str, dict[str, object]], comparison["models"])["b"]["serving"] = "local-router"
    cast(dict[str, object], comparison)["router"] = {
        "checksum": "f" * 64,
        "english_delta": 0.0,
        "run_id": "run-b",
    }

    with pytest.raises(ValueError, match="router|checksum"):
        build_figure_data(comparison)


def test_generic_router_source_cannot_be_used_by_a_base_family() -> None:
    comparison = comparison_fixture()
    runs = cast(dict[str, dict[str, object]], comparison["runs"])
    runs["router"] = {
        "checksum": "e" * 64,
        "model": "laya-router",
        "run_id": "router-test",
    }
    cast(dict[str, object], comparison)["risk_coverage"] = [
        {
            "checksum": "e" * 64,
            "coverage": 1.0,
            "metric": "risk",
            "model": "router",
            "primitive": "choice",
            "run_id": "router-test",
            "value": 0.1,
        }
    ]

    with pytest.raises(ValueError, match="router|base"):
        build_figure_data(comparison)


def test_generic_router_rows_require_router_source_for_custom_display_labels() -> None:
    comparison = comparison_fixture()
    runs = cast(dict[str, dict[str, object]], comparison["runs"])
    runs["router"] = {
        "checksum": "e" * 64,
        "model": "laya-router",
        "run_id": "router-test",
    }
    cast(dict[str, object], comparison)["router"] = [
        {
            "checksum": "b" * 64,
            "metric": "route_count",
            "model": "router-panel",
            "run_id": "laya-test",
            "value": 1,
        }
    ]

    with pytest.raises(ValueError, match="router.*base|base.*router"):
        build_figure_data(comparison)

    cast(dict[str, object], comparison)["router"] = [
        {
            "checksum": "e" * 64,
            "metric": "route_count",
            "model": "router-panel",
            "run_id": "router-test",
            "value": 1,
        }
    ]
    row = build_figure_data(comparison)["router"][0]

    assert row["model"] == "router-panel"
    assert row["source_run_ids"] == ["router-test"]


def test_router_family_rows_are_rejected_instead_of_dropped_without_a_router_source() -> None:
    comparison = comparison_fixture()
    cast(dict[str, object], comparison)["router"] = [
        {
            "checksum": "b" * 64,
            "metric": "route_count",
            "model": "router-panel",
            "run_id": "laya-test",
            "value": 1,
        }
    ]

    with pytest.raises(
        ValueError, match="router family rows require a Router-attributed source run"
    ):
        build_figure_data(comparison)

    del comparison["router"]

    assert build_figure_data(comparison)["router"] == []


def test_write_figure_data_sorts_reordered_valid_input_to_identical_bytes(
    tmp_path: Path,
) -> None:
    data = build_figure_data(comparison_fixture())
    reordered = {
        family: list(reversed(cast(list[dict[str, Any]], rows))) for family, rows in data.items()
    }
    first_root = tmp_path / "ordered"
    second_root = tmp_path / "reordered"

    first_source = write_figure_data(data, first_root)
    second_source = write_figure_data(reordered, second_root)

    assert first_source.read_bytes() == second_source.read_bytes()
    for family in FIGURE_FAMILIES:
        assert (first_root / "data" / f"{family}.csv").read_bytes() == (
            second_root / "data" / f"{family}.csv"
        ).read_bytes()


def test_release_figure_fixture_has_valid_writer_shape(tmp_path: Path) -> None:
    fixture = figure_fixture()
    source_path = write_figure_data(fixture, tmp_path / "fixture")
    written = cast(
        dict[str, list[dict[str, Any]]],
        json.loads(source_path.read_text(encoding="utf-8")),
    )

    assert set(written) == set(FIGURE_FAMILIES)
    for family in FIGURE_FAMILIES:
        assert sorted(canonical_json(row) for row in written[family]) == sorted(
            canonical_json(row) for row in cast(list[dict[str, Any]], fixture[family])
        )
