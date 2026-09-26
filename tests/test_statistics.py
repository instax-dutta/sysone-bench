from __future__ import annotations

import json
import math
from typing import Any, cast

import pytest

from benchmark.statistics import (
    apply_temperature,
    fit_temperature,
    holm_adjust,
    paired_cluster_bootstrap,
    paired_cluster_permutation,
)


def test_bootstrap_is_deterministic_and_reports_protocol() -> None:
    first = paired_cluster_bootstrap(
        [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], replicates=1000, seed=42
    )
    second = paired_cluster_bootstrap(
        [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], replicates=1000, seed=42
    )

    assert first == second
    assert set(first) == {"point_delta", "ci_low", "ci_high", "replicates", "seed"}
    assert first["replicates"] == 1000
    assert first["seed"] == 42
    assert all(math.isfinite(float(value)) for value in first.values())
    json.dumps(first, allow_nan=False)


def test_bootstrap_point_delta_uses_b_minus_a() -> None:
    result = paired_cluster_bootstrap(
        [0, 0, 1, 1], [1, 1, 1, 1], [10, 10, 11, 11], replicates=1000, seed=42
    )

    assert result["point_delta"] == pytest.approx(0.5)


def test_bootstrap_uses_linear_percentile_interpolation() -> None:
    result = paired_cluster_bootstrap(
        [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1], replicates=2, seed=42
    )

    assert result["ci_low"] == pytest.approx(-0.4875)
    assert result["ci_high"] == pytest.approx(-0.0125)


def test_bootstrap_resamples_whole_clusters() -> None:
    result = paired_cluster_bootstrap(
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [10, 10, 11, 11],
        replicates=1000,
        seed=42,
    )

    assert result["point_delta"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)


def test_bootstrap_weights_unequal_clusters_by_row_count() -> None:
    result = paired_cluster_bootstrap(
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1], replicates=2, seed=42
    )

    assert result["point_delta"] == pytest.approx(0.25)
    assert result["ci_low"] == pytest.approx(0.00625)
    assert result["ci_high"] == pytest.approx(0.24375)


def test_bootstrap_preserves_first_seen_cluster_order() -> None:
    result = paired_cluster_bootstrap(
        [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1], replicates=2, seed=7
    )
    reordered = paired_cluster_bootstrap(
        [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], replicates=2, seed=7
    )

    assert result != reordered
    assert math.isfinite(result["ci_low"])
    assert math.isfinite(result["ci_high"])


def test_bootstrap_rejects_empty_or_mismatched_data() -> None:
    cases: list[tuple[list[Any], list[Any], list[Any]]] = [
        ([], [], []),
        ([0], [], [0]),
        ([0], [0], []),
        ([0], [0, 1], [0, 1]),
    ]
    for a_rows, b_rows, cluster_ids in cases:
        with pytest.raises(ValueError):
            paired_cluster_bootstrap(a_rows, b_rows, cluster_ids, replicates=10, seed=42)


def test_bootstrap_rejects_nonfinite_or_non_numeric_rows() -> None:
    for value in (math.nan, math.inf, -math.inf, True, "0"):
        with pytest.raises((TypeError, ValueError)):
            paired_cluster_bootstrap(cast(Any, [value]), [0], [0], replicates=10, seed=42)


def test_bootstrap_rejects_empty_or_unhashable_cluster_ids() -> None:
    cluster_ids: tuple[object, ...] = ("", None, " ", [])
    for cluster_id in cluster_ids:
        with pytest.raises((TypeError, ValueError)):
            paired_cluster_bootstrap([0, 1], [0, 0], [cluster_id, 1], replicates=10, seed=42)


def test_bootstrap_rejects_nonpositive_or_noninteger_replicates() -> None:
    for replicates in (0, -1, 1.5, True):
        with pytest.raises((TypeError, ValueError)):
            paired_cluster_bootstrap([0], [0], [1], replicates=cast(Any, replicates), seed=42)


def test_cluster_statistics_reject_invalid_seeds() -> None:
    for seed in (True, 1.5, "42", None):
        with pytest.raises((TypeError, ValueError)):
            paired_cluster_bootstrap([0], [0], [1], seed=cast(Any, seed))


def test_cluster_statistics_reject_invalid_cluster_ids() -> None:
    cluster_ids: tuple[object, ...] = (None, "", " ", True, [], {})
    for cluster_id in cluster_ids:
        with pytest.raises((TypeError, ValueError)):
            paired_cluster_bootstrap([0, 1], [0, 0], [cluster_id, 1], seed=42)


def test_fit_temperature_rejects_invalid_label_lengths_and_bools() -> None:
    with pytest.raises(ValueError):
        fit_temperature([0.1, 0.9], [0])
    for label in (True, False, 1.5, "0"):
        with pytest.raises((TypeError, ValueError)):
            fit_temperature([0.1, 0.9], [cast(Any, label), 1])


def test_permutation_is_deterministic_and_finite() -> None:
    first = paired_cluster_permutation(
        [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], replicates=1000, seed=42
    )
    second = paired_cluster_permutation(
        [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], replicates=1000, seed=42
    )

    assert first == second
    assert "p_value" in first
    assert 0.0 < first["p_value"] <= 1.0
    assert math.isfinite(first["p_value"])
    json.dumps(first, allow_nan=False)


def test_permutation_flips_cluster_differences_independently() -> None:
    result = paired_cluster_permutation(
        [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], replicates=1000, seed=42
    )

    assert result["p_value"] < 1.0


def test_permutation_weights_unequal_clusters_by_row_count() -> None:
    result = paired_cluster_permutation(
        [0, 0, 0, 0], [1, 1, 1, -1], [0, 1, 1, 2], replicates=100, seed=42
    )

    assert result["p_value"] == pytest.approx(0.7524752475247525)


def test_permutation_uses_exact_finite_add_one_count() -> None:
    result = paired_cluster_permutation([0, 0, 0], [1, 0.1, 0.9], [0, 1, 2], replicates=1, seed=42)

    assert result["p_value"] == pytest.approx(0.5)


def test_permutation_zero_effect_has_finite_p_value() -> None:
    result = paired_cluster_permutation(
        [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], replicates=100, seed=42
    )

    assert result["p_value"] == 1.0


def test_permutation_rejects_same_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        paired_cluster_permutation([], [], [], replicates=10, seed=42)
    with pytest.raises((TypeError, ValueError)):
        paired_cluster_permutation([math.nan], [0], [1], replicates=10, seed=42)
    with pytest.raises((TypeError, ValueError)):
        paired_cluster_permutation([0], [0], ["bad"], replicates=cast(Any, 1.5), seed=42)


def test_holm_adjustment_is_monotone_and_restores_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm_adjust([0.04, 0.01, 0.03]) == [0.06, 0.03, 0.06]


def test_holm_adjustment_caps_and_handles_empty_input() -> None:
    assert holm_adjust([]) == []
    assert holm_adjust([0.0, 0.5, 1.0]) == [0.0, 1.0, 1.0]
    assert all(0.0 <= value <= 1.0 for value in holm_adjust([0.2, 0.8, 0.4]))


def test_holm_adjustment_rejects_invalid_p_values() -> None:
    for value in (-0.1, 1.1, math.nan, math.inf, True, "0.1"):
        with pytest.raises((TypeError, ValueError)):
            holm_adjust([0.1, cast(Any, value)])


def test_fit_temperature_returns_positive_finite_binary_calibration() -> None:
    temperature = fit_temperature(
        [0.9, 0.8, 0.2, 0.1],
        [0, 0, 1, 1],
    )

    assert math.isfinite(temperature)
    assert temperature > 0.0
    assert temperature > 1.0
    fitted = apply_temperature([0.9, 0.1], temperature)
    assert 0.1 < fitted[0] < 0.9
    assert 0.1 < fitted[1] < 0.9


def test_fit_temperature_is_input_order_invariant() -> None:
    forward = fit_temperature([0.1, 0.4, 0.8, 0.9], [0, 1, 0, 1])
    reverse = fit_temperature([0.9, 0.8, 0.4, 0.1], [1, 0, 1, 0])

    assert forward == pytest.approx(reverse)


def test_fit_temperature_finds_binary_nll_optimum() -> None:
    temperature = fit_temperature(
        [1 / (1 + math.exp(1)), 1 / (1 + math.exp(2))],
        [1, 0],
    )

    assert temperature == pytest.approx(2.383122074728365, rel=1e-10)


def test_fit_temperature_is_stable_at_extreme_probabilities() -> None:
    temperature = fit_temperature([0.0, 0.01, 0.99, 1.0], [0, 0, 1, 1])
    fitted = apply_temperature([0.0, 1e-300, 1.0 - 1e-15, 1.0], temperature)

    assert math.isfinite(temperature)
    assert temperature > 0.0
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fitted)
    assert fitted[0] == 0.0
    assert fitted[-1] == 1.0


def test_fit_temperature_rejects_invalid_one_class_and_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        fit_temperature([], [])
    with pytest.raises(ValueError):
        fit_temperature([0.2, 0.8], [0])
    with pytest.raises(ValueError):
        fit_temperature([0.5, 0.5], [0, 1])
    with pytest.raises(ValueError):
        fit_temperature([0.2, 0.8], [0, 2])
    with pytest.raises(ValueError):
        fit_temperature([0.2, 1.2], [0, 1])
    with pytest.raises((TypeError, ValueError)):
        fit_temperature([0.2, math.nan], [0, 1])


def test_apply_temperature_does_not_mutate_inputs() -> None:
    probabilities = [0.0, 0.2, 0.8, 1.0]
    original = probabilities.copy()

    fitted = apply_temperature(probabilities, 2.0)

    assert probabilities == original
    assert fitted[0] == 0.0
    assert fitted[-1] == 1.0
    assert 0.0 < fitted[1] < 0.5
    assert 0.5 < fitted[2] < 1.0


def test_apply_temperature_identity_preserves_probabilities() -> None:
    probabilities = [0.1, 0.2, 0.8, 0.9]

    assert apply_temperature(probabilities, 1.0) == probabilities


def test_apply_temperature_uses_binary_logit_transform() -> None:
    probability = 0.8
    temperature = 2.0
    expected = 1.0 / (1.0 + math.exp(-math.log(probability / (1.0 - probability)) / temperature))

    assert apply_temperature([probability], temperature)[0] == pytest.approx(expected)


def test_apply_temperature_rejects_nonpositive_or_invalid_temperature() -> None:
    for temperature in (0.0, -1.0, math.nan, math.inf, True, "2"):
        with pytest.raises((TypeError, ValueError)):
            apply_temperature([0.5], cast(Any, temperature))


def test_apply_temperature_rejects_empty_or_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        apply_temperature([], 1.0)
    with pytest.raises((TypeError, ValueError)):
        apply_temperature([math.inf], 1.0)
    with pytest.raises(ValueError):
        apply_temperature([1.1], 1.0)
