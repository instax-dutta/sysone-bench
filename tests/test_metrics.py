from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest

from benchmark import statistics
from benchmark.canonical import canonical_json
from benchmark.metrics import (
    decision_correct,
    macro_f1,
    score_correct,
    score_level,
    summarize_predictions,
)


def _row(
    *,
    expected: Mapping[str, Any],
    answers: Mapping[str, Any],
    questions: list[dict[str, Any]],
    split: str = "evaluation",
) -> dict[str, Any]:
    return {
        "answers": dict(answers),
        "case_id": "case-1",
        "expected": dict(expected),
        "execution_phase": "benchmark",
        "phase": split,
        "question_ids": [question["qid"] for question in questions],
        "questions": questions,
        "schema_version": 2,
        "split": split,
        "suite_id": "fixture",
    }


def _choice_question() -> dict[str, Any]:
    return {
        "qid": "choice",
        "type": "choice",
        "instructions": "Choose a label.",
        "criteria": {"a": "label a", "b": "label b"},
    }


def _noul_question() -> dict[str, Any]:
    return {
        "qid": "noul",
        "type": "noul",
        "instructions": "Answer true or false.",
    }


def _score_question() -> dict[str, Any]:
    return {
        "qid": "score",
        "type": "score",
        "instructions": "Give a score.",
        "criteria": ["0", "1", "2", "3", "4"],
        "max_score": 4,
    }


def _choice_rows() -> list[dict[str, Any]]:
    values = [
        ("a", "a", {"a": 0.9, "b": 0.1}),
        ("a", "b", {"a": 0.2, "b": 0.8}),
        ("b", "b", {"a": 0.7, "b": 0.3}),
        ("b", "a", {"a": 0.4, "b": 0.6}),
    ]
    rows: list[dict[str, Any]] = []
    for index, (expected, choice, probabilities) in enumerate(values):
        row = _row(
            expected={"choice": expected},
            answers={
                "choice": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": max(probabilities.values()),
                }
            },
            questions=[_choice_question()],
        )
        row["case_id"] = f"choice-{index}"
        rows.append(row)
    return rows


def _noul_rows() -> list[dict[str, Any]]:
    values = [(0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)]
    rows: list[dict[str, Any]] = []
    for index, (expected, probability) in enumerate(values):
        row = _row(
            expected={"noul": expected},
            answers={"noul": {"type": "noul", "noul": probability}},
            questions=[_noul_question()],
        )
        row["case_id"] = f"noul-{index}"
        rows.append(row)
    return rows


def _score_rows() -> list[dict[str, Any]]:
    values = [(0, 0.4), (2, 1.5), (2, 2.6), (3, 3.5)]
    rows: list[dict[str, Any]] = []
    for index, (expected, prediction) in enumerate(values):
        row = _row(
            expected={"score": expected},
            answers={"score": {"type": "score", "score": prediction, "confidence": None}},
            questions=[_score_question()],
        )
        row["case_id"] = f"score-{index}"
        rows.append(row)
    return rows


def test_choice_requires_exact_label() -> None:
    assert decision_correct("choice", "yes", {"type": "choice", "choice": "yes"})
    assert decision_correct("choice", "yes", {"choice": "yes"})
    assert not decision_correct("choice", "yes", {"type": "choice", "choice": "no"})


def test_noul_uses_half_threshold() -> None:
    assert decision_correct("noul", 1, {"type": "noul", "noul": 0.5})
    assert not decision_correct("noul", 1, {"type": "noul", "noul": 0.4999})


def test_score_uses_half_up_nearest_level() -> None:
    assert score_level(0.5) == 1
    assert score_level(1.5) == 2
    assert score_correct(2, 1.6)


def test_score_level_rejects_invalid_predictions() -> None:
    for prediction in (-0.1, math.nan, math.inf, -math.inf, True, "1.0"):
        with pytest.raises((TypeError, ValueError), match="score"):
            score_level(cast(Any, prediction))


def test_decision_correct_rejects_invalid_or_incoherent_values() -> None:
    cases = [
        ("choice", "yes", {"type": "choice"}),
        ("noul", 1, {"type": "noul", "noul": "0.5"}),
        ("noul", 1, {"type": "noul", "noul": math.nan}),
        ("score", 1, {"type": "score", "score": "1.0", "confidence": None}),
        ("unknown", "yes", {"type": "choice", "choice": "yes"}),
    ]
    for question_type, expected, answer in cases:
        with pytest.raises((TypeError, ValueError)):
            decision_correct(question_type, expected, cast(Mapping[str, Any], answer))


def test_macro_f1_averages_over_declared_labels() -> None:
    assert macro_f1(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        ["a", "b"],
    ) == pytest.approx(11 / 15)


def test_macro_f1_rejects_length_mismatch_and_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="length"):
        macro_f1(["a"], ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        macro_f1(["a"], ["a"], ["a", "a"])


def test_summarize_choice_metrics_are_golden() -> None:
    summary = summarize_predictions(_choice_rows())
    choice = summary["choice"]

    assert choice["accuracy"] == pytest.approx(0.5)
    assert choice["macro_f1"] == pytest.approx(0.5)
    assert choice["balanced_accuracy"] == pytest.approx(0.5)
    assert choice["per_class_recall"] == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}
    assert choice["brier"] == pytest.approx(0.65)
    assert choice["nll"] == pytest.approx(
        sum(-math.log(value) for value in (0.9, 0.2, 0.3, 0.6)) / 4
    )
    assert choice["ece_raw"] == pytest.approx(0.45)
    assert "ece_fitted" in choice


def test_summarize_noul_metrics_are_golden() -> None:
    summary = summarize_predictions(_noul_rows())
    noul = summary["noul"]

    assert noul["accuracy"] == pytest.approx(0.75)
    assert noul["balanced_accuracy"] == pytest.approx(0.75)
    assert noul["auroc"] == pytest.approx(0.75)
    assert noul["auprc"] == pytest.approx(5 / 6)
    assert noul["brier"] == pytest.approx(0.158125)
    assert noul["nll"] == pytest.approx(
        sum(-math.log(value) for value in (0.9, 0.6, 0.35, 0.8)) / 4
    )
    assert noul["ece_raw"] == pytest.approx(0.3375)
    assert "ece_fitted" in noul


def test_summarize_score_metrics_are_golden_and_exclude_probability_metrics() -> None:
    summary = summarize_predictions(_score_rows())
    score = summary["score"]

    assert score["mae"] == pytest.approx(0.5)
    assert score["rmse"] == pytest.approx(math.sqrt(0.255))
    assert score["nearest_level_accuracy"] == pytest.approx(0.5)
    assert score["within_one_accuracy"] == pytest.approx(1.0)
    assert score["raw_equality_rate"] == pytest.approx(0.0)
    assert score["spearman"] == pytest.approx(0.9486832980505138)
    assert score["ordinal_confusion"] == {
        "labels": [0, 1, 2, 3, 4],
        "matrix": [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
    }
    assert "ece_raw" not in score
    assert "ece_fitted" not in score
    assert "risk_coverage" not in score


def test_summarize_accepts_mixed_primitives_and_preserves_raw_decisions() -> None:
    questions = [_choice_question(), _noul_question(), _score_question()]
    row = _row(
        expected={"choice": "a", "noul": 1, "score": 1},
        answers={
            "choice": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.8, "b": 0.2},
                "confidence": 0.8,
            },
            "noul": {"type": "noul", "noul": 0.75},
            "score": {"type": "score", "score": 1.25, "confidence": None},
        },
        questions=questions,
    )

    summary = summarize_predictions([row])

    assert summary["choice"]["accuracy"] == 1.0
    assert summary["noul"]["accuracy"] == 1.0
    assert summary["score"]["nearest_level_accuracy"] == 1.0
    assert summary["raw"]["choice"][0]["answer"]["probabilities"] == {"a": 0.8, "b": 0.2}
    assert summary["raw"]["noul"][0]["answer"]["noul"] == 0.75
    assert summary["raw"]["score"][0]["answer"]["score"] == 1.25
    assert summary["raw"]["choice"][0]["case_id"] == "case-1"


def test_empty_summary_is_explicit_and_does_not_invent_metrics() -> None:
    summary = summarize_predictions([])

    assert summary["choice"]["accuracy"] is None
    assert summary["noul"]["auroc"] is None
    assert summary["score"]["mae"] is None
    assert summary["raw"] == {"choice": [], "noul": [], "score": []}


def test_missing_probability_is_not_coerced_to_zero() -> None:
    row = _row(
        expected={"choice": "a"},
        answers={"choice": {"type": "choice", "choice": "a", "confidence": 1.0}},
        questions=[_choice_question()],
    )

    with pytest.raises(ValueError, match="probabilit"):
        summarize_predictions([row])


def test_degenerate_probability_ranking_is_explicit() -> None:
    rows = [
        _row(
            expected={"noul": 0},
            answers={"noul": {"type": "noul", "noul": 0.1}},
            questions=[_noul_question()],
        ),
        _row(
            expected={"noul": 0},
            answers={"noul": {"type": "noul", "noul": 0.2}},
            questions=[_noul_question()],
        ),
    ]

    summary = summarize_predictions(rows)

    assert summary["noul"]["balanced_accuracy"] is None
    assert summary["noul"]["auroc"] is None
    assert summary["noul"]["auprc"] is None


def test_score_degenerate_spearman_is_explicit() -> None:
    rows = [
        _row(
            expected={"score": 0},
            answers={"score": {"type": "score", "score": 0.1, "confidence": None}},
            questions=[_score_question()],
        ),
        _row(
            expected={"score": 1},
            answers={"score": {"type": "score", "score": 0.1, "confidence": None}},
            questions=[_score_question()],
        ),
    ]

    summary = summarize_predictions(rows)

    assert summary["score"]["spearman"] is None


def test_summarize_uses_evaluation_rows_for_headline_score_metrics() -> None:
    calibration = _score_rows()[0]
    calibration["split"] = "calibration"
    calibration["phase"] = "calibration"
    evaluation = _score_rows()[1:]

    summary = summarize_predictions([calibration, *evaluation])

    assert summary["score"]["mae"] == pytest.approx((0.5 + 0.6 + 0.5) / 3)
    assert len(summary["raw"]["score"]) == 4


def test_summarize_fits_noul_ece_on_calibration_rows() -> None:
    noul_rows = _noul_rows()
    calibration = [noul_rows[0], noul_rows[2]]
    for row in calibration:
        row["split"] = "calibration"
        row["phase"] = "calibration"
    evaluation = [noul_rows[1], noul_rows[3]]

    summary = summarize_predictions([*calibration, *evaluation])

    assert summary["noul"]["ece_fitted"] is not None
    assert 0.0 <= summary["noul"]["ece_fitted"] <= 1.0


def test_summarize_uses_shared_noul_calibration_only_on_split_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noul_rows = _noul_rows()
    calibration = [noul_rows[0], noul_rows[2]]
    for row in calibration:
        row["split"] = "calibration"
        row["phase"] = "calibration"
    evaluation = [noul_rows[1], noul_rows[3]]
    fit_calls: list[tuple[list[float], list[int]]] = []
    apply_calls: list[tuple[list[float], float]] = []

    def fake_fit(probabilities: Sequence[float], labels: Sequence[int]) -> float:
        fit_calls.append((list(probabilities), list(labels)))
        return 2.0

    def fake_apply(probabilities: Sequence[float], temperature: float) -> list[float]:
        apply_calls.append((list(probabilities), temperature))
        return [0.25, 0.75]

    monkeypatch.setattr(statistics, "fit_temperature", fake_fit)
    monkeypatch.setattr(statistics, "apply_temperature", fake_apply)

    summary = summarize_predictions([*calibration, *evaluation])
    raw = summary["raw"]["noul"]

    assert fit_calls == [([0.1, 0.35], [0, 1])]
    assert apply_calls == [([0.4, 0.8], 2.0)]
    assert raw[0]["raw_probability"] == 0.1
    assert raw[0]["probability"] == 0.1
    assert raw[0]["fitted_probability"] is None
    assert raw[0]["temperature"] == 2.0
    assert raw[1]["raw_probability"] == 0.35
    assert raw[1]["fitted_probability"] is None
    assert raw[2]["raw_probability"] == 0.4
    assert raw[2]["fitted_probability"] == 0.25
    assert raw[2]["temperature"] == 2.0
    assert raw[3]["raw_probability"] == 0.8
    assert raw[3]["fitted_probability"] == 0.75
    assert summary["noul"]["ece_fitted"] == pytest.approx(0.25)
    canonical_json(summary)


def test_summarize_rejects_score_without_declared_range() -> None:
    question = _score_question()
    del question["max_score"]
    row = _row(
        expected={"score": 1},
        answers={"score": {"type": "score", "score": 1.0, "confidence": None}},
        questions=[question],
    )

    with pytest.raises(ValueError, match="max_score"):
        summarize_predictions([row])


def test_summarize_rejects_invalid_score_confidence() -> None:
    row = _row(
        expected={"score": 1},
        answers={"score": {"type": "score", "score": 1.0, "confidence": "high"}},
        questions=[_score_question()],
    )

    with pytest.raises((TypeError, ValueError), match="confidence"):
        summarize_predictions([row])


def test_summarize_rejects_choice_without_declared_criteria() -> None:
    question = _choice_question()
    del question["criteria"]
    row = _row(
        expected={"choice": "a"},
        answers={
            "choice": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 1.0},
                "confidence": 1.0,
            }
        },
        questions=[question],
    )

    with pytest.raises(ValueError, match="criteria"):
        summarize_predictions([row])


@pytest.mark.parametrize("confidence", [None, math.nan, math.inf, -math.inf, "high"])
def test_summarize_requires_finite_choice_confidence(confidence: object) -> None:
    row = _row(
        expected={"choice": "a"},
        answers={
            "choice": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 1, "b": 0},
                "confidence": confidence,
            }
        },
        questions=[_choice_question()],
    )

    with pytest.raises((TypeError, ValueError), match="confidence"):
        summarize_predictions([row])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_phase", None),
        ("phase", None),
        ("phase", "calibration"),
    ],
)
def test_summarize_requires_v2_phase_identity(field: str, value: str | None) -> None:
    row = _row(
        expected={"noul": 1},
        answers={"noul": {"type": "noul", "noul": 0.75}},
        questions=[_noul_question()],
    )
    if value is None:
        del row[field]
    else:
        row[field] = value

    with pytest.raises(ValueError, match="phase"):
        summarize_predictions([row])


def test_zero_probability_nll_is_unavailable_and_summary_is_canonical() -> None:
    noul_rows = [
        _row(
            expected={"noul": 0},
            answers={"noul": {"type": "noul", "noul": 0}},
            questions=[_noul_question()],
        ),
        _row(
            expected={"noul": 1},
            answers={"noul": {"type": "noul", "noul": 0}},
            questions=[_noul_question()],
        ),
    ]
    choice_row = _row(
        expected={"choice": "a"},
        answers={
            "choice": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0, "b": 1},
                "confidence": 1,
            }
        },
        questions=[_choice_question()],
    )

    noul_summary = summarize_predictions(noul_rows)
    choice_summary = summarize_predictions([choice_row])

    assert noul_summary["noul"]["nll"] is None
    assert choice_summary["choice"]["nll"] is None
    assert noul_summary["raw"]["noul"][0]["probability"] == 0
    assert type(choice_summary["raw"]["choice"][0]["probabilities"]["a"]) is int
    canonical_json(noul_summary)
    canonical_json(choice_summary)


def test_auprc_ties_are_order_invariant() -> None:
    rows = [
        _row(
            expected={"noul": 1},
            answers={"noul": {"type": "noul", "noul": 0.5}},
            questions=[_noul_question()],
        ),
        _row(
            expected={"noul": 0},
            answers={"noul": {"type": "noul", "noul": 0.5}},
            questions=[_noul_question()],
        ),
    ]

    forward = summarize_predictions(rows)["noul"]["auprc"]
    reverse = summarize_predictions(list(reversed(rows)))["noul"]["auprc"]

    assert forward == pytest.approx(0.5)
    assert reverse == pytest.approx(0.5)


@pytest.mark.parametrize("labels", [(0, 0), (1, 1)])
def test_auprc_is_none_for_one_class_cases(labels: tuple[int, int]) -> None:
    rows = [
        _row(
            expected={"noul": label},
            answers={"noul": {"type": "noul", "noul": 0.2 if label == 0 else 0.8}},
            questions=[_noul_question()],
        )
        for label in labels
    ]

    assert summarize_predictions(rows)["noul"]["auprc"] is None


def test_one_class_noul_calibration_has_no_fitted_ece() -> None:
    calibration = _noul_rows()[:2]
    for row in calibration:
        row["split"] = "calibration"
        row["phase"] = "calibration"
    evaluation = _noul_rows()[2:]

    summary = summarize_predictions([*calibration, *evaluation])

    assert summary["noul"]["ece_fitted"] is None


def test_one_class_choice_calibration_has_no_fitted_ece() -> None:
    calibration = _choice_rows()[:2]
    for row in calibration:
        row["split"] = "calibration"
        row["phase"] = "calibration"
    evaluation = _choice_rows()[2:]

    summary = summarize_predictions([*calibration, *evaluation])

    assert summary["choice"]["ece_fitted"] is None


def test_raw_score_inputs_preserve_numeric_forms() -> None:
    question = _score_question()
    question["max_score"] = 4.0
    row = _row(
        expected={"score": 2.0},
        answers={"score": {"type": "score", "score": 1, "confidence": None}},
        questions=[question],
    )

    raw = summarize_predictions([row])["raw"]["score"][0]

    assert raw["expected"] == 2.0
    assert type(raw["expected"]) is float
    assert raw["prediction"] == 1
    assert type(raw["prediction"]) is int
    assert raw["max_score"] == 4.0
    assert type(raw["max_score"]) is float
    assert raw["answer"]["score"] == 1


def test_metric_summaries_are_row_order_deterministic() -> None:
    choice_cases = [
        ("b", 0.0001),
        ("b", 0.9),
        ("a", 0.8),
        ("b", 0.3),
        ("b", 0.7),
        ("a", 0.9),
        ("a", 0.3),
        ("b", 0.99),
        ("a", 0.8),
        ("a", 0.8),
        ("b", 0.8),
        ("a", 0.9),
        ("a", 0.5),
        ("b", 0.9),
        ("a", 0.9),
        ("b", 0.1),
        ("a", 0.3),
        ("b", 0.9),
        ("a", 0.7),
        ("a", 0.5),
    ]
    noul_cases = [
        (0, 0.1),
        (1, 0.5),
        (1, 0.2),
        (0, 0.3),
        (0, 0.1),
        (1, 0.5),
        (0, 0.2),
        (1, 0.3),
        (1, 0.4),
        (0, 0.4),
    ] * 2
    score_cases = [
        (0, 0.0),
        (4, 3.999999),
        (1, 0.000001),
        (3, 2.999999),
        (2, 1.5),
    ] * 4
    questions = [_choice_question(), _noul_question(), _score_question()]
    rows: list[dict[str, Any]] = []
    for index, (choice_case, noul_case, score_case) in enumerate(
        zip(choice_cases, noul_cases, score_cases, strict=True)
    ):
        expected_choice, choice_probability = choice_case
        expected_noul, noul_probability = noul_case
        expected_score, score_prediction = score_case
        split = "calibration" if index % 5 in (0, 3) else "evaluation"
        choice_prediction = (
            expected_choice if index % 4 else ("b" if expected_choice == "a" else "a")
        )
        row = _row(
            expected={
                "choice": expected_choice,
                "noul": expected_noul,
                "score": expected_score,
            },
            answers={
                "choice": {
                    "type": "choice",
                    "choice": choice_prediction,
                    "probabilities": {
                        "a": choice_probability,
                        "b": 1.0 - choice_probability,
                    },
                    "confidence": max(choice_probability, 1.0 - choice_probability),
                },
                "noul": {"type": "noul", "noul": noul_probability},
                "score": {
                    "type": "score",
                    "score": score_prediction,
                    "confidence": None,
                },
            },
            questions=questions,
            split=split,
        )
        row["case_id"] = f"mixed-{index}"
        rows.append(row)

    forward = summarize_predictions(rows)
    reverse = summarize_predictions(list(reversed(rows)))

    for key in ("choice", "noul", "score"):
        assert canonical_json(forward[key]) == canonical_json(reverse[key])
