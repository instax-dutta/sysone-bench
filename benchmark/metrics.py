from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from benchmark import statistics
from benchmark.canonical import canonical_json

_CHOICE_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "per_class_recall",
    "brier",
    "nll",
    "ece_raw",
    "ece_fitted",
)
_NOUL_METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "brier",
    "nll",
    "ece_raw",
    "ece_fitted",
)
_SCORE_METRIC_KEYS = (
    "mae",
    "rmse",
    "nearest_level_accuracy",
    "within_one_accuracy",
    "raw_equality_rate",
    "spearman",
    "ordinal_confusion",
)
_CHOICE_ANSWER_FIELDS = frozenset({"type", "choice", "probabilities", "confidence"})
_NOUL_ANSWER_FIELDS = frozenset({"type", "noul"})
_SCORE_ANSWER_FIELDS = frozenset({"type", "score", "confidence"})


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: object, name: str) -> float:
    if not _is_number(value):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(cast(int | float, value))
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _integer_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer")
    number = _finite_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _probability(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, Any] = {}
    for key, child in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        result[key] = child
    return result


def _sequence(value: object, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _question_type(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in {"choice", "noul", "score"}:
        raise ValueError(f"unsupported question type: {value}")
    return value


def _check_answer_type(answer: Mapping[str, Any], question_type: str) -> None:
    answer_type = answer.get("type")
    if not isinstance(answer_type, str):
        raise TypeError("answer type is missing")
    if answer_type != question_type:
        raise ValueError("answer type does not match question type")


def _validate_answer_fields(answer: Mapping[str, Any], question_type: str) -> None:
    if question_type == "choice":
        required = _CHOICE_ANSWER_FIELDS
    elif question_type == "noul":
        required = _NOUL_ANSWER_FIELDS
    else:
        required = _SCORE_ANSWER_FIELDS
    fields = set(answer)
    unknown = fields - required
    missing = required - fields
    if unknown:
        raise ValueError("answer has unsupported fields: " + ", ".join(sorted(unknown)))
    if missing:
        raise ValueError("answer has missing fields: " + ", ".join(sorted(missing)))


def _choice_label_order(labels: Sequence[Any], criteria: object) -> list[str]:
    if criteria is not None:
        criteria_map = _require_mapping(criteria, "choice criteria")
        if not criteria_map:
            raise ValueError("choice criteria must not be empty")
        result: list[str] = []
        for label, description in criteria_map.items():
            if not isinstance(label, str) or not label:
                raise ValueError("choice labels must be non-empty strings")
            if not isinstance(description, str):
                raise TypeError("choice criteria descriptions must be strings")
            result.append(label)
        return result
    try:
        ordered = sorted(labels, key=lambda value: (type(value).__name__, repr(value)))
    except TypeError as error:
        raise TypeError("choice labels must be hashable and sortable") from error
    result = []
    for label in ordered:
        if not isinstance(label, str) or not label:
            raise ValueError("choice labels must be non-empty strings")
        result.append(label)
    return result


def _probability_map(value: object, labels: Sequence[str], name: str) -> dict[str, float]:
    probabilities = _require_mapping(value, name)
    if set(probabilities) != set(labels):
        raise ValueError(f"{name} labels do not match choices")
    result: dict[str, float] = {}
    values: list[float] = []
    for label in labels:
        probability = _probability(probabilities[label], f"{name}[{label!r}]")
        result[label] = probability
        values.append(probability)
    if abs(math.fsum(values) - 1.0) > 1e-6:
        raise ValueError(f"{name} must sum to one")
    return result


def _expected_choice(value: object, labels: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise TypeError("choice expected value must be a string")
    if value not in labels:
        raise ValueError("choice expected value must be a legal label")
    return value


def _expected_noul(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("noul expected value must be integer 0 or 1")
    return value


def _score_bound(value: object, name: str) -> int:
    return _integer_value(value, name)


def score_level(prediction: float) -> int:
    try:
        number = _finite_number(prediction, "score")
    except (TypeError, ValueError) as error:
        raise ValueError("score must be finite and nonnegative") from error
    if number < 0.0:
        raise ValueError("score must be finite and nonnegative")
    return math.floor(number + 0.5)


def score_correct(expected: int, prediction: float) -> bool:
    expected_level = _integer_value(expected, "expected score")
    if expected_level < 0:
        raise ValueError("expected score must be nonnegative")
    return score_level(prediction) == expected_level


def decision_correct(question_type: str, expected: Any, answer: Mapping[str, Any]) -> bool:
    selected_type = _question_type(question_type, "question type")
    answer_object = _require_mapping(answer, "answer")
    if "type" in answer_object:
        _check_answer_type(answer_object, selected_type)
    if selected_type == "choice":
        expected_label = expected if isinstance(expected, str) else None
        if expected_label is None:
            raise TypeError("choice expected value must be a string")
        actual_label = answer_object.get("choice")
        if not isinstance(actual_label, str):
            raise TypeError("choice answer must contain a string label")
        return actual_label == expected_label
    if selected_type == "noul":
        expected_value = _expected_noul(expected)
        probability = _probability(answer_object.get("noul"), "noul probability")
        return (probability >= 0.5) == (expected_value == 1)
    expected_level = _integer_value(expected, "expected score")
    if expected_level < 0:
        raise ValueError("expected score must be nonnegative")
    return score_correct(expected_level, cast(float, answer_object.get("score")))


def _hashable_labels(labels: Sequence[Any]) -> set[Any]:
    try:
        result = set(labels)
    except TypeError as error:
        raise TypeError("macro-F1 labels must be hashable") from error
    if len(result) != len(labels):
        raise ValueError("macro-F1 labels must be unique")
    return result


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float:
    true_values = _sequence(y_true, "y_true")
    pred_values = _sequence(y_pred, "y_pred")
    label_values = _sequence(labels, "labels")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred length must match")
    if not true_values:
        raise ValueError("macro-F1 requires at least one observation")
    if not label_values:
        raise ValueError("macro-F1 requires at least one label")
    label_set = _hashable_labels(label_values)
    for value in true_values:
        if value not in label_set:
            raise ValueError("y_true contains a value outside labels")
    for value in pred_values:
        if value not in label_set:
            raise ValueError("y_pred contains a value outside labels")
    true_counts = Counter(true_values)
    pred_counts = Counter(pred_values)
    true_positive = Counter(
        value
        for value, prediction in zip(true_values, pred_values, strict=True)
        if value == prediction
    )
    scores: list[float] = []
    for label in label_values:
        true_positive_count = true_positive[label]
        false_positive = pred_counts[label] - true_positive_count
        false_negative = true_counts[label] - true_positive_count
        denominator = 2 * true_positive_count + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive_count / denominator)
    return float(math.fsum(scores) / len(scores))


def _mean(values: Sequence[float]) -> float | None:
    if not values or any(not math.isfinite(value) for value in values):
        return None
    return float(math.fsum(sorted(values)) / len(values))


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positive_count = sum(label == 1 for label in labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    ordered = sorted(
        range(len(probabilities)), key=lambda index: probabilities[index], reverse=True
    )
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area_terms: list[float] = []
    start = 0
    while start < len(ordered):
        threshold = probabilities[ordered[start]]
        end = start + 1
        while end < len(ordered) and probabilities[ordered[end]] == threshold:
            end += 1
        for index in ordered[start:end]:
            if labels[index] == 1:
                true_positive += 1
            else:
                false_positive += 1
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        area_terms.append((recall - previous_recall) * precision)
        previous_recall = recall
        start = end
    return float(math.fsum(area_terms))


def _binary_nll(probability: float, label: int) -> float:
    positive = probability if label == 1 else 1.0 - probability
    return math.inf if positive == 0.0 else -math.log(positive)


def _choice_nll(probabilities: Mapping[str, float], expected: str) -> float:
    probability = probabilities[expected]
    return math.inf if probability == 0.0 else -math.log(probability)


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = values[order[start]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(expected: Sequence[float], predicted: Sequence[float]) -> float | None:
    if len(expected) != len(predicted) or not expected:
        return None
    expected_rank = _rank(expected)
    predicted_rank = _rank(predicted)
    expected_mean = math.fsum(sorted(expected_rank)) / len(expected_rank)
    predicted_mean = math.fsum(sorted(predicted_rank)) / len(predicted_rank)
    expected_deviation = [value - expected_mean for value in expected_rank]
    predicted_deviation = [value - predicted_mean for value in predicted_rank]
    denominator = math.sqrt(
        math.fsum(sorted(value * value for value in expected_deviation))
        * math.fsum(sorted(value * value for value in predicted_deviation))
    )
    if denominator == 0.0:
        return None
    return float(
        math.fsum(
            sorted(
                left * right
                for left, right in zip(expected_deviation, predicted_deviation, strict=True)
            )
        )
        / denominator
    )


def _ece(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    confidence_values: list[list[float]] = [[] for _ in range(bins)]
    correct_values: list[list[float]] = [[] for _ in range(bins)]
    for confidence, correct in sorted(pairs, key=lambda pair: (pair[0], pair[1])):
        index = min(int(confidence * bins), bins - 1)
        confidence_values[index].append(confidence)
        correct_values[index].append(float(correct))
    total = len(pairs)
    return float(
        math.fsum(
            abs(
                math.fsum(confidence_values[index]) / len(confidence_values[index])
                - math.fsum(correct_values[index]) / len(correct_values[index])
            )
            for index in range(bins)
            if confidence_values[index]
        )
        / total
    )


def _fit_choice_temperature(items: Sequence[Mapping[str, Any]]) -> float:
    if not items:
        return 1.0
    values = [
        (cast(dict[str, float], item["probabilities"]), cast(str, item["expected"]))
        for item in items
    ]
    values.sort(key=lambda value: (value[1], tuple(sorted(value[0].items()))))
    if all(len(probabilities) < 2 for probabilities, _ in values):
        return 1.0

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        terms: list[float] = []
        for probabilities, expected in values:
            labels = sorted(probabilities)
            logits = [
                math.log(probabilities[label]) if probabilities[label] > 0.0 else -745.0
                for label in labels
            ]
            maximum = max(logits)
            exponentials = [
                math.exp(value / temperature - maximum / temperature) for value in logits
            ]
            denominator = math.fsum(exponentials)
            expected_probability = exponentials[labels.index(expected)] / denominator
            terms.append(
                math.inf if expected_probability == 0.0 else -math.log(expected_probability)
            )
        return math.fsum(sorted(terms))

    left = -5.0
    right = 5.0
    for _ in range(80):
        first = (2.0 * left + right) / 3.0
        second = (left + 2.0 * right) / 3.0
        if objective(first) <= objective(second):
            right = second
        else:
            left = first
    return math.exp((left + right) / 2.0)


def _choice_fitted_confidence(
    probabilities: Mapping[str, float], temperature: float
) -> tuple[dict[str, float], float]:
    labels = sorted(probabilities)
    logits = [
        math.log(probabilities[label]) if probabilities[label] > 0.0 else -745.0 for label in labels
    ]
    maximum = max(logits)
    exponentials = [math.exp(value / temperature - maximum / temperature) for value in logits]
    denominator = math.fsum(exponentials)
    fitted = {
        label: exponential / denominator
        for label, exponential in zip(labels, exponentials, strict=True)
    }
    predicted = max(labels, key=lambda label: (fitted[label], label))
    return fitted, fitted[predicted]


def _choice_items_for_metrics(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [item for item in items if item["split"] == "evaluation"]


def _noul_items_for_metrics(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [item for item in items if item["split"] == "evaluation"]


def _score_items_for_metrics(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [item for item in items if item["split"] == "evaluation"]


def _choice_summary(
    items: Sequence[Mapping[str, Any]],
    calibration_items: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    evaluation_items = [item for item in items if item["split"] == "evaluation"]
    if not evaluation_items:
        return {key: None for key in _CHOICE_METRIC_KEYS}
    labels = sorted(
        {
            label
            for item in evaluation_items
            for label in cast(dict[str, float], item["probabilities"])
        }
    )
    true_values = [cast(str, item["expected"]) for item in evaluation_items]
    predicted_values = [cast(str, item["prediction"]) for item in evaluation_items]
    true_counts = Counter(true_values)
    recalls: dict[str, float | None] = {}
    for label in labels:
        support = true_counts[label]
        recalls[label] = (
            None
            if support == 0
            else _true_positive_count(label, true_values, predicted_values) / support
        )
    supported_recalls = [value for value in recalls.values() if value is not None]
    brier_values: list[float] = []
    nll_values: list[float] = []
    raw_pairs: list[tuple[float, bool]] = []
    for item in evaluation_items:
        probabilities = cast(dict[str, float], item["probabilities"])
        expected = cast(str, item["expected"])
        brier_values.append(
            math.fsum(
                (probabilities[label] - float(label == expected)) ** 2
                for label in sorted(probabilities)
            )
        )
        nll_values.append(_choice_nll(probabilities, expected))
        raw_pairs.append((cast(float, item["confidence"]), cast(bool, item["correct"])))
    calibration_source = items if calibration_items is None else calibration_items
    fit_items = [item for item in calibration_source if item["split"] == "calibration"]
    if fit_items and len({cast(str, item["expected"]) for item in fit_items}) >= 2:
        temperature = _fit_choice_temperature(fit_items)
        fitted_pairs: list[tuple[float, bool]] = []
        for item in evaluation_items:
            _, confidence = _choice_fitted_confidence(
                cast(dict[str, float], item["probabilities"]), temperature
            )
            fitted_pairs.append((confidence, cast(bool, item["correct"])))
        fitted_ece = _ece(fitted_pairs)
    else:
        fitted_ece = None
    return {
        "accuracy": _mean([float(cast(bool, item["correct"])) for item in evaluation_items]),
        "macro_f1": macro_f1(true_values, predicted_values, labels),
        "balanced_accuracy": (
            None
            if not supported_recalls
            else float(math.fsum(sorted(supported_recalls)) / len(supported_recalls))
        ),
        "per_class_recall": recalls,
        "brier": _mean(brier_values),
        "nll": _mean(nll_values),
        "ece_raw": _ece(raw_pairs),
        "ece_fitted": fitted_ece,
    }


def _true_positive_count(
    label: str, true_values: Sequence[str], predicted_values: Sequence[str]
) -> int:
    return sum(
        true == label and predicted == label
        for true, predicted in zip(true_values, predicted_values, strict=True)
    )


def _fit_noul_calibration(
    fit_items: Sequence[Mapping[str, Any]],
    evaluation_items: Sequence[Mapping[str, Any]],
) -> tuple[float | None, list[float] | None]:
    if not fit_items:
        return None, None
    fit_probabilities = [cast(float, item["probability"]) for item in fit_items]
    fit_labels = [cast(int, item["expected"]) for item in fit_items]
    if len(set(fit_labels)) < 2 or len(set(fit_probabilities)) < 2:
        return None, None
    try:
        temperature = statistics.fit_temperature(fit_probabilities, fit_labels)
        fitted_probabilities = (
            statistics.apply_temperature(
                [cast(float, item["probability"]) for item in evaluation_items],
                temperature,
            )
            if evaluation_items
            else []
        )
    except (TypeError, ValueError, OverflowError):
        return None, None
    return temperature, fitted_probabilities


def _noul_summary(
    items: Sequence[Mapping[str, Any]],
    calibration_items: Sequence[Mapping[str, Any]] | None = None,
    calibration_result: tuple[float | None, list[float] | None] | None = None,
) -> dict[str, Any]:
    evaluation_items = [item for item in items if item["split"] == "evaluation"]
    labels = [cast(int, item["expected"]) for item in evaluation_items]
    predictions = [cast(int, item["prediction"]) for item in evaluation_items]
    probabilities = [cast(float, item["probability"]) for item in evaluation_items]
    positive_count = sum(label == 1 for label in labels)
    negative_count = len(labels) - positive_count
    recalls: list[float] = []
    if positive_count:
        recalls.append(
            sum(
                label == 1 and prediction == 1
                for label, prediction in zip(labels, predictions, strict=True)
            )
            / positive_count
        )
    if negative_count:
        recalls.append(
            sum(
                label == 0 and prediction == 0
                for label, prediction in zip(labels, predictions, strict=True)
            )
            / negative_count
        )
    if positive_count and negative_count:
        ordered = sorted(
            zip(probabilities, labels, strict=True), key=lambda pair: (pair[0], pair[1])
        )
        positive_probabilities = [probability for probability, label in ordered if label == 1]
        negative_probabilities = [probability for probability, label in ordered if label == 0]
        pair_scores = [
            1.0 if probability > other else 0.5 if probability == other else 0.0
            for probability in positive_probabilities
            for other in negative_probabilities
        ]
        auroc = math.fsum(pair_scores) / (positive_count * negative_count)
    else:
        auroc = None
    auprc = _average_precision(labels, probabilities)
    raw_pairs = [
        (probability, label == 1) for probability, label in zip(probabilities, labels, strict=True)
    ]
    if calibration_result is None:
        calibration_source = items if calibration_items is None else calibration_items
        fit_items = [item for item in calibration_source if item["split"] == "calibration"]
        calibration_result = _fit_noul_calibration(fit_items, evaluation_items)
    _, fitted_probabilities = calibration_result
    if fitted_probabilities is not None and len(fitted_probabilities) == len(evaluation_items):
        fitted_pairs = [
            (fitted_probability, cast(int, item["expected"]) == 1)
            for fitted_probability, item in zip(fitted_probabilities, evaluation_items, strict=True)
        ]
        fitted_ece = _ece(fitted_pairs)
    else:
        fitted_ece = None
    return {
        "accuracy": _mean(
            [
                float(correct)
                for correct in (
                    label == prediction
                    for label, prediction in zip(labels, predictions, strict=True)
                )
            ]
        ),
        "balanced_accuracy": (
            None
            if not positive_count or not negative_count
            else float(math.fsum(sorted(recalls)) / len(recalls))
        ),
        "auroc": auroc,
        "auprc": auprc,
        "brier": _mean(
            [
                (probability - label) ** 2
                for probability, label in zip(probabilities, labels, strict=True)
            ]
        ),
        "nll": _mean(
            [
                _binary_nll(probability, label)
                for probability, label in zip(probabilities, labels, strict=True)
            ]
        ),
        "ece_raw": _ece(raw_pairs),
        "ece_fitted": fitted_ece,
    }


def _score_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        return {key: None for key in _SCORE_METRIC_KEYS}
    expected_values = [cast(int, item["expected"]) for item in items]
    predictions = [cast(float, item["prediction"]) for item in items]
    levels = [cast(int, item["rounded_level"]) for item in items]
    errors = [
        prediction - expected
        for prediction, expected in zip(predictions, expected_values, strict=True)
    ]
    maximum = max(
        [
            *expected_values,
            *levels,
            *[cast(int, item["max_score"]) for item in items if item["max_score"] is not None],
        ]
    )
    labels = list(range(maximum + 1))
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, level in zip(expected_values, levels, strict=True):
        matrix[expected][level] += 1
    squared_error_mean = _mean([error * error for error in errors])
    return {
        "mae": _mean([abs(error) for error in errors]),
        "rmse": None if squared_error_mean is None else squared_error_mean**0.5,
        "nearest_level_accuracy": _mean(
            [
                float(expected == level)
                for expected, level in zip(expected_values, levels, strict=True)
            ]
        ),
        "within_one_accuracy": _mean(
            [
                float(abs(expected - level) <= 1)
                for expected, level in zip(expected_values, levels, strict=True)
            ]
        ),
        "raw_equality_rate": _mean(
            [
                float(prediction == expected)
                for prediction, expected in zip(predictions, expected_values, strict=True)
            ]
        ),
        "spearman": _spearman([float(value) for value in expected_values], predictions),
        "ordinal_confusion": {"labels": labels, "matrix": matrix},
    }


def _empty_raw() -> dict[str, list[dict[str, Any]]]:
    return {"choice": [], "noul": [], "score": []}


def _annotate_noul_raw(
    items: Sequence[Mapping[str, Any]],
    records: list[dict[str, Any]],
    calibration_result: tuple[float | None, list[float] | None],
) -> None:
    temperature, fitted_probabilities = calibration_result
    evaluation_indices = [
        index for index, item in enumerate(items) if item["split"] == "evaluation"
    ]
    for record in records:
        record["raw_probability"] = deepcopy(record["probability"])
        record["fitted_probability"] = None
        record["temperature"] = temperature
    if fitted_probabilities is not None:
        for index, fitted_probability in zip(evaluation_indices, fitted_probabilities, strict=True):
            records[index]["fitted_probability"] = fitted_probability


def _row_question_values(
    row: Mapping[str, Any], row_index: int
) -> tuple[list[Mapping[str, Any]], list[str], Mapping[str, Any], Mapping[str, Any], str]:
    questions_value = row.get("questions")
    if questions_value is None:
        raise ValueError(f"row {row_index}: questions are missing")
    questions_raw = _sequence(questions_value, f"row {row_index} questions")
    questions: list[Mapping[str, Any]] = []
    for question_index, question in enumerate(questions_raw):
        question_object = _require_mapping(question, f"row {row_index} question {question_index}")
        qid = question_object.get("qid")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"row {row_index}: question {question_index} has no qid")
        _question_type(question_object.get("type"), f"row {row_index} question {qid} type")
        questions.append(question_object)
    question_ids_value = row.get("question_ids")
    if question_ids_value is None:
        raise ValueError(f"row {row_index}: question_ids are missing")
    question_ids = _sequence(question_ids_value, f"row {row_index} question_ids")
    if any(not isinstance(qid, str) or not qid for qid in question_ids):
        raise ValueError(f"row {row_index}: question IDs must be non-empty strings")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError(f"row {row_index}: question IDs must be unique")
    if [cast(str, question["qid"]) for question in questions] != question_ids:
        raise ValueError(f"row {row_index}: question definition order is invalid")
    expected_value = row.get("expected")
    answers_value = row.get("answers")
    if expected_value is None or answers_value is None:
        raise ValueError(f"row {row_index}: expected and answers are required")
    expected = _require_mapping(expected_value, f"row {row_index} expected")
    answers = _require_mapping(answers_value, f"row {row_index} answers")
    expected_ids = set(expected)
    answer_ids = set(answers)
    question_id_set = set(question_ids)
    if expected_ids != question_id_set or answer_ids != question_id_set:
        raise ValueError(f"row {row_index}: question IDs differ")
    split_value = row.get("split")
    if not isinstance(split_value, str) or split_value not in {"calibration", "evaluation"}:
        raise ValueError(f"row {row_index}: split must be calibration or evaluation")
    if row.get("execution_phase") != "benchmark":
        raise ValueError(f"row {row_index}: execution phase must be benchmark")
    if row.get("phase") != split_value:
        raise ValueError(f"row {row_index}: phase and split differ")
    return questions, cast(list[str], question_ids), expected, answers, split_value


def _extract_choice(
    question: Mapping[str, Any], expected: Any, answer: Mapping[str, Any]
) -> dict[str, Any]:
    question_type = _question_type(question.get("type"), "choice question type")
    _check_answer_type(answer, question_type)
    _validate_answer_fields(answer, question_type)
    criteria = question.get("criteria")
    if criteria is None:
        raise ValueError("choice criteria are missing")
    labels = _choice_label_order([], criteria)
    expected_label = _expected_choice(expected, labels)
    actual_label = answer.get("choice")
    if not isinstance(actual_label, str) or actual_label not in labels:
        raise ValueError("choice answer must contain a legal label")
    probabilities_value = answer.get("probabilities")
    if probabilities_value is None:
        raise ValueError("choice probabilities are missing")
    probabilities = _probability_map(probabilities_value, labels, "choice probabilities")
    confidence_value = answer.get("confidence")
    _probability(confidence_value, "choice confidence")
    return {
        "expected": expected_label,
        "prediction": actual_label,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
        "correct": actual_label == expected_label,
    }


def _extract_noul(expected: Any, answer: Mapping[str, Any]) -> dict[str, Any]:
    _check_answer_type(answer, "noul")
    _validate_answer_fields(answer, "noul")
    expected_value = _expected_noul(expected)
    probability = _probability(answer.get("noul"), "noul probability")
    prediction = int(probability >= 0.5)
    return {
        "expected": expected_value,
        "prediction": prediction,
        "probability": probability,
        "correct": prediction == expected_value,
    }


def _extract_score(
    question: Mapping[str, Any], expected: Any, answer: Mapping[str, Any]
) -> dict[str, Any]:
    _check_answer_type(answer, "score")
    _validate_answer_fields(answer, "score")
    expected_value = _integer_value(expected, "expected score")
    if expected_value < 0:
        raise ValueError("expected score must be nonnegative")
    maximum_value = question.get("max_score")
    if maximum_value is None:
        raise ValueError("max_score is missing")
    maximum: int | None
    maximum = _score_bound(maximum_value, "max_score")
    if maximum < 0:
        raise ValueError("max_score must be nonnegative")
    prediction = _finite_number(answer.get("score"), "score")
    if prediction < 0.0:
        raise ValueError("score must be finite and nonnegative")
    if maximum is not None and prediction > maximum:
        raise ValueError("score must be inside the declared range")
    if maximum is not None and expected_value > maximum:
        raise ValueError("expected score must be inside the declared range")
    confidence_value = answer.get("confidence")
    if confidence_value is not None:
        _probability(confidence_value, "score confidence")
    level = score_level(prediction)
    return {
        "expected": expected_value,
        "prediction": prediction,
        "rounded_level": level,
        "max_score": maximum,
        "correct": level == expected_value,
    }


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = _sequence(rows, "rows")
    choice_items: list[dict[str, Any]] = []
    noul_items: list[dict[str, Any]] = []
    score_items: list[dict[str, Any]] = []
    raw = _empty_raw()
    for row_index, row_value in enumerate(materialized):
        row = _require_mapping(row_value, f"row {row_index}")
        questions, question_ids, expected, answers, split = _row_question_values(row, row_index)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"row {row_index}: case_id must be a non-empty string")
        for question, qid in zip(questions, question_ids, strict=True):
            answer = _require_mapping(answers[qid], f"row {row_index} answer {qid}")
            question_type = _question_type(
                question.get("type"), f"row {row_index} question {qid} type"
            )
            if question_type == "choice":
                item = _extract_choice(question, expected[qid], answer)
                item.update(
                    {"split": split, "case_id": case_id, "row_index": row_index, "qid": qid}
                )
                choice_items.append(item)
                raw["choice"].append(
                    {
                        "answer": deepcopy(dict(answer)),
                        "case_id": case_id,
                        "expected": deepcopy(expected[qid]),
                        "prediction": deepcopy(answer["choice"]),
                        "probabilities": deepcopy(dict(answer["probabilities"])),
                        "qid": qid,
                        "row_index": row_index,
                        "split": split,
                    }
                )
            elif question_type == "noul":
                item = _extract_noul(expected[qid], answer)
                item.update(
                    {"split": split, "case_id": case_id, "row_index": row_index, "qid": qid}
                )
                noul_items.append(item)
                raw["noul"].append(
                    {
                        "answer": deepcopy(dict(answer)),
                        "case_id": case_id,
                        "expected": deepcopy(expected[qid]),
                        "prediction": item["prediction"],
                        "probability": deepcopy(answer["noul"]),
                        "qid": qid,
                        "row_index": row_index,
                        "split": split,
                    }
                )
            else:
                item = _extract_score(question, expected[qid], answer)
                item.update(
                    {"split": split, "case_id": case_id, "row_index": row_index, "qid": qid}
                )
                score_items.append(item)
                raw["score"].append(
                    {
                        "answer": deepcopy(dict(answer)),
                        "case_id": case_id,
                        "expected": deepcopy(expected[qid]),
                        "prediction": deepcopy(answer["score"]),
                        "max_score": deepcopy(question["max_score"]),
                        "qid": qid,
                        "row_index": row_index,
                        "rounded_level": item["rounded_level"],
                        "split": split,
                    }
                )
    metric_choice = _choice_items_for_metrics(choice_items)
    metric_noul = _noul_items_for_metrics(noul_items)
    metric_score = _score_items_for_metrics(score_items)
    noul_calibration = _fit_noul_calibration(
        [item for item in noul_items if item["split"] == "calibration"],
        [item for item in noul_items if item["split"] == "evaluation"],
    )
    _annotate_noul_raw(noul_items, raw["noul"], noul_calibration)
    summary = {
        "choice": _choice_summary(metric_choice, choice_items),
        "noul": _noul_summary(metric_noul, noul_items, noul_calibration),
        "score": _score_summary(metric_score),
        "raw": raw,
    }
    canonical_json(summary)
    return summary
