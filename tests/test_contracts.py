import math

import pytest

from benchmark.contracts import validate_answer, validate_answers


def choice_question() -> dict[str, object]:
    return {
        "qid": "intent",
        "type": "choice",
        "instructions": "Choose one label.",
        "criteria": {"yes": "yes means yes", "no": "no means no"},
    }


def noul_question() -> dict[str, object]:
    return {
        "qid": "flag",
        "type": "noul",
        "instructions": "Answer whether the flag is true.",
    }


def score_question() -> dict[str, object]:
    return {
        "qid": "intensity",
        "type": "score",
        "instructions": "Give the intensity.",
        "criteria": ["low", "medium", "high"],
        "max_score": 4,
    }


def choice_answer(**overrides: object) -> dict[str, object]:
    answer: dict[str, object] = {
        "type": "choice",
        "choice": "yes",
        "probabilities": {"yes": 0.7, "no": 0.3},
        "confidence": 0.7,
    }
    answer.update(overrides)
    return answer


def test_valid_choice_answer():
    validate_answers([choice_question()], {"intent": choice_answer()})


def test_choice_accepts_boundary_probabilities():
    answer = choice_answer(probabilities={"yes": 0, "no": 1}, confidence=0)
    validate_answers([choice_question()], {"intent": answer})


def test_choice_probability_sum_allows_declared_tolerance():
    answer = choice_answer(probabilities={"yes": 0.5000005, "no": 0.5})
    validate_answer(choice_question(), answer)


def test_valid_noul_boundaries():
    for value in (0, 1):
        validate_answers([noul_question()], {"flag": {"type": "noul", "noul": value}})


def test_sentinel_text_is_a_valid_question_id():
    question = {"qid": "<missing-question-id>", "type": "noul"}
    answer = {"type": "noul", "noul": 0.5}
    validate_answers([question], {"<missing-question-id>": answer})


def test_valid_score_boundaries_and_null_confidence():
    for value in (0, 4):
        answer = {"type": "score", "score": value, "confidence": None}
        validate_answers([score_question()], {"intensity": answer})


def test_missing_answer_fails_with_question_id():
    with pytest.raises(ValueError, match=r"intent: missing answer"):
        validate_answers([choice_question()], {})


def test_extra_answer_fails_with_question_id():
    with pytest.raises(ValueError, match=r"extra answer"):
        validate_answers(
            [choice_question()], {"intent": choice_answer(), "extra": {"type": "noul", "noul": 0.5}}
        )


def test_extra_only_ids_report_the_expected_question_id():
    with pytest.raises(ValueError, match=r"flag:.*extra"):
        validate_answers(
            [noul_question()],
            {"flag": {"type": "noul", "noul": 0.5}, "extra": {"type": "noul", "noul": 0.5}},
        )


def test_wrong_answer_type_fails_with_question_id():
    with pytest.raises(ValueError, match=r"intent:.*type"):
        validate_answers([choice_question()], {"intent": {"type": "noul", "noul": 0.5}})


def test_extra_ids_without_questions_use_answer_set_context():
    with pytest.raises(ValueError, match=r"answer set:.*extra"):
        validate_answers([], {"orphan": {"type": "noul", "noul": 0.5}})


def test_choice_rejects_illegal_label_without_coercion():
    answer = choice_answer(choice=1)
    with pytest.raises(ValueError, match=r"intent:.*choice"):
        validate_answer(choice_question(), answer)


def test_choice_requires_exact_probability_labels():
    for probabilities in ({"yes": 1}, {"yes": 0.6, "no": 0.3, "other": 0.1}):
        with pytest.raises(ValueError, match=r"intent:.*probabilit"):
            validate_answer(choice_question(), choice_answer(probabilities=probabilities))


def test_choice_rejects_probability_values_outside_unit_interval():
    for value in (-0.1, 1.1):
        with pytest.raises(ValueError, match=r"intent:.*probabilit"):
            validate_answer(
                choice_question(), choice_answer(probabilities={"yes": value, "no": 1 - value})
            )


@pytest.mark.parametrize("value", ["0.5", True, math.nan, math.inf, -math.inf])
def test_noul_does_not_coerce_or_accept_nonfinite_values(value):
    with pytest.raises(ValueError, match=r"flag:.*probabilit"):
        validate_answer(noul_question(), {"type": "noul", "noul": value})


def test_choice_rejects_nonfinite_probability_value():
    with pytest.raises(ValueError, match=r"intent:.*probabilit"):
        validate_answer(
            choice_question(),
            choice_answer(probabilities={"yes": math.nan, "no": 0.5}),
        )


def test_choice_requires_confidence_and_rejects_bad_types():
    missing = choice_answer()
    del missing["confidence"]
    with pytest.raises(ValueError, match=r"intent:.*confidence"):
        validate_answer(choice_question(), missing)
    for value in (None, "0.7", True, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match=r"intent:.*confidence"):
            validate_answer(choice_question(), choice_answer(confidence=value))


def test_noul_rejects_unknown_fields():
    with pytest.raises(ValueError, match=r"flag:.*field"):
        validate_answer(noul_question(), {"type": "noul", "noul": 0.5, "confidence": 0.5})


def test_score_requires_declared_range():
    question = score_question()
    del question["max_score"]
    with pytest.raises(ValueError, match=r"intensity:.*max_score"):
        validate_answer(question, {"type": "score", "score": 1})


def test_score_does_not_coerce_or_accept_nonfinite_values():
    for value in ("1", True, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"intensity:.*score"):
            validate_answer(
                score_question(),
                {"type": "score", "score": value, "confidence": 0.5},
            )


def test_score_rejects_values_outside_declared_range():
    for value in (-0.1, 4.1):
        with pytest.raises(ValueError, match=r"intensity:.*score"):
            validate_answer(
                score_question(),
                {"type": "score", "score": value, "confidence": 0.5},
            )


def test_score_requires_confidence_and_rejects_invalid_values():
    with pytest.raises(ValueError, match=r"intensity:.*confidence"):
        validate_answer(score_question(), {"type": "score", "score": 1})
    for value in ("0.5", True, math.nan, -0.1, 1.1):
        with pytest.raises(ValueError, match=r"intensity:.*confidence"):
            validate_answer(score_question(), {"type": "score", "score": 1, "confidence": value})


def test_unknown_question_type_is_rejected_before_answer_validation():
    question = {"qid": "mystery", "type": "boolean", "instructions": "Answer."}
    with pytest.raises(ValueError, match=r"mystery:.*type"):
        validate_answer(question, {"type": "boolean", "boolean": True})


def test_unknown_answer_type_is_rejected():
    with pytest.raises(ValueError, match=r"flag:.*type"):
        validate_answer(noul_question(), {"type": "text", "text": "hello"})


def test_unknown_question_field_is_rejected():
    question = {"qid": "q", "type": "noul", "instructions": "Answer.", "extra": True}
    with pytest.raises(ValueError, match=r"q:.*field"):
        validate_answer(question, {"type": "noul", "noul": 0.5})


def test_expected_type_is_rejected_as_an_unknown_question_field():
    question = {"qid": "q", "type": "noul", "instructions": "Answer.", "expected_type": "noul"}
    with pytest.raises(ValueError, match=r"q:.*field"):
        validate_answer(question, {"type": "noul", "noul": 0.5})


@pytest.mark.parametrize(
    "question",
    [
        {},
        {"qid": "", "type": "noul"},
        {"qid": 1, "type": "noul"},
        {"qid": "q", "type": "noul", "instructions": 3},
        {"qid": "q", "type": "choice", "instructions": "Choose.", "criteria": []},
        {"qid": "q", "type": "choice", "instructions": "Choose.", "criteria": {"yes": 1}},
        {
            "qid": "q",
            "type": "score",
            "instructions": "Score.",
            "criteria": ["low", 2],
            "max_score": 1,
        },
    ],
)
def test_malformed_question_objects_raise_value_error(question):
    with pytest.raises(ValueError):
        validate_answers([question], {})


def test_non_mapping_inputs_raise_value_error():
    with pytest.raises(ValueError, match="question"):
        validate_answer([], {})
    with pytest.raises(ValueError, match="q: answer"):
        validate_answer({"qid": "q", "type": "noul"}, [])


@pytest.mark.parametrize("answers", [[], None, "not-a-mapping"])
def test_malformed_answer_set_has_deterministic_context(answers):
    with pytest.raises(ValueError, match=r"answer set.*mapping"):
        validate_answers([noul_question()], answers)


def test_malformed_answer_mapping_reports_its_question_id():
    with pytest.raises(ValueError, match=r"flag:.*answer.*mapping"):
        validate_answers([noul_question()], {"flag": []})


def test_duplicate_question_ids_are_rejected_deterministically():
    questions = [noul_question(), noul_question()]
    with pytest.raises(ValueError, match="flag.*duplicate"):
        validate_answers(questions, {"flag": {"type": "noul", "noul": 0.5}})
