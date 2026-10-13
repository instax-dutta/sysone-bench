"""Fake-engine tests for the Largha adapter.

No weights are loaded and `phocinae-server` is not imported: each test installs a fake
`phocinae.engine` module whose engine returns scripted probability vectors, so the
projection logic runs without inference.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

import runners.largha_runner as _largha_module
from benchmark.orchestrator import _answer_is_correct
from runners.largha_runner import DEFAULT_REVISION, LarghaRunner

largha_module = cast(Any, _largha_module)

CHOICE_QUESTION: dict[str, Any] = {
    "qid": "intent",
    "type": "choice",
    "instructions": "What does the customer want in `message`?",
    "criteria": {
        "billing_question": "a question about an invoice, plan or payment method",
        "cancellation": "wants to cancel or downgrade",
        "refund": "money returned or a duplicate charge reversed",
    },
}
NOUL_QUESTION: dict[str, Any] = {
    "qid": "refund_requested",
    "type": "noul",
    "instructions": "Does the customer ask for money back?",
}
SCORE_QUESTION: dict[str, Any] = {
    "qid": "sentiment",
    "type": "score",
    "instructions": "How positive is the sentiment of this movie review?",
    "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
    "max_score": 4,
}
STATE = {"message": "please refund my order"}


@dataclass
class FakeResult:
    """One scripted engine answer: the canonical answer plus the probability vector."""

    answer: Any
    confidence: float
    probabilities: list[float]


class FakeEngine:
    def __init__(self, results: Mapping[str, FakeResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        state: str,
        questions: Sequence[Mapping[str, Any]],
        *,
        perm: bool | None = None,
        usage: bool = True,
        with_scores: bool = False,
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any], dict[str, int], dict[str, Any]]:
        self.calls.append(
            {
                "state": state,
                "questions": deepcopy([dict(question) for question in questions]),
                "with_scores": with_scores,
            }
        )
        answers: dict[str, Any] = {}
        confidences: dict[str, float] = {}
        raw_scores: dict[str, list[float]] = {}
        for question in questions:
            qid = str(question["id"])
            result = self.results[qid]
            answers[qid] = result.answer
            confidences[qid] = result.confidence
            raw_scores[qid] = list(result.probabilities)
        return answers, confidences, {}, {"input_tokens": 41, "output_tokens": 0}, raw_scores


def _install_fake_phocinae(
    monkeypatch: pytest.MonkeyPatch, engine: FakeEngine
) -> list[tuple[str, str, int | None]]:
    loaded: list[tuple[str, str, int | None]] = []

    def fake_engine(
        model_dir: str,
        device: str = "auto",
        *,
        perm_avg: bool = False,
        warm: bool = True,
        threads: int | None = None,
    ) -> FakeEngine:
        loaded.append((model_dir, device, threads))
        return engine

    root = types.ModuleType("phocinae")
    root.__dict__["__version__"] = "9.9.9"
    engine_module = types.ModuleType("phocinae.engine")
    engine_module.__dict__["Engine"] = fake_engine
    for name, module in (("phocinae", root), ("phocinae.engine", engine_module)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(largha_module.importlib_metadata, "version", lambda name: "9.9.9")
    return loaded


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("PHOC_MODEL_DIR", "PHOC_DEVICE", "PHOC_THREADS", "PHOC_REVISION"):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def model_dir(tmp_path: Path) -> str:
    (tmp_path / "model.safetensors").write_bytes(b"fake")
    return str(tmp_path)


def _runner(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: str,
    results: Mapping[str, FakeResult],
    *,
    device: str | None = None,
) -> tuple[LarghaRunner, FakeEngine, list[tuple[str, str, int | None]]]:
    engine = FakeEngine(results)
    loaded = _install_fake_phocinae(monkeypatch, engine)
    monkeypatch.setenv("PHOC_MODEL_DIR", model_dir)
    return LarghaRunner(device=device), engine, loaded


def _choice_results() -> dict[str, FakeResult]:
    return {"intent": FakeResult(answer=2, confidence=0.7, probabilities=[0.1, 0.2, 0.70001])}


def test_info_records_package_version_and_revision_pin(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    runner, _, _ = _runner(monkeypatch, model_dir, _choice_results())
    info = runner.info()
    assert info["runner"] == "largha"
    assert info["model"] == "Phocinae/Phocinae-Largha-150M-v1"
    assert info["revision"] == DEFAULT_REVISION
    assert len(DEFAULT_REVISION) == 40
    assert info["device"] == "cpu"
    assert info["serving"] == "local"
    assert info["adapter_version"] == "1"
    assert info["package"] == "phocinae-server"
    assert info["package_version"] == "9.9.9"


def test_model_dir_device_and_threads_resolution(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    _, _, loaded = _runner(monkeypatch, model_dir, _choice_results())
    assert loaded == [(model_dir, "cpu", None)]
    monkeypatch.setenv("PHOC_THREADS", "4")
    monkeypatch.setenv("PHOC_DEVICE", "cuda")
    runner, _, loaded = _runner(monkeypatch, model_dir, _choice_results())
    assert loaded == [(model_dir, "cuda", 4)]
    assert runner.info()["device"] == "cuda"
    runner = LarghaRunner(model_dir=model_dir, device="cpu")
    assert loaded[-1] == (model_dir, "cpu", 4)
    assert runner.info()["device"] == "cpu"
    monkeypatch.setenv("PHOC_REVISION", "f" * 40)
    runner = LarghaRunner(model_dir=model_dir, device="cpu")
    assert runner.info()["revision"] == "f" * 40


def test_missing_model_dir_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_phocinae(monkeypatch, FakeEngine({}))
    monkeypatch.delenv("PHOC_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="PHOC_MODEL_DIR must point"):
        LarghaRunner()
    monkeypatch.setenv("PHOC_MODEL_DIR", "/nonexistent")
    with pytest.raises(ValueError, match="has no model.safetensors"):
        LarghaRunner()


def test_choice_maps_labels_and_normalizes_probabilities(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    runner, engine, _ = _runner(monkeypatch, model_dir, _choice_results())
    response = runner.predict(STATE, {"intent": CHOICE_QUESTION})
    answer = response["answers"]["intent"]
    assert answer["choice"] == "refund"
    assert set(answer["probabilities"]) == set(CHOICE_QUESTION["criteria"])
    assert math.fsum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-12)
    assert answer["confidence"] == 0.7
    call = engine.calls[0]
    assert call["state"] == "message: please refund my order"
    assert call["with_scores"] is True
    assert call["questions"] == [
        {
            "id": "intent",
            "type": "choice",
            "options": [
                "billing_question: a question about an invoice, plan or payment method",
                "cancellation: wants to cancel or downgrade",
                "refund: money returned or a duplicate charge reversed",
            ],
        }
    ]


def test_noul_keeps_the_yes_probability(monkeypatch: pytest.MonkeyPatch, model_dir: str) -> None:
    results = {"refund_requested": FakeResult(True, 0.93, [0.07, 0.93])}
    runner, _, _ = _runner(monkeypatch, model_dir, results)
    response = runner.predict(STATE, {"refund_requested": NOUL_QUESTION})
    assert response["answers"]["refund_requested"] == {"type": "noul", "noul": 0.93}


@pytest.mark.parametrize(
    ("probabilities", "expected_score"),
    [
        ([1.0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0),
        ([0, 0, 0, 0, 1.0, 0, 0, 0, 0], 2.0),
        ([0, 0, 0, 0, 0, 1.0, 0, 0, 0], 2.5),
        ([0, 0, 0, 0, 0, 0, 0, 0, 1.0], 4.0),
        ([0.1] * 9, 2.0),
    ],
)
def test_score_is_the_rescaled_expectation(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: str,
    probabilities: list[float],
    expected_score: float,
) -> None:
    results = {"sentiment": FakeResult(6, 0.5, probabilities)}
    runner, _, _ = _runner(monkeypatch, model_dir, results)
    answer = runner.predict(STATE, {"sentiment": SCORE_QUESTION})["answers"]["sentiment"]
    assert answer["type"] == "score"
    assert answer["score"] == pytest.approx(expected_score)
    assert answer["confidence"] == 0.5


def test_score_expectation_on_a_half_boundary_rounds_up_under_the_contract(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    results = {"sentiment": FakeResult(7, 0.5, [0, 0, 0, 0, 0, 1.0, 0, 0, 0])}
    runner, _, _ = _runner(monkeypatch, model_dir, results)
    answer = runner.predict(STATE, {"sentiment": SCORE_QUESTION})["answers"]["sentiment"]
    assert answer["score"] == 2.5
    assert _answer_is_correct(SCORE_QUESTION, answer, 3)
    assert not _answer_is_correct(SCORE_QUESTION, answer, 2)


def test_multi_key_states_render_with_field_names(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    results = {"refund_requested": FakeResult(True, 0.5, [0.5, 0.5])}
    runner, engine, _ = _runner(monkeypatch, model_dir, results)
    state = {"premise": "A man is sleeping.", "hypothesis": "Someone snores."}
    runner.predict(state, {"refund_requested": NOUL_QUESTION})
    assert engine.calls[0]["state"] == "premise: A man is sleeping.\nhypothesis: Someone snores."


def test_predict_records_usage_and_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    results = {
        "intent": FakeResult(0, 0.6, [0.6, 0.3, 0.1]),
        "refund_requested": FakeResult(False, 0.8, [0.8, 0.2]),
    }
    runner, _, _ = _runner(monkeypatch, model_dir, results)
    questions = {"intent": CHOICE_QUESTION, "refund_requested": NOUL_QUESTION}
    before = deepcopy((STATE, questions))
    response = runner.predict(STATE, questions)
    assert (STATE, questions) == before
    assert response["_usage"] == {"input_tokens": 41, "output_tokens": 0}
    assert set(response["answers"]) == {"intent", "refund_requested"}
    assert response["_raw_model"]["option_scores"] == {
        "intent": [0.6, 0.3, 0.1],
        "refund_requested": [0.8, 0.2],
    }


def test_phase_is_validated(monkeypatch: pytest.MonkeyPatch, model_dir: str) -> None:
    runner, _, _ = _runner(monkeypatch, model_dir, _choice_results())
    assert runner.predict(STATE, {"intent": CHOICE_QUESTION}, phase="warmup")["answers"]
    with pytest.raises(ValueError, match="phase must be one of"):
        runner.predict(STATE, {"intent": CHOICE_QUESTION}, phase="evaluation")


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({**CHOICE_QUESTION, "criteria": {}}, "q: choice criteria must be a non-empty mapping"),
        ({**SCORE_QUESTION, "criteria": []}, "q: score criteria must be a non-empty list"),
        ({**SCORE_QUESTION, "max_score": -1}, "q: max_score must be a finite nonnegative"),
        ({"type": "ranking", "instructions": "x"}, "q: unsupported question type 'ranking'"),
    ],
)
def test_malformed_questions_fail_with_the_question_id(
    monkeypatch: pytest.MonkeyPatch,
    model_dir: str,
    question: dict[str, Any],
    message: str,
) -> None:
    runner, _, _ = _runner(monkeypatch, model_dir, _choice_results())
    with pytest.raises(ValueError, match=message):
        runner.predict(STATE, {"q": question})


def test_engine_vector_mismatch_fails_with_the_question_id(
    monkeypatch: pytest.MonkeyPatch, model_dir: str
) -> None:
    results = {"intent": FakeResult(0, 0.5, [0.5, 0.5])}
    runner, _, _ = _runner(monkeypatch, model_dir, results)
    with pytest.raises(ValueError, match="engine returned 2 option probabilities"):
        runner.predict(STATE, {"intent": CHOICE_QUESTION})
