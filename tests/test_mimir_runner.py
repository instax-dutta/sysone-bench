"""Fake-engine tests for the MIMIR adapter.

No weights are loaded and `mimir-decisions` is not imported: each test installs a fake `mimir`
module whose engine returns scripted results, so the projection logic runs without inference.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import pytest

import runners.mimir_runner as _mimir_module
from benchmark.orchestrator import _answer_is_correct
from runners.mimir_runner import MimirRunner

mimir_module = cast(Any, _mimir_module)

CHOICE_QUESTION: dict[str, Any] = {
    "qid": "intent",
    "type": "choice",
    "instructions": "What does the customer want?",
    "criteria": {"refund": "money returned", "cancel": "", "info": None},
}
NOUL_QUESTION: dict[str, Any] = {
    "qid": "refund_requested",
    "type": "noul",
    "instructions": "Does the customer ask for money back?",
}
SCORE_QUESTION: dict[str, Any] = {
    "qid": "sentiment",
    "type": "score",
    "instructions": "How positive is the review?",
    "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
}
STATE = {"message": "please refund my order"}


@dataclass
class FakeResult:
    answer: Any
    probabilities: dict[str, float]
    status: str = "decided"


@dataclass
class FakeSpec:
    instructions: str
    options: dict[str, str] = field(default_factory=dict)


class FakeChoice(FakeSpec):
    pass


class FakeRate(FakeSpec):
    pass


class FakeYesNo(FakeSpec):
    pass


class FakeMode:
    CERTIFIED = "certified"
    STANDARD = "standard"


class FakeEngine:
    def __init__(self, results: Mapping[str, FakeResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def info(self) -> Any:
        return types.SimpleNamespace(
            model="Mythologic/MIMIR-1",
            revision="a" * 40,
            device=self.device,
            variant="fp32",
            certification="equivalent",
            runtime=types.SimpleNamespace(torch="2.14.0"),
        )

    def decide(self, context: object, spec: FakeSpec, **kwargs: Any) -> FakeResult:
        self.calls.append({"context": context, "spec": spec, **kwargs})
        return self.results[spec.instructions]

    def count_tokens(self, context: object, spec: FakeSpec) -> int:
        return len(spec.instructions)

    device = "cpu"


class FakeMimir:
    engine: ClassVar[FakeEngine]
    loaded: ClassVar[list[tuple[str, str]]] = []

    @classmethod
    def from_pretrained(cls, model: str, *, device: str) -> FakeEngine:
        cls.loaded.append((model, device))
        cls.engine.device = device
        return cls.engine


def _install_fake_mimir(monkeypatch: pytest.MonkeyPatch, engine: FakeEngine) -> None:
    FakeMimir.engine = engine
    FakeMimir.loaded = []
    root = types.ModuleType("mimir")
    root.__dict__.update(
        Mimir=FakeMimir, Choice=FakeChoice, Rate=FakeRate, YesNo=FakeYesNo, Mode=FakeMode
    )
    systemone = types.ModuleType("mimir.compat.systemone.v1")
    systemone.__dict__["state_context"] = lambda state: dict(state)
    for name, module in (
        ("mimir", root),
        ("mimir.compat", types.ModuleType("mimir.compat")),
        ("mimir.compat.systemone", types.ModuleType("mimir.compat.systemone")),
        ("mimir.compat.systemone.v1", systemone),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(mimir_module.importlib_metadata, "version", lambda name: "9.9.9")
    monkeypatch.delenv("MIMIR_MODEL", raising=False)
    monkeypatch.delenv("MIMIR_DEVICE", raising=False)


def _runner(
    monkeypatch: pytest.MonkeyPatch, results: Mapping[str, FakeResult]
) -> tuple[MimirRunner, FakeEngine]:
    engine = FakeEngine(results)
    _install_fake_mimir(monkeypatch, engine)
    return MimirRunner(), engine


@pytest.fixture
def intent_results() -> Iterator[dict[str, FakeResult]]:
    yield {
        CHOICE_QUESTION["instructions"]: FakeResult(
            "refund", {"refund": 0.7, "cancel": 0.2, "info": 0.1}
        )
    }


def test_info_records_package_version_device_mode_and_risk(
    monkeypatch: pytest.MonkeyPatch, intent_results: dict[str, FakeResult]
) -> None:
    runner, _ = _runner(monkeypatch, intent_results)
    info = runner.info()
    assert info["package"] == "mimir-decisions"
    assert info["package_version"] == "9.9.9"
    assert info["torch"] == "2.14.0"
    assert info["revision"] == "a" * 40
    assert info["device"] == "cpu"
    assert info["mode"] == "certified"
    assert info["risk"] == 0.01
    assert info["adapter_version"] == "2"


def test_device_defaults_to_cpu_and_is_overridable(
    monkeypatch: pytest.MonkeyPatch, intent_results: dict[str, FakeResult]
) -> None:
    _runner(monkeypatch, intent_results)
    assert FakeMimir.loaded == [("Mythologic/MIMIR-1", "cpu")]
    monkeypatch.setenv("MIMIR_DEVICE", "cuda")
    monkeypatch.setenv("MIMIR_MODEL", "/releases/mimir")
    runner = MimirRunner()
    assert FakeMimir.loaded[-1] == ("/releases/mimir", "cuda")
    assert runner.info()["device"] == "cuda"
    assert MimirRunner(model="other", device="cpu").info()["device"] == "cpu"
    assert FakeMimir.loaded[-1] == ("other", "cpu")


def test_choice_decided_keeps_the_model_answer_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        CHOICE_QUESTION["instructions"]: FakeResult(
            "refund", {"refund": 0.50001, "cancel": 0.29999, "info": 0.2}
        )
    }
    runner, engine = _runner(monkeypatch, results)
    response = runner.predict(STATE, {"intent": CHOICE_QUESTION})
    answer = response["answers"]["intent"]
    assert answer["choice"] == "refund"
    assert math.fsum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)
    assert answer["confidence"] == pytest.approx(0.50001)
    assert response["_raw_model"] == {
        "decisions": [{"status": "decided", "projected_to_argmax": False}]
    }
    call = engine.calls[0]
    assert call["mode"] == "certified"
    assert call["risk"] == 0.01
    assert call["spec"].options == {
        "refund": "refund: money returned",
        "cancel": "cancel",
        "info": "info",
    }


def test_abstained_choice_is_projected_to_the_argmax_option_and_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        CHOICE_QUESTION["instructions"]: FakeResult(
            None, {"refund": 0.1, "cancel": 0.5, "info": 0.4}, status="abstained"
        )
    }
    runner, _ = _runner(monkeypatch, results)
    response = runner.predict(STATE, {"intent": CHOICE_QUESTION})
    assert response["answers"]["intent"]["choice"] == "cancel"
    assert response["_raw_model"]["decisions"] == [
        {"status": "abstained", "projected_to_argmax": True}
    ]


def test_deferred_choice_keeps_its_own_answer_and_is_not_projected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        CHOICE_QUESTION["instructions"]: FakeResult(
            "info", {"refund": 0.4, "cancel": 0.1, "info": 0.5}, status="deferred"
        )
    }
    runner, _ = _runner(monkeypatch, results)
    response = runner.predict(STATE, {"intent": CHOICE_QUESTION})
    assert response["answers"]["intent"]["choice"] == "info"
    assert response["_raw_model"]["decisions"] == [
        {"status": "deferred", "projected_to_argmax": False}
    ]


def test_noul_reports_the_yes_probability_whatever_the_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        NOUL_QUESTION["instructions"]: FakeResult(
            True, {"yes": 0.93, "no": 0.07}, status="deferred"
        )
    }
    runner, engine = _runner(monkeypatch, results)
    response = runner.predict(STATE, {"refund_requested": NOUL_QUESTION})
    assert response["answers"]["refund_requested"] == {"type": "noul", "noul": 0.93}
    assert isinstance(engine.calls[0]["spec"], FakeYesNo)
    assert response["_raw_model"]["decisions"] == [
        {"status": "deferred", "projected_to_argmax": False}
    ]


def test_score_is_the_expectation_over_levels(monkeypatch: pytest.MonkeyPatch) -> None:
    probabilities = {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.3}
    results = {SCORE_QUESTION["instructions"]: FakeResult(3, probabilities)}
    runner, engine = _runner(monkeypatch, results)
    response = runner.predict(STATE, {"sentiment": SCORE_QUESTION})
    answer = response["answers"]["sentiment"]
    assert answer["type"] == "score"
    assert answer["score"] == pytest.approx(2.6)
    assert answer["confidence"] == 0.3
    assert isinstance(engine.calls[0]["spec"], FakeRate)
    assert list(engine.calls[0]["spec"].options) == ["0", "1", "2", "3", "4"]


def test_score_expectation_on_a_half_boundary_rounds_up_under_the_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        SCORE_QUESTION["instructions"]: FakeResult(
            2, {"0": 0.0, "1": 0.0, "2": 0.5, "3": 0.5, "4": 0.0}
        )
    }
    runner, _ = _runner(monkeypatch, results)
    answer = runner.predict(STATE, {"sentiment": SCORE_QUESTION})["answers"]["sentiment"]
    assert answer["score"] == 2.5
    assert _answer_is_correct(SCORE_QUESTION, answer, 3)
    assert not _answer_is_correct(SCORE_QUESTION, answer, 2)


def test_two_level_score_is_asked_as_a_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    question = {**SCORE_QUESTION, "criteria": ["bad", "good"]}
    results = {question["instructions"]: FakeResult(1, {"0": 0.25, "1": 0.75})}
    runner, engine = _runner(monkeypatch, results)
    answer = runner.predict(STATE, {"sentiment": question})["answers"]["sentiment"]
    assert isinstance(engine.calls[0]["spec"], FakeChoice)
    assert answer["score"] == pytest.approx(0.75)


def test_predict_counts_usage_per_question_and_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {
        CHOICE_QUESTION["instructions"]: FakeResult("refund", {"refund": 1.0}),
        NOUL_QUESTION["instructions"]: FakeResult(False, {"yes": 0.2, "no": 0.8}),
    }
    runner, _ = _runner(monkeypatch, results)
    questions = {"intent": CHOICE_QUESTION, "refund_requested": NOUL_QUESTION}
    before = deepcopy((STATE, questions))
    response = runner.predict(STATE, questions)
    assert (STATE, questions) == before
    expected_tokens = len(CHOICE_QUESTION["instructions"]) + len(NOUL_QUESTION["instructions"])
    assert response["_usage"] == {"input_tokens": expected_tokens, "output_tokens": 0, "calls": 2}
    assert set(response["answers"]) == {"intent", "refund_requested"}
    assert len(response["_raw_model"]["decisions"]) == 2


@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({**CHOICE_QUESTION, "criteria": {}}, "q: choice criteria must be a non-empty mapping"),
        ({**SCORE_QUESTION, "criteria": []}, "q: score criteria must be a non-empty list"),
        ({"type": "ranking", "instructions": "x"}, "q: unsupported question type 'ranking'"),
    ],
)
def test_malformed_questions_fail_with_the_question_id(
    monkeypatch: pytest.MonkeyPatch,
    intent_results: dict[str, FakeResult],
    question: dict[str, Any],
    message: str,
) -> None:
    runner, _ = _runner(monkeypatch, intent_results)
    with pytest.raises(ValueError, match=message):
        runner.predict(STATE, {"q": question})


def test_unknown_phase_is_rejected(
    monkeypatch: pytest.MonkeyPatch, intent_results: dict[str, FakeResult]
) -> None:
    runner, _ = _runner(monkeypatch, intent_results)
    with pytest.raises(ValueError, match="phase must be one of"):
        runner.predict(STATE, {"intent": CHOICE_QUESTION}, phase="evaluation")
