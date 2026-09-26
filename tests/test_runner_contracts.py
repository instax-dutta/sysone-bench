from __future__ import annotations

import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

fake_laya = cast(Any, ModuleType("laya"))
fake_laya.__version__ = "9.8.7"
fake_laya.load = None
fake_laya.Router = None
sys.modules["laya"] = fake_laya

import runners.laya_runner as laya_module
from runners.base import BaseRunner
from runners.laya_runner import LayaRunner, _build_laya_metadata
from runners.router_runner import RouterRunner


class FakeLayaAgent:
    def __init__(
        self,
        answer_ids: tuple[str, ...] = ("intent",),
        *,
        revision: str | None = "resolved-revision",
        device: str | None = "cpu",
        checkpoint: str | None = "root",
        resolved_repo: str | None = "resolved/laya-test",
    ) -> None:
        if revision is not None:
            self.revision = revision
        if device is not None:
            self.device = device
        if checkpoint is not None:
            self.checkpoint = checkpoint
        config: dict[str, Any] = {"_commit_hash": revision}
        if resolved_repo is not None:
            config["_name_or_path"] = resolved_repo
        self.model = SimpleNamespace(config=SimpleNamespace(**config))
        self.answer_ids = answer_ids
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []

    def set_results(self, *results: dict[str, Any]) -> None:
        self.results = [deepcopy(result) for result in results]

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        state["adapter_mutation"] = True
        questions["intent"]["adapter_mutation"] = True
        if self.results:
            return deepcopy(self.results.pop(0))
        return {
            "answers": {answer_id: {"type": "noul", "noul": 1.0} for answer_id in self.answer_ids},
            "usage": {"input_tokens": 3, "output_tokens": 0},
            "model": "fake-laya-model",
            "ignored": {"not": "part of the core contract"},
        }


class RealisticLayaAgent:
    """Mirrors laya 0.3.11: exposes model_id/cfg/device but no revision or checkpoint."""

    def __init__(self, model_id: str = "convaiinnovations/laya") -> None:
        self.model_id = model_id
        self.device = "cpu"
        self.cfg = {"model_name": "rl-agent"}
        self.model = SimpleNamespace(config=SimpleNamespace())

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        return {"answers": {qid: {"type": "noul", "noul": 1.0} for qid in questions}}


class FakeRouter:
    def __init__(self, preload: bool = False, max_loaded: int = 2) -> None:
        self.preload_enabled = preload
        self.max_loaded = max_loaded
        self.preloaded: list[str] = []
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []
        self._agents = {
            "english": FakeLayaAgent(
                revision="english-revision",
                device="cpu",
                checkpoint="english",
                resolved_repo="resolved/english",
            ),
            "multilingual": FakeLayaAgent(
                revision="multilingual-revision",
                device="cpu",
                checkpoint="multilingual",
                resolved_repo="resolved/multilingual",
            ),
        }

    def preload(self, names: list[str]) -> None:
        self.preloaded = list(names)

    def set_results(self, *results: dict[str, Any]) -> None:
        self.results = [deepcopy(result) for result in results]

    def predict(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        state["router_mutation"] = True
        questions["intent"]["router_mutation"] = True
        if self.results:
            return deepcopy(self.results.pop(0))
        return {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {
                "model": "english",
                "reason": "English Latin text",
            },
            "usage": {"input_tokens": 4, "output_tokens": 1},
            "model": "fake-router-model",
        }


class MetadataOnlyRunner(BaseRunner):
    name = "metadata-only"

    def __init__(self) -> None:
        self._metadata = {
            "runner": self.name,
            "model": "model-id",
            "revision": "revision-id",
            "serving": "fake",
            "device": "cpu",
            "adapter_version": "test",
            "details": {"nested": ["original"]},
        }


def get_laya_api() -> Any:
    return cast(Any, laya_module)._get_laya()


def install_fake_laya(monkeypatch: pytest.MonkeyPatch, agent: FakeLayaAgent) -> list[str]:
    loaded_repos: list[str] = []

    def load(repo: str) -> FakeLayaAgent:
        loaded_repos.append(repo)
        return agent

    api = get_laya_api()
    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(api, "__version__", "9.8.7", raising=False)
    return loaded_repos


def install_kwargs_laya(
    monkeypatch: pytest.MonkeyPatch, agent: Any, version: str = "0.3.11"
) -> list[tuple[str, dict[str, Any]]]:
    loaded: list[tuple[str, dict[str, Any]]] = []

    def load(repo: str, **kwargs: Any) -> Any:
        loaded.append((repo, kwargs))
        return agent

    api = get_laya_api()
    monkeypatch.setattr(api, "load", load)
    monkeypatch.setattr(api, "__version__", version, raising=False)
    return loaded


def install_fake_router(monkeypatch: pytest.MonkeyPatch) -> None:
    api = get_laya_api()
    monkeypatch.setattr(api, "Router", FakeRouter)
    monkeypatch.setattr(api, "__version__", "9.8.7", raising=False)


def test_base_info_has_required_keys_and_returns_nested_snapshots() -> None:
    runner = MetadataOnlyRunner()
    info = runner.info()

    assert {
        "runner",
        "model",
        "revision",
        "serving",
        "device",
        "adapter_version",
    } <= set(info)
    info["details"]["nested"].append("changed")
    info["model"] = "changed"
    assert runner.info()["details"] == {"nested": ["original"]}
    assert runner.info()["model"] == "model-id"


def test_laya_does_not_mutate_inputs_and_normalizes_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeLayaAgent()
    loaded_repos = install_fake_laya(monkeypatch, agent)
    runner = LayaRunner(repo="example/laya-test")
    state = {"message": "hello", "nested": {"value": 1}}
    questions = {"intent": {"type": "noul", "nested": {"value": 2}}}
    original_state = deepcopy(state)
    original_questions = deepcopy(questions)

    result = runner.predict(state, questions)

    assert state == original_state
    assert questions == original_questions
    assert loaded_repos == ["example/laya-test"]
    assert agent.calls[0][0] is not state
    assert agent.calls[0][1] is not questions
    assert result == {
        "answers": {"intent": {"type": "noul", "noul": 1.0}},
        "_usage": {"input_tokens": 3, "output_tokens": 0},
        "_raw_model": "fake-laya-model",
    }


def test_laya_info_records_identity_from_the_requested_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeLayaAgent(checkpoint="english")
    loaded = install_kwargs_laya(monkeypatch, agent, version="9.8.7")
    runner = LayaRunner(repo="example/laya-test", subfolder="english")

    info = runner.info()

    assert loaded == [("example/laya-test", {"subfolder": "english"})]
    assert info["runner"] == "laya"
    assert info["model"] == "example/laya-test"
    assert info["requested_repo"] == "example/laya-test"
    assert info["repo"] == "example/laya-test"
    assert info["resolved_repo"] == "resolved/laya-test"
    assert info["resolved"]["revision"] == "resolved-revision"
    assert info["resolved"]["device"] == "cpu"
    assert info["resolved"]["checkpoint"] == "english"
    assert info["resolved"]["package_version"] == "9.8.7"
    assert info["revision"] == "resolved-revision"
    assert info["package_version"] == "9.8.7"
    assert info["device"] == "cpu"
    assert info["serving"] == "local"
    assert info["adapter_version"]
    assert info["checkpoint"] == "english"
    assert info["requested_checkpoint"] == "english"


def test_laya_info_returns_an_isolated_metadata_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeLayaAgent()
    install_fake_laya(monkeypatch, agent)
    runner = LayaRunner(repo="example/laya-test")

    info = runner.info()
    info["model"] = "changed"
    info["checkpoint"] = "changed"

    assert runner.info()["model"] == "example/laya-test"
    assert runner.info()["checkpoint"] == "root"


def test_laya_info_resolves_revision_from_the_cached_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    agent = RealisticLayaAgent()
    install_kwargs_laya(monkeypatch, agent)
    monkeypatch.setattr(
        laya_module, "_cached_snapshot_revision", lambda repo, subfolder=None: commit
    )

    info = LayaRunner().info()

    assert info["revision"] == commit
    assert info["resolved"]["revision"] == commit
    assert info["resolved_repo"] == "convaiinnovations/laya"
    assert info["resolved_model"] == "convaiinnovations/laya"
    assert info["checkpoint"] == "root"
    assert info["package_version"] == "0.3.11"
    assert info["device"] == "cpu"


def test_laya_info_fails_closed_when_no_revision_can_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_kwargs_laya(monkeypatch, RealisticLayaAgent())
    monkeypatch.setattr(
        laya_module, "_cached_snapshot_revision", lambda repo, subfolder=None: None
    )

    with pytest.raises(ValueError, match="Laya identity missing revision"):
        LayaRunner()


def test_laya_passes_the_requested_subfolder_to_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = install_kwargs_laya(monkeypatch, RealisticLayaAgent())
    monkeypatch.setattr(
        laya_module, "_cached_snapshot_revision", lambda repo, subfolder=None: "a" * 40
    )

    info = LayaRunner(subfolder="multilingual").info()

    assert loaded == [("convaiinnovations/laya", {"subfolder": "multilingual"})]
    assert info["checkpoint"] == "multilingual"
    assert info["requested_checkpoint"] == "multilingual"


def test_cached_snapshot_revision_ignores_local_paths_and_non_commit_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    assert laya_module._cached_snapshot_revision(str(tmp_path)) is None
    assert laya_module._cached_snapshot_revision("./local/checkpoint") is None

    requested: list[tuple[str, str]] = []

    def fake_lookup(repo_id: str, filename: str, **kwargs: Any) -> str:
        requested.append((repo_id, filename))
        return str(tmp_path / "refs" / "main" / "rl_agent_config.json")

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.try_to_load_from_cache = fake_lookup  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert laya_module._cached_snapshot_revision("convaiinnovations/laya") is None
    assert laya_module._cached_snapshot_revision("convaiinnovations/laya", "multilingual") is None
    assert requested == [
        ("convaiinnovations/laya", "rl_agent_config.json"),
        ("convaiinnovations/laya", "multilingual/rl_agent_config.json"),
    ]


def test_cached_snapshot_revision_returns_the_cached_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    commit = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.try_to_load_from_cache = lambda repo_id, filename, **kwargs: str(  # type: ignore[attr-defined]
        tmp_path / "models--org--name" / "snapshots" / commit / filename
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert laya_module._cached_snapshot_revision("org/name") == commit


def test_cached_snapshot_revision_returns_none_when_the_cache_entry_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.try_to_load_from_cache = lambda repo_id, filename, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert laya_module._cached_snapshot_revision("org/name") is None


def test_cached_snapshot_revision_survives_a_cache_lookup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_lookup_error(repo_id: str, filename: str, **kwargs: Any) -> str:
        raise OSError("cache unavailable")

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.try_to_load_from_cache = raise_lookup_error  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert laya_module._cached_snapshot_revision("org/name") is None


def test_laya_normalizes_vendor_answer_extras_onto_the_benchmark_contract() -> None:
    questions = {
        "intent": {"qid": "intent", "type": "noul"},
        "topic": {
            "qid": "topic",
            "type": "choice",
            "criteria": {"billing": "billing", "other": "other"},
        },
        "severity": {"qid": "severity", "type": "score", "max_score": 3},
    }
    answers = {
        "intent": {"type": "noul", "noul": 0.72, "confidence": 0.72, "action": {"act_probability": 0.1}},
        "topic": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "other": 0.2},
            "confidence": 0.8,
            "action": {"act_probability": 0.1},
        },
        "severity": {
            "type": "score",
            "score": 2,
            "legend": {"0": "low", "1": "mid", "2": "high", "3": "max"},
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2},
            "confidence": 0.5,
            "action": {"act_probability": 0.1},
        },
    }

    normalized = laya_module.normalize_answers(questions, answers)

    assert normalized == {
        "intent": {"type": "noul", "noul": 0.72},
        "topic": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.8, "other": 0.2},
            "confidence": 0.8,
        },
        "severity": {"type": "score", "score": 2, "confidence": 0.5},
    }
    assert list(normalized["severity"]) == ["type", "score", "confidence"]


def test_laya_normalization_resums_rounded_probabilities() -> None:
    criteria = {
        "billing_question": "",
        "cancellation": "",
        "information": "",
        "other": "",
        "technical": "",
    }
    # Laya rounds each probability to four decimals, so the raw values sum to 0.9999.
    raw = {
        "billing_question": 0.1234,
        "cancellation": 0.2,
        "information": 0.2765,
        "other": 0.2,
        "technical": 0.2,
    }
    assert abs(sum(raw.values()) - 1.0) > 1e-6

    normalized = laya_module.normalize_answers(
        {"intent": {"qid": "intent", "type": "choice", "criteria": criteria}},
        {
            "intent": {
                "type": "choice",
                "choice": "information",
                "probabilities": raw,
                "confidence": 0.2765,
                "action": {"act_probability": 0.0},
            }
        },
    )

    projected = normalized["intent"]["probabilities"]
    assert set(projected) == set(criteria)
    assert abs(sum(projected.values()) - 1.0) <= 1e-12
    assert max(projected, key=lambda label: projected[label]) == "information"
    assert normalized["intent"]["choice"] == "information"
    assert "action" not in normalized["intent"]


def test_laya_normalization_rejects_impossible_probability_payloads() -> None:
    questions = {"intent": {"qid": "intent", "type": "choice", "criteria": {"a": "", "b": ""}}}
    template = {"type": "choice", "choice": "a", "confidence": 0.5}

    for probabilities, match in (
        ({}, "must not be empty"),
        ({"a": 0.0, "b": 0.0}, "must not sum to zero"),
        ({"a": -0.5, "b": 1.5}, "finite and nonnegative"),
        ({"a": 0.5, "c": 0.5}, "do not match the question criteria"),
    ):
        with pytest.raises(ValueError, match=match):
            laya_module.normalize_answers(
                questions, {"intent": {**template, "probabilities": probabilities}}
            )


def test_laya_normalization_rejects_mismatched_or_incomplete_answers() -> None:
    questions = {"intent": {"qid": "intent", "type": "noul"}}

    with pytest.raises(ValueError, match="declared question type"):
        laya_module.normalize_answers(
            questions, {"intent": {"type": "choice", "choice": "a", "probabilities": {}, "confidence": 1.0}}
        )
    with pytest.raises(ValueError, match="missing declared field"):
        laya_module.normalize_answers(questions, {"intent": {"type": "noul"}})
    with pytest.raises(ValueError, match="not a supported Laya answer type"):
        laya_module.normalize_answers(
            {"intent": {"qid": "intent", "type": "ranking"}},
            {"intent": {"type": "ranking"}},
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        laya_module.normalize_answers(questions, {"intent": "noul"})


def test_laya_predict_returns_the_normalized_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = FakeLayaAgent()
    agent.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.9, "action": {"act_probability": 0.0}}},
            "usage": {"input_tokens": 3, "output_tokens": 0},
            "model": "fake-laya-model",
        }
    )
    install_fake_laya(monkeypatch, agent)
    runner = LayaRunner(repo="example/laya-test")

    result = runner.predict(
        {"message": "hello"}, {"intent": {"qid": "intent", "type": "noul"}}
    )

    assert result["answers"] == {"intent": {"type": "noul", "noul": 0.9}}
    assert result["_raw_model"] == "fake-laya-model"


def test_laya_rejects_non_exact_answer_ids_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for answer_ids in ((), ("intent", "extra")):
        agent = FakeLayaAgent(answer_ids=answer_ids)
        install_fake_laya(monkeypatch, agent)
        runner = LayaRunner(repo="example/laya-test")
        state = {"message": "hello"}
        questions = {"intent": {"type": "noul"}}
        original_state = deepcopy(state)
        original_questions = deepcopy(questions)

        with pytest.raises(ValueError, match="answer IDs"):
            runner.predict(state, questions)

        assert state == original_state
        assert questions == original_questions


def test_router_preserves_laya_identity_and_isolates_route_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    state = {"message": "hello", "nested": {"value": 1}}
    questions = {"intent": {"type": "noul", "nested": {"value": 2}}}
    original_state = deepcopy(state)
    original_questions = deepcopy(questions)

    result = runner.predict(state, questions)
    snapshot = runner.info()
    snapshot["route_counts"]["english"] = 99
    snapshot["route_reasons"]["english"] = "changed"
    snapshot["route_reason_counts"]["english"]["English Latin text"] = 99
    info = runner.info()

    assert state == original_state
    assert questions == original_questions
    assert result == {
        "answers": {"intent": {"type": "noul", "noul": 0.25}},
        "_usage": {"input_tokens": 4, "output_tokens": 1},
        "_routing": {"model": "english", "reason": "English Latin text"},
        "_raw_model": "fake-router-model",
    }
    assert info["runner"] == "laya-router"
    assert info["model"] == "laya-router"
    assert info["revision"] == "english:english-revision,multilingual:multilingual-revision"
    assert info["package_version"] == "9.8.7"
    assert info["device"] == "cpu"
    assert info["preloaded"] == ["english", "multilingual"]
    assert set(info["checkpoint_identities"]) == {"english", "multilingual"}
    assert info["checkpoint_identities"]["english"]["revision"] == "english-revision"
    assert info["checkpoint_identities"]["multilingual"]["resolved_repo"] == "resolved/multilingual"
    assert "convaiinnovations/laya" not in str(info)
    assert info["route_counts"] == {"english": 1}
    assert info["route_reasons"] == {"english": "English Latin text"}
    assert info["route_reason_counts"] == {"english": {"English Latin text": 1}}
    assert runner.info()["route_counts"] == {"english": 1}
    assert runner.info()["route_reasons"] == {"english": "English Latin text"}
    assert runner.info()["route_reason_counts"] == {"english": {"English Latin text": 1}}


def test_router_counts_are_deterministic_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()

    runner.predict({"message": "one"}, {"intent": {"type": "noul"}})
    runner.predict({"message": "two"}, {"intent": {"type": "noul"}})

    assert runner.info()["route_counts"] == {"english": 2}
    assert runner.info()["route_reason_counts"] == {"english": {"English Latin text": 2}}


def test_router_separates_route_counts_and_reasons_by_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    runner.router.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "english", "reason": "warmup reason"},
        },
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "multilingual", "reason": "benchmark reason"},
        },
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "english", "reason": "speed reason"},
        },
    )

    runner.predict({"message": "warm"}, {"intent": {"type": "noul"}}, phase="warmup")
    runner.predict({"message": "bench"}, {"intent": {"type": "noul"}}, phase="benchmark")
    runner.predict({"message": "speed"}, {"intent": {"type": "noul"}}, phase="speed")
    info = runner.info()

    assert info["route_counts_by_phase"] == {
        "warmup": {"english": 1},
        "benchmark": {"multilingual": 1},
        "speed": {"english": 1},
    }
    assert info["route_reason_counts_by_phase"] == {
        "warmup": {"english": {"warmup reason": 1}},
        "benchmark": {"multilingual": {"benchmark reason": 1}},
        "speed": {"english": {"speed reason": 1}},
    }
    assert info["route_counts"] == {"english": 2, "multilingual": 1}
    assert info["route_reason_counts"] == {
        "english": {"speed reason": 1, "warmup reason": 1},
        "multilingual": {"benchmark reason": 1},
    }


def test_router_rejects_unknown_phase_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()

    with pytest.raises(ValueError, match="phase"):
        runner.predict({"message": "bad"}, {"intent": {"type": "noul"}}, phase="invalid")

    assert runner.router.calls == []
    assert runner.info()["route_counts"] == {}


def test_router_missing_routing_is_counter_neutral_and_omits_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    runner.router.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "english", "reason": "English Latin text"},
        },
        {"answers": {"intent": {"type": "noul", "noul": 0.25}}},
        {"answers": {"intent": {"type": "noul", "noul": 0.25}}, "routing": None},
    )

    first = runner.predict({"message": "one"}, {"intent": {"type": "noul"}})
    missing = runner.predict({"message": "two"}, {"intent": {"type": "noul"}})
    null_routing = runner.predict({"message": "three"}, {"intent": {"type": "noul"}})

    assert first["_routing"] == {"model": "english", "reason": "English Latin text"}
    assert "_routing" not in missing
    assert "_routing" not in null_routing
    assert runner.info()["route_counts"] == {"english": 1}
    assert runner.info()["route_reason_counts"] == {"english": {"English Latin text": 1}}
    assert "?" not in runner.info()["route_counts"]


def test_router_rejects_malformed_model_without_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    runner.router.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": None, "reason": "bad"},
        }
    )

    with pytest.raises(ValueError, match="routing model"):
        runner.predict({"message": "bad"}, {"intent": {"type": "noul"}})

    info = runner.info()
    assert info["route_counts"] == {}
    assert info["route_reason_counts"] == {}
    assert info["route_reasons"] == {}


def test_router_rejects_placeholder_routing_model_without_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for model in (
        "?",
        "-",
        "n/a",
        "N/A",
        "none",
        "None",
        "null",
        "NULL",
        "unknown",
        "UNKNOWN",
        "",
        "   ",
    ):
        install_fake_router(monkeypatch)
        runner = RouterRunner()
        runner.router.set_results(
            {
                "answers": {"intent": {"type": "noul", "noul": 0.25}},
                "routing": {"model": model, "reason": "bad"},
            }
        )

        with pytest.raises(ValueError, match="routing model"):
            runner.predict({"message": "bad"}, {"intent": {"type": "noul"}})

        info = runner.info()
        assert info["route_counts"] == {}
        assert info["route_counts_by_phase"] == {"warmup": {}, "benchmark": {}, "speed": {}}
        assert info["route_reasons"] == {}
        assert info["route_reason_counts"] == {}
        assert info["route_reason_collections"] == {}


def test_router_accepts_valid_routing_model_after_placeholder_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    runner.router.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "?", "reason": "bad"},
        },
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "multilingual", "reason": "valid reason"},
        },
    )

    with pytest.raises(ValueError, match="routing model"):
        runner.predict({"message": "bad"}, {"intent": {"type": "noul"}})

    result = runner.predict({"message": "valid"}, {"intent": {"type": "noul"}})

    assert result["_routing"] == {"model": "multilingual", "reason": "valid reason"}
    assert runner.info()["route_counts"] == {"multilingual": 1}
    assert runner.info()["route_counts_by_phase"]["benchmark"] == {"multilingual": 1}


def test_router_accepts_underscore_routing_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    runner.router.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "_routing": {"model": "english", "reason": "alias reason"},
        }
    )

    result = runner.predict({"message": "alias"}, {"intent": {"type": "noul"}})

    assert result["_routing"] == {"model": "english", "reason": "alias reason"}
    assert runner.info()["route_counts"] == {"english": 1}
    assert runner.info()["route_reason_counts"] == {"english": {"alias reason": 1}}


def test_router_reason_collections_are_sorted_and_call_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots: list[dict[str, Any]] = []
    for reasons in (["zeta reason", "alpha reason"], ["alpha reason", "zeta reason"]):
        install_fake_router(monkeypatch)
        runner = RouterRunner()
        runner.router.set_results(
            {
                "answers": {"intent": {"type": "noul", "noul": 0.25}},
                "routing": {"model": "english", "reasons": reasons},
            }
        )
        runner.predict({"message": "one"}, {"intent": {"type": "noul"}})
        snapshots.append(runner.info())

    assert snapshots[0]["route_reasons"] == {"english": "alpha reason"}
    assert snapshots[1]["route_reasons"] == {"english": "alpha reason"}
    assert snapshots[0]["route_reason_collections"] == {"english": ["alpha reason", "zeta reason"]}
    assert snapshots[1]["route_reason_collections"] == {"english": ["alpha reason", "zeta reason"]}
    assert snapshots[0]["route_reasons_by_phase"] == {
        "warmup": {},
        "benchmark": {"english": ["alpha reason", "zeta reason"]},
        "speed": {},
    }
    assert snapshots[0]["route_reason_counts_by_phase"]["benchmark"] == {
        "english": {"alpha reason": 1, "zeta reason": 1}
    }
    assert snapshots[0]["route_reason_counts"] == snapshots[1]["route_reason_counts"]


def test_laya_identity_fails_closed_for_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for missing in ("revision", "device"):
        kwargs: dict[str, str | None] = {
            "revision": "resolved-revision",
            "device": "cpu",
            "checkpoint": "english",
        }
        kwargs[missing] = None
        agent = FakeLayaAgent(
            revision=kwargs["revision"],
            device=kwargs["device"],
            checkpoint=kwargs["checkpoint"],
        )
        install_fake_laya(monkeypatch, agent)

        with pytest.raises(ValueError, match=missing):
            LayaRunner(repo="example/laya-test")


def test_laya_checkpoint_identity_requires_a_source_of_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeLayaAgent(checkpoint=None)
    install_fake_laya(monkeypatch, agent)

    with pytest.raises(ValueError, match="Laya identity missing checkpoint"):
        _build_laya_metadata(
            "example/laya-test",
            agent,
            "laya",
            "local",
            package=get_laya_api(),
        )


def test_laya_identity_fails_closed_for_missing_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeLayaAgent()
    install_fake_laya(monkeypatch, agent)
    monkeypatch.delattr(get_laya_api(), "__version__")

    with pytest.raises(ValueError, match="package version"):
        LayaRunner(repo="example/laya-test")


def test_router_requires_identity_for_every_preloaded_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FakeRouter()
    del router._agents["multilingual"]
    install_fake_router(monkeypatch)
    monkeypatch.setattr(get_laya_api(), "Router", lambda **kwargs: router)

    with pytest.raises(ValueError, match="multilingual"):
        RouterRunner()


def test_router_records_verified_checkpoint_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()

    info = runner.info()

    assert info["model"] == "laya-router"
    assert info["checkpoint_identities"]["english"]["resolved_repo"] == "resolved/english"
    assert info["checkpoint_identities"]["english"]["revision"] == "english-revision"
    assert info["checkpoint_identities"]["multilingual"]["checkpoint"] == "multilingual"
    assert "convaiinnovations/laya" not in str(info)


def test_router_metadata_snapshots_are_deeply_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_router(monkeypatch)
    runner = RouterRunner()
    runner.router.set_results(
        {
            "answers": {"intent": {"type": "noul", "noul": 0.25}},
            "routing": {"model": "english", "reason": "English Latin text"},
        }
    )
    runner.predict({"message": "one"}, {"intent": {"type": "noul"}}, phase="benchmark")
    snapshot = runner.info()
    snapshot["route_counts"]["english"] = 99
    snapshot["route_counts_by_phase"]["benchmark"]["english"] = 99
    snapshot["route_reason_collections"]["english"].append("changed")
    snapshot["route_reasons_by_phase"]["benchmark"]["english"].append("changed")
    snapshot["checkpoint_identities"]["english"]["revision"] = "changed"

    fresh = runner.info()

    assert fresh["route_counts"] == {"english": 1}
    assert fresh["route_counts_by_phase"]["benchmark"] == {"english": 1}
    assert fresh["route_reason_collections"] == {"english": ["English Latin text"]}
    assert fresh["route_reasons_by_phase"]["benchmark"] == {"english": ["English Latin text"]}
    assert fresh["checkpoint_identities"]["english"]["revision"] == "english-revision"
