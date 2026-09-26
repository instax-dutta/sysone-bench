from __future__ import annotations

import importlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import requests
from fakes import (
    make_fake_laya_runner,
    make_fake_router,
    make_runner_with_key,
    manifest_fixture,
    run_backend_import_test,
    run_collision_fixture,
    score_question,
    write_fake_run,
)

from runners.base import BaseRunner


def test_fake_laya_runner_preserves_inputs_and_returns_deterministic_answers() -> None:
    runner, state, questions = make_fake_laya_runner()
    original_state = deepcopy(state)
    original_questions = deepcopy(questions)

    first = runner.predict(state, questions)
    state_after_first = deepcopy(state)
    questions_after_first = deepcopy(questions)
    assert state == state_after_first
    assert questions == questions_after_first
    state["message"] = "changed"
    questions["intent"]["type"] = "choice"
    second = runner.predict(state, questions)

    assert isinstance(runner, BaseRunner)
    assert first == {"answers": {"intent": {"type": "noul", "noul": 1.0}}}
    assert second == first
    assert first is not second
    assert runner.seen[0] == (original_state, original_questions)
    assert runner.seen[1] == (state, questions)
    assert state["message"] == "changed"
    assert questions["intent"]["type"] == "choice"


def test_fake_laya_runner_exposes_stable_metadata() -> None:
    runner, _, _ = make_fake_laya_runner()

    assert runner.info() == {
        "runner": "fake-laya",
        "model": "fake",
        "revision": "test",
        "serving": "fake",
        "device": "cpu",
        "adapter_version": "test",
    }


def test_fake_router_tracks_scoped_route_counts() -> None:
    runner = make_fake_router()
    first_info = runner.info()

    runner.predict({"message": "hello"}, {"intent": {"type": "noul"}})
    runner.predict({"message": "bonjour"}, {"intent": {"type": "noul"}})
    info = runner.info()
    info["route_counts"]["english"] = 99

    assert isinstance(runner, BaseRunner)
    assert first_info["route_counts"] == {}
    assert runner.info()["route_counts"] == {"english": 2}


def test_make_runner_with_key_uses_only_a_test_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    from runners.jev_runner import JevRunner

    sentinel = "real-key-that-must-not-be-used"
    monkeypatch.setenv("TYPESAFE_API_KEY", sentinel)

    def fail_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("fixture construction must not make a request")

    with (
        patch("runners.jev_runner.requests.post", side_effect=fail_request),
        patch("runners.jev_runner.requests.sessions.Session.post", side_effect=fail_request),
        patch("runners.jev_runner.requests.sessions.Session.request", side_effect=fail_request),
    ):
        runner = make_runner_with_key()

    assert isinstance(runner, JevRunner)
    assert runner.api_key == "test-only-placeholder"
    assert os.environ["TYPESAFE_API_KEY"] == sentinel


def test_make_runner_with_key_default_session_transport_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = Mock(side_effect=AssertionError("network transport reached"))
    monkeypatch.setattr(requests.sessions.Session, "send", send)

    runner = make_runner_with_key()

    with pytest.raises(AssertionError):
        runner.session.post("https://api.typesafe.ai/v1/systemone")
    send.assert_not_called()


def test_make_runner_with_key_never_loads_dotenv_or_reads_real_env() -> None:
    import runners.jev_runner as jev_module
    from runners.jev_runner import JevRunner

    load = Mock(side_effect=AssertionError("fixture must not call load_dotenv"))
    original_open = open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            path = Path(os.fspath(file))
        except (TypeError, ValueError):
            return original_open(file, *args, **kwargs)
        if path == jev_module.REPOSITORY_ROOT / ".env":
            raise AssertionError("fixture must not read the repository .env")
        return original_open(file, *args, **kwargs)

    with (
        patch.object(jev_module, "load_dotenv", load),
        patch("builtins.open", guarded_open),
    ):
        runner = make_runner_with_key()

    assert isinstance(runner, JevRunner)
    load.assert_not_called()


def test_run_backend_import_test_uses_the_requested_module_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def import_module(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(__name__=name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert run_backend_import_test("torch") == {
        "backend": "torch",
        "module": "runners.pcd.engine_torch",
    }
    assert run_backend_import_test("mlx") == {
        "backend": "mlx",
        "module": "runners.pcd.engine_mlx",
    }
    assert imported == ["runners.pcd.engine_torch", "runners.pcd.engine_mlx"]


def test_run_collision_fixture_raises_the_future_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidGenerationError(Exception):
        pass

    module = ModuleType("runners.pcd.engine_common")
    module.__dict__["InvalidGenerationError"] = InvalidGenerationError
    monkeypatch.setitem(sys.modules, "runners.pcd.engine_common", module)

    with pytest.raises(InvalidGenerationError, match="collision fixture"):
        run_collision_fixture()


def test_score_question_is_deterministic_and_independent() -> None:
    first = score_question()
    second = score_question()
    cast(list[str], first["criteria"]).append("extra")

    assert second == {
        "qid": "sentiment",
        "type": "score",
        "instructions": "How positive is the sentiment?",
        "criteria": [
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
        "max_score": 4,
    }


def test_manifest_fixture_writes_only_deterministic_temporary_jsonl(tmp_path: Path) -> None:
    path = manifest_fixture(tmp_path)
    first_bytes = path.read_bytes()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert path == tmp_path / "manifest.jsonl"
    assert path.parent == tmp_path
    assert len(records) == 3
    assert records[0] == {
        "schema_version": 2,
        "dataset_version": "2.0.0",
        "suite_id": "fixture",
        "case_id": "fixture-0000",
        "order_index": 0,
        "split": "evaluation",
        "state": {"text": "0"},
        "questions": [{"qid": "q", "type": "noul", "instructions": "fixture question"}],
        "expected": {"q": 0},
        "provenance_id": "fixture",
        "label_provenance": "fixture",
    }
    assert [record["expected"] for record in records] == [
        {"q": 0},
        {"q": 1},
        {"q": 0},
    ]
    assert manifest_fixture(tmp_path).read_bytes() == first_bytes


def test_write_fake_run_writes_deterministic_files_under_tmp_path(tmp_path: Path) -> None:
    run_root = write_fake_run(tmp_path, "digest-a")
    other_root = write_fake_run(tmp_path, "digest-b")

    assert run_root == tmp_path / "run-digest-a"
    assert other_root == tmp_path / "run-digest-b"
    assert json.loads((run_root / "metadata.json").read_text(encoding="utf-8")) == {
        "run_id": "run-digest-a",
        "manifest_digest": "digest-a",
        "model": {"id": "fake", "revision": "test"},
        "suites": {},
    }
    assert (run_root / "predictions.jsonl").read_bytes() == b""
    assert json.loads((run_root / "summary.json").read_text(encoding="utf-8")) == {
        "decisions": 0,
        "suites": {},
    }


def test_write_fake_run_does_not_overwrite_an_existing_run(tmp_path: Path) -> None:
    run_root = write_fake_run(tmp_path, "digest")
    sentinel = run_root / "metadata.json"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_fake_run(tmp_path, "digest")

    assert sentinel.read_text(encoding="utf-8") == "keep"
