from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

from runners.base import VALID_PHASES, BaseRunner

if TYPE_CHECKING:
    from runners.jev_runner import JevRunner

_TEST_API_KEY = "test-only-placeholder"


def _fail_closed_session_post(*args: object, **kwargs: object) -> None:
    raise AssertionError("test helper session transport is disabled")


class FakeLayaRunner(BaseRunner):
    name = "fake-laya"

    def __init__(self) -> None:
        self.seen: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        self.seen.append((deepcopy(state), deepcopy(questions)))
        return {"answers": {"intent": {"type": "noul", "noul": 1.0}}}

    def info(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "model": "fake",
            "revision": "test",
            "serving": "fake",
            "device": "cpu",
            "adapter_version": "test",
        }


class FakeRouter(FakeLayaRunner):
    name = "fake-router"

    def __init__(self) -> None:
        super().__init__()
        self.route_counts: dict[str, int] = {}
        self.route_counts_by_phase: dict[str, dict[str, int]] = {
            phase: {} for phase in VALID_PHASES
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        self.route_counts["english"] = self.route_counts.get("english", 0) + 1
        phase_counts = self.route_counts_by_phase[phase]
        phase_counts["english"] = phase_counts.get("english", 0) + 1
        return super().predict(state, questions, phase=phase)

    def info(self) -> dict[str, Any]:
        info = super().info()
        info["route_counts"] = dict(self.route_counts)
        info["route_counts_by_phase"] = {
            phase: dict(self.route_counts_by_phase[phase]) for phase in VALID_PHASES
        }
        return info


def make_fake_laya_runner() -> tuple[BaseRunner, dict[str, Any], dict[str, Any]]:
    state: dict[str, Any] = {"message": "hello"}
    questions: dict[str, Any] = {"intent": {"type": "noul"}}
    return FakeLayaRunner(), state, questions


def make_fake_router() -> BaseRunner:
    return FakeRouter()


def make_runner_with_key() -> JevRunner:
    from runners.jev_runner import JevRunner

    with (
        patch.dict(os.environ, {"TYPESAFE_API_KEY": _TEST_API_KEY}, clear=True),
        patch("runners.jev_runner.load_dotenv", return_value=False),
    ):
        runner = JevRunner(
            model="jev-1.13.0",
            base_url="https://api.typesafe.ai/v1/systemone",
        )
    cast(Any, runner.session).post = _fail_closed_session_post
    return runner


def run_backend_import_test(backend: str) -> dict[str, str]:
    import importlib

    module_name = "runners.pcd.engine_torch" if backend == "torch" else "runners.pcd.engine_mlx"
    module = importlib.import_module(module_name)
    return {"backend": backend, "module": module.__name__}


def run_collision_fixture() -> None:
    from runners.pcd.engine_common import InvalidGenerationError

    raise InvalidGenerationError("collision fixture")


def score_question() -> dict[str, object]:
    return {
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


def manifest_fixture(tmp_path: Path) -> Path:
    records = [
        {
            "schema_version": 2,
            "dataset_version": "2.0.0",
            "suite_id": "fixture",
            "case_id": f"fixture-{index:04d}",
            "order_index": index,
            "split": "evaluation",
            "state": {"text": str(index)},
            "questions": [{"qid": "q", "type": "noul", "instructions": "fixture question"}],
            "expected": {"q": index % 2},
            "provenance_id": "fixture",
            "label_provenance": "fixture",
        }
        for index in range(3)
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def write_fake_run(tmp_path: Path, manifest_digest_value: str) -> Path:
    run_root = tmp_path / f"run-{manifest_digest_value}"
    run_root.mkdir()
    (run_root / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_root.name,
                "manifest_digest": manifest_digest_value,
                "model": {"id": "fake", "revision": "test"},
                "suites": {},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "predictions.jsonl").write_text("", encoding="utf-8")
    (run_root / "summary.json").write_text(
        json.dumps({"decisions": 0, "suites": {}}),
        encoding="utf-8",
    )
    return run_root
