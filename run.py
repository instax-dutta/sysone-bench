"""Run sealed v2 manifests through the append-only orchestration boundary."""

import argparse
import copy
import hashlib
import inspect
import json
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.timing import measure_call
from benchmark.usage import UsageCounter
from runners.base import BaseRunner
from runners.laya_runner import LayaRunner

DEFAULT_MANIFEST = Path("datasets/v2/manifest.jsonl")
DEFAULT_MANIFEST_CHECKSUM = Path("datasets/v2/manifest.sha256")
DEFAULT_OUTPUT_ROOT = Path("results/v2/runs")


class _MissingLaya:
    @staticmethod
    def ece_score(confidences: Any, correctness: Any) -> float:
        pairs = [
            (float(confidence), float(correct))
            for confidence, correct in zip(confidences, correctness, strict=True)
        ]
        if not pairs:
            return 0.0
        return sum(abs(confidence - correct) for confidence, correct in pairs) / len(pairs)

    @staticmethod
    def triage_questions() -> dict[str, Any]:
        raise RuntimeError("laya package is required for legacy speed scaling")


laya: Any
try:
    import laya as laya_module
except ModuleNotFoundError as error:
    if error.name != "laya":
        raise
    laya = _MissingLaya()
else:
    laya = laya_module


def noul_prob(a: Mapping[str, Any]) -> float:
    return float(a.get("noul", a.get("probability", a.get("boolean", 0.0))))


def question_hash(questions: object) -> str:
    return hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:12]


def _call_predict(
    predict: Callable[..., dict[str, Any]],
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    phase: str = "benchmark",
) -> dict[str, Any]:
    parameters = inspect.signature(predict).parameters.values()
    accepts_phase = any(
        parameter.name == "phase"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for parameter in parameters
    ) or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_phase:
        return predict(state, questions, phase=phase)
    return predict(state, questions)


def run_suite(
    runner: Any,
    questions: Mapping[str, Any],
    cases: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    phase: str = "benchmark",
) -> dict[str, Any]:
    lat: list[float] = []
    correct: list[int] = []
    confs: list[float] = []
    y_true: list[Any] = []
    y_prob: list[float] = []
    per_q: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []
    score_err: list[float] = []
    for state, expected in cases:
        res, elapsed = measure_call(
            time.perf_counter,
            partial(_call_predict, runner.predict, state, questions, phase),
        )
        lat.append(elapsed * 1000)
        for qid, exp in expected.items():
            a = res["answers"][qid]
            if a["type"] == "choice":
                ok = int(a["choice"] == exp)
                c = float(a["confidence"])
            elif a["type"] == "score":
                pred = float(a["score"])
                score_err.append(abs(pred - float(exp)))
                ok = int(abs(pred - float(exp)) <= 0.5)
                c = float(a.get("confidence", 0.0))
            else:
                p = noul_prob(a)
                ok = int(int(p >= 0.5) == exp)
                c = max(p, 1 - p)
                y_true.append(exp)
                y_prob.append(p)
            correct.append(ok)
            confs.append(c)
            per_q.setdefault(qid, []).append(ok)
            rows.append({"qid": qid, "ok": ok, "conf": round(c, 3)})
    acc = float(np.mean(correct))
    ece = float(laya.ece_score(np.array(confs), np.array(correct)))
    brier = float(np.mean([(p - t) ** 2 for p, t in zip(y_prob, y_true)])) if y_prob else None
    mae = round(float(np.mean(score_err)), 4) if score_err else None
    return {
        "states": len(cases),
        "decisions": len(correct),
        "accuracy": round(acc, 4),
        "ece": round(ece, 4),
        "brier_noul": round(brier, 4) if brier is not None else None,
        "score_mae": mae,
        "ms_per_call_mean": round(statistics.mean(lat), 1),
        "ms_per_call_p50": round(statistics.median(lat), 1),
        "ms_per_call_p95": round(float(np.percentile(lat, 95)), 1),
        "per_question_acc": {k: round(float(np.mean(v)), 4) for k, v in per_q.items()},
        "rows": rows,
    }


def speed_scaling(runner: Any) -> dict[str, dict[str, float]]:
    base = {"message": "My payment failed twice, error 500, need help urgently."}
    tq = laya.triage_questions()
    keys = list(tq.keys())
    out: dict[str, dict[str, float]] = {}
    for nq in [1, 5, 10, 20]:
        qs = {
            f"q{i}_{keys[i % len(keys)]}": copy.deepcopy(tq[keys[i % len(keys)]]) for i in range(nq)
        }
        for _ in range(2):
            _call_predict(runner.predict, base, qs, "warmup")
        ts = []
        for _ in range(10):
            _, elapsed = measure_call(
                time.perf_counter,
                partial(_call_predict, runner.predict, base, qs, "speed"),
            )
            ts.append(elapsed * 1000)
        out[str(nq)] = {
            "ms_per_call_p50": round(statistics.median(ts), 1),
            "ms_per_q": round(statistics.median(ts) / nq, 1),
        }
    return out


class TrackUsage:
    """Wraps a runner, accumulating per-call token usage into a shared counter."""

    def __init__(self, runner: Any, counter: UsageCounter | dict[str, int]) -> None:
        self.runner = runner
        self.counter = counter

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        res = _call_predict(self.runner.predict, state, questions, phase)
        usage = res.get("_usage")
        if usage is None:
            usage = {}
        if not isinstance(usage, Mapping):
            raise TypeError("runner usage must be a mapping")
        if isinstance(self.counter, UsageCounter):
            self.counter.record(phase, usage)
        elif isinstance(self.counter, dict):
            self.counter["input_tokens"] += int(usage.get("input_tokens", 0))
            self.counter["output_tokens"] += int(usage.get("output_tokens", 0))
            self.counter["calls"] += 1
        else:
            raise TypeError("counter must be a UsageCounter or dictionary")
        return res


def build_runner(name: str) -> BaseRunner:
    if name == "laya":
        return LayaRunner()
    if name == "laya-router":
        from runners.router_runner import RouterRunner

        return RouterRunner()
    if name == "qwen":
        from runners.qwen_runner import QwenRunner

        return QwenRunner()
    if name == "jev":
        from runners.jev_runner import JevRunner

        return JevRunner()
    raise ValueError(f"unknown model: {name}")


def _print_run_progress(model: str, suite_id: str, split: str, decisions: int) -> None:
    print(f"[{model}] {suite_id}/{split}: {decisions} decisions")


def main(argv: Sequence[str] | None = None) -> int:
    from benchmark.orchestrator import run_all_v2

    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("--models", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-checksum", type=Path, default=DEFAULT_MANIFEST_CHECKSUM)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        return 1
    if not args.manifest.is_file():
        print("sealed manifest is required", file=sys.stderr)
        return 1
    if not args.manifest_checksum.is_file():
        print("sealed manifest checksum is required", file=sys.stderr)
        return 1
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if not models:
        print("run failed: --models must contain at least one model", file=sys.stderr)
        return 2
    try:
        paths = run_all_v2(
            models,
            args.manifest,
            args.output_root,
            manifest_checksum_path=args.manifest_checksum,
            run_id=args.run_id,
            progress=_print_run_progress,
        )
    except RuntimeError:
        print("run failed: runner execution failed", file=sys.stderr)
        return 1
    except (ImportError, OSError, TypeError, ValueError) as error:
        print(f"run failed: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"run artifacts: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
