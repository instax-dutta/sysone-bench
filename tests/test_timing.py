import json
from collections.abc import Iterator, Mapping
from typing import Any

import pytest

import run
from benchmark import timing
from benchmark.timing import measure_call, speed_scaling
from benchmark.usage import UsageCounter
from runners.base import BaseRunner


class FakeTimingRunner(BaseRunner):
    name = "fake-timing"

    def __init__(self) -> None:
        self.observed_sizes: list[int] = []
        self.observed_phases: list[str] = []

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        self.observed_sizes.append(len(questions))
        self.observed_phases.append(phase)
        questions.clear()
        usage = {
            "input_tokens": 1 if phase == "warmup" else 2,
            "output_tokens": 1 if phase == "warmup" else 3,
        }
        return {"answers": {}, "_usage": usage}


class FakeSuiteRunner(BaseRunner):
    name = "fake-suite"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.phases: list[str] = []

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        self.events.append("predict")
        self.phases.append(phase)
        return {
            "answers": {"q": {"type": "noul", "noul": 0.75}},
            "_usage": {"input_tokens": 3, "output_tokens": 1},
        }


class FakeLegacyRunner(BaseRunner):
    name = "fake-legacy"

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls = 0
        self.events = events if events is not None else []

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        self.events.append("legacy_predict")
        return {
            "answers": {},
            "_usage": {"input_tokens": 4, "output_tokens": 2},
        }


class FakeLegacySuiteRunner(FakeLegacyRunner):
    name = "fake-legacy-suite"

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        self.events.append("legacy_predict")
        return {
            "answers": {"q": {"type": "noul", "noul": 0.75}},
            "_usage": {"input_tokens": 4, "output_tokens": 2},
        }


class FakeKwargsRunner:
    name = "fake-kwargs"

    def __init__(self) -> None:
        self.phases: list[str] = []

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.phases.append(kwargs["phase"])
        return {
            "answers": {},
            "_usage": {"input_tokens": 5, "output_tokens": 1},
        }


class TypeErrorRunner:
    name = "fake-type-error"

    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        raise TypeError("model body failed")


def clock_values(durations: list[float]) -> Iterator[float]:
    for duration in durations:
        yield 0.0
        yield duration


def test_usage_is_separated_by_phase():
    counter = UsageCounter()
    counter.record("benchmark", {"input_tokens": 10, "output_tokens": 2})
    counter.record("speed", {"input_tokens": 3, "output_tokens": 1})

    assert counter.as_dict() == {
        "warmup": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "benchmark": {"input_tokens": 10, "output_tokens": 2, "calls": 1},
        "speed": {"input_tokens": 3, "output_tokens": 1, "calls": 1},
    }


def test_usage_aggregates_calls_within_each_phase():
    counter = UsageCounter()
    counter.record("warmup", {})
    counter.record("warmup", {"input_tokens": 2, "output_tokens": 1})
    counter.record("benchmark", {"input_tokens": 4, "output_tokens": 3})

    assert counter.as_dict()["warmup"] == {
        "input_tokens": 2,
        "output_tokens": 1,
        "calls": 2,
    }
    assert counter.as_dict()["benchmark"] == {
        "input_tokens": 4,
        "output_tokens": 3,
        "calls": 1,
    }
    assert counter.as_dict()["speed"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "calls": 0,
    }


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_usage_rejects_negative_token_counts(field):
    counter = UsageCounter()

    with pytest.raises(ValueError, match=field):
        counter.record("benchmark", {field: -1})

    assert counter.as_dict()["benchmark"]["calls"] == 0


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_usage_rejects_non_integer_token_counts(field, value):
    counter = UsageCounter()

    with pytest.raises(TypeError, match=field):
        counter.record("benchmark", {field: value})

    assert counter.as_dict()["benchmark"]["calls"] == 0


def test_usage_rejects_unknown_phase():
    counter = UsageCounter()

    with pytest.raises(ValueError, match="phase"):
        counter.record("calibration", {})

    assert counter.as_dict() == {
        "warmup": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "benchmark": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "speed": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
    }


def test_usage_rejects_non_mapping_usage():
    counter = UsageCounter()

    with pytest.raises(TypeError, match="usage"):
        counter.record("benchmark", [("input_tokens", 1)])

    assert counter.as_dict()["benchmark"]["calls"] == 0


def test_usage_snapshots_are_deterministic_json_and_input_isolated():
    counter = UsageCounter()
    usage = {
        "input_tokens": 7,
        "output_tokens": 4,
        "provider": {"cached_tokens": 3},
    }

    counter.record("speed", usage)
    usage["input_tokens"] = 99
    usage["provider"]["cached_tokens"] = 88
    first = counter.as_dict()
    first["speed"]["input_tokens"] = 100
    first["speed"]["provider"] = "mutated"
    second = counter.as_dict()

    assert second == {
        "warmup": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "benchmark": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "speed": {"input_tokens": 7, "output_tokens": 4, "calls": 1},
    }
    assert json.dumps(second, sort_keys=True, separators=(",", ":")) == (
        '{"benchmark":{"calls":0,"input_tokens":0,"output_tokens":0},'
        '"speed":{"calls":1,"input_tokens":7,"output_tokens":4},'
        '"warmup":{"calls":0,"input_tokens":0,"output_tokens":0}}'
    )
    assert first is not second
    assert first["speed"] is not second["speed"]


def test_measure_call_wraps_the_call_and_preserves_unrounded_result_and_identity():
    events: list[str] = []
    values = iter([10.0, 10.000001])
    result = object()

    def clock() -> float:
        events.append("clock")
        return next(values)

    def call() -> object:
        events.append("call")
        return result

    returned, elapsed = measure_call(clock, call)

    assert returned is result
    assert elapsed == 9.999999992515995e-07
    assert isinstance(elapsed, float)
    assert events == ["clock", "call", "clock"]


def test_speed_scaling_uses_two_warmups_and_ten_timed_calls_per_size(monkeypatch):
    durations = [float(value) for value in range(1, 21)]
    values = iter(clock_values(durations))
    monkeypatch.setattr(timing, "perf_counter", lambda: next(values))
    runner = FakeTimingRunner()
    records: list[tuple[str, dict[str, int]]] = []

    class RecordingUsageCounter(UsageCounter):
        def record(self, phase: str, usage: Mapping[str, Any]) -> None:
            records.append((phase, dict(usage)))
            super().record(phase, usage)

    monkeypatch.setattr(timing, "UsageCounter", RecordingUsageCounter)

    result = speed_scaling(runner, [2, 3])

    assert result == {
        "2": {"p50": 5.5, "p95": 9.549999999999999, "sample_count": 10.0},
        "3": {"p50": 15.5, "p95": 19.549999999999997, "sample_count": 10.0},
    }
    assert list(result) == ["2", "3"]
    assert runner.observed_sizes == [2] * 12 + [3] * 12
    assert (
        runner.observed_phases == ["warmup"] * 2 + ["speed"] * 10 + ["warmup"] * 2 + ["speed"] * 10
    )
    assert [phase for phase, _ in records] == runner.observed_phases
    assert records[0] == ("warmup", {"input_tokens": 1, "output_tokens": 1})
    assert records[2] == ("speed", {"input_tokens": 2, "output_tokens": 3})
    with pytest.raises(StopIteration):
        next(values)


def test_speed_scaling_percentiles_are_linear_and_unrounded(monkeypatch):
    values = iter(clock_values([0.25, 0.5, 0.75]))
    monkeypatch.setattr(timing, "perf_counter", lambda: next(values))
    runner = FakeTimingRunner()

    result = speed_scaling(runner, [1], repetitions=3)

    assert result == {"1": {"p50": 0.5, "p95": 0.725, "sample_count": 3.0}}
    assert runner.observed_phases == ["warmup", "warmup", "speed", "speed", "speed"]


def test_percentile_sorts_unsorted_samples_before_linear_interpolation():
    samples = [3.0, 1.0, 2.0]

    assert timing._percentile(samples, 0.50) == 2.0
    assert timing._percentile(samples, 0.95) == 2.9
    assert samples == [3.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "sizes",
    [None, [], [0], [-1], [True], [1.0], ["1"], [1, 1]],
)
def test_speed_scaling_rejects_invalid_sizes_before_calling_runner(monkeypatch, sizes):
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0

    monkeypatch.setattr(timing, "perf_counter", clock)
    runner = FakeTimingRunner()

    with pytest.raises((TypeError, ValueError), match="size|sequence"):
        speed_scaling(runner, sizes)

    assert runner.observed_phases == []
    assert clock_calls == 0


@pytest.mark.parametrize("repetitions", [None, 0, -1, True, 1.0, "10"])
def test_speed_scaling_rejects_invalid_repetitions_before_calling_runner(monkeypatch, repetitions):
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0

    monkeypatch.setattr(timing, "perf_counter", clock)
    runner = FakeTimingRunner()

    with pytest.raises((TypeError, ValueError), match="repetitions"):
        speed_scaling(runner, [1], repetitions=repetitions)

    assert runner.observed_phases == []
    assert clock_calls == 0


def test_speed_scaling_returns_deterministic_json_snapshots(monkeypatch):
    first_values = iter(clock_values([1.0] * 30))
    second_values = iter(clock_values([1.0] * 30))
    monkeypatch.setattr(timing, "perf_counter", lambda: next(first_values))
    first = speed_scaling(FakeTimingRunner(), [1, 2, 4])
    monkeypatch.setattr(timing, "perf_counter", lambda: next(second_values))
    second = speed_scaling(FakeTimingRunner(), [1, 2, 4])

    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_track_usage_routes_explicit_phases_without_replacing_runner_result():
    runner = FakeTimingRunner()
    counter = UsageCounter()
    tracked = run.TrackUsage(runner, counter)

    returned = tracked.predict({}, {}, phase="speed")

    assert returned == {"answers": {}, "_usage": {"input_tokens": 2, "output_tokens": 3}}
    assert runner.observed_phases == ["speed"]
    assert counter.as_dict()["speed"] == {
        "input_tokens": 2,
        "output_tokens": 3,
        "calls": 1,
    }
    assert counter.as_dict()["benchmark"]["calls"] == 0


def test_track_usage_omits_phase_for_legacy_runner_and_records_requested_phase():
    runner = FakeLegacyRunner()
    counter = UsageCounter()
    tracked = run.TrackUsage(runner, counter)

    returned = tracked.predict({}, {}, phase="warmup")

    assert returned == {"answers": {}, "_usage": {"input_tokens": 4, "output_tokens": 2}}
    assert runner.calls == 1
    assert counter.as_dict()["warmup"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "calls": 1,
    }
    assert counter.as_dict()["benchmark"]["calls"] == 0


def test_track_usage_supports_historical_dictionary_counter() -> None:
    runner = FakeTimingRunner()
    counter = {"input_tokens": 3, "output_tokens": 1, "calls": 0}
    tracked = run.TrackUsage(runner, counter)

    returned = tracked.predict({}, {}, phase="benchmark")

    assert returned == {"answers": {}, "_usage": {"input_tokens": 2, "output_tokens": 3}}
    assert counter == {"input_tokens": 5, "output_tokens": 4, "calls": 1}


def test_call_predict_uses_the_legacy_two_argument_form():
    runner = FakeLegacyRunner()

    returned = run._call_predict(runner.predict, {}, {}, phase="warmup")

    assert returned == {"answers": {}, "_usage": {"input_tokens": 4, "output_tokens": 2}}
    assert runner.calls == 1
    assert runner.events == ["legacy_predict"]


def test_call_predict_forwards_phase_to_explicit_and_kwargs_runners():
    explicit_runner = FakeTimingRunner()
    kwargs_runner = FakeKwargsRunner()

    run._call_predict(explicit_runner.predict, {}, {}, phase="speed")
    run._call_predict(kwargs_runner.predict, {}, {}, phase="benchmark")

    assert explicit_runner.observed_phases == ["speed"]
    assert kwargs_runner.phases == ["benchmark"]


def test_call_predict_does_not_retry_model_type_errors():
    runner = TypeErrorRunner()

    with pytest.raises(TypeError, match="model body failed"):
        run._call_predict(runner.predict, {}, {}, phase="speed")

    assert runner.calls == 1


def test_run_suite_supports_a_raw_legacy_runner_with_exact_timing(monkeypatch):
    events: list[str] = []
    values = iter([1.0, 1.125])
    runner = FakeLegacySuiteRunner(events)
    monkeypatch.setattr(run.time, "perf_counter", lambda: events.append("clock") or next(values))
    monkeypatch.setitem(run.laya.__dict__, "ece_score", lambda confidences, correct: 0.0)

    suite = run.run_suite(
        runner,
        {"q": {"type": "noul"}},
        [({"message": "hello"}, {"q": 1})],
        phase="warmup",
    )

    assert suite["ms_per_call_mean"] == 125.0
    assert suite["decisions"] == 1
    assert events == ["clock", "legacy_predict", "clock"]
    assert runner.calls == 1


def test_legacy_speed_scaling_uses_two_argument_calls_and_exact_counts(monkeypatch):
    values = iter(clock_values([0.001] * 40))
    runner = FakeLegacySuiteRunner()
    monkeypatch.setattr(run.time, "perf_counter", lambda: next(values))
    monkeypatch.setattr(
        run.laya,
        "triage_questions",
        lambda: {"q0": {"type": "noul"}},
        raising=False,
    )

    result = run.speed_scaling(runner)

    assert result == {
        "1": {"ms_per_call_p50": 1.0, "ms_per_q": 1.0},
        "5": {"ms_per_call_p50": 1.0, "ms_per_q": 0.2},
        "10": {"ms_per_call_p50": 1.0, "ms_per_q": 0.1},
        "20": {"ms_per_call_p50": 1.0, "ms_per_q": 0.1},
    }
    assert runner.calls == 48
    assert runner.events == ["legacy_predict"] * 48


def test_run_suite_measures_and_propagates_the_explicit_phase(monkeypatch):
    events: list[str] = []
    values = iter([1.0, 1.125])
    runner = run.TrackUsage(FakeSuiteRunner(events), UsageCounter())
    monkeypatch.setattr(run.time, "perf_counter", lambda: events.append("clock") or next(values))
    monkeypatch.setitem(run.laya.__dict__, "ece_score", lambda confidences, correct: 0.0)

    suite = run.run_suite(
        runner,
        {"q": {"type": "noul"}},
        [({"message": "hello"}, {"q": 1})],
        phase="warmup",
    )

    assert suite["ms_per_call_mean"] == 125.0
    assert suite["decisions"] == 1
    assert events == ["clock", "predict", "clock"]
    assert runner.runner.phases == ["warmup"]
    assert runner.counter.as_dict()["warmup"] == {
        "input_tokens": 3,
        "output_tokens": 1,
        "calls": 1,
    }
