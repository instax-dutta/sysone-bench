from collections.abc import Mapping
from typing import Any

_PHASES = ("warmup", "benchmark", "speed")
_COUNTER_FIELDS = ("input_tokens", "output_tokens", "calls")


class UsageCounter:
    def __init__(self) -> None:
        self._counts = {phase: {field: 0 for field in _COUNTER_FIELDS} for phase in _PHASES}

    @staticmethod
    def _token_count(usage: Mapping[str, Any], field: str) -> int:
        value = usage.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value

    def record(self, phase: str, usage: Mapping[str, Any]) -> None:
        if not isinstance(phase, str) or phase not in _PHASES:
            raise ValueError(f"phase must be one of: {', '.join(_PHASES)}")
        if not isinstance(usage, Mapping):
            raise TypeError("usage must be a mapping")
        additions = {
            "input_tokens": self._token_count(usage, "input_tokens"),
            "output_tokens": self._token_count(usage, "output_tokens"),
        }
        counts = self._counts[phase]
        for field in _COUNTER_FIELDS:
            counts[field] += additions.get(field, 1)

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {phase: dict(counts) for phase, counts in self._counts.items()}
