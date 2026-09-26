"""Runner interface. The v2 orchestrator owns this boundary; legacy run.py migration is deferred to Task 6."""

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

VALID_PHASES = ("warmup", "benchmark", "speed")
_ROUTING_MODEL_SENTINELS = frozenset(
    {
        "?",
        "-",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "undefined",
        "missing",
        "unspecified",
    }
)


def renormalize_choice_probabilities(probabilities: object, label: str) -> dict[str, float]:
    """Re-sum a vendor probability map onto the contract's unit-sum requirement.

    Model providers round probabilities for transport, so the values that arrive drift away from
    one by up to ``len(labels) * 5e-5`` and fail the contract's 1e-6 tolerance. Dividing by the
    observed total and absorbing the residual on the largest entry is order-preserving, so the
    reported choice stays the argmax and the projected values sum to one. No ordering or ranking
    changes; only transport rounding error is removed.
    """
    if isinstance(probabilities, (str, bytes, bytearray)) or not isinstance(probabilities, Mapping):
        raise TypeError(f"{label}: probabilities must be a mapping")
    if not probabilities:
        raise ValueError(f"{label}: probabilities must not be empty")
    labels: list[str] = []
    values: list[float] = []
    for name, value in probabilities.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"{label}: probability labels must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label}: probability {name!r} must be a number")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{label}: probability {name!r} must be finite and nonnegative")
        labels.append(name)
        values.append(number)
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError(f"{label}: probabilities must not sum to zero")
    scaled = [value / total for value in values]
    top = max(range(len(scaled)), key=lambda index: (scaled[index], -index))
    scaled[top] = 1.0 - math.fsum(scaled[:top] + scaled[top + 1 :])
    return dict(zip(labels, scaled))


class BaseRunner:
    name = "base"
    _metadata: Mapping[str, Any] | None = None

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        """Return answers keyed by the exact question IDs.

        Adapters deep-copy both inputs before calling model code, normalize vendor output at
        their edge, and return only ``answers`` plus optional ``_usage``, ``_routing``, and
        ``_raw_model`` fields. ``phase`` is optional and defaults to ``benchmark``. ``info()``
        records the actual model identity and adapter version. Usage fields are per-call and
        phase-independent. Callers retain ownership of their input mappings.
        """
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        """Return a fresh metadata snapshot for result records."""
        metadata = self._metadata
        if metadata is None:
            metadata = {
                "runner": self.name,
                "model": getattr(self, "model", self.name),
                "revision": getattr(self, "revision", "unknown"),
                "serving": getattr(self, "serving", "unknown"),
                "device": getattr(self, "device", "unknown"),
                "adapter_version": getattr(self, "adapter_version", "unknown"),
            }
        snapshot = deepcopy(dict(metadata))
        defaults = {
            "runner": self.name,
            "model": self.name,
            "revision": "unknown",
            "serving": "unknown",
            "device": "unknown",
            "adapter_version": "unknown",
        }
        for key, value in defaults.items():
            snapshot.setdefault(key, deepcopy(value))
        return snapshot

    def _copy_inputs(
        self, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return deepcopy(state), deepcopy(questions)

    @staticmethod
    def _validate_phase(phase: str) -> str:
        if not isinstance(phase, str) or phase not in VALID_PHASES:
            allowed = ", ".join(VALID_PHASES)
            raise ValueError(f"phase must be one of: {allowed}")
        return phase

    @staticmethod
    def _routing_metadata(
        snapshot: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any] | None, str | None, tuple[str, ...]]:
        if "routing" in snapshot:
            raw_routing = snapshot["routing"]
        elif "_routing" in snapshot:
            raw_routing = snapshot["_routing"]
        else:
            return None, None, ()
        if raw_routing is None:
            return None, None, ()
        if not isinstance(raw_routing, Mapping):
            raise TypeError("Router routing metadata must be a mapping")
        model = raw_routing.get("model")
        if (
            not isinstance(model, str)
            or not model.strip()
            or model.strip().casefold() in _ROUTING_MODEL_SENTINELS
        ):
            raise ValueError("Router routing model must be a non-empty, non-placeholder string")
        model_key = model.strip()
        raw_reasons: object
        if "reasons" in raw_routing:
            raw_reasons = raw_routing["reasons"]
        elif "reason" in raw_routing:
            raw_reasons = raw_routing["reason"]
        else:
            raw_reasons = ""
        if raw_reasons is None:
            return raw_routing, model_key, ("",)
        if isinstance(raw_reasons, str):
            reason_values = (raw_reasons,)
        elif isinstance(raw_reasons, (list, tuple, set, frozenset)):
            if not all(isinstance(reason, str) for reason in raw_reasons):
                raise TypeError("Router routing reasons must be strings")
            reason_values = tuple(raw_reasons)
        else:
            raise TypeError("Router routing reason must be a string or string collection")
        normalized = tuple(sorted({reason.strip() or "" for reason in reason_values}))
        return raw_routing, model_key, normalized or ("",)

    @staticmethod
    def _validate_answer_ids(questions: Mapping[str, Any], answers: object) -> Mapping[Any, Any]:
        if not isinstance(answers, Mapping):
            raise TypeError("answers must be a mapping keyed by question ID")
        expected = set(questions)
        actual = set(answers)
        missing = [qid for qid in questions if qid not in actual]
        extra = [qid for qid in answers if qid not in expected]
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={list(missing)}")
            if extra:
                details.append(f"extra={list(extra)}")
            raise ValueError("answer IDs must exactly match question IDs; " + "; ".join(details))
        return answers
