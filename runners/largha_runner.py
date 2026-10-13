"""Largha adapter: local CPU inference through the released phocinae-server runtime.

`phocinae-server` serves the released `Phocinae/Phocinae-Largha-150M-v1` weights with a
self-contained pure-torch forward pass and no network calls. The engine reads one string
state plus a typed question list and answers every question of a case in one batched
forward per call. `PHOC_MODEL_DIR` points at a downloaded copy of the released model
repository, `PHOC_DEVICE` selects the device (default CPU), `PHOC_THREADS` caps CPU
intra-op threads, and `PHOC_REVISION` overrides the recorded revision pin.

Question mapping (sealed manifest shape to engine shape and back):

- `choice`: each criteria label renders one engine option, `"<label>: <description>"` in
  criteria order. The engine returns per-option probabilities in request order; the
  adapter keys them back to the labels and renormalizes with
  `renormalize_choice_probabilities`.
- `noul`: the engine returns `[p(false), p(true)]`; the answer keeps `p(true)`.
- `score`: the engine scores nine ordinal levels, "level 2" .. "level 10". The answer
  reports the expectation over those levels, linearly rescaled onto the question's
  `[0, max_score]` range, which the contract rounds half-up.

The engine answers every question with its canonical per-type instruction; the manifest
`instructions` string is recorded in the predictions but is not part of the released
protocol. States render with their field names preserved, matching
`runners.qwen_runner.state_to_context`, because the sealed questions refer to those names.
"""

import math
import os
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from typing import Any

from runners.base import BaseRunner, renormalize_choice_probabilities

ADAPTER_VERSION = "1"
DEFAULT_MODEL = "Phocinae/Phocinae-Largha-150M-v1"
DEFAULT_REVISION = "6fd31334ad461265578a8b52cc38969b3960a9dc"
DEFAULT_DEVICE = "cpu"
PACKAGE = "phocinae-server"
SCORE_LEVEL_MIN = 2
SCORE_LEVEL_COUNT = 9
_SCORE_LEVEL_SPAN = SCORE_LEVEL_COUNT - 1


def state_to_context(state: object) -> str:
    """Render a state mapping as labelled text, matching the repository convention.

    The sealed questions refer to state field names directly, so the labels are kept.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return "\n".join(f"{key}: {value}" for key, value in state.items())
    raise TypeError("state must be a string or a mapping")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _option_text(label: str, description: object) -> str:
    description_text = _text(description)
    if not description_text:
        return label
    return f"{label}: {description_text}"


def _choice_labels(question: Mapping[str, Any], qid: str) -> tuple[str, ...]:
    criteria = question.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError(f"{qid}: choice criteria must be a non-empty mapping")
    labels: list[str] = []
    for label in criteria:
        if not isinstance(label, str) or not label:
            raise ValueError(f"{qid}: choice criteria labels must be non-empty strings")
        labels.append(label)
    return tuple(labels)


class LarghaRunner(BaseRunner):
    name = "largha"

    def __init__(self, model_dir: str | None = None, device: str | None = None) -> None:
        from phocinae.engine import Engine

        directory = os.environ.get("PHOC_MODEL_DIR") if model_dir is None else model_dir
        if not directory:
            raise ValueError(
                "PHOC_MODEL_DIR must point at a downloaded copy of the released model repository"
            )
        if not os.path.isfile(os.path.join(directory, "model.safetensors")):
            raise ValueError(f"PHOC_MODEL_DIR {directory!r} has no model.safetensors")
        target = os.environ.get("PHOC_DEVICE", DEFAULT_DEVICE) if device is None else device
        threads_text = os.environ.get("PHOC_THREADS", "")
        threads = int(threads_text) if threads_text.strip() else None
        self.engine = Engine(directory, device=target, threads=threads)
        self._metadata = {
            "runner": self.name,
            "model": DEFAULT_MODEL,
            "revision": os.environ.get("PHOC_REVISION", DEFAULT_REVISION),
            "serving": "local",
            "device": target,
            "adapter_version": ADAPTER_VERSION,
            "package": PACKAGE,
            "package_version": importlib_metadata.version(PACKAGE),
        }

    @staticmethod
    def _score_max_score(question: Mapping[str, Any], qid: str) -> float:
        criteria = question.get("criteria")
        if (
            isinstance(criteria, (str, bytes, bytearray))
            or not isinstance(criteria, Sequence)
            or not criteria
        ):
            raise ValueError(f"{qid}: score criteria must be a non-empty list")
        max_score = question.get("max_score")
        if (
            isinstance(max_score, bool)
            or not isinstance(max_score, (int, float))
            or not math.isfinite(max_score)
            or max_score < 0
        ):
            raise ValueError(f"{qid}: max_score must be a finite nonnegative number")
        return float(max_score)

    def _engine_question(self, question: Mapping[str, Any], qid: str) -> dict[str, Any]:
        question_type = question.get("type")
        if question_type == "choice":
            criteria = question["criteria"]
            labels = _choice_labels(question, qid)
            options = [_option_text(label, criteria[label]) for label in labels]
            return {"id": qid, "type": "choice", "options": options}
        if question_type == "noul":
            return {"id": qid, "type": "noul"}
        if question_type == "score":
            self._score_max_score(question, qid)
            return {"id": qid, "type": "score"}
        raise ValueError(f"{qid}: unsupported question type {question_type!r}")

    def _score_answer(
        self, question: Mapping[str, Any], probabilities: list[float], confidence: float, qid: str
    ) -> dict[str, Any]:
        max_score = self._score_max_score(question, qid)
        if len(probabilities) != SCORE_LEVEL_COUNT:
            raise ValueError(
                f"{qid}: engine returned {len(probabilities)} score levels, "
                f"expected {SCORE_LEVEL_COUNT}"
            )
        total = math.fsum(probabilities)
        if total <= 0.0:
            raise ValueError(f"{qid}: engine score probabilities must not sum to zero")
        expectation = (
            math.fsum(
                (SCORE_LEVEL_MIN + index) * value for index, value in enumerate(probabilities)
            )
            / total
        )
        score = (expectation - SCORE_LEVEL_MIN) / _SCORE_LEVEL_SPAN * max_score
        score = min(max(score, 0.0), max_score)
        return {"type": "score", "score": score, "confidence": float(confidence)}

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        state_text = state_to_context(copied_state)
        engine_questions: list[dict[str, Any]] = []
        for qid, question in copied_questions.items():
            if not isinstance(question, Mapping):
                raise TypeError(f"{qid}: question must be a mapping")
            engine_questions.append(self._engine_question(question, qid))
        answers_raw, confidences, _, usage, option_scores = self.engine.run(
            state_text, engine_questions, with_scores=True
        )
        answers: dict[str, Any] = {}
        raw_scores: dict[str, list[float]] = {}
        for qid, question in copied_questions.items():
            probabilities = [float(value) for value in option_scores[qid]]
            raw_scores[qid] = probabilities
            question_type = question.get("type")
            if question_type == "choice":
                labels = _choice_labels(question, qid)
                if len(probabilities) != len(labels):
                    raise ValueError(
                        f"{qid}: engine returned {len(probabilities)} option probabilities "
                        f"for {len(labels)} criteria"
                    )
                index = answers_raw[qid]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError(f"{qid}: engine choice index must be an integer")
                if not 0 <= index < len(labels):
                    raise ValueError(f"{qid}: engine choice index is outside the option list")
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[index],
                    "probabilities": renormalize_choice_probabilities(
                        dict(zip(labels, probabilities)), qid
                    ),
                    "confidence": float(confidences[qid]),
                }
            elif question_type == "noul":
                if len(probabilities) < 2:
                    raise ValueError(
                        f"{qid}: engine returned {len(probabilities)} noul probabilities"
                    )
                probability = probabilities[1]
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"{qid}: noul probability must be finite and in [0, 1]")
                answers[qid] = {"type": "noul", "noul": probability}
            elif question_type == "score":
                answers[qid] = self._score_answer(
                    question, probabilities, float(confidences[qid]), qid
                )
            else:
                raise ValueError(f"{qid}: unsupported question type {question_type!r}")
        return {
            "answers": self._validate_answer_ids(copied_questions, answers),
            "_usage": {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            },
            "_raw_model": {"option_scores": raw_scores},
        }
