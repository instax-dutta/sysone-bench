from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from runners.base import BaseRunner
from runners.pcd import engine
from runners.pcd.engine_common import (
    InvalidGenerationError,
    parse_legal_value,
    validate_probability_map,
)
from runners.pcd.schema import StructuredSchema

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PCD_SOURCE_REVISION = "6345318eea8cf81b68fa21962aae3ef688a50284"
PCD_SOURCE_REVISION_ROLE = "vendored_upstream_base_revision"
PCD_IMPLEMENTATION_FILES = (
    "engine_common.py",
    "engine_torch.py",
    "engine.py",
    "prompt_builder.py",
    "schema.py",
)
PCD_IMPLEMENTATION_DIGEST_ALGORITHM = "sha256"
ADAPTER_VERSION = "2"


def pcd_implementation_digest() -> str:
    digest = hashlib.sha256()
    digest.update(b"sysone-bench-pcd-implementation-v1\0")
    source_root = Path(__file__).resolve().parent / "pcd"
    for filename in PCD_IMPLEMENTATION_FILES:
        filename_bytes = filename.encode("utf-8")
        content = (source_root / filename).read_bytes()
        digest.update(len(filename_bytes).to_bytes(8, "big"))
        digest.update(filename_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


PCD_IMPLEMENTATION_DIGEST = pcd_implementation_digest()


def state_to_context(state: object) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        return "\n".join(f"{key}: {value}" for key, value in state.items())
    return "\n".join(str(value) for value in cast(Iterable[object], state))


def questions_to_schema(questions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        question_type = question.get("type")
        instructions = str(question.get("instructions", ""))
        if question_type == "choice":
            criteria = question.get("criteria", {})
            if not isinstance(criteria, Mapping) or not criteria:
                raise ValueError(f"choice question {question_id} requires criteria")
            choices = [str(label) for label in criteria]
            descriptions = "; ".join(
                f"{label}: {criteria[label]}" for label in choices if criteria[label]
            )
            schema[question_id] = {
                "type": "enum",
                "description": f"{instructions}; {descriptions}".rstrip("; "),
                "choices": choices,
            }
        elif question_type == "noul":
            schema[question_id] = {
                "type": "boolean",
                "description": instructions,
            }
        elif question_type == "score":
            criteria = question.get("criteria", [])
            if (
                not isinstance(criteria, Sequence)
                or isinstance(criteria, (str, bytes))
                or not criteria
            ):
                raise ValueError(f"score question {question_id} requires ordered criteria")
            levels = [str(level) for level in criteria]
            ordered_levels = ", ".join(levels)
            schema[question_id] = {
                "type": "enum",
                "description": (
                    f"Ordered levels: {ordered_levels}. {instructions} "
                    "Answer with exactly one level."
                ),
                "choices": levels,
                "_score_levels": True,
            }
        else:
            raise ValueError(f"unsupported question type: {question_type!r}")
    return schema


def _probabilities(
    field: Mapping[str, Any], choices: list[str], question_id: str
) -> dict[str, float]:
    raw = field.get("probabilities")
    if not isinstance(raw, Mapping):
        legacy = field.get("top_choices")
        if not isinstance(legacy, list):
            raise InvalidGenerationError(
                "field telemetry is missing probability map", field=question_id
            )
        raw = {}
        for item in legacy:
            if not isinstance(item, Mapping) or "choice" not in item or "probability" not in item:
                raise InvalidGenerationError(
                    "field telemetry contains an invalid choice probability",
                    field=question_id,
                )
            choice = str(item["choice"])
            if choice in raw:
                raise InvalidGenerationError(
                    "field telemetry contains duplicate choices", field=question_id
                )
            raw[choice] = item["probability"]
    return validate_probability_map(raw, choices, question_id)


def _selected_value(
    field: Mapping[str, Any],
    choices: list[str],
    question_id: str,
    *,
    boolean: bool = False,
) -> str:
    value = field.get("value")
    if boolean and isinstance(value, bool):
        value = "true" if value else "false"
    return parse_legal_value(value, choices, question_id, boolean=boolean)


class QwenRunner(BaseRunner):
    name = "qwen-pcd-15b"

    def __init__(self) -> None:
        backend = getattr(engine, "_mlx_engine", None) or getattr(engine, "_torch_engine", None)
        if backend is not None:
            backend.MODEL_ID = MODEL_ID
        self._metadata = {
            "runner": self.name,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "serving": "local",
            "device": "cpu",
            "adapter_version": ADAPTER_VERSION,
            "pcd_source_revision": PCD_SOURCE_REVISION,
            "pcd_source_revision_role": PCD_SOURCE_REVISION_ROLE,
            "pcd_implementation_digest": pcd_implementation_digest(),
            "engine": "parallel-constrained-decoding",
            "score_mapping": "enum-over-levels",
        }

    def predict(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        phase: str = "benchmark",
    ) -> dict[str, Any]:
        self._validate_phase(phase)
        copied_state, copied_questions = self._copy_inputs(state, questions)
        if not copied_questions:
            return {"answers": {}}
        schema = questions_to_schema(copied_questions)
        clean_schema = {
            field_name: {key: value for key, value in field.items() if not key.startswith("_")}
            for field_name, field in schema.items()
        }
        result = engine.run_parallel_generation(
            state_to_context(copied_state),
            StructuredSchema(clean_schema),
        )
        if not isinstance(result, Mapping):
            raise InvalidGenerationError("parallel generation result must be a mapping")
        telemetry = result.get("field_telemetry")
        if not isinstance(telemetry, Mapping):
            raise InvalidGenerationError("parallel generation result is missing field telemetry")
        answers: dict[str, Any] = {}
        for question_id, question in copied_questions.items():
            field = telemetry.get(question_id)
            if not isinstance(field, Mapping):
                raise InvalidGenerationError(
                    "parallel generation omitted a field", field=question_id
                )
            question_type = question["type"]
            if question_type == "choice":
                choices = [str(label) for label in question.get("criteria", {})]
                probabilities = _probabilities(field, choices, question_id)
                value = _selected_value(field, choices, question_id)
                answers[question_id] = {
                    "type": "choice",
                    "choice": value,
                    "probabilities": probabilities,
                    "confidence": probabilities[value],
                }
            elif question_type == "noul":
                probabilities = _probabilities(field, ["true", "false"], question_id)
                _selected_value(field, ["true", "false"], question_id, boolean=True)
                answers[question_id] = {
                    "type": "noul",
                    "noul": probabilities["true"],
                }
            else:
                levels = [str(level) for level in question["criteria"]]
                value = _selected_value(field, levels, question_id)
                confidence = _probabilities(field, levels, question_id)[value]
                answers[question_id] = {
                    "type": "score",
                    "score": float(levels.index(value)),
                    "confidence": confidence,
                }
        return {"answers": self._validate_answer_ids(copied_questions, answers)}

    def info(self) -> dict[str, Any]:
        return dict(super().info())


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "PCD_IMPLEMENTATION_DIGEST",
    "PCD_IMPLEMENTATION_DIGEST_ALGORITHM",
    "PCD_IMPLEMENTATION_FILES",
    "PCD_SOURCE_REVISION",
    "PCD_SOURCE_REVISION_ROLE",
    "QwenRunner",
    "engine",
    "pcd_implementation_digest",
    "questions_to_schema",
    "state_to_context",
]
