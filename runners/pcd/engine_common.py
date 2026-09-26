from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class InvalidGenerationError(ValueError):
    def __init__(
        self,
        message: str = "invalid generated output",
        field: str | None = None,
        generated_text: str | None = None,
        *,
        field_name: str | None = None,
    ) -> None:
        self.field = field_name if field_name is not None else field
        self.field_name = self.field
        self.generated_text = generated_text
        self.generated = generated_text
        self.reason = message
        details: list[str] = []
        if self.field is not None:
            details.append(f"field={self.field}")
        if generated_text is not None:
            details.append(f"generated_text={generated_text!r}")
        suffix = f" ({'; '.join(details)})" if details else ""
        super().__init__(f"{message}{suffix}")


@dataclass
class TrieNode:
    children: dict[int, TrieNode] = field(default_factory=dict)
    terminals: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class LegalChoice:
    value: str
    token_ids: tuple[int, ...]


def _probability_values(probabilities: object, field_name: str) -> list[float]:
    if isinstance(probabilities, (str, bytes)) or not isinstance(probabilities, Sequence):
        raise InvalidGenerationError("probabilities must be a sequence", field=field_name)
    values: list[float] = []
    for value in probabilities:
        if isinstance(value, bool):
            raise InvalidGenerationError("probabilities must be numeric", field=field_name)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidGenerationError("probabilities must be numeric", field=field_name) from exc
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise InvalidGenerationError(
                "probabilities must be finite and between zero and one", field=field_name
            )
        values.append(number)
    if not values:
        raise InvalidGenerationError("probabilities must not be empty", field=field_name)
    return values


def validate_probability_vector(
    probabilities: object,
    choices: Sequence[str] | None = None,
    field_name: str = "field",
    tolerance: float = 1e-6,
) -> list[float]:
    values = _probability_values(probabilities, field_name)
    if choices is not None and len(values) != len(choices):
        raise InvalidGenerationError(
            "probability count does not match legal choices",
            field=field_name,
        )
    total = math.fsum(values)
    if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise InvalidGenerationError("probabilities must sum to one", field=field_name)
    return values


def validate_probability_map(
    probabilities: object,
    choices: Sequence[str],
    field_name: str = "field",
    tolerance: float = 1e-6,
) -> dict[str, float]:
    if not isinstance(probabilities, Mapping):
        raise InvalidGenerationError("probabilities must be a mapping", field=field_name)
    expected = [str(choice) for choice in choices]
    if len(set(expected)) != len(expected):
        raise InvalidGenerationError("legal choices must be unique", field=field_name)
    if set(probabilities) != set(expected):
        raise InvalidGenerationError(
            "probability keys do not match legal choices", field=field_name
        )
    values = _probability_values([probabilities[key] for key in expected], field_name)
    total = math.fsum(values)
    if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise InvalidGenerationError("probabilities must sum to one", field=field_name)
    return {key: value for key, value in zip(expected, values)}


def build_field_telemetry(
    field_name: str,
    value: str,
    field_type: str,
    choices: Sequence[str],
    probabilities: Mapping[str, object] | Sequence[object],
) -> dict[str, Any]:
    normalized_choices = [str(choice) for choice in choices]
    if isinstance(probabilities, Mapping):
        normalized = validate_probability_map(probabilities, normalized_choices, field_name)
    else:
        values = validate_probability_vector(probabilities, normalized_choices, field_name)
        normalized = dict(zip(normalized_choices, values))
    if value not in normalized:
        raise InvalidGenerationError(
            "selected value is not a legal choice", field=field_name, generated_text=value
        )
    ordered = sorted(
        normalized_choices,
        key=lambda choice: (-normalized[choice], normalized_choices.index(choice)),
    )
    return {
        "value": value,
        "type": field_type,
        "confidence": normalized[value],
        "cardinality": len(normalized_choices),
        "probabilities": normalized,
        "choices": list(normalized_choices),
        "top_choices": [
            {"choice": choice, "probability": normalized[choice]} for choice in ordered
        ],
    }


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, int):
        encoded = [encoded]
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise TypeError("tokenizer.encode must return a sequence")
    result: list[int] = []
    for token_id in encoded:
        if isinstance(token_id, bool):
            raise TypeError("tokenizer returned a non-integer token id")
        try:
            integer = int(token_id)
        except (TypeError, ValueError) as exc:
            raise TypeError("tokenizer returned a non-integer token id") from exc
        if integer < 0:
            raise TypeError("tokenizer returned a negative token id")
        result.append(integer)
    if not result:
        raise ValueError("tokenizer returned no tokens")
    return result


def compile_legal_choices(
    tokenizer: Any,
    choices: Sequence[str],
    field_name: str,
    *,
    boolean: bool = False,
) -> list[LegalChoice]:
    normalized = ["true", "false"] if boolean else [str(choice) for choice in choices]
    if not normalized or len(set(normalized)) != len(normalized):
        raise InvalidGenerationError("legal choices must be non-empty and unique", field=field_name)
    compiled: list[LegalChoice] = []
    seen: dict[tuple[int, ...], str] = {}
    for choice in normalized:
        variants = [choice, f" {choice}"]
        if boolean:
            variants.extend([choice.capitalize(), f" {choice.capitalize()}"])
        token_sequences: list[tuple[int, ...]] = []
        for variant in variants:
            try:
                token_sequences.append(tuple(_encode(tokenizer, variant)))
            except (TypeError, ValueError, InvalidGenerationError):
                continue
        if not token_sequences:
            raise InvalidGenerationError(
                "legal choice has no encodable continuation",
                field=field_name,
                generated_text=choice,
            )
        selected = token_sequences[0]
        previous = seen.get(selected)
        if previous is not None and previous != choice:
            raise InvalidGenerationError(
                "legal token collision", field=field_name, generated_text=choice
            )
        seen[selected] = choice
        compiled.append(LegalChoice(value=choice, token_ids=selected))
    return compiled


def build_legal_trie(choices: Sequence[LegalChoice]) -> TrieNode:
    root = TrieNode()
    for index, choice in enumerate(choices):
        node = root
        for token_id in choice.token_ids:
            node = node.children.setdefault(token_id, TrieNode())
        node.terminals.append(index)
    return root


def parse_legal_value(
    text: object,
    choices: Sequence[str],
    field_name: str,
    *,
    boolean: bool = False,
) -> str:
    if not isinstance(text, str):
        raise InvalidGenerationError(
            "generated value is not text", field=field_name, generated_text=str(text)
        )
    candidate = text.strip()
    if not candidate:
        raise InvalidGenerationError("generated value is empty", field=field_name)
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'":
        candidate = candidate[1:-1].strip()
    normalized = [str(choice) for choice in choices]
    if boolean:
        folded = candidate.casefold()
        matches = [choice for choice in normalized if choice.casefold() == folded]
    else:
        matches = [choice for choice in normalized if choice == candidate]
    if len(matches) != 1:
        reason = (
            "generated value is unparseable"
            if not matches
            else "generated value collides with legal choices"
        )
        raise InvalidGenerationError(reason, field=field_name, generated_text=text)
    return matches[0]


def parse_json_value(
    value: object,
    choices: Sequence[str],
    field_name: str,
    *,
    boolean: bool = False,
) -> str:
    if boolean and isinstance(value, bool):
        return "true" if value else "false"
    return parse_legal_value(value, choices, field_name, boolean=boolean)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "InvalidGenerationError",
    "LegalChoice",
    "TrieNode",
    "build_field_telemetry",
    "build_legal_trie",
    "compile_legal_choices",
    "json_text",
    "parse_json_value",
    "parse_legal_value",
    "validate_probability_map",
    "validate_probability_vector",
]
