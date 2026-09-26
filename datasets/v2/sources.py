from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

_FULL_COMMIT_REVISION_LENGTHS = frozenset({40, 64})
_HEX_DIGITS = frozenset("0123456789abcdef")


def require_resolved_revision(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in _FULL_COMMIT_REVISION_LENGTHS
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{context} must use a full lowercase hexadecimal resolved revision")
    return value


SUITE_SOURCE_IDS: dict[str, str] = {
    "agnews": "agnews",
    "emotion": "emotion",
    "banking77_12": "banking77",
    "mnli": "mnli",
    "sst5": "sst5",
}

SOURCES: dict[str, dict[str, Any]] = {
    "agnews": {
        "canonical": "fancyzhx/ag_news",
        "wrapper": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "split": "test",
        "license": "unknown",
        "citation": "https://huggingface.co/datasets/fancyzhx/ag_news",
    },
    "emotion": {
        "canonical": "dair-ai/emotion",
        "wrapper": "dair-ai/emotion",
        "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
        "split": "test",
        "license": "other",
        "citation": "https://huggingface.co/datasets/dair-ai/emotion",
    },
    "banking77": {
        "canonical": "PolyAI/banking77",
        "wrapper": "mteb/banking77",
        "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2",
        "split": "test",
        "license": "mit",
        "citation": "https://huggingface.co/datasets/PolyAI/banking77",
    },
    "mnli": {
        "canonical": "nyu-mll/glue",
        "wrapper": "nyu-mll/glue",
        "config": "mnli",
        "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        "split": "validation_matched",
        "license": "other",
        "citation": "https://huggingface.co/datasets/nyu-mll/glue",
    },
    "sst5": {
        "canonical": "SetFit/sst5",
        "wrapper": "SetFit/sst5",
        "revision": "e51bdcd8cd3a30da231967c1a249ba59361279a3",
        "split": "test",
        "license": "not_declared",
        "citation": "https://huggingface.co/datasets/SetFit/sst5",
    },
}

_LOCK_PATH = Path(__file__).with_name("sources.lock.json")
_REQUIRED_FIELDS = frozenset({"canonical", "wrapper", "revision", "split", "license", "citation"})
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"config"}


def _load_lock() -> dict[str, dict[str, Any]]:
    try:
        payload: object = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read source lock: {error}") from error

    if not isinstance(payload, dict):
        raise TypeError("source lock must contain an object")
    if set(payload) != set(SOURCES):
        raise ValueError("source lock source IDs do not match SOURCES")

    records: dict[str, dict[str, Any]] = {}
    for source_id, raw_record in payload.items():
        if not isinstance(source_id, str) or not isinstance(raw_record, dict):
            raise TypeError("source lock entries must be objects")
        record = {str(key): value for key, value in raw_record.items()}
        missing = _REQUIRED_FIELDS - set(record)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"source lock entry {source_id!r} is missing: {names}")
        unknown = set(record) - _ALLOWED_FIELDS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"source lock entry {source_id!r} has unknown fields: {names}")
        for field in _REQUIRED_FIELDS:
            value = record[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"source lock entry {source_id!r} has invalid {field}")
        require_resolved_revision(record["revision"], f"source lock entry {source_id!r}")
        if source_id == "mnli":
            if record.get("config") != "mnli":
                raise ValueError("MNLI source lock entry must use config 'mnli'")
        elif "config" in record:
            raise ValueError(f"source lock entry {source_id!r} has unexpected config")
        records[source_id] = record
    return records


_LOCKED_SOURCES = _load_lock()
if _LOCKED_SOURCES != SOURCES:
    raise ValueError("source lock metadata does not match SOURCES")


def load_sources() -> dict[str, dict[str, Any]]:
    return deepcopy(SOURCES)


def source_record(source_id: str) -> Mapping[str, Any]:
    if not isinstance(source_id, str) or source_id not in SOURCES:
        raise KeyError(f"unknown source ID: {source_id}")
    return deepcopy(SOURCES[source_id])


def derive_seed(suite_id: str, base_seed: int = 42) -> int:
    digest = hashlib.sha256(f"{base_seed}:{suite_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_rows(
    rows: Sequence[Mapping[str, Any]], count: int, suite_id: str
) -> list[Mapping[str, Any]]:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("sample count must be an integer")
    if count < 0 or count > len(rows):
        raise ValueError("sample count is outside the source row count")
    generator = random.Random(derive_seed(suite_id))
    return [rows[index] for index in generator.sample(range(len(rows)), count)]
