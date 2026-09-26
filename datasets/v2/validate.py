from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, cast

from benchmark.canonical import digest_value
from benchmark.manifest import validate_manifest

from .build_manifest import (
    PROVISIONAL_CALIBRATION_CASES,
    PUBLIC_SUITE_IDS,
    SUITE_COUNTS,
    SUITE_DECISIONS,
)
from .sources import SUITE_SOURCE_IDS, derive_seed, require_resolved_revision

CALIBRATION_CASES: dict[str, int] = dict(PROVISIONAL_CALIBRATION_CASES)
EXPECTED_CASES = sum(SUITE_COUNTS.values())
EXPECTED_DECISIONS = sum(
    SUITE_COUNTS[suite_id] * SUITE_DECISIONS[suite_id] for suite_id in SUITE_COUNTS
)
EXPECTED_CALIBRATION_DECISIONS = sum(
    CALIBRATION_CASES[suite_id] * SUITE_DECISIONS[suite_id] for suite_id in SUITE_COUNTS
)
EXPECTED_EVALUATION_DECISIONS = EXPECTED_DECISIONS - EXPECTED_CALIBRATION_DECISIONS
NEAR_DUPLICATE_THRESHOLD = 0.78
PROVENANCE_NAME = "provenance.json"

_METADATA_FIELDS = ("canonical", "wrapper", "revision", "split", "license", "citation")
_ROW_PROVENANCE_FIELDS = (
    "canonical",
    "wrapper",
    "revision",
    "split",
    "source_label",
    "local_modification",
    "license",
    "citation",
    "checksum",
)
_DEFAULT_PROVENANCE_PATH = Path(__file__).with_name(PROVENANCE_NAME)
_MISSING = object()
_PUBLIC_SOURCE_IDS = frozenset(SUITE_SOURCE_IDS[suite_id] for suite_id in PUBLIC_SUITE_IDS)
_PRIMARY_STRATUM_FIELDS: dict[str, str] = {
    "triage": "intent",
    "guardrails": "jailbreak",
    "moderation": "toxic",
    "agnews": "topic",
    "emotion": "emotion",
    "banking77_12": "intent",
    "mnli": "relation",
    "sst5": "sentiment",
}


def _validate_json_value(value: object, name: str, active: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{name} contains a cycle")
        active.add(marker)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{name} object keys must be strings")
                _validate_json_value(child, f"{name}.{key}", active)
        finally:
            active.remove(marker)
        return
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{name} contains a cycle")
        active.add(marker)
        try:
            for index, child in enumerate(value):
                _validate_json_value(child, f"{name}[{index}]", active)
        finally:
            active.remove(marker)
        return
    raise TypeError(f"{name} must contain only JSON values")


def normalized_state(value: object) -> str:
    _validate_json_value(value, "state", set())
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"state cannot be serialized as canonical JSON: {error}") from error


def _stratum_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    suite_id = record.get("suite_id")
    if not isinstance(suite_id, str):
        raise TypeError("case suite_id must be a string")
    expected = record.get("expected")
    if not isinstance(expected, Mapping):
        raise TypeError("case expected answers must be a mapping")
    if suite_id == "multilingual_intent":
        state = record.get("state")
        if not isinstance(state, Mapping) or "language" not in state:
            raise ValueError("multilingual cases must include state.language")
        if "intent" not in expected:
            raise ValueError("multilingual cases must include expected.intent")
        return (
            normalized_state(state["language"]),
            normalized_state(expected["intent"]),
        )
    field = _PRIMARY_STRATUM_FIELDS.get(suite_id)
    if field is None:
        raise KeyError(f"unknown suite: {suite_id}")
    if field not in expected:
        raise ValueError(f"{suite_id} cases must include expected.{field}")
    return (normalized_state(expected[field]),)


def _stratum_name(key: tuple[str, ...]) -> str:
    return "|".join(key)


def _quota_by_stratum(
    groups: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
    target: int,
    total: int,
) -> dict[tuple[str, ...], int]:
    if target < 0 or target > total:
        raise ValueError("stratified target is outside the suite count")
    keys = sorted(groups)
    numerators = {key: target * len(groups[key]) for key in keys}
    quotas = {key: numerators[key] // total for key in keys}
    remaining = target - sum(quotas.values())
    ranked = sorted(keys, key=lambda item: (-(numerators[item] % total), item))
    for key in ranked:
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("unable to allocate the exact stratified target")
    return quotas


def _require_record_sequence(records: Sequence[Mapping[str, Any]]) -> None:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("manifest records must be a sequence")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")


def assign_splits(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _require_record_sequence(records)
    copied = [deepcopy(dict(record)) for record in records]
    case_ids: list[str] = []
    suite_counts: Counter[str] = Counter()
    for index, record in enumerate(copied):
        suite_id = record.get("suite_id")
        case_id = record.get("case_id")
        if not isinstance(suite_id, str) or not suite_id:
            raise ValueError(f"record {index}: suite_id must be a non-empty string")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"record {index}: case_id must be a non-empty string")
        case_ids.append(case_id)
        suite_counts[suite_id] += 1
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("split assignment requires unique case IDs")
    if set(suite_counts) != set(CALIBRATION_CASES):
        raise ValueError("split assignment requires the complete approved suite set")
    if dict(suite_counts) != SUITE_COUNTS:
        raise ValueError("split assignment requires exact suite counts")

    selected_ids: set[str] = set()
    for suite_id in sorted(CALIBRATION_CASES):
        suite_records = [record for record in copied if record["suite_id"] == suite_id]
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for record in suite_records:
            groups.setdefault(_stratum_key(record), []).append(record)
        quotas = _quota_by_stratum(groups, CALIBRATION_CASES[suite_id], len(suite_records))
        generator = random.Random(derive_seed(f"split:{suite_id}"))
        suite_selected = 0
        for stratum in sorted(groups):
            candidates = sorted(groups[stratum], key=lambda record: record["case_id"])
            chosen = generator.sample(candidates, quotas[stratum])
            selected_ids.update(record["case_id"] for record in chosen)
            suite_selected += len(chosen)
        if suite_selected != CALIBRATION_CASES[suite_id]:
            raise ValueError(f"split assignment did not produce the exact {suite_id} total")

    expected_total = sum(CALIBRATION_CASES.values())
    if len(selected_ids) != expected_total:
        raise ValueError("split assignment did not produce the exact calibration total")

    calibration_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    for record in copied:
        if record["case_id"] in selected_ids:
            record["split"] = "calibration"
            calibration_ids.add(record["case_id"])
        else:
            record["split"] = "evaluation"
            evaluation_ids.add(record["case_id"])
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration and evaluation IDs overlap")
    if len(calibration_ids) != expected_total:
        raise ValueError("split assignment calibration IDs are not disjoint and exact")
    if len(evaluation_ids) != EXPECTED_CASES - expected_total:
        raise ValueError("split assignment evaluation IDs are not exact")
    return copied


def _read_registry_payload(registry: Mapping[str, Any] | Path) -> Any:
    if isinstance(registry, Path):
        try:
            return json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read provenance registry: {error}") from error
    return registry


def _load_registry(
    registry: Mapping[str, Any] | Path | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if registry is None:
        if not _DEFAULT_PROVENANCE_PATH.exists():
            return {}, {}
        payload = _read_registry_payload(_DEFAULT_PROVENANCE_PATH)
    else:
        payload = _read_registry_payload(registry)
    if not isinstance(payload, Mapping):
        raise TypeError("provenance registry must contain an object")
    entries: Any = payload
    for key in ("records", "provenance", "entries"):
        if key in payload:
            entries = payload[key]
            break
    if not isinstance(entries, Mapping):
        raise TypeError("provenance registry records must be an object")
    registry_sources = payload.get("sources", {})
    if not isinstance(registry_sources, Mapping):
        raise TypeError("provenance registry sources must be an object")
    return cast(Mapping[str, Any], entries), cast(Mapping[str, Any], registry_sources)


def _source_container(sources: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = sources.get("sources")
    if isinstance(nested, Mapping):
        return cast(Mapping[str, Any], nested)
    return sources


def _sources_with_registry_metadata(
    sources: Mapping[str, Any], registry_sources: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not registry_sources:
        return sources
    merged = dict(sources)
    source_container = _source_container(sources)
    for source_id, fallback in registry_sources.items():
        if not isinstance(source_id, str) or not isinstance(fallback, Mapping):
            continue
        existing = source_container.get(source_id)
        if existing is None:
            merged[source_id] = deepcopy(dict(fallback))
            continue
        if not isinstance(existing, Mapping):
            merged[source_id] = deepcopy(dict(fallback))
            continue
        combined = deepcopy(dict(fallback))
        nested = existing.get("metadata")
        if isinstance(nested, Mapping):
            combined.update(deepcopy(dict(nested)))
        for key, value in existing.items():
            if key != "metadata":
                combined[key] = deepcopy(value)
        existing_metadata = nested if isinstance(nested, Mapping) else existing
        fallback_metadata = fallback.get("metadata")
        if not isinstance(fallback_metadata, Mapping):
            fallback_metadata = fallback
        if all(field in existing_metadata for field in _METADATA_FIELDS) and all(
            field in fallback_metadata for field in _METADATA_FIELDS
        ):
            for field in _METADATA_FIELDS:
                if field == "revision" and source_id in _PUBLIC_SOURCE_IDS:
                    require_resolved_revision(existing_metadata[field], f"source {source_id!r}")
                    require_resolved_revision(fallback_metadata[field], f"source {source_id!r}")
                if existing_metadata[field] != fallback_metadata[field]:
                    raise ValueError(
                        f"source metadata for {source_id!r} differs between registries"
                    )
            if existing_metadata.get("config") != fallback_metadata.get("config"):
                raise ValueError(f"source metadata for {source_id!r} differs between registries")
        merged[source_id] = combined
    return merged


def _source_value(sources: Mapping[str, Any], source_id: str) -> Any:
    container = _source_container(sources)
    value = container.get(source_id)
    if value is None and source_id == "banking77":
        value = container.get("banking77_12")
    if value is None:
        raise ValueError(f"public source metadata is missing for {source_id!r}")
    return value


def _source_metadata(sources: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    raw = _source_value(sources, source_id)
    if not isinstance(raw, Mapping):
        raise TypeError(f"source metadata for {source_id!r} must be a mapping")
    nested = raw.get("metadata")
    metadata = raw if nested is None else nested
    if not isinstance(metadata, Mapping):
        raise TypeError(f"source metadata for {source_id!r} must be a mapping")
    result = {field: deepcopy(metadata[field]) for field in _METADATA_FIELDS if field in metadata}
    missing = [field for field in _METADATA_FIELDS if field not in result]
    if missing:
        raise ValueError(
            f"public source metadata for {source_id!r} is missing: {', '.join(missing)}"
        )
    for field in _METADATA_FIELDS:
        value = result[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"public source metadata for {source_id!r} has invalid {field}")
    require_resolved_revision(result["revision"], f"public source {source_id!r}")
    if "config" in metadata:
        config = metadata["config"]
        if not isinstance(config, str) or not config:
            raise ValueError(f"public source metadata for {source_id!r} has invalid config")
        result["config"] = deepcopy(config)
    return result


def _source_rows(sources: Mapping[str, Any], source_id: str) -> list[Mapping[str, Any]] | None:
    raw = _source_value(sources, source_id)
    if isinstance(raw, Mapping):
        value: Any = raw.get("rows", raw.get("data"))
    elif isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        return None
    else:
        value = raw
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"source rows for {source_id!r} must be a sequence")
    rows = list(value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError(f"source rows for {source_id!r} must contain mappings")
    return cast(list[Mapping[str, Any]], rows)


def _source_row_id(row: Mapping[str, Any], index: int) -> Any:
    for field in ("row_id", "id", "idx", "index"):
        if field in row:
            return row[field]
    return index


def _source_label(suite_id: str, row: Mapping[str, Any]) -> Any:
    if suite_id == "banking77_12" and "label_text" in row:
        return row["label_text"]
    for field in ("label", "labels", "target", "class"):
        if field in row:
            return row[field]
    if "expected" in row and isinstance(row["expected"], Mapping):
        return row["expected"]
    return _MISSING


def _entry_for_case(entries: Mapping[str, Any], record: Mapping[str, Any]) -> Mapping[str, Any]:
    case_id = cast(str, record["case_id"])
    if case_id not in entries:
        raise ValueError(f"row-level provenance registry is missing public case {case_id!r}")
    entry = entries[case_id]
    if not isinstance(entry, Mapping):
        raise TypeError(f"provenance entry for {case_id!r} must be a mapping")
    return cast(Mapping[str, Any], entry)


def _entry_row_id(entry: Mapping[str, Any], case_id: str) -> Any:
    for field in ("row_id", "original_row_id", "original_row_index"):
        if field in entry:
            value = entry[field]
            if value is None or (isinstance(value, str) and not value):
                raise ValueError(f"provenance for {case_id!r} has an empty row_id")
            normalized_state(value)
            return value
    raise ValueError(f"provenance for {case_id!r} is missing row_id/original row ID")


def _validate_checksum(value: Any, case_id: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"provenance for {case_id!r} has an invalid checksum")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"provenance for {case_id!r} has an invalid checksum")
    return value.lower()


def _find_source_row(rows: Sequence[Mapping[str, Any]], row_id: Any) -> Mapping[str, Any] | None:
    for index, row in enumerate(rows):
        if _source_row_id(row, index) == row_id:
            return row
    return None


def _validate_public_provenance(
    records: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    entries: Mapping[str, Any],
) -> dict[str, Any]:
    public_count = 0
    checksum_verified = 0
    source_rows_verified = 0
    source_ids: set[str] = set()
    metadata_cache: dict[str, dict[str, Any]] = {}
    rows_cache: dict[str, list[Mapping[str, Any]] | None] = {}
    for record in records:
        suite_id = cast(str, record["suite_id"])
        if suite_id not in PUBLIC_SUITE_IDS:
            continue
        public_count += 1
        source_id = SUITE_SOURCE_IDS[suite_id]
        source_ids.add(source_id)
        if source_id not in metadata_cache:
            metadata_cache[source_id] = _source_metadata(sources, source_id)
        metadata = metadata_cache[source_id]
        if source_id not in rows_cache:
            rows_cache[source_id] = _source_rows(sources, source_id)
        rows = rows_cache[source_id]
        entry = _entry_for_case(entries, record)
        case_id = cast(str, record["case_id"])
        provenance_id = cast(str, record["provenance_id"])
        if provenance_id != source_id:
            raise ValueError(
                f"provenance ID for {case_id!r} does not identify source {source_id!r}"
            )
        if "source_id" in entry and entry["source_id"] != source_id:
            raise ValueError(f"provenance for {case_id!r} does not identify source {source_id!r}")
        if "provenance_id" in entry and entry["provenance_id"] != provenance_id:
            raise ValueError(f"provenance ID mismatch for {case_id!r}")
        missing = [field for field in _ROW_PROVENANCE_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"provenance for {case_id!r} is missing: {', '.join(missing)}")
        for field in (
            "canonical",
            "wrapper",
            "revision",
            "split",
            "local_modification",
            "license",
            "citation",
        ):
            value = entry[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"provenance for {case_id!r} has invalid {field}")
        require_resolved_revision(entry["revision"], f"provenance for {case_id!r}")
        for field in _METADATA_FIELDS:
            if entry[field] != metadata[field]:
                raise ValueError(
                    f"provenance for {case_id!r} has {field} that differs from the source registry"
                )
        if "config" in metadata and (
            "config" not in entry or entry["config"] != metadata["config"]
        ):
            raise ValueError(f"provenance for {case_id!r} has invalid config")
        if "config" not in metadata and "config" in entry:
            raise ValueError(f"provenance for {case_id!r} has unexpected config")
        if entry["source_label"] is None or (
            isinstance(entry["source_label"], str) and not entry["source_label"]
        ):
            raise ValueError(f"provenance for {case_id!r} has an empty source_label")
        normalized_state(entry["source_label"])
        row_id = _entry_row_id(entry, case_id)
        checksum = _validate_checksum(entry["checksum"], case_id)
        entry_source_row = entry.get("source_row", _MISSING)
        if entry_source_row is not _MISSING:
            if not isinstance(entry_source_row, Mapping):
                raise TypeError(f"provenance for {case_id!r} has an invalid source_row")
            normalized_state(entry_source_row)
        source_row = _find_source_row(rows, row_id) if rows is not None else None
        if rows is not None and source_row is None:
            raise ValueError(f"original row ID {row_id!r} is missing for {case_id!r}")
        checksum_row = entry_source_row if entry_source_row is not _MISSING else source_row
        checksum_checked = checksum_row is not _MISSING
        if checksum_checked:
            expected_checksum = hashlib.sha256(
                normalized_state(checksum_row).encode("utf-8")
            ).hexdigest()
            if checksum != expected_checksum:
                raise ValueError(f"checksum mismatch for {case_id!r}")
        if (
            entry_source_row is not _MISSING
            and source_row is not None
            and normalized_state(entry_source_row) != normalized_state(source_row)
        ):
            raise ValueError(f"source row/checksum mismatch for {case_id!r}")
        if checksum_checked:
            checksum_verified += 1
        if source_row is not None:
            source_rows_verified += 1
            raw_label = _source_label(suite_id, source_row)
            if raw_label is not _MISSING and entry["source_label"] != raw_label:
                raise ValueError(f"source label mismatch for {case_id!r}")
    return {
        "public_cases": public_count,
        "source_ids": sorted(source_ids),
        "checksums_verified": checksum_verified,
        "source_rows_verified": source_rows_verified,
        "registry_entries": len(entries),
    }


def _expected_core_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("suite_id"), str):
            return dict(SUITE_COUNTS)
        counts[record["suite_id"]] += 1
    if set(counts) == set(SUITE_COUNTS):
        return dict(SUITE_COUNTS)
    return dict(counts)


def _near_duplicate_clusters(
    records: Sequence[Mapping[str, Any]], normalized_states: Sequence[str]
) -> list[dict[str, Any]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    character_counts = [Counter(value) for value in normalized_states]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if normalized_states[left] == normalized_states[right]:
                continue
            left_length = len(normalized_states[left])
            right_length = len(normalized_states[right])
            if 200 * min(left_length, right_length) < 78 * (left_length + right_length):
                continue
            shared_characters = sum((character_counts[left] & character_counts[right]).values())
            if 200 * shared_characters < 78 * (
                len(normalized_states[left]) + len(normalized_states[right])
            ):
                continue
            ratio = SequenceMatcher(
                None,
                normalized_states[left],
                normalized_states[right],
                autojunk=False,
            ).ratio()
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                union(left, right)
    components: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        root = find(index)
        components.setdefault(root, []).append(cast(str, record["case_id"]))
    ordered = sorted(
        (sorted(members) for members in components.values() if len(members) > 1),
        key=lambda members: tuple(members),
    )
    return [
        {"cluster_id": f"near-duplicate-{index:04d}", "members": members}
        for index, members in enumerate(ordered, start=1)
    ]


def _distribution_summary(
    records: Sequence[Mapping[str, Any]], strata: Sequence[tuple[str, ...]]
) -> dict[str, Any]:
    by_suite = Counter(cast(str, record["suite_id"]) for record in records)
    by_split = Counter(cast(str, record["split"]) for record in records)
    by_suite_split: dict[str, dict[str, int]] = {}
    by_stratum: dict[str, dict[str, dict[str, int]]] = {}
    for record, stratum in zip(records, strata, strict=True):
        suite_id = cast(str, record["suite_id"])
        split = cast(str, record["split"])
        by_suite_split.setdefault(suite_id, {"calibration": 0, "evaluation": 0})[split] += 1
        stratum_counts = by_stratum.setdefault(suite_id, {})
        values = stratum_counts.setdefault(
            _stratum_name(stratum), {"total": 0, "calibration": 0, "evaluation": 0}
        )
        values["total"] += 1
        values[split] += 1
    return {
        "by_suite": {suite_id: by_suite[suite_id] for suite_id in sorted(by_suite)},
        "by_split": {split: by_split[split] for split in sorted(by_split)},
        "by_suite_and_split": {
            suite_id: by_suite_split[suite_id] for suite_id in sorted(by_suite_split)
        },
        "by_stratum": {
            suite_id: {name: by_stratum[suite_id][name] for name in sorted(by_stratum[suite_id])}
            for suite_id in sorted(by_stratum)
        },
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | Path | None = None,
) -> dict[str, Any]:
    _require_record_sequence(records)
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping")
    if registry is None:
        possible_registry = sources.get("provenance_registry")
        if possible_registry is not None:
            registry = cast(Mapping[str, Any], possible_registry)
        elif isinstance(sources.get("records"), Mapping):
            registry = sources
    manifest_summary = validate_manifest(records, _expected_core_counts(records))
    if set(manifest_summary.by_suite) != set(SUITE_COUNTS):
        raise ValueError("dataset suite set does not match the approved inventory")
    if manifest_summary.total_cases != EXPECTED_CASES:
        raise ValueError("dataset case count does not match the approved inventory")
    if manifest_summary.total_decisions != EXPECTED_DECISIONS:
        raise ValueError("dataset decision count does not match the approved inventory")
    if manifest_summary.by_suite != SUITE_COUNTS:
        raise ValueError("dataset suite counts do not match the approved inventory")

    records_by_suite: dict[str, list[Mapping[str, Any]]] = {
        suite_id: [] for suite_id in SUITE_COUNTS
    }
    calibration_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    calibration_decisions = 0
    evaluation_decisions = 0
    for record in records:
        suite_id = cast(str, record["suite_id"])
        split = cast(str, record["split"])
        case_id = cast(str, record["case_id"])
        expected = cast(Mapping[str, Any], record["expected"])
        records_by_suite[suite_id].append(record)
        if split == "calibration":
            calibration_ids.add(case_id)
            calibration_decisions += len(expected)
        else:
            evaluation_ids.add(case_id)
            evaluation_decisions += len(expected)
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration and evaluation case IDs overlap")
    for suite_id, expected_count in SUITE_COUNTS.items():
        suite_records = records_by_suite[suite_id]
        actual_calibration = sum(record["split"] == "calibration" for record in suite_records)
        actual_evaluation = sum(record["split"] == "evaluation" for record in suite_records)
        if actual_calibration != CALIBRATION_CASES[suite_id]:
            raise ValueError(f"suite {suite_id!r} calibration count does not match")
        if actual_evaluation != expected_count - CALIBRATION_CASES[suite_id]:
            raise ValueError(f"suite {suite_id!r} evaluation count does not match")
        if len(suite_records) != expected_count:
            raise ValueError(f"suite {suite_id!r} case count does not match")
        if any(
            len(cast(Mapping[str, Any], record["expected"])) != SUITE_DECISIONS[suite_id]
            for record in suite_records
        ):
            raise ValueError(f"suite {suite_id!r} decision count does not match")
    if len(calibration_ids) != 238 or len(evaluation_ids) != 952:
        raise ValueError("dataset split case totals do not match the approved inventory")
    if calibration_decisions != EXPECTED_CALIBRATION_DECISIONS:
        raise ValueError("calibration decision total does not match the approved inventory")
    if evaluation_decisions != EXPECTED_EVALUATION_DECISIONS:
        raise ValueError("evaluation decision total does not match the approved inventory")

    normalized_states = [
        normalized_state(cast(Mapping[str, Any], record["state"])) for record in records
    ]
    first_by_state: dict[str, str] = {}
    for record, state in zip(records, normalized_states, strict=True):
        case_id = cast(str, record["case_id"])
        previous = first_by_state.get(state)
        if previous is not None:
            raise ValueError(f"normalized state duplicate between {previous!r} and {case_id!r}")
        first_by_state[state] = case_id
    entries, registry_sources = _load_registry(registry)
    effective_sources = _sources_with_registry_metadata(sources, registry_sources)
    provenance_summary = _validate_public_provenance(records, effective_sources, entries)
    strata = [_stratum_key(record) for record in records]
    near_duplicate_clusters = _near_duplicate_clusters(records, normalized_states)
    distribution = _distribution_summary(records, strata)
    provenance_digest = digest_value(entries) if entries else None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": "2.0.0",
        "total_cases": manifest_summary.total_cases,
        "total_decisions": manifest_summary.total_decisions,
        "by_suite": {
            suite_id: manifest_summary.by_suite[suite_id]
            for suite_id in sorted(manifest_summary.by_suite)
        },
        "calibration_cases": len(calibration_ids),
        "evaluation_cases": len(evaluation_ids),
        "calibration_decisions": calibration_decisions,
        "evaluation_decisions": evaluation_decisions,
        "split_counts": {
            "calibration": len(calibration_ids),
            "evaluation": len(evaluation_ids),
        },
        "decision_counts": {
            "calibration": calibration_decisions,
            "evaluation": evaluation_decisions,
        },
        "split_by_suite": distribution["by_suite_and_split"],
        "distribution": distribution,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_clusters": near_duplicate_clusters,
        "provenance": provenance_summary,
        "provenance_digest": provenance_digest,
        "digest": manifest_summary.digest,
    }
    return summary
