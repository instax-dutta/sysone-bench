from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from benchmark.canonical import canonical_json
from benchmark.manifest import manifest_digest, validate_manifest

from .labeling import _loads_strict_json, _read_manifest_jsonl, _read_strict_json
from .validate import validate_dataset

ASSISTED_LABEL_PROVENANCE = "human-reviewed-ai-assisted-v1"
ASSISTED_REVIEW_PROTOCOL_ID = "human-reviewed-ai-assisted-v1"
ASSISTED_EXPORT_SOURCE = "cerebras-assistant"
ASSISTED_EXPORT_SCHEMA_VERSION = 1
ASSISTED_EXPORT_NAME = "assistant_confirmed.json"
ASSISTED_REVIEWER_COUNT = 1
ASSISTED_REVIEWED_QUESTIONS = 1550
MANIFEST_NAME = "manifest.jsonl"
PROVENANCE_NAME = "provenance.json"
SEALED_NAME = "manifest.sha256"
PROVISIONAL_MANIFEST_NAME = "manifest.provisional.jsonl"
PROVISIONAL_PROVENANCE_NAME = "provenance.provisional.json"
ASSISTED_EXPORT_RELATIVE_PATH = Path("labels") / ASSISTED_EXPORT_NAME
_ASSISTED_ENVELOPE_FIELDS = frozenset(
    {"schema_version", "source", "official_labels", "model", "entries"}
)
_ASSISTED_ENTRY_FIELDS = frozenset({"case_id", "qid", "answer", "question_type", "model"})
_ASSISTED_METHOD = (
    "One human reviewer read every case and question and corrected the AI draft answers; "
    "there was no second independent review and no adjudication."
)


def _value_error(message: str) -> ValueError:
    return ValueError(message)


def _format_keys(keys: Iterable[object]) -> str:
    ordered = sorted(keys, key=lambda key: (type(key).__name__, repr(key)))
    return ", ".join(repr(key) for key in ordered)


def _require_text(container: Mapping[str, Any], field: str, context: str) -> str:
    value = container.get(field)
    if not isinstance(value, str) or not value.strip():
        raise _value_error(f"{context} {field} must be a nonempty string")
    return value


def _field_mismatch(context: str, missing: AbstractSet[str], extra: AbstractSet[str]) -> ValueError:
    details: list[str] = []
    if missing:
        details.append(f"missing={_format_keys(missing)}")
    if extra:
        details.append(f"extra={_format_keys(extra)}")
    return _value_error(f"{context} fields are invalid: " + "; ".join(details))


@dataclass(frozen=True)
class AssistedExport:
    model: str
    answers: Mapping[tuple[str, str], Any]


def _question_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {record_index} must be a mapping")
        case_id = _require_text(record, "case_id", f"record {record_index}")
        questions = record.get("questions")
        if isinstance(questions, (str, bytes, bytearray)) or not isinstance(questions, Sequence):
            raise _value_error(f"record {record_index}: questions must be an ordered sequence")
        for position, question in enumerate(questions):
            if not isinstance(question, Mapping):
                raise _value_error(f"record {record_index}: question {position} must be an object")
            qid = _require_text(question, "qid", f"record {record_index} question {position}")
            key = (case_id, qid)
            if key in index:
                raise _value_error(
                    f"record {record_index}: question {position} repeats an existing question"
                )
            index[key] = question
    return index


def parse_assisted_export(
    payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> AssistedExport:
    if not isinstance(payload, Mapping):
        raise TypeError("assistant export must be a mapping")
    fields = set(payload)
    missing = _ASSISTED_ENVELOPE_FIELDS - fields
    extra = fields - _ASSISTED_ENVELOPE_FIELDS
    if missing or extra:
        raise _field_mismatch("assistant export", missing, extra)
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ASSISTED_EXPORT_SCHEMA_VERSION
    ):
        raise _value_error("assistant export schema_version must equal 1")
    if payload["source"] != ASSISTED_EXPORT_SOURCE:
        raise _value_error("assistant export source must be cerebras-assistant")
    if payload["official_labels"] is not False:
        raise _value_error("assistant export official_labels must be exactly false")
    model = _require_text(payload, "model", "assistant export")
    entries = payload["entries"]
    if isinstance(entries, (str, bytes, bytearray)) or not isinstance(entries, Sequence):
        raise _value_error("assistant export entries must be a list")
    index = _question_index(records)
    answers: dict[tuple[str, str], Any] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise _value_error(f"assistant export entry {entry_index} must be an object")
        entry_fields = set(entry)
        missing_entry = _ASSISTED_ENTRY_FIELDS - entry_fields
        extra_entry = entry_fields - _ASSISTED_ENTRY_FIELDS
        if missing_entry or extra_entry:
            raise _field_mismatch(
                f"assistant export entry {entry_index}", missing_entry, extra_entry
            )
        case_id = _require_text(entry, "case_id", f"assistant export entry {entry_index}")
        qid = _require_text(entry, "qid", f"assistant export entry {entry_index}")
        _require_text(entry, "model", f"assistant export entry {entry_index}")
        key = (case_id, qid)
        if key in answers:
            raise _value_error(f"assistant export entry {entry_index} duplicates an earlier entry")
        question = index.get(key)
        if question is None:
            raise _value_error(
                f"assistant export entry {entry_index} does not match a manifest question"
            )
        question_type = _require_text(
            entry, "question_type", f"assistant export entry {entry_index}"
        )
        if question_type != question.get("type"):
            raise _value_error(
                f"assistant export entry {entry_index} question_type must match the manifest"
            )
        answers[key] = deepcopy(entry["answer"])
    return AssistedExport(model=model, answers=answers)


def _apply_export(
    records: Sequence[Mapping[str, Any]], export: AssistedExport
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
        copied.append(deepcopy(dict(record)))
    suite_counts: Counter[str] = Counter()
    for index, record in enumerate(copied):
        suite_id = _require_text(record, "suite_id", f"record {index}")
        suite_counts[suite_id] += 1
        case_id = cast(str, record["case_id"])
        expected = record.get("expected")
        if not isinstance(expected, dict):
            raise _value_error(f"record {index}: expected must be a mapping")
        for qid in expected:
            if (case_id, qid) not in export.answers:
                raise _value_error(
                    f"record {index}: assistant export is missing a manifest question"
                )
            expected[qid] = deepcopy(export.answers[(case_id, qid)])
        record["label_provenance"] = ASSISTED_LABEL_PROVENANCE
    validate_manifest(copied, dict(suite_counts))
    return copied


def apply_assisted_review(
    records: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence of mappings")
    copied: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise _value_error(f"record {index} must be a mapping")
        copied.append(deepcopy(dict(record)))
    export = parse_assisted_export(payload, copied)
    return _apply_export(copied, export)


def _stage_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _discard(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _restore_bytes(path: Path, content: bytes) -> None:
    temporary_path = _stage_bytes(path, content)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _capture_targets(targets: Sequence[Path]) -> dict[Path, bytes | None]:
    snapshots: dict[Path, bytes | None] = {}
    for target in targets:
        if not os.path.lexists(target):
            snapshots[target] = None
            continue
        try:
            snapshots[target] = target.read_bytes()
        except OSError:
            raise _value_error("unable to snapshot finalize output") from None
    return snapshots


def _restore_target(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
        return
    _restore_bytes(path, content)


def _write_checksum_exclusive(sealed_path: Path, content: bytes, output_root: Path) -> None:
    try:
        checksum_file = sealed_path.open("xb")
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite sealed manifest in {output_root}") from error
    try:
        with checksum_file:
            checksum_file.write(content)
            checksum_file.flush()
            os.fsync(checksum_file.fileno())
    except BaseException:
        sealed_path.unlink(missing_ok=True)
        raise


def _refuse_existing(path: Path, message: str) -> None:
    if os.path.lexists(path):
        raise FileExistsError(message)


def _require_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise _value_error(f"finalize input is not valid: symbolic links are refused in {path}")
    if not path.is_file():
        raise _value_error(f"finalize requires dataset and assistant artifacts; missing: {path}")


def _require_provisional_draft(provenance: Mapping[str, Any], output_root: Path) -> None:
    if "provisional" not in provenance:
        raise _value_error("provenance.json must include provisional=true")
    provisional = provenance["provisional"]
    if type(provisional) is bool and provisional is False:
        raise FileExistsError(f"refusing to overwrite sealed manifest in {output_root}")
    if type(provisional) is not bool or provisional is not True:
        raise _value_error("provenance.json provisional must be boolean true")


def _manifest_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(record).decode("utf-8") + "\n" for record in records).encode(
        "utf-8"
    )


def _assisted_export_payload(content: bytes) -> Any:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _value_error(f"{ASSISTED_EXPORT_NAME} is not valid UTF-8 JSON: {error}") from None
    return _loads_strict_json(text, ASSISTED_EXPORT_NAME)


def _label_review(export: AssistedExport, export_digest: str) -> dict[str, Any]:
    return {
        "protocol": ASSISTED_REVIEW_PROTOCOL_ID,
        "reviewer_count": ASSISTED_REVIEWER_COUNT,
        "independent_human_review": False,
        "adjudication": False,
        "assistant_model": export.model,
        "assistant_export_sha256": export_digest,
        "questions_reviewed": ASSISTED_REVIEWED_QUESTIONS,
        "entries_applied": ASSISTED_REVIEWED_QUESTIONS,
        "method": _ASSISTED_METHOD,
    }


def finalize_assisted_dataset(
    output_root: Path,
    assistant_path: Path,
    validator: Callable[..., Any] = validate_dataset,
) -> str:
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if not isinstance(assistant_path, Path):
        raise TypeError("assistant_path must be a Path")
    if not output_root.is_dir():
        raise _value_error("--output must be an existing directory")
    manifest_path = output_root / MANIFEST_NAME
    provenance_path = output_root / PROVENANCE_NAME
    sealed_path = output_root / SEALED_NAME
    manifest_backup_path = output_root / PROVISIONAL_MANIFEST_NAME
    provenance_backup_path = output_root / PROVISIONAL_PROVENANCE_NAME
    assistant_copy_path = output_root / ASSISTED_EXPORT_RELATIVE_PATH
    for required in (manifest_path, provenance_path, assistant_path):
        _require_regular_file(required)
    _refuse_existing(sealed_path, f"refusing to overwrite sealed manifest in {output_root}")
    manifest_before = manifest_path.read_bytes()
    provenance_before = provenance_path.read_bytes()
    assistant_before = assistant_path.read_bytes()
    records = _read_manifest_jsonl(manifest_path)
    provenance_value = _read_strict_json(provenance_path, PROVENANCE_NAME)
    if not isinstance(provenance_value, Mapping):
        raise _value_error("provenance.json must contain a JSON object")
    provenance = cast(Mapping[str, Any], provenance_value)
    _require_provisional_draft(provenance, output_root)
    _refuse_existing(
        manifest_backup_path,
        f"refusing to overwrite existing backup {PROVISIONAL_MANIFEST_NAME}",
    )
    _refuse_existing(
        provenance_backup_path,
        f"refusing to overwrite existing backup {PROVISIONAL_PROVENANCE_NAME}",
    )
    export = parse_assisted_export(_assisted_export_payload(assistant_before), records)
    final_records = _apply_export(records, export)
    reviewed = sum(len(cast(Mapping[str, Any], record["expected"])) for record in final_records)
    if reviewed != ASSISTED_REVIEWED_QUESTIONS or len(export.answers) != reviewed:
        raise _value_error(
            f"finalize requires exactly {ASSISTED_REVIEWED_QUESTIONS} reviewed questions"
        )
    finalized = deepcopy(dict(provenance))
    finalized["provisional"] = False
    finalized["label_review"] = _label_review(export, hashlib.sha256(assistant_before).hexdigest())
    summary = validator(final_records, finalized, registry=finalized)
    if not isinstance(summary, Mapping):
        raise TypeError("validator must return a mapping summary")
    digest = cast(str, manifest_digest(final_records))
    if summary.get("digest") != digest:
        raise _value_error("manifest digest changed during dataset validation")
    manifest_content = _manifest_jsonl_bytes(final_records)
    provenance_content = canonical_json(finalized) + b"\n"
    checksum_digest = hashlib.sha256(manifest_content).hexdigest()
    checksum_content = f"{checksum_digest}  {MANIFEST_NAME}\n".encode()
    labels_dir = assistant_copy_path.parent
    labels_dir_created = not labels_dir.is_dir()
    targets = (manifest_backup_path, provenance_backup_path, assistant_copy_path)
    snapshots = _capture_targets(targets)
    staged: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    manifest_temporary: Path | None = None
    provenance_temporary: Path | None = None
    checksum_created = False
    manifest_replaced = False
    provenance_replaced = False
    try:
        for target, content in (
            (manifest_backup_path, manifest_before),
            (provenance_backup_path, provenance_before),
            (assistant_copy_path, assistant_before),
        ):
            staged.append((target, _stage_bytes(target, content)))
        manifest_temporary = _stage_bytes(manifest_path, manifest_content)
        provenance_temporary = _stage_bytes(provenance_path, provenance_content)
        _refuse_existing(sealed_path, f"refusing to overwrite sealed manifest in {output_root}")
        _write_checksum_exclusive(sealed_path, checksum_content, output_root)
        checksum_created = True
        for target, temporary in staged:
            os.replace(temporary, target)
            committed.append(target)
        _discard(temporary for _, temporary in staged)
        staged = []
        os.replace(manifest_temporary, manifest_path)
        manifest_temporary = None
        manifest_replaced = True
        os.replace(provenance_temporary, provenance_path)
        provenance_temporary = None
        provenance_replaced = True
    except BaseException as error:
        _discard(
            [temporary for _, temporary in staged]
            + [path for path in (manifest_temporary, provenance_temporary) if path is not None]
        )
        staged = []
        manifest_temporary = None
        provenance_temporary = None
        rollback_error: BaseException | None = None
        for target in reversed(committed):
            try:
                _restore_target(target, snapshots[target])
            except OSError as restore_error:
                rollback_error = restore_error
        if checksum_created:
            try:
                sealed_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                rollback_error = cleanup_error
        for replaced, path, content in (
            (manifest_replaced, manifest_path, manifest_before),
            (provenance_replaced, provenance_path, provenance_before),
        ):
            if not replaced:
                continue
            try:
                _restore_bytes(path, content)
            except OSError as restore_error:
                rollback_error = restore_error
        if labels_dir_created:
            try:
                os.rmdir(labels_dir)
            except OSError:
                pass
        if rollback_error is not None:
            raise RuntimeError("unable to roll back failed dataset finalize") from error
        raise
    finally:
        _discard(temporary for _, temporary in staged)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        if provenance_temporary is not None:
            provenance_temporary.unlink(missing_ok=True)
    return checksum_digest


_CLI_PATH_OPTIONS = (("--output", "output"), ("--assistant", "assistant"))
_CLI_SAFE_OPTIONS = tuple(option for option, _ in _CLI_PATH_OPTIONS)
_CLI_SAFE_TERMS = (
    "collision",
    "required",
    "requires",
    "not valid",
    "existing directory",
    "duplicate JSON object key",
    "path resolution failed",
    "manifest.jsonl",
    "manifest.provisional.jsonl",
    "provenance.json",
    "provenance.provisional.json",
    "assistant_confirmed.json",
    "manifest.sha256",
    "provisional",
    "sealed",
    "symbolic link",
)


class _CliArgumentError(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise _CliArgumentError


def _safe_cli_error(error: BaseException) -> str:
    if isinstance(error, RuntimeError):
        return "error: path resolution failed"
    try:
        message = str(error).casefold()
    except BaseException as conversion_error:
        if isinstance(conversion_error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        return "error: CLI operation failed"
    parts = [option for option in _CLI_SAFE_OPTIONS if option.casefold() in message]
    parts.extend(term for term in _CLI_SAFE_TERMS if term.casefold() in message)
    if not parts:
        return "error: CLI operation failed"
    return "error: " + " ".join(dict.fromkeys(parts))


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="python -m datasets.v2.assisted_labels",
        description="Finalize the v2 dataset from one human-reviewed AI-assisted label export.",
    )
    parser.add_argument("--output", type=Path, help="canonical v2 dataset directory")
    parser.add_argument("--assistant", type=Path, help="assistant_confirmed.json export path")
    return parser


def _required_cli_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise _value_error(f"{name} is required")
    return value


@dataclass(frozen=True)
class _PathIdentity:
    canonical: str
    file_id: tuple[int, int] | None


def _path_identity(path: Path) -> _PathIdentity:
    try:
        resolved_path = path.resolve()
    except RuntimeError:
        raise RuntimeError("path resolution failed") from None
    canonical = os.path.normcase(os.path.realpath(str(resolved_path)))
    if sys.platform == "darwin":
        canonical = canonical.casefold()
    try:
        stat_result = path.stat()
    except OSError:
        return _PathIdentity(canonical, None)
    return _PathIdentity(canonical, (stat_result.st_dev, stat_result.st_ino))


def _path_identities_collide(left: _PathIdentity, right: _PathIdentity) -> bool:
    if left.canonical == right.canonical:
        return True
    return left.file_id is not None and left.file_id == right.file_id


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = _build_cli_parser()
        arguments = parser.parse_args(argv)
        output = _required_cli_path(arguments.output, "--output")
        assistant = _required_cli_path(arguments.assistant, "--assistant")
        if _path_identities_collide(_path_identity(output), _path_identity(assistant)):
            raise _value_error("--output path collision with --assistant")
        digest = finalize_assisted_dataset(output, assistant)
        sys.stdout.write(f"{digest}\n")
        return 0
    except _CliArgumentError:
        sys.stderr.write("error: CLI operation failed\n")
        return 1
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        sys.stderr.write(f"{_safe_cli_error(error)}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
