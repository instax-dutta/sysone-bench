from __future__ import annotations

import ntpath
import posixpath
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmark.canonical import canonical_json, digest_file


def _validate_run_id(run_id: str) -> None:
    if (
        not run_id
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or "\x00" in run_id
        or posixpath.isabs(run_id)
        or ntpath.isabs(run_id)
        or bool(ntpath.splitdrive(run_id)[0])
    ):
        raise ValueError(f"invalid run_id {run_id!r}: must be one safe path component")


def create_run_directory(results_root: Path, run_id: str) -> Path:
    _validate_run_id(run_id)
    path = results_root / run_id
    path.mkdir(exist_ok=False)
    return path


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    content = canonical_json(payload).decode("utf-8") + "\n"
    with path.open("x", encoding="utf-8") as file:
        file.write(content)


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    content = "".join(canonical_json(row).decode("utf-8") + "\n" for row in rows)
    with path.open("x", encoding="utf-8") as file:
        file.write(content)
    return path


def write_checksums(directory: Path) -> Path:
    checksum_path = directory / "checksums.sha256"
    all_paths = list(directory.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise ValueError("checksum directory must not contain symlinks")
    files = [path for path in all_paths if path != checksum_path and path.is_file()]
    files.sort(key=lambda path: path.relative_to(directory).as_posix())
    lines = [f"{digest_file(path)}  {path.relative_to(directory).as_posix()}" for path in files]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    with checksum_path.open("x", encoding="utf-8") as file:
        file.write(content)
    return checksum_path
