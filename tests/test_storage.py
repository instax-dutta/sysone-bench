import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from benchmark.canonical import canonical_json
from benchmark.storage import (
    create_run_directory,
    write_checksums,
    write_json_exclusive,
    write_jsonl_exclusive,
)


def test_json_write_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_json_exclusive(path, {"run_id": "one"})
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"run_id": "two"})

    assert path.read_bytes() == original


def test_json_write_returns_none_and_canonical_utf8_bytes(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    payload = {"z": "雪", "a": 1}

    result = write_json_exclusive(path, payload)

    assert result is None
    assert path.read_bytes() == canonical_json(payload) + b"\n"


def test_json_write_rejects_non_json_value_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"

    with pytest.raises((TypeError, ValueError)):
        write_json_exclusive(path, {"value": object()})

    assert not path.exists()


def test_json_write_rejects_nonfinite_value_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"

    with pytest.raises(ValueError, match="non-finite"):
        write_json_exclusive(path, {"value": math.nan})

    assert not path.exists()


def test_jsonl_write_returns_path_and_canonical_utf8_records(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows: list[Mapping[str, Any]] = [
        {"case_id": "a", "text": "é", "ok": 1},
        {"case_id": "b", "text": "雪", "ok": 0},
    ]

    result = write_jsonl_exclusive(path, rows)

    assert result == path
    assert path.read_bytes() == b"".join(canonical_json(row) + b"\n" for row in rows)
    assert [json.loads(line)["case_id"] for line in path.read_text().splitlines()] == ["a", "b"]


def test_jsonl_write_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_jsonl_exclusive(path, [{"case_id": "one"}])
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_jsonl_exclusive(path, [{"case_id": "two"}])

    assert path.read_bytes() == original


def test_jsonl_write_rejects_malformed_row_without_creating_file(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows: Iterable[Mapping[str, Any]] = [{"case_id": "a"}, {"case_id": object()}]

    with pytest.raises((TypeError, ValueError)):
        write_jsonl_exclusive(path, rows)

    assert not path.exists()


def test_jsonl_write_of_empty_iterable_creates_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"

    result = write_jsonl_exclusive(path, [])

    assert result == path
    assert path.read_bytes() == b""


def test_create_run_directory_accepts_single_component_id(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()

    path = create_run_directory(results_root, "run-001")

    assert path == results_root / "run-001"
    assert path.parent == results_root
    assert path.name == "run-001"
    assert path.is_dir()


def test_create_run_directory_refuses_collision(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_path = results_root / "run-001"
    run_path.mkdir(parents=True)
    sentinel = run_path / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_run_directory(results_root, "run-001")

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_create_run_directory_does_not_create_missing_root(tmp_path: Path) -> None:
    results_root = tmp_path / "missing" / "results"

    with pytest.raises(FileNotFoundError):
        create_run_directory(results_root, "run-001")

    assert not results_root.parent.exists()


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../outside",
        r"..\outside",
        "nested/run",
        r"nested\run",
        r"/absolute-run",
        r"\absolute-run",
        r"C:\absolute-run",
        r"\\server\share\absolute-run",
        "C:relative-run",
    ],
)
def test_create_run_directory_rejects_unsafe_id_without_side_effects(
    tmp_path: Path, run_id: str
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    existing = results_root / "existing"
    existing.mkdir()
    sentinel = existing / "artifact.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    children_before = sorted(results_root.iterdir())

    with pytest.raises(ValueError) as error:
        create_run_directory(results_root, run_id)

    assert str(error.value) == f"invalid run_id {run_id!r}: must be one safe path component"
    assert sorted(results_root.iterdir()) == children_before
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_create_run_directory_rejects_absolute_id_without_creating_outside(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    existing = results_root / "existing"
    existing.mkdir()
    sentinel = existing / "artifact.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    outside = tmp_path / "outside"
    run_id = str(outside)

    with pytest.raises(ValueError) as error:
        create_run_directory(results_root, run_id)

    assert str(error.value) == f"invalid run_id {run_id!r}: must be one safe path component"
    assert not outside.exists()
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_checksums_are_sorted_relative_posix_paths_and_exclude_output(tmp_path: Path) -> None:
    contents = {
        "z.txt": b"z\n",
        "nested/a.json": b'{"a":1}\n',
        "nested/deep/b.bin": b"b",
        "nested/checksums.sha256": b"nested checksum\n",
    }
    for relative_path, value in contents.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    checksum_path = write_checksums(tmp_path)

    assert checksum_path == tmp_path / "checksums.sha256"
    expected = "".join(
        f"{hashlib.sha256(contents[relative_path]).hexdigest()}  {relative_path}\n"
        for relative_path in sorted(contents)
    )
    assert checksum_path.read_text(encoding="utf-8") == expected
    assert all(
        line.split("  ", 1)[1] != "checksums.sha256"
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
    )


def test_checksums_ignore_directories_and_include_nested_regular_files(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "data.bin").write_bytes(b"data")
    (nested / "empty").mkdir()

    write_checksums(tmp_path)

    lines = (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert lines == [f"{hashlib.sha256(b'data').hexdigest()}  nested/data.bin"]


def test_checksums_refuse_to_overwrite_existing_output(tmp_path: Path) -> None:
    (tmp_path / "artifact.txt").write_bytes(b"artifact")
    checksum_path = tmp_path / "checksums.sha256"
    checksum_path.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_checksums(tmp_path)

    assert checksum_path.read_text(encoding="utf-8") == "keep\n"


def test_checksums_reject_symlinks_before_reading_link_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("outside", encoding="utf-8")
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "artifact.txt").symlink_to(target)
    monkeypatch.setattr(
        "benchmark.storage.digest_file",
        lambda path: pytest.fail(f"followed symlink target: {path}"),
    )

    with pytest.raises(ValueError, match="symlink"):
        write_checksums(directory)

    assert not (directory / "checksums.sha256").exists()
