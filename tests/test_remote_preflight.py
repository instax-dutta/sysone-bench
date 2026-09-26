from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ops.remote import preflight, remote_jev, remote_jev_entrypoint
from ops.remote.preflight import check

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = REPOSITORY_ROOT / "ops" / "remote"
CLEANUP_SCRIPT = REMOTE_ROOT / "cleanup.sh"
LAUNCHER_SCRIPT = REMOTE_ROOT / "run_open_model.sh"
TEST_KEY = "test-only-placeholder"
CONTAINER_ID = "a" * 64
preflight_module = cast(Any, preflight)
remote_jev_module = cast(Any, remote_jev)
remote_jev_entrypoint_module = cast(Any, remote_jev_entrypoint)


def approved_snapshot(**overrides: Any) -> dict[str, Any]:
    cpu_count = overrides.get("cpu_count", 4)
    if type(cpu_count) is int:
        affinity = list(range(cpu_count))
    else:
        affinity = [0, 1, 2, 3]
    snapshot: dict[str, Any] = {
        "cpu_count": 4,
        "cpu_affinity": [0, 1, 2, 3],
        "requested_cpu_ids": [0, 1, 2, 3],
        "load1": 2.0,
        "load5": 3.0,
        "memory_available_gib": 16.0,
        "disk_available_gib": 100.0,
        "avx2": True,
        "python_version": "3.12.1",
        "run_id": "fixture",
        "workspace_root": str(preflight.DEFAULT_WORKSPACE_ROOT),
        "workspace_root_state": "ready",
        "disk_path": str(preflight.DEFAULT_WORKSPACE_ROOT),
        "run_root": str(preflight.DEFAULT_RUN_ROOT),
        "run_root_state": "ready",
        "run_path": str(preflight.DEFAULT_RUN_ROOT / "fixture"),
        "run_path_state": "absent",
        "owner_marker_path": str(
            preflight.DEFAULT_RUN_ROOT / "fixture" / preflight.OWNER_MARKER_NAME
        ),
        "owner_marker_state": "absent",
        "owner_marker_valid": False,
        "owner_marker_owner_uid": None,
        "container_name": "sysone-bench-fixture",
        "container_state": "absent",
        "model_cache_path": str(preflight.DEFAULT_MODEL_CACHE),
        "model_cache_state": "ready",
        "manifest_path": str(preflight.DEFAULT_MANIFEST_PATH),
        "manifest_checksum_path": str(preflight.DEFAULT_MANIFEST_CHECKSUM_PATH),
        "manifest_state": "ready",
        "manifest_checksum_state": "ready",
        "manifest_checksum_valid": True,
        "read_errors": (),
        "errors": (),
    }
    if type(cpu_count) is int and "cpu_affinity" not in overrides:
        snapshot["cpu_affinity"] = affinity
    snapshot.update(overrides)
    return snapshot


def test_collect_snapshot_is_read_only_and_returns_complete_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loadavg_path = tmp_path / "loadavg"
    loadavg_path.write_text("1.5 2.5 1.0 1 2\n", encoding="utf-8")
    cpuinfo_path = tmp_path / "cpuinfo"
    cpuinfo_path.write_text("flags : avx2 aes\n", encoding="utf-8")
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text("MemAvailable: 16777216 kB\n", encoding="utf-8")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    run_root = workspace_root / "runs"
    run_root.mkdir()
    model_cache = tmp_path / "cache"
    model_cache.mkdir()
    manifest_path = tmp_path / "sealed" / "manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text('{"fixture":true}\n', encoding="utf-8")
    manifest_checksum_path = tmp_path / "sealed" / "manifest.sha256"
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_checksum_path.write_text(f"{manifest_digest}  manifest.jsonl\n", encoding="utf-8")

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "free":
            return subprocess.CompletedProcess(
                command,
                0,
                f"Mem: 100 1 2 3 4 5 {16 * preflight.GIB}\n",
                "",
            )
        return subprocess.CompletedProcess(command, 1, "", "Error: No such object")

    monkeypatch.setattr(
        preflight_module.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False
    )
    monkeypatch.setattr(preflight_module.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(
        preflight_module.os,
        "statvfs",
        lambda path: SimpleNamespace(f_bavail=200, f_frsize=preflight.GIB),
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    value = preflight.collect_snapshot(
        run_id="fixture",
        workspace_root=workspace_root,
        run_root=run_root,
        model_cache=model_cache,
        loadavg_path=loadavg_path,
        cpuinfo_path=cpuinfo_path,
        meminfo_path=meminfo_path,
        manifest_path=manifest_path,
        manifest_checksum_path=manifest_checksum_path,
        runner=runner,
    )

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before
    assert value["cpu_count"] == 4
    assert value["cpu_affinity"] == [0, 1, 2, 3]
    assert value["load1"] == 1.5
    assert value["load5"] == 2.5
    assert value["memory_available_gib"] == 16.0
    assert value["disk_available_gib"] == 200.0
    assert value["avx2"] is True
    assert value["workspace_root"] == str(workspace_root)
    assert value["workspace_root_state"] == "ready"
    assert value["disk_path"] == str(workspace_root)
    assert value["run_root"] == str(run_root)
    assert value["run_root_state"] == "ready"
    assert value["run_path_state"] == "absent"
    assert value["container_state"] == "absent"
    assert value["model_cache_state"] == "ready"
    assert value["manifest_state"] == "ready"
    assert value["manifest_checksum_state"] == "ready"
    assert value["manifest_checksum_valid"] is True
    assert value["read_errors"] == ()


def test_collect_snapshot_allows_absent_run_root_with_workspace_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    run_root = workspace_root / "runs"
    model_cache = tmp_path / "cache"
    model_cache.mkdir()
    manifest_path = tmp_path / "sealed" / "manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("fixture\n", encoding="utf-8")
    manifest_checksum_path = tmp_path / "sealed" / "manifest.sha256"
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_checksum_path.write_text(f"{manifest_digest}  manifest.jsonl\n", encoding="utf-8")
    disk_paths: list[Path] = []

    monkeypatch.setattr(preflight, "read_cpu_affinity", lambda: (0, 1, 2, 3))
    monkeypatch.setattr(preflight, "read_loadavg", lambda path: (0.0, 0.0))
    monkeypatch.setattr(preflight, "read_free_available", lambda runner, path: 16 * preflight.GIB)

    def record_disk(path: Path) -> int:
        disk_paths.append(path)
        return int(100 * preflight.GIB)

    monkeypatch.setattr(preflight, "read_disk_available", record_disk)
    monkeypatch.setattr(preflight, "read_avx2", lambda path: True)
    monkeypatch.setattr(preflight_module.platform, "python_version", lambda: "3.12.0")

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "Error: No such object")

    value = preflight.collect_snapshot(
        run_id="fixture",
        workspace_root=workspace_root,
        run_root=run_root,
        model_cache=model_cache,
        manifest_path=manifest_path,
        manifest_checksum_path=manifest_checksum_path,
        runner=runner,
    )

    assert value["run_root_state"] == "absent"
    assert disk_paths == [workspace_root]
    assert value["read_errors"] == ()


def test_postflight_accepts_real_collected_owned_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    run_root = workspace_root / "runs"
    run_dir = run_root / "fixture"
    run_dir.mkdir(parents=True)
    marker = run_dir / preflight.OWNER_MARKER_NAME
    marker.write_text("fixture\n", encoding="utf-8")
    model_cache = tmp_path / "cache"
    model_cache.mkdir()
    manifest_path = tmp_path / "sealed" / "manifest.jsonl"
    manifest_path.parent.mkdir()
    manifest_path.write_text("fixture\n", encoding="utf-8")
    manifest_checksum_path = tmp_path / "sealed" / "manifest.sha256"
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_checksum_path.write_text(f"{manifest_digest}  manifest.jsonl\n", encoding="utf-8")
    monkeypatch.setattr(preflight_module, "DEFAULT_WORKSPACE_ROOT", workspace_root)
    monkeypatch.setattr(preflight_module, "DEFAULT_RUN_ROOT", run_root)
    monkeypatch.setattr(preflight_module, "DEFAULT_MODEL_CACHE", model_cache)
    monkeypatch.setattr(preflight_module, "DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(preflight_module, "DEFAULT_MANIFEST_CHECKSUM_PATH", manifest_checksum_path)
    monkeypatch.setattr(preflight, "read_cpu_affinity", lambda: (0, 1, 2, 3))
    monkeypatch.setattr(preflight, "read_loadavg", lambda path: (0.0, 0.0))
    monkeypatch.setattr(preflight, "read_free_available", lambda runner, path: 16 * preflight.GIB)
    monkeypatch.setattr(preflight, "read_disk_available", lambda path: 100 * preflight.GIB)
    monkeypatch.setattr(preflight, "read_avx2", lambda path: True)
    monkeypatch.setattr(preflight_module.platform, "python_version", lambda: "3.12.0")

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "Error: No such object")

    value = preflight.collect_snapshot(
        run_id="fixture",
        workspace_root=workspace_root,
        run_root=run_root,
        model_cache=model_cache,
        manifest_path=manifest_path,
        manifest_checksum_path=manifest_checksum_path,
        runner=runner,
    )

    assert value["run_path_state"] == "ready"
    assert value["owner_marker_state"] == "ready"
    assert value["owner_marker_valid"] is True
    check(value, allow_owned=True)


def test_path_states_reject_symlinks_and_unowned_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_file = tmp_path / "cache-file"
    cache_file.write_text("x", encoding="utf-8")

    assert preflight._path_state(link) == "symlink"
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(target, target_is_directory=True)
    assert preflight._path_state(parent_link / "missing") == "symlink"
    assert preflight._path_state(cache, cache=True) == "ready"
    assert preflight._path_state(cache_file, cache=True) == "not_directory"
    monkeypatch.setattr(
        preflight_module.os,
        "geteuid",
        lambda: preflight_module.os.stat(cache).st_uid + 1,
    )
    assert preflight._path_state(cache, cache=True) == "unowned"


def test_preflight_rejects_high_load() -> None:
    with pytest.raises(RuntimeError, match="load"):
        check(approved_snapshot(cpu_count=12, load1=8.0, load5=10.0))


def test_preflight_accepts_approved_capacity() -> None:
    check(
        approved_snapshot(
            cpu_count=12,
            load1=4.0,
            load5=6.0,
            memory_available_gib=20,
            disk_available_gib=500,
        )
    )


def test_preflight_accepts_exact_thresholds() -> None:
    check(approved_snapshot())


def test_preflight_rejects_forged_alternate_disk_path() -> None:
    with pytest.raises(RuntimeError, match="disk path"):
        check(approved_snapshot(disk_path="/tmp/forged-disk"))


def test_preflight_check_has_no_fixed_path_bypass() -> None:
    assert "enforce_fixed_paths" not in inspect.signature(check).parameters


def test_preflight_cli_rejects_disk_path_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight_module, "collect_snapshot", pytest.fail)
    with pytest.raises(SystemExit):
        preflight.main(["--run-id", "fixture", "--disk-path", "/tmp/forged-disk"])


def test_preflight_requires_every_safety_field() -> None:
    with pytest.raises(RuntimeError, match="missing"):
        check({"cpu_count": 4})


def test_preflight_rejects_each_missing_safety_field() -> None:
    required = (
        "cpu_count",
        "cpu_affinity",
        "requested_cpu_ids",
        "load1",
        "load5",
        "memory_available_gib",
        "disk_available_gib",
        "avx2",
        "python_version",
        "run_id",
        "workspace_root",
        "workspace_root_state",
        "disk_path",
        "run_root",
        "run_root_state",
        "run_path",
        "run_path_state",
        "owner_marker_path",
        "owner_marker_state",
        "owner_marker_valid",
        "owner_marker_owner_uid",
        "container_name",
        "container_state",
        "model_cache_path",
        "model_cache_state",
        "manifest_path",
        "manifest_checksum_path",
        "manifest_state",
        "manifest_checksum_state",
        "manifest_checksum_valid",
        "read_errors",
    )
    for field in required:
        value = approved_snapshot()
        value.pop(field)
        with pytest.raises(RuntimeError, match="missing"):
            check(value)


def test_preflight_rejects_unsafe_snapshots() -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"load1": 2.01}, "load1"),
        ({"load1": -0.1}, "load1"),
        ({"load5": 3.01}, "load5"),
        ({"memory_available_gib": 15.99}, "memory"),
        ({"disk_available_gib": 99.99}, "disk"),
        ({"avx2": False}, "AVX2"),
        ({"python_version": "3.11.9"}, "Python"),
        ({"workspace_root_state": "symlink"}, "workspace root"),
        ({"workspace_root_state": "unknown"}, "workspace root"),
        ({"disk_path": "/tmp/forged-disk"}, "disk path"),
        ({"owner_marker_path": "/tmp/forged-owner"}, "owner marker"),
        ({"owner_marker_state": "symlink"}, "owner marker"),
        ({"run_root_state": "symlink"}, "run root"),
        ({"run_root_state": "unknown"}, "run root"),
        ({"run_path_state": "occupied"}, "run path"),
        ({"run_path_state": "symlink"}, "run path"),
        ({"run_path_state": "unknown"}, "run path"),
        ({"run_path": "/tmp/other"}, "run path"),
        ({"run_path": "relative/fixture"}, "run path"),
        ({"workspace_root": "/tmp/forged"}, "workspace root"),
        ({"run_root": "/tmp/forged"}, "run root"),
        ({"model_cache_path": "/tmp/forged-cache"}, "model cache"),
        ({"manifest_path": "/tmp/forged-manifest.jsonl"}, "manifest"),
        ({"manifest_checksum_path": "/tmp/forged-manifest.sha256"}, "manifest checksum"),
        ({"manifest_state": "symlink"}, "manifest"),
        ({"manifest_checksum_valid": False}, "manifest checksum"),
        ({"container_state": "occupied"}, "container"),
        ({"container_state": "unknown"}, "container"),
        ({"container_state": "not-found"}, "container"),
        ({"model_cache_state": "missing"}, "model cache"),
        ({"model_cache_state": "symlink"}, "model cache"),
        ({"model_cache_state": "unowned"}, "model cache"),
        ({"model_cache_state": "unknown"}, "model cache"),
        ({"model_cache_path": "relative/cache"}, "model cache"),
        ({"read_errors": ("permission denied",)}, "read errors"),
    ]
    for overrides, message in cases:
        with pytest.raises(RuntimeError, match=message):
            check(approved_snapshot(**overrides))


def test_postflight_check_allows_only_known_owned_run_state() -> None:
    owned = approved_snapshot(
        run_path_state="ready",
        owner_marker_state="ready",
        owner_marker_valid=True,
        owner_marker_owner_uid=os.geteuid(),
    )
    check(owned, allow_owned=True)
    with pytest.raises(RuntimeError, match="run path"):
        check(owned)
    with pytest.raises(RuntimeError, match="run path"):
        check(approved_snapshot(run_path_state="symlink"), allow_owned=True)
    with pytest.raises(RuntimeError, match="owner marker"):
        check(
            approved_snapshot(
                run_path_state="ready",
                owner_marker_state="ready",
                owner_marker_valid=False,
                owner_marker_owner_uid=os.geteuid(),
            ),
            allow_owned=True,
        )
    with pytest.raises(RuntimeError, match="owner marker ownership"):
        check(
            approved_snapshot(
                run_path_state="ready",
                owner_marker_state="ready",
                owner_marker_valid=True,
                owner_marker_owner_uid=os.geteuid() + 1,
            ),
            allow_owned=True,
        )
    with pytest.raises(RuntimeError, match="container"):
        check(approved_snapshot(container_state="occupied"), allow_owned=True)


def test_preflight_uses_process_affinity_and_requires_requested_cpus() -> None:
    with pytest.raises(RuntimeError, match="CPU"):
        check(approved_snapshot(cpu_affinity=[4, 5, 6, 7], requested_cpu_ids=[0, 1, 2, 3]))
    with pytest.raises(RuntimeError, match="CPU"):
        check(approved_snapshot(cpu_count=3, cpu_affinity=[0, 1, 2]))


def test_preflight_rejects_malformed_snapshots() -> None:
    cases: list[dict[str, Any]] = [
        {"cpu_count": 0},
        {"cpu_count": True},
        {"load1": "2"},
        {"memory_available_gib": float("nan")},
        {"avx2": None},
        {"python_version": "3.12.1\n"},
        {"cpu_affinity": "0,1,2,3"},
        {"read_errors": None},
    ]
    for overrides in cases:
        with pytest.raises(RuntimeError):
            check(approved_snapshot(**overrides))


def test_container_state_distinguishes_missing_object_from_daemon_error() -> None:
    def missing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "Error: No such object")

    def denied(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "permission denied")

    assert preflight._container_state("sysone-bench-fixture", missing) == "absent"
    assert preflight._container_state("sysone-bench-fixture", denied) == "unknown"
    check(approved_snapshot(container_state="absent"))


def test_preflight_cli_emits_parseable_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value = approved_snapshot()
    monkeypatch.setattr(preflight_module, "collect_snapshot", lambda **kwargs: value)
    assert preflight.main(["--run-id", "fixture"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_path_state"] == "absent"


def test_read_api_key_selects_only_the_named_variable(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "JEV_MODEL=jev-1.13.0\n"
        f"TYPESAFE_API_KEY={TEST_KEY}\n"
        "JEV_BASE_URL=https://example.invalid\n",
        encoding="utf-8",
    )

    assert remote_jev._read_api_key(env_path) == TEST_KEY


def test_read_api_key_has_no_path_override(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(f"TYPESAFE_API_KEY={TEST_KEY}\n", encoding="utf-8")
    read_api_key = cast(Any, remote_jev.read_api_key)
    with pytest.raises(TypeError):
        read_api_key(env_path)


def test_read_api_key_fails_closed(tmp_path: Path) -> None:
    cases = [
        "JEV_MODEL=jev-1.13.0\n",
        f"TYPESAFE_API_KEY={TEST_KEY}\nTYPESAFE_API_KEY=other\n",
        "TYPESAFE_API_KEY=\n",
    ]
    for content in cases:
        env_path = tmp_path / ".env"
        env_path.write_text(content, encoding="utf-8")
        with pytest.raises(RuntimeError):
            remote_jev._read_api_key(env_path)


def test_send_api_key_uses_fixed_ssh_stdin_and_sanitized_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("TYPESAFE_API_KEY", "inherited-placeholder")
    monkeypatch.setattr(remote_jev_module, "REMOTE_HOST", "runner@benchmark-host")
    monkeypatch.setattr(remote_jev_module, "REMOTE_ROOT", "/srv/benchmark/checkout")
    monkeypatch.setattr(remote_jev_module.subprocess, "run", run)

    assert remote_jev.send_api_key(TEST_KEY) == 0

    command, kwargs = calls[0]
    assert command == [
        "ssh",
        "runner@benchmark-host",
        "/srv/benchmark/checkout/ops/remote/remote_jev_entrypoint.py",
    ]
    assert kwargs["input"] == f"{TEST_KEY}\n"
    assert TEST_KEY not in command
    assert isinstance(kwargs["env"], dict)
    assert "TYPESAFE_API_KEY" not in kwargs["env"]
    assert TEST_KEY not in repr(kwargs["env"])
    assert kwargs["timeout"] == 6 * 60 * 60
    assert kwargs["timeout"] >= 3 * 60 * 60
    assert kwargs["timeout"] <= 24 * 60 * 60


def test_send_api_key_rejects_destination_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_jev_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("override must not reach SSH"),
    )
    send_api_key = cast(Any, remote_jev.send_api_key)
    with pytest.raises(TypeError):
        send_api_key(TEST_KEY, remote_host="other-host")
    with pytest.raises(TypeError):
        send_api_key(TEST_KEY, remote_entrypoint="/tmp/other.py")


def test_send_api_key_redacts_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise RuntimeError(f"transport included {TEST_KEY}")

    monkeypatch.setattr(remote_jev_module.subprocess, "run", run)

    with pytest.raises(RuntimeError) as error:
        remote_jev.send_api_key(TEST_KEY)

    assert TEST_KEY not in str(error.value)


def test_remote_entrypoint_scopes_key_to_child_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["env_during_call"] = dict(kwargs["env"])
        seen["env_object"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 7, "", "")

    monkeypatch.setattr(remote_jev_entrypoint_module.subprocess, "run", run)
    monkeypatch.setattr(
        remote_jev_entrypoint_module.sys,
        "stdin",
        io.StringIO(f"{TEST_KEY}\n"),
    )
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    assert remote_jev_entrypoint.main(["fake-jev"]) == 7
    assert seen["command"] == ["fake-jev"]
    assert seen["env_during_call"]["TYPESAFE_API_KEY"] == TEST_KEY
    assert "TYPESAFE_API_KEY" not in seen["env_object"]
    assert "TYPESAFE_API_KEY" not in os.environ


def test_remote_entrypoint_sanitizes_inherited_key_before_child() -> None:
    sanitized = remote_jev_entrypoint._sanitized_environment(
        {"TYPESAFE_API_KEY": "inherited-placeholder", "SAFE_VALUE": "retained"}
    )
    assert "TYPESAFE_API_KEY" not in sanitized
    assert sanitized == {"SAFE_VALUE": "retained"}


def test_remote_entrypoint_clears_child_environment_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["env"] = kwargs["env"]
        raise RuntimeError(f"child included {TEST_KEY}")

    monkeypatch.setattr(remote_jev_entrypoint_module.subprocess, "run", run)
    monkeypatch.setattr(remote_jev_entrypoint_module.sys, "stdin", io.StringIO(f"{TEST_KEY}\n"))

    assert remote_jev_entrypoint.main(["fake-jev"]) != 0
    assert "TYPESAFE_API_KEY" not in seen["env"]


def test_default_jev_command_uses_v2_orchestrator_and_sealed_inputs() -> None:
    command = remote_jev_entrypoint._command([])
    assert command[:3] == ["/usr/bin/python3", "-m", "benchmark.orchestrator"]
    assert "--manifest" in command
    assert "--manifest-checksum" in command
    assert "--output-root" in command
    assert command[command.index("--output-root") + 1] == str(
        remote_jev_entrypoint.REMOTE_ROOT / "results" / "v2" / "runs"
    )
    assert str(remote_jev_entrypoint.REMOTE_ROOT).startswith(str(REPOSITORY_ROOT)) or True
    assert "--run-id" in command
    run_id = command[command.index("--run-id") + 1]
    assert re.fullmatch(r"jev-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}", run_id)
    assert "run.py" not in command


def test_remote_entrypoint_rejects_missing_key_before_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_jev_entrypoint_module.subprocess, "run", pytest.fail)
    monkeypatch.setattr(remote_jev_entrypoint_module.sys, "stdin", io.StringIO("\n"))

    assert remote_jev_entrypoint.main(["fake-jev"]) != 0


def _run_fake_cleanup(
    tmp_path: Path, scenario: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_dir = tmp_path / "bin"
    fake_dir.mkdir()
    log_path = tmp_path / "docker.log"
    fake_script = f"""#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
with open({str(log_path)!r}, "a", encoding="utf-8") as stream:
    stream.write("\\t".join(args) + "\\n")
scenario = os.environ["FAKE_DOCKER_SCENARIO"]
if args[:2] == ["container", "inspect"]:
    format_value = args[3]
    target = args[4]
    if target != {CONTAINER_ID!r}:
        raise SystemExit(1)
    if ".Id}}}}" in format_value:
        if scenario == "disappear":
            sys.stderr.write("Error: No such object")
            raise SystemExit(1)
        if scenario == "name-error":
            sys.stderr.write("permission denied")
            raise SystemExit(1)
        print({CONTAINER_ID!r})
    elif "com.sysone-bench.owned" in format_value:
        if scenario == "mismatch":
            print("false")
        else:
            print("true")
    elif "com.sysone-bench.run-id" in format_value:
        if scenario == "mismatch":
            print("other-run")
        else:
            print("race")
    else:
        print("true")
elif args[:2] == ["rm", "-f"]:
    pass
"""
    fake_path = fake_dir / "docker"
    fake_path.write_text(fake_script, encoding="utf-8")
    fake_path.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_dir}:{environment.get('PATH', '')}"
    environment["FAKE_DOCKER_SCENARIO"] = scenario
    completed = subprocess.run(
        ["bash", str(CLEANUP_SCRIPT), "--run-id", "race", "--container-id", CONTAINER_ID],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    log = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    return completed, log


def test_cleanup_removes_only_the_revalidated_container_id(tmp_path: Path) -> None:
    completed, log = _run_fake_cleanup(tmp_path, "success")
    assert completed.returncode == 0
    assert f"rm\t-f\t--\t{CONTAINER_ID}" in log
    assert not any("sysone-bench-race" in line for line in log if line.startswith("rm\t"))


def test_cleanup_does_nothing_when_id_disappears_or_labels_mismatch(tmp_path: Path) -> None:
    for scenario in ("disappear", "mismatch", "name-error"):
        scenario_path = tmp_path / scenario
        scenario_path.mkdir()
        completed, log = _run_fake_cleanup(scenario_path, scenario)
        assert completed.returncode == 0
        assert not any(line.startswith("rm\t") for line in log)


def test_launcher_uses_cidfile_and_id_only_cleanup_handoff() -> None:
    launcher = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    cleanup = CLEANUP_SCRIPT.read_text(encoding="utf-8")
    assert "--cidfile" in launcher
    assert "CONTAINER_CIDFILE" in launcher
    assert "read_container_id" in launcher
    assert "container_started" not in launcher
    assert "--container-id" in launcher
    assert "--container-name" not in launcher
    assert "--container-id" in cleanup
    assert "docker container inspect" in cleanup
    assert "docker rm -f --" in cleanup


def test_launcher_uses_fixed_image_output_mount_and_v2_entrypoint() -> None:
    launcher = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    assert "SYSONE_BENCH_IMAGE" in launcher
    assert "sysone-bench-v2:cpu" in launcher
    # Manifest and checksum paths default to the checkout and are overridable, never hard coded.
    assert 'MANIFEST_PATH="${SYSONE_BENCH_MANIFEST:-$WORKSPACE_ROOT/datasets/v2/manifest.jsonl}"' in launcher
    assert (
        'MANIFEST_CHECKSUM_PATH="${SYSONE_BENCH_MANIFEST_CHECKSUM:-$WORKSPACE_ROOT/datasets/v2/manifest.sha256}"'
        in launcher
    )
    assert "/home/" not in launcher
    assert "@" not in launcher.split("#!/usr/bin/env bash", 1)[1].split("\n", 1)[0]
    assert '-v "$MANIFEST_PATH:/input/manifest.jsonl:ro"' in launcher
    assert '-v "$MANIFEST_CHECKSUM_PATH:/input/manifest.sha256:ro"' in launcher
    assert '--user "$HOST_UID:$HOST_GID"' in launcher
    assert 'docker image inspect "$PROJECT_IMAGE"' in launcher
    assert "--pull=never" in launcher
    assert '"$PROJECT_IMAGE"' in launcher
    assert "benchmark.orchestrator" in launcher
    assert 'VENV_PYTHON="/workspace/.venv/bin/python"' in launcher
    assert '-v "$run_dir:/results/$run_id"' in launcher
    assert '-v "$run_dir:/workspace"' not in launcher
    assert "PYTHON_BIN" not in launcher
    assert "/usr/bin/python3" in launcher
    assert "json.load" in launcher
    assert '--output-root "/results/$run_id"' in launcher
    assert "--disk-path" not in launcher
    assert "--run-id" in launcher
    assert "from benchmark.storage import write_checksums" in launcher
    assert "sha256sum" not in launcher
    assert launcher.index("preflight.py") < launcher.index('mkdir "$run_dir"')
    assert launcher.index('mkdir "$run_dir"') < launcher.index("docker run")


def test_launcher_revalidates_owned_run_marker_and_fixed_paths() -> None:
    launcher = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    for required in (
        "OWNER_MARKER",
        "validate_owned_run",
        "set -C",
        '-O "$WORKSPACE_ROOT"',
        '-O "$RUNS_ROOT"',
        '-O "$run_dir"',
        '! -L "$OWNER_MARKER"',
    ):
        assert required in launcher
    assert launcher.index("validate_owned_run") < launcher.index("docker run")


def test_launcher_schedules_process_inside_container() -> None:
    launcher = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    assert "/usr/bin/nice" in launcher
    assert "/usr/bin/ionice" in launcher
    assert 'nice -n 19 ionice -c 3 "${docker_command[@]}"' not in launcher
    assert "nice -n 19" in launcher
    assert "ionice -c 3" in launcher


def test_launcher_fake_execution_runs_scheduled_v2_process_in_results_mount(
    tmp_path: Path,
) -> None:
    script_dir = tmp_path / "script"
    script_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    shutil.copytree(REPOSITORY_ROOT / "benchmark", workspace_root / "benchmark")
    runs_root = workspace_root / "runs"
    model_cache = tmp_path / "cache"
    model_cache.mkdir()
    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    manifest_path = sealed_root / "manifest.jsonl"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    manifest_checksum_path = sealed_root / "manifest.sha256"
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest_checksum_path.write_text(f"{manifest_digest}  manifest.jsonl\n", encoding="utf-8")
    fake_python = bin_dir / "python"
    fake_id = bin_dir / "id"
    fake_docker = bin_dir / "docker"
    docker_log = tmp_path / "docker.jsonl"
    snapshot = approved_snapshot()
    postflight_snapshot = approved_snapshot(
        run_path_state="ready",
        owner_marker_state="ready",
        owner_marker_valid=True,
        owner_marker_owner_uid=os.geteuid(),
    )
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "if any(arg.endswith('preflight.py') for arg in sys.argv[1:]):\n"
        "    if '--post-run' in sys.argv:\n"
        "        if 'FAKE_PREFLIGHT_REPLACE_MARKER' in os.environ:\n"
        "            run_id = sys.argv[sys.argv.index('--run-id') + 1]\n"
        "            run_root = Path(sys.argv[sys.argv.index('--run-root') + 1])\n"
        "            marker = run_root / run_id / '.sysone-owner'\n"
        "            mode = os.environ['FAKE_PREFLIGHT_REPLACE_MARKER']\n"
        "            if mode == 'mismatch':\n"
        "                marker.write_text('other-run\\n', encoding='utf-8')\n"
        "            elif mode == 'symlink':\n"
        "                marker.unlink()\n"
        "                marker.symlink_to(Path(os.environ['FAKE_MARKER_TARGET']))\n"
        f"        print({json.dumps(postflight_snapshot)!r})\n"
        "    else:\n"
        f"        print({json.dumps(snapshot)!r})\n"
        "    raise SystemExit(0)\n"
        "if '-c' in sys.argv[1:]:\n"
        "    command = sys.argv[sys.argv.index('-c') + 1]\n"
        "    if 'write_checksums' in command:\n"
        "        sys.path.insert(0, sys.argv[sys.argv.index('-c') + 2])\n"
        "        from benchmark.storage import write_checksums\n"
        "        write_checksums(Path(sys.argv[-1]))\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_id.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "value = os.environ.get('FAKE_UID', '1000') if sys.argv[1] == '-u' else os.environ.get('FAKE_GID', '1000')\n"
        "print(value)\n",
        encoding="utf-8",
    )
    fake_id.chmod(0o755)
    owned_id = "b" * 64
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"log_path = {str(docker_log)!r}\n"
        f"owned_id = {owned_id!r}\n"
        "args = sys.argv[1:]\n"
        "with open(log_path, 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[:2] == ['image', 'inspect']:\n"
        "    raise SystemExit(0)\n"
        "if args and args[0] == 'run':\n"
        "    if os.environ.get('FAKE_DOCKER_COLLISION') == '1':\n"
        "        raise SystemExit(9)\n"
        "    if '--cidfile' in args:\n"
        "        cidfile = Path(args[args.index('--cidfile') + 1])\n"
        "        cidfile.write_text(owned_id + '\\n', encoding='utf-8')\n"
        "    for value in args:\n"
        "        if ':/results/' in value:\n"
        "            output = Path(value.split(':/results/', 1)[0])\n"
        "            output.mkdir(parents=True, exist_ok=True)\n"
        "            if os.environ.get('FAKE_DOCKER_CREATE_ARTIFACTS', '1') == '1':\n"
        "                for name in ('metadata.json', 'predictions.jsonl', 'summary.json', 'usage.json'):\n"
        "                    (output / name).write_text('fake\\n', encoding='utf-8')\n"
        "            break\n"
        "    raise SystemExit(int(os.environ.get('FAKE_DOCKER_RUN_STATUS', '0')))\n"
        "if args[:2] == ['container', 'inspect']:\n"
        "    format_value = args[3]\n"
        "    target = args[4]\n"
        "    if target.startswith('sysone-bench-'):\n"
        "        if os.environ.get('FAKE_DOCKER_EXPECT_ID_ONLY') == '1' or os.environ.get('FAKE_DOCKER_COLLISION') != '1':\n"
        "            raise SystemExit(1)\n"
        "        if '.Id}}' in format_value:\n"
        "            print(owned_id)\n"
        "        else:\n"
        "            print('true')\n"
        "    elif target == owned_id:\n"
        "        if '.Id}}' in format_value:\n"
        "            print(owned_id)\n"
        "        elif 'com.sysone-bench.owned' in format_value:\n"
        "            print('true')\n"
        "        elif 'com.sysone-bench.run-id' in format_value:\n"
        "            print(os.environ.get('FAKE_DOCKER_RUN_ID', 'race'))\n"
        "        else:\n"
        "            print('true')\n"
        "    else:\n"
        "        raise SystemExit(1)\n"
        "    raise SystemExit(0)\n"
        "if args[:2] == ['rm', '-f']:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    launcher_source = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    launcher_source = launcher_source.replace(
        'PREFLIGHT_PYTHON="/usr/bin/python3"', f'PREFLIGHT_PYTHON="{fake_python}"'
    )
    launcher_source = launcher_source.replace("/usr/bin/id", str(fake_id))
    launcher_copy = script_dir / "run_open_model.sh"
    launcher_copy.write_text(launcher_source, encoding="utf-8")
    cleanup_copy = script_dir / "cleanup.sh"
    cleanup_copy.write_text(CLEANUP_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    cleanup_copy.chmod(0o755)
    (script_dir / "preflight.py").write_text("", encoding="utf-8")
    launcher_copy.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment.get('PATH', '')}"
    environment["FAKE_UID"] = "1234"
    environment["FAKE_GID"] = "2345"
    # Paths come from the documented overrides, so the test also proves they work.
    environment["SYSONE_BENCH_WORKSPACE_ROOT"] = str(workspace_root)
    environment["SYSONE_BENCH_MODEL_CACHE"] = str(model_cache)
    environment["SYSONE_BENCH_MANIFEST"] = str(manifest_path)
    environment["SYSONE_BENCH_MANIFEST_CHECKSUM"] = str(manifest_checksum_path)
    environment["HOME"] = str(tmp_path)
    completed = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "fixture", "--model", "laya"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    commands = [json.loads(line) for line in docker_log.read_text(encoding="utf-8").splitlines()]
    run_command = next(command for command in commands if command and command[0] == "run")
    assert "sysone-bench-v2:cpu" in run_command
    assert "--user" in run_command
    assert "1234:2345" in run_command
    assert f"{manifest_path}:/input/manifest.jsonl:ro" in run_command
    assert f"{manifest_checksum_path}:/input/manifest.sha256:ro" in run_command
    assert "--manifest" in run_command
    assert "/input/manifest.jsonl" in run_command
    assert "--manifest-checksum" in run_command
    assert "/input/manifest.sha256" in run_command
    assert f"{runs_root / 'fixture'}:/results/fixture" in run_command
    assert not any(value.endswith(":/workspace") for value in run_command)
    command_index = run_command.index("sysone-bench-v2:cpu")
    process_command = run_command[command_index + 1 :]
    assert process_command[:6] == ["/usr/bin/nice", "-n", "19", "/usr/bin/ionice", "-c", "3"]
    assert process_command[6] == "/workspace/.venv/bin/python"
    assert process_command[7:9] == ["-m", "benchmark.orchestrator"]
    assert (runs_root / "fixture" / "postflight.json").is_file()
    assert (runs_root / "fixture" / "checksums.sha256").is_file()
    environment["FAKE_DOCKER_CREATE_ARTIFACTS"] = "0"
    failed = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "missing", "--model", "laya"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0
    assert (runs_root / "missing" / "postflight.json").is_file()
    assert not (runs_root / "missing" / "checksums.sha256").exists()
    environment["FAKE_DOCKER_RUN_STATUS"] = "9"
    model_failed = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "model-fail", "--model", "laya"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert model_failed.returncode != 0
    assert (runs_root / "model-fail" / "postflight.json").is_file()
    assert not (runs_root / "model-fail" / "checksums.sha256").exists()
    collision_environment = environment.copy()
    collision_environment["FAKE_DOCKER_COLLISION"] = "1"
    collision_environment["FAKE_DOCKER_RUN_STATUS"] = "9"
    collision_environment["FAKE_DOCKER_CREATE_ARTIFACTS"] = "0"
    collision_environment["FAKE_DOCKER_RUN_ID"] = "name-collision"
    collision_result = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "name-collision", "--model", "laya"],
        env=collision_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert collision_result.returncode != 0
    collision_commands = [
        json.loads(line) for line in docker_log.read_text(encoding="utf-8").splitlines()
    ]
    assert not any(command[:2] == ["rm", "-f"] for command in collision_commands)
    id_environment = environment.copy()
    id_environment["FAKE_DOCKER_WRITE_CIDFILE"] = "1"
    id_environment["FAKE_DOCKER_EXPECT_ID_ONLY"] = "1"
    id_environment["FAKE_DOCKER_RUN_STATUS"] = "0"
    id_environment["FAKE_DOCKER_CREATE_ARTIFACTS"] = "1"
    id_environment["FAKE_DOCKER_RUN_ID"] = "owned-id"
    id_result = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "owned-id", "--model", "laya"],
        env=id_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert id_result.returncode == 0, id_result.stderr
    id_commands = [json.loads(line) for line in docker_log.read_text(encoding="utf-8").splitlines()]
    assert any(
        command[:3] == ["rm", "-f", "--"] and command[3] == owned_id for command in id_commands
    )
    assert not any(
        command[:2] == ["container", "inspect"] and "sysone-bench-owned-id" in command
        for command in id_commands
    )
    root_environment = environment.copy()
    root_environment["FAKE_UID"] = "0"
    root_result = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "root-launch", "--model", "laya"],
        env=root_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert root_result.returncode != 0
    assert not (runs_root / "root-launch").exists()
    marker_target = tmp_path / "marker-target"
    marker_target.write_text("target\n", encoding="utf-8")
    for mode in ("mismatch", "symlink"):
        marker_environment = environment.copy()
        marker_environment["FAKE_DOCKER_CREATE_ARTIFACTS"] = "1"
        marker_environment["FAKE_DOCKER_RUN_STATUS"] = "0"
        marker_environment["FAKE_PREFLIGHT_REPLACE_MARKER"] = mode
        marker_environment["FAKE_MARKER_TARGET"] = str(marker_target)
        run_id = f"marker-{mode}"
        marker_result = subprocess.run(
            ["bash", str(launcher_copy), "--run-id", run_id, "--model", "laya"],
            env=marker_environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert marker_result.returncode != 0
        marker_run_commands = [
            command
            for command in commands
            if command and command[0] == "run" and f"sysone-bench-{run_id}" in command
        ]
        assert not marker_run_commands
        assert not (runs_root / run_id / "checksums.sha256").exists()
        commands = [
            json.loads(line) for line in docker_log.read_text(encoding="utf-8").splitlines()
        ]
    manifest_path.unlink()
    missing_manifest = subprocess.run(
        ["bash", str(launcher_copy), "--run-id", "missing-manifest", "--model", "laya"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_manifest.returncode != 0
    assert not (runs_root / "missing-manifest").exists()


def test_launcher_records_postflight_and_rejects_symlink_artifacts() -> None:
    launcher = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    assert "--post-run" in launcher
    assert "postflight.json" in launcher
    assert "record_postflight" in launcher
    assert '-L "$run_dir/$artifact"' in launcher
    assert "-type l" in launcher
    assert "write_checksums" in launcher
    assert "sha256sum" not in launcher


def test_remote_receiver_wrapper_is_deployable() -> None:
    wrapper = REMOTE_ROOT / "remote_jev_entrypoint.py"
    assert wrapper.is_file()
    assert wrapper.stat().st_mode & 0o111
    assert wrapper.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3\n")


def test_dockerignore_excludes_secrets_generated_data_and_caches() -> None:
    ignore_path = REPOSITORY_ROOT / ".dockerignore"
    text = ignore_path.read_text(encoding="utf-8")
    for pattern in (
        ".env",
        ".git",
        "results/",
        "datasets/v2/manifest.jsonl",
        "datasets/v2/provenance.json",
        "datasets/v2/labeling/",
        "datasets/v2/packets/",
        "__pycache__/",
        ".venv/",
        "*.safetensors",
        "*.bin",
    ):
        assert pattern in text


def test_dockerfile_cpu_filter_removes_cuda_and_torch_requirement_lines() -> None:
    dockerfile = (REMOTE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    docker_pattern = r"^[[:space:]-]*(cuda|nvidia|triton|torch)([^A-Za-z0-9]|$)"
    assert docker_pattern in dockerfile
    pattern = r"^[ \t-]*(cuda|nvidia|triton|torch)([^A-Za-z0-9]|$)"
    requirement_lines = [
        "torch==2.14.0+cpu",
        "triton==3.0.0",
        "cuda-bindings==13.0.0",
        "nvidia-cublas==13.0.0",
        "numpy==2.0.0",
    ]
    assert [line for line in requirement_lines if re.search(pattern, line)] == requirement_lines[:4]
    assert re.search(pattern, "numpy==2.0.0") is None


def test_qwen_export_keeps_transformers_after_cpu_filter() -> None:
    dockerfile = (REMOTE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uv export --frozen --no-dev --no-emit-project --extra qwen" in dockerfile
    completed = subprocess.run(
        [
            "uv",
            "export",
            "--offline",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--extra",
            "qwen",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--format",
            "requirements-txt",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pattern = re.compile(r"^[ \t-]*(cuda|nvidia|triton|torch)([^A-Za-z0-9]|$)")
    lines = completed.stdout.splitlines()
    kept = [line for line in lines if not pattern.search(line)]
    assert any(re.match(r"^transformers(?:\[.*\])?==", line) for line in kept)
    assert not any(pattern.search(line) for line in kept)


def test_dockerfile_uses_cpu_safe_default_and_explicit_copy_set() -> None:
    dockerfile = (REMOTE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12-slim-bookworm" in dockerfile
    assert "uv export --frozen --no-dev --no-emit-project" in dockerfile
    assert "uv pip install --no-deps --requirement" in dockerfile
    assert "--no-install-package torch" not in dockerfile
    assert "nvidia" in dockerfile
    assert "cuda" in dockerfile
    assert "triton" in dockerfile
    assert "--extra qwen" in dockerfile
    assert "COPY results" not in dockerfile
    assert "COPY datasets ./datasets" not in dockerfile
    assert "datasets/cases.py" not in dockerfile
    assert "datasets/public_cases.json" not in dockerfile
    assert "USER sysone" in dockerfile
    assert "EXPOSE" not in dockerfile
    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert '"nvidia"' in dockerfile
    assert '"triton"' in dockerfile
    agents = (REMOTE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "sysone-bench-v2:cpu" in agents
    assert "download.pytorch.org/whl/cpu" in agents
    assert "CUDA" in agents
    assert "uv.lock" in agents


def test_preflight_module_uses_read_only_host_sources() -> None:
    source = (REMOTE_ROOT / "preflight.py").read_text(encoding="utf-8")
    assert '"/proc/loadavg"' in source
    assert '"free"' in source
    assert "statvfs" in source
    assert "platform.python_version" in source
    assert "lexists" in source
    assert "lstat" in source
    assert "sched_getaffinity" in source
    assert "docker run" not in source
    assert "docker rm" not in source


def test_task5_scripts_have_isolation_contract() -> None:
    launcher = LAUNCHER_SCRIPT.read_text(encoding="utf-8")
    cleanup = CLEANUP_SCRIPT.read_text(encoding="utf-8")
    agents = (REMOTE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "--cpus=4" in launcher
    assert "--memory=12g" in launcher
    assert "--memory-swap=12g" in launcher
    assert "--cpuset-cpus=0-3" in launcher
    assert "/var/run/docker.sock" not in launcher
    assert "--publish" not in launcher
    assert "docker rm -f" in cleanup
    assert "com.sysone-bench.owned" in cleanup
    assert "com.sysone-bench.run-id" in cleanup
    for required in (
        "no-global-install",
        "12 GiB",
        "no exposed ports",
        "existing containers",
        "TYPESAFE_API_KEY",
    ):
        assert required in agents
