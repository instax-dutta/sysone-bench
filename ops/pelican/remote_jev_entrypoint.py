#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

API_KEY_NAME = "TYPESAFE_API_KEY"
REMOTE_ROOT = Path("/home/tejes/sysone-bench-v2")
REMOTE_MANIFEST_PATH = "/home/tejes/sysone-bench-v2/datasets/v2/manifest.jsonl"
REMOTE_MANIFEST_CHECKSUM_PATH = "/home/tejes/sysone-bench-v2/datasets/v2/manifest.sha256"
REMOTE_OUTPUT_ROOT = "/home/tejes/sysone-bench-v2/results/v2/runs"
DEFAULT_JEV_COMMAND = (
    "/usr/bin/python3",
    "-m",
    "benchmark.orchestrator",
    "--models",
    "jev",
    "--manifest",
    REMOTE_MANIFEST_PATH,
    "--manifest-checksum",
    REMOTE_MANIFEST_CHECKSUM_PATH,
    "--output-root",
    REMOTE_OUTPUT_ROOT,
)


def _sanitized_environment(environ: Mapping[str, str]) -> dict[str, str]:
    sanitized = dict(environ)
    sanitized.pop(API_KEY_NAME, None)
    return sanitized


def _read_key(stream: Any) -> str:
    line = stream.readline()
    if not isinstance(line, str) or not line:
        raise RuntimeError("remote Jev credential is missing")
    line = line.removesuffix("\n")
    line = line.removesuffix("\r")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in line):
        raise RuntimeError("remote Jev credential is invalid")
    extra = stream.readline()
    if isinstance(extra, str) and extra.strip():
        raise RuntimeError("remote Jev credential input must contain one line")
    return line


def _generated_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"jev-{timestamp}-{uuid4().hex[:12]}"


def _command(argv: Sequence[str] | None) -> list[str]:
    if argv is None:
        selected = list(sys.argv[1:])
    else:
        selected = list(argv)
    if not selected:
        selected = [*DEFAULT_JEV_COMMAND, "--run-id", _generated_run_id()]
    if any(not isinstance(value, str) or "\x00" in value for value in selected):
        raise RuntimeError("remote Jev command is invalid")
    return selected


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    environ: Mapping[str, str] | None = None,
    runner: Any = None,
) -> int:
    source = sys.stdin if stdin is None else stdin
    try:
        api_key = _read_key(source)
        command = _command(argv)
        if any(api_key in value for value in command):
            raise RuntimeError("remote Jev credential cannot be an argument")
        base_environment = _sanitized_environment(os.environ if environ is None else environ)
        child_environment = dict(base_environment)
        child_environment[API_KEY_NAME] = api_key
        command_runner = subprocess.run if runner is None else runner
        try:
            completed = command_runner(
                command,
                cwd=str(REMOTE_ROOT),
                env=child_environment,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            child_environment.pop(API_KEY_NAME, None)
        returncode = getattr(completed, "returncode", 1)
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            return 1
        return returncode
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
