from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPOSITORY_ROOT / ".env"
# Where the benchmark runs is deployment configuration, not a repository default, so nothing
# personal is baked in. Set both variables to run the Jev leg:
#   SYSONE_BENCH_SSH_HOST      for example "runner@benchmark-host"
#   SYSONE_BENCH_REMOTE_ROOT   the checkout path on that host
REMOTE_HOST = os.environ.get("SYSONE_BENCH_SSH_HOST", "")
REMOTE_ROOT = os.environ.get("SYSONE_BENCH_REMOTE_ROOT", "")


def remote_entrypoint() -> str:
    """Absolute path of the entrypoint inside the remote checkout."""
    if not REMOTE_ROOT:
        raise RuntimeError("SYSONE_BENCH_REMOTE_ROOT is not set")
    return f"{REMOTE_ROOT.rstrip('/')}/ops/remote/remote_jev_entrypoint.py"
API_KEY_NAME = "TYPESAFE_API_KEY"
SSH_TIMEOUT_SECONDS = 6 * 60 * 60


def _validate_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("remote Jev credential is missing")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise RuntimeError("remote Jev credential is invalid")
    return value


def _read_api_key(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise RuntimeError("local Jev environment could not be read") from None
    found: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != API_KEY_NAME:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            raise RuntimeError("remote Jev credential is missing")
        if found is not None:
            raise RuntimeError("local Jev environment contains duplicate credential entries")
        found = value
    if found is None:
        raise RuntimeError("remote Jev credential is missing")
    return _validate_key(found)


def read_api_key() -> str:
    return _read_api_key(DOTENV_PATH)


def send_api_key(api_key: str, *, runner: Any = None) -> int:
    key = _validate_key(api_key)
    if not REMOTE_HOST:
        raise RuntimeError("SYSONE_BENCH_SSH_HOST is not set")
    command = ["ssh", REMOTE_HOST, remote_entrypoint()]
    command_runner = subprocess.run if runner is None else runner
    sanitized_environment = dict(os.environ)
    sanitized_environment.pop(API_KEY_NAME, None)
    try:
        completed = command_runner(
            command,
            input=f"{key}\n",
            env=sanitized_environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=SSH_TIMEOUT_SECONDS,
        )
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError):
        raise RuntimeError("remote Jev transport failed") from None
    returncode = getattr(completed, "returncode", 1)
    if not isinstance(returncode, int) or returncode != 0:
        raise RuntimeError("remote Jev command failed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream the Jev credential over SSH stdin")
    parser.parse_args(argv)
    try:
        api_key = read_api_key()
        send_api_key(api_key)
    except (OSError, RuntimeError, TypeError, ValueError):
        print("remote Jev invocation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
