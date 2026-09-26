from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from benchmark.orchestrator import compare_v2

DEFAULT_OUTPUT_ROOT = Path("results/v2/comparisons")


def ci(acc: float, n: int, z: float = 1.96) -> tuple[float, float]:
    se = math.sqrt(acc * (1 - acc) / n)
    return round(max(0, acc - z * se), 3), round(min(1, acc + z * se), 3)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare.py")
    parser.add_argument("path_a", type=Path)
    parser.add_argument("path_b", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if error.code is None:
            return 0
        if isinstance(error.code, int):
            return error.code
        return 1
    try:
        output = compare_v2(args.path_a, args.path_b, args.output_root)
    except (OSError, TypeError, ValueError) as error:
        print(f"compare failed: {error}", file=sys.stderr)
        return 1
    print(f"comparison artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
