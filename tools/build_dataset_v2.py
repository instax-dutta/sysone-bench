from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _external_datasets_module(root: Path) -> ModuleType:
    module = importlib.import_module("datasets")
    origin = getattr(module, "__file__", None)
    local_init = (root / "datasets" / "__init__.py").resolve()
    if origin is not None and Path(origin).resolve() == local_init:
        raise RuntimeError(f"external datasets package resolved to local file {local_init}")
    assert origin is not None and Path(origin).resolve() != local_init
    return module


def _extend_package_path(module: ModuleType, root: Path) -> None:
    package_path = root / "datasets"
    paths = getattr(module, "__path__", None)
    if paths is None:
        raise RuntimeError("external datasets package has no import path")
    local_path = str(package_path)
    if local_path not in list(paths):
        paths.append(local_path)
    sys.path.insert(0, str(root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the provisional sysone-bench v2 manifest")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = _repository_root()
    external_datasets = _external_datasets_module(root)
    _extend_package_path(external_datasets, root)
    build_module = importlib.import_module("datasets.v2.build_manifest")
    sources_module = importlib.import_module("datasets.v2.sources")
    records = build_module.build_records(args.output, sources_module.load_sources())
    digest = build_module.write_manifest(records, args.output)
    print(f"wrote {len(records)} records to {args.output / 'manifest.jsonl'}")
    print(f"manifest digest {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
