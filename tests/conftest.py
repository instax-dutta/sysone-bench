import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_json(tmp_path: Path):
    def write(name: str, value: object) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    return write
