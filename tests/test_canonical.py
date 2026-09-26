import hashlib
import math
from pathlib import Path

import pytest

from benchmark.canonical import canonical_json, digest_bytes, digest_file, digest_value


def test_object_keys_are_sorted_and_separators_are_compact():
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_none_is_serialized_as_an_explicit_unavailable_value():
    assert canonical_json({"nll": None}) == b'{"nll":null}'


def test_array_order_is_preserved():
    assert canonical_json(["b", "a"]) == b'["b","a"]'


def test_unicode_is_utf8_without_escaping():
    assert canonical_json({"b": "é", "a": "雪"}) == '{"a":"雪","b":"é"}'.encode()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nested_nonfinite_numbers_are_rejected(value):
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"nested": [value]})


def test_digest_bytes_is_full_lowercase_sha256():
    assert digest_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_digest_value_hashes_canonical_json_bytes():
    value = {"b": 2, "a": 1}
    assert digest_value(value) == digest_bytes(canonical_json(value))


def test_file_digest_matches_bytes(tmp_path: Path):
    path = tmp_path / "value.json"
    path.write_bytes(b"abc")
    assert digest_file(path) == digest_bytes(b"abc")


def test_file_digest_matches_canonical_value(tmp_path: Path):
    path = tmp_path / "value.json"
    path.write_bytes(canonical_json("abc"))
    assert digest_file(path) == digest_value("abc")
