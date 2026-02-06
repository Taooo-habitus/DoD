"""Tests for generic JSON artifact IO helpers."""

from pathlib import Path

from DoD.io.artifacts import read_json, write_json


def test_write_and_read_json_roundtrip(tmp_path: Path) -> None:
    """write_json and read_json should roundtrip nested structures."""
    path = tmp_path / "nested" / "artifact.json"
    payload = {"a": 1, "b": {"x": [1, 2, 3]}, "c": "text"}

    write_json(path, payload)

    assert path.exists()
    assert read_json(path) == payload
