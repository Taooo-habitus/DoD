"""Additional tests for page table image-related IO."""

import base64
import json
from pathlib import Path

from DoD.page_table import (
    PageRecord,
    load_page_range,
    write_image_page_table,
    write_page_table,
)


def test_write_image_page_table_encodes_base64(tmp_path: Path) -> None:
    """Image page table should contain base64 bytes for each image."""
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"abc")
    image_b.write_bytes(b"xyz")

    path = tmp_path / "image_page_table.jsonl"
    write_image_page_table(path, [image_a, image_b])

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["page_id"] == 1
    assert second["page_id"] == 2
    assert base64.b64decode(first["image_b64"]) == b"abc"
    assert base64.b64decode(second["image_b64"]) == b"xyz"


def test_load_page_range_wrapper(tmp_path: Path) -> None:
    """load_page_range should wrap iter_page_table as a list."""
    path = tmp_path / "page_table.jsonl"
    write_page_table(
        path,
        [
            PageRecord(page_id=1, text="A"),
            PageRecord(page_id=2, text="B"),
            PageRecord(page_id=3, text="C"),
        ],
    )
    subset = load_page_range(path, page_range=(1, 2))
    assert [item.page_id for item in subset] == [1, 2]
