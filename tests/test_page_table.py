"""Tests for page table IO."""

from pathlib import Path

from DoD.page_table import PageRecord, iter_page_table, write_page_table


def test_page_table_write_and_iter_range(tmp_path: Path) -> None:
    """Write JSONL and read a page range."""
    path = tmp_path / "page_table.jsonl"
    records = [
        PageRecord(page_id=1, text="A"),
        PageRecord(page_id=2, text="B"),
        PageRecord(page_id=3, text="C"),
    ]

    write_page_table(path, records)

    subset = list(iter_page_table(path, page_range=(2, 3)))
    assert [rec.page_id for rec in subset] == [2, 3]
    assert [rec.text for rec in subset] == ["B", "C"]
