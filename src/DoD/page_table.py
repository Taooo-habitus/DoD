"""Page table representation and IO helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence


@dataclass
class PageRecord:
    """Represents extracted text for a single page."""

    page_id: int
    text: str
    image_path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Return JSONL string for the page record."""
        return json.dumps(asdict(self), ensure_ascii=True)


def write_page_table(path: Path, records: Sequence[PageRecord]) -> None:
    """Write page records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_jsonl() + "\n")


def iter_page_table(
    path: Path, page_range: Optional[tuple[int, int]] = None
) -> Iterator[PageRecord]:
    """Iterate page records from JSONL. Optionally filter by page range (inclusive)."""
    start, end = page_range if page_range else (None, None)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            record = PageRecord(
                page_id=int(data["page_id"]),
                text=data.get("text", ""),
                image_path=data.get("image_path"),
                metadata=data.get("metadata", {}),
            )
            if start is not None and record.page_id < start:
                continue
            if end is not None and record.page_id > end:
                continue
            yield record


def load_page_range(
    path: Path, page_range: Optional[tuple[int, int]] = None
) -> list[PageRecord]:
    """Load a list of page records, optionally filtered by page range."""
    return list(iter_page_table(path, page_range=page_range))
