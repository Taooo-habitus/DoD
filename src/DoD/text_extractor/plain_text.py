"""Plain text extractor for Markdown or TXT files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


class PlainTextExtractor(TextExtractor):
    """Extract text directly from plain text files."""

    requires_images = False

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract text directly from the file."""
        text = input_path.read_text(encoding="utf-8")
        pages = _split_pages(text)
        records: List[PageRecord] = []
        for idx, page_text in enumerate(pages, start=1):
            records.append(
                PageRecord(
                    page_id=idx,
                    text=page_text.strip(),
                    image_path=None,
                    metadata={"backend": "plain_text"},
                )
            )
        return records


def _split_pages(text: str) -> List[str]:
    if "\f" in text:
        return [chunk for chunk in text.split("\f") if chunk.strip()]
    return [text]
