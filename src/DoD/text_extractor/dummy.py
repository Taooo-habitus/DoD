"""Dummy text extraction backend for testing without dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


class DummyExtractor(TextExtractor):
    """Return empty text for each page."""

    requires_images = True

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Return empty text per page."""
        if not image_paths:
            raise ValueError("Dummy extractor requires image paths.")
        return [
            PageRecord(
                page_id=idx,
                text="",
                image_path=str(image_path),
                metadata={"backend": "dummy"},
            )
            for idx, image_path in enumerate(image_paths, start=1)
        ]
