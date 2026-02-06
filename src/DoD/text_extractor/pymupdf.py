"""Text extraction via PyMuPDF (pymupdf4llm dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


class PyMuPDFExtractor(TextExtractor):
    """Extract text directly from PDFs using PyMuPDF."""

    # Keep images in the pipeline for downstream encoding needs.
    requires_images = True

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract text from each PDF page."""
        try:
            import pymupdf4llm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pymupdf4llm is required for the PyMuPDF text extractor. "
                "Install from https://pypi.org/project/pymupdf4llm/."
            ) from exc

        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF (fitz) is required for the PyMuPDF text extractor. "
                "Install via pymupdf4llm or pymupdf."
            ) from exc

        doc = fitz.open(str(input_path))
        try:
            page_count = getattr(doc, "page_count", len(doc))
            records: List[PageRecord] = []
            for page_index in self._progress(
                range(page_count), total=page_count, desc="Text (PyMuPDF)"
            ):
                page = doc.load_page(page_index)
                text = page.get_text("text").strip()
                image_path = None
                if image_paths and page_index < len(image_paths):
                    image_path = str(image_paths[page_index])
                records.append(
                    PageRecord(
                        page_id=page_index + 1,
                        text=text,
                        image_path=image_path,
                        metadata={"backend": "pymupdf"},
                    )
                )
            return records
        finally:
            doc.close()
