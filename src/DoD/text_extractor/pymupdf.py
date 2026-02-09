"""Text extraction via PyMuPDF (pymupdf4llm dependency)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


class PyMuPDFExtractor(TextExtractor):
    """Extract text directly from PDFs using PyMuPDF."""

    # Keep images in the pipeline for downstream encoding needs.
    requires_images = True

    def __init__(self, batch: int = 1) -> None:
        """Batching Config."""
        self.batch = max(1, int(batch))

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

        def _extract_page(page_index: int) -> str:
            doc = fitz.open(str(input_path))
            try:
                page = doc.load_page(page_index)
                return page.get_text("text").strip()
            finally:
                doc.close()

        doc = fitz.open(str(input_path))
        try:
            page_count = getattr(doc, "page_count", len(doc))
        finally:
            doc.close()

        results: dict[int, str] = {}
        if self.batch <= 1:
            doc = fitz.open(str(input_path))
            try:
                for page_index in self._progress(
                    range(page_count), total=page_count, desc="Text (PyMuPDF)"
                ):
                    page = doc.load_page(page_index)
                    results[page_index] = page.get_text("text").strip()
            finally:
                doc.close()
        else:
            with ThreadPoolExecutor(max_workers=self.batch) as executor:
                futures = {
                    executor.submit(_extract_page, page_index): page_index
                    for page_index in range(page_count)
                }
                for future in self._progress(
                    as_completed(futures), total=len(futures), desc="Text (PyMuPDF)"
                ):
                    results[futures[future]] = future.result()

        records: List[PageRecord] = []
        for page_index in range(page_count):
            image_path = None
            if image_paths and page_index < len(image_paths):
                image_path = str(image_paths[page_index])
            records.append(
                PageRecord(
                    page_id=page_index + 1,
                    text=results.get(page_index, ""),
                    image_path=image_path,
                    metadata={"backend": "pymupdf"},
                )
            )
        return records
