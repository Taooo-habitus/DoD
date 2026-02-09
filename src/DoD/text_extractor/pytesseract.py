"""Text extraction via Tesseract OCR (pytesseract)."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


class PyTesseractExtractor(TextExtractor):
    """Extract text from page images using pytesseract."""

    requires_images = True

    def __init__(self, batch: int = 1) -> None:
        """Batching Config."""
        self.batch = max(1, int(batch))

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract text from each page image."""
        if not image_paths:
            raise ValueError("PyTesseract extractor requires image paths.")

        try:
            pytesseract = importlib.import_module("pytesseract")
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is required for the PyTesseract text extractor. "
                "Install from https://pypi.org/project/pytesseract/."
            ) from exc

        try:
            pil_image = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for the PyTesseract text extractor. "
                "Install from https://pypi.org/project/Pillow/."
            ) from exc

        def _ocr_image(path: Path) -> str:
            with pil_image.open(path) as image:
                return pytesseract.image_to_string(image) or ""

        results: dict[int, str] = {}
        if self.batch <= 1:
            for idx, image_path in self._progress(
                enumerate(image_paths, start=1),
                total=len(image_paths),
                desc="Text (PyTesseract)",
            ):
                results[idx] = _ocr_image(image_path)
        else:
            with ThreadPoolExecutor(max_workers=self.batch) as executor:
                futures = {
                    executor.submit(_ocr_image, path): idx
                    for idx, path in enumerate(image_paths, start=1)
                }
                for future in self._progress(
                    as_completed(futures), total=len(futures), desc="Text (PyTesseract)"
                ):
                    results[futures[future]] = future.result()

        records: List[PageRecord] = []
        for idx, image_path in enumerate(image_paths, start=1):
            text = results.get(idx, "")
            records.append(
                PageRecord(
                    page_id=idx,
                    text=text.strip(),
                    image_path=str(image_path),
                    metadata={"backend": "pytesseract"},
                )
            )
        return records
