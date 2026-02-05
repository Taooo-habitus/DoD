"""OCR / text extraction interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord


class TextExtractor(ABC):
    """Base class for text extraction backends."""

    requires_images: bool = True

    @staticmethod
    def _progress(iterable, total: int | None = None, desc: str = "OCR"):
        try:
            from tqdm import tqdm
        except ImportError:
            return iterable
        return tqdm(iterable, total=total, desc=desc)

    @abstractmethod
    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract page-level text."""
        raise NotImplementedError
