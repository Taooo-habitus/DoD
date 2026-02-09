"""Tests for extractor backend selection behavior."""

from pathlib import Path

import pytest

from DoD.config import PipelineConfig
from DoD.pipeline import _select_extractor
from DoD.text_extractor.plain_text import PlainTextExtractor
from DoD.text_extractor.pymupdf import PyMuPDFExtractor
from DoD.text_extractor.pytesseract import PyTesseractExtractor


def test_select_extractor_uses_plain_text_for_text_inputs() -> None:
    """Text files should always use the plain text extractor."""
    cfg = PipelineConfig(input_path="doc.txt")
    cfg.text_extractor.backend = "pymupdf"

    extractor = _select_extractor(cfg, Path("doc.txt"))

    assert isinstance(extractor, PlainTextExtractor)


def test_select_extractor_accepts_pymupdf_alias() -> None:
    """The pymupdf4llm alias should map to the PyMuPDF extractor."""
    cfg = PipelineConfig(input_path="doc.pdf")
    cfg.text_extractor.backend = "pymupdf4llm"

    extractor = _select_extractor(cfg, Path("doc.pdf"))

    assert isinstance(extractor, PyMuPDFExtractor)


def test_select_extractor_removed_backend_raises() -> None:
    """Removed backends should raise a clear error."""
    cfg = PipelineConfig(input_path="doc.pdf")
    cfg.text_extractor.backend = "glm_ocr"

    with pytest.raises(ValueError, match="has been removed"):
        _select_extractor(cfg, Path("doc.pdf"))


def test_select_extractor_accepts_pytesseract_backend() -> None:
    """Pytesseract should map to the PyTesseract extractor."""
    cfg = PipelineConfig(input_path="doc.pdf")
    cfg.text_extractor.backend = "pytesseract"

    extractor = _select_extractor(cfg, Path("doc.pdf"))

    assert isinstance(extractor, PyTesseractExtractor)


def test_select_extractor_unknown_backend_raises() -> None:
    """Unknown backends should raise a clear error."""
    cfg = PipelineConfig(input_path="doc.pdf")
    cfg.text_extractor.backend = "unknown_backend"

    with pytest.raises(ValueError, match="Unsupported text extractor backend"):
        _select_extractor(cfg, Path("doc.pdf"))
