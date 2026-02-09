"""Tests for the PyTesseract text extractor."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from DoD.text_extractor.pytesseract import PyTesseractExtractor


def test_pytesseract_requires_images(tmp_path: Path) -> None:
    """Extractor should reject calls without image paths."""
    extractor = PyTesseractExtractor(batch=2)

    with pytest.raises(ValueError, match="requires image paths"):
        extractor.extract(tmp_path / "doc.pdf", image_paths=None)


def test_pytesseract_extracts_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Extractor should OCR each image path."""
    image_paths = [tmp_path / "page_0001.png", tmp_path / "page_0002.png"]
    for path in image_paths:
        path.write_bytes(b"image")

    class FakeImage:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(path):
        return FakeImage(path)

    fake_pil = SimpleNamespace(open=fake_open)
    fake_pytesseract = SimpleNamespace(
        image_to_string=lambda image: f"Text for {image.path.name}"
    )

    real_import = importlib.import_module

    def fake_import(name):
        if name == "pytesseract":
            return fake_pytesseract
        if name == "PIL.Image":
            return fake_pil
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    extractor = PyTesseractExtractor()
    records = extractor.extract(tmp_path / "doc.pdf", image_paths=image_paths)

    assert [record.page_id for record in records] == [1, 2]
    assert records[0].text == "Text for page_0001.png"
    assert records[0].metadata["backend"] == "pytesseract"
    assert records[1].text == "Text for page_0002.png"
