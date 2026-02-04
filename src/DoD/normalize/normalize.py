"""Document normalization into an image collection."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Iterable, List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_to_images(
    input_path: Path,
    output_dir: Path,
    dpi: int = 200,
    image_format: str = "png",
    max_pages: int | None = None,
) -> List[Path]:
    """Normalize a document into a directory of images.

    Supported inputs:
    - PDF file (requires pdf2image)
    - Image file
    - Directory of images
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir() if p.is_file() and _is_image(p)
        )
        if not image_paths:
            raise ValueError(f"No images found in directory: {input_path}")
        return _copy_images(image_paths, output_dir)

    if _is_image(input_path):
        return _copy_images([input_path], output_dir)

    if input_path.suffix.lower() == ".pdf":
        return _pdf_to_images(
            input_path,
            output_dir,
            dpi=dpi,
            image_format=image_format,
            max_pages=max_pages,
        )

    raise ValueError(f"Unsupported input type for normalization: {input_path}")


def _copy_images(image_paths: Iterable[Path], output_dir: Path) -> List[Path]:
    copied = []
    for idx, path in enumerate(image_paths, start=1):
        target = output_dir / f"page_{idx:04d}{path.suffix.lower()}"
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _pdf_to_images(
    pdf_path: Path, output_dir: Path, dpi: int, image_format: str, max_pages: int | None
) -> List[Path]:
    try:
        pdf2image = importlib.import_module("pdf2image")
        convert_from_path = pdf2image.convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "pdf2image is required for PDF normalization. Install it and ensure "
            "poppler is available on the system."
        ) from exc

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if max_pages is not None:
        pages = pages[:max_pages]

    image_paths: List[Path] = []
    for idx, page in enumerate(pages, start=1):
        target = output_dir / f"page_{idx:04d}.{image_format}"
        page.save(target, image_format.upper())
        image_paths.append(target)
    return image_paths
