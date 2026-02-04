"""Tests for document normalization."""

from pathlib import Path

from DoD.normalize.normalize import normalize_to_images


def test_normalize_single_image(tmp_path: Path) -> None:
    """Normalize a single image file."""
    source_path = tmp_path / "sample.jpg"
    source_path.write_bytes(b"fake")

    output_dir = tmp_path / "images"
    images = normalize_to_images(source_path, output_dir)

    assert len(images) == 1
    assert images[0].name.startswith("page_0001")
    assert images[0].exists()
    assert images[0].read_bytes() == source_path.read_bytes()


def test_normalize_directory(tmp_path: Path) -> None:
    """Normalize a directory of images."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.png").write_bytes(b"one")
    (input_dir / "b.png").write_bytes(b"two")

    output_dir = tmp_path / "images"
    images = normalize_to_images(input_dir, output_dir)

    assert len(images) == 2
    assert [img.name for img in images] == ["page_0001.png", "page_0002.png"]
    assert images[0].read_bytes() == (input_dir / "a.png").read_bytes()
    assert images[1].read_bytes() == (input_dir / "b.png").read_bytes()


def test_normalize_unsupported(tmp_path: Path) -> None:
    """Raise on unsupported input types."""
    source_path = tmp_path / "sample.docx"
    source_path.write_text("nope", encoding="utf-8")

    output_dir = tmp_path / "images"
    try:
        normalize_to_images(source_path, output_dir)
    except ValueError as exc:
        assert "Unsupported input type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported input")
