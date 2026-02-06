"""Unit tests for pipeline helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import DoD.pipeline as pipeline
from DoD.config import PipelineConfig
from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


def test_resolve_input_path_relative_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative paths should resolve via original cwd; missing should raise."""
    file_path = tmp_path / "doc.txt"
    file_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(pipeline, "get_original_cwd", lambda: str(tmp_path))
    assert pipeline._resolve_input_path("doc.txt") == file_path

    with pytest.raises(FileNotFoundError):
        pipeline._resolve_input_path("missing.txt")


def test_prepare_output_dirs_without_hydra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When Hydra is not initialized, cwd should be used."""
    cfg = PipelineConfig(input_path="x")
    cfg.artifacts.output_dir = "artifacts_test"

    monkeypatch.setattr(
        pipeline.HydraConfig, "initialized", staticmethod(lambda: False)
    )
    monkeypatch.chdir(tmp_path)

    output_dir, images_dir = pipeline._prepare_output_dirs(cfg)
    assert output_dir == tmp_path / "artifacts_test"
    assert images_dir == output_dir / "images"
    assert output_dir.exists()
    assert images_dir.exists()


def test_prepare_output_dirs_with_hydra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When Hydra is initialized, runtime output_dir should be used."""
    cfg = PipelineConfig(input_path="x")
    cfg.artifacts.output_dir = "artifacts_test"

    monkeypatch.setattr(pipeline.HydraConfig, "initialized", staticmethod(lambda: True))
    monkeypatch.setattr(
        pipeline.HydraConfig,
        "get",
        staticmethod(
            lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))
        ),
    )

    output_dir, images_dir = pipeline._prepare_output_dirs(cfg)
    assert output_dir == tmp_path / "artifacts_test"
    assert images_dir == output_dir / "images"


def test_normalize_if_needed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Normalization should only run when extractor requires images."""
    cfg = PipelineConfig(input_path="x")
    cfg.normalize.max_pages = 2
    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF")
    images_dir = tmp_path / "images"

    calls = {}

    def fake_normalize(path, out_dir, dpi, image_format, max_pages):
        calls["args"] = (path, out_dir, dpi, image_format, max_pages)
        return [out_dir / "page_0001.png"]

    monkeypatch.setattr(pipeline, "normalize_to_images", fake_normalize)

    class RequiresImages:
        requires_images = True

    class NoImages:
        requires_images = False

    result = pipeline._normalize_if_needed(
        cfg, input_path, images_dir, cast(TextExtractor, RequiresImages())
    )
    assert result == [images_dir / "page_0001.png"]
    assert calls["args"][-1] == 2

    assert (
        pipeline._normalize_if_needed(
            cfg, input_path, images_dir, cast(TextExtractor, NoImages())
        )
        is None
    )


def test_generate_toc_raises_on_adapter_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TOC generation should fail in strict mode if PageIndex adapter fails."""
    cfg = PipelineConfig(input_path=str(tmp_path / "doc.pdf"))
    cfg.toc.backend = "pageindex"
    page_records = [PageRecord(page_id=1, text="hello")]

    class FailingAdapter:
        def __init__(self, config):
            self.config = config

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "PageIndexAdapter", FailingAdapter)
    with pytest.raises(RuntimeError, match="PageIndex TOC generation failed"):
        pipeline._generate_toc(cfg, tmp_path / "doc.pdf", page_records)


def test_generate_toc_raises_on_unsupported_backend(tmp_path: Path) -> None:
    """Unsupported TOC backends should fail explicitly."""
    cfg = PipelineConfig(input_path=str(tmp_path / "doc.pdf"))
    cfg.toc.backend = "unknown"
    page_records = [PageRecord(page_id=1, text="hello")]

    with pytest.raises(ValueError, match="Unsupported TOC backend"):
        pipeline._generate_toc(cfg, tmp_path / "doc.pdf", page_records)


def test_write_artifacts_writes_manifest_and_tables(tmp_path: Path) -> None:
    """Artifact writer should emit page/toc/manifest and optional image table."""
    cfg = PipelineConfig(input_path="x")
    output_dir = tmp_path / "artifacts"
    input_path = tmp_path / "doc.pdf"
    input_path.write_bytes(b"%PDF")
    page_records = [PageRecord(page_id=1, text="hello")]
    toc_tree = {"doc_name": "doc", "structure": []}
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"img")

    out = pipeline._write_artifacts(
        cfg, input_path, output_dir, page_records, toc_tree, [image_path]
    )

    assert Path(out["page_table"]).exists()
    assert Path(out["toc_tree"]).exists()
    assert Path(out["manifest"]).exists()
    manifest = json.loads(Path(out["manifest"]).read_text(encoding="utf-8"))
    assert "image_page_table" in manifest["artifacts"]
