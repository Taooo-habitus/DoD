"""Document digestion pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra.utils import get_original_cwd

from DoD.config import PipelineConfig
from DoD.io.artifacts import write_json
from DoD.normalize.normalize import normalize_to_images
from DoD.ocr.base import TextExtractor
from DoD.ocr.dummy import DummyExtractor
from DoD.ocr.glm_ocr import GlmOcrExtractor
from DoD.ocr.glm_ocr_transformers import GlmOcrTransformersExtractor
from DoD.ocr.plain_text import PlainTextExtractor
from DoD.page_table import PageRecord, write_page_table
from DoD.toc.pageindex_adapter import PageIndexAdapter, fallback_toc

logger = logging.getLogger(__name__)


def digest_document(cfg: PipelineConfig) -> Dict[str, str]:
    """Run the document digestion pipeline and return artifact paths."""
    input_path = _resolve_input_path(cfg.input_path)
    output_dir, images_dir = _prepare_output_dirs(cfg)
    extractor = _select_extractor(cfg, input_path)
    image_paths = _normalize_if_needed(cfg, input_path, images_dir, extractor)
    page_records = extractor.extract(input_path, image_paths=image_paths)
    logger.info("Extracted %s pages.", len(page_records))

    toc_tree = _generate_toc(cfg, input_path, page_records)
    return _write_artifacts(cfg, input_path, output_dir, page_records, toc_tree)


def _resolve_input_path(input_path: str) -> Path:
    path = Path(input_path)
    if not path.is_absolute():
        path = Path(get_original_cwd()) / path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return path


def _prepare_output_dirs(cfg: PipelineConfig) -> Tuple[Path, Path]:
    output_dir = Path.cwd() / cfg.artifacts.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, images_dir


def _normalize_if_needed(
    cfg: PipelineConfig, input_path: Path, images_dir: Path, extractor: TextExtractor
) -> Optional[List[Path]]:
    if not getattr(extractor, "requires_images", False):
        return None
    return normalize_to_images(
        input_path,
        images_dir,
        dpi=cfg.normalize.dpi,
        image_format=cfg.normalize.image_format,
        max_pages=cfg.normalize.max_pages,
    )


def _select_extractor(cfg: PipelineConfig, input_path: Path):
    backend = cfg.ocr.backend.lower()

    if backend == "plain_text" or input_path.suffix.lower() in {".md", ".txt"}:
        return PlainTextExtractor()

    if backend == "dummy":
        return DummyExtractor()

    if backend == "glm_ocr":
        return GlmOcrExtractor(
            batch_size=cfg.ocr.batch_size,
            device=cfg.ocr.device,
            api_host=cfg.ocr.glmocr_api_host,
            api_port=cfg.ocr.glmocr_api_port,
            maas_enabled=cfg.ocr.glmocr_maas_enabled,
            api_key=cfg.ocr.glmocr_api_key,
        )
    if backend == "glm_ocr_transformers":
        return GlmOcrTransformersExtractor(
            model_name=cfg.ocr.glmocr_model,
            prompt=cfg.ocr.glmocr_prompt,
            device=cfg.ocr.device,
            max_new_tokens=cfg.ocr.glmocr_max_new_tokens,
        )

    raise ValueError(f"Unsupported OCR backend: {cfg.ocr.backend}")


def _generate_toc(
    cfg: PipelineConfig, input_path: Path, page_records: List[PageRecord]
) -> Dict[str, object]:
    if cfg.toc.backend.lower() == "pageindex":
        try:
            adapter = PageIndexAdapter(config=asdict(cfg.toc))
            return adapter.generate(input_path, page_records=page_records)
        except RuntimeError as exc:
            logger.warning("PageIndex unavailable, using fallback TOC. %s", exc)

    return fallback_toc(total_pages=len(page_records))


def _write_artifacts(
    cfg: PipelineConfig,
    input_path: Path,
    output_dir: Path,
    page_records: List[PageRecord],
    toc_tree: Dict[str, object],
) -> Dict[str, str]:
    page_table_path = output_dir / cfg.artifacts.page_table_filename
    write_page_table(page_table_path, page_records)

    toc_path = output_dir / cfg.artifacts.toc_filename
    write_json(toc_path, toc_tree)

    manifest = {
        "input_path": str(input_path),
        "page_count": len(page_records),
        "artifacts": {"page_table": str(page_table_path), "toc_tree": str(toc_path)},
        "config": asdict(cfg),
    }
    manifest_path = output_dir / cfg.artifacts.manifest_filename
    write_json(manifest_path, manifest)

    return {
        "page_table": str(page_table_path),
        "toc_tree": str(toc_path),
        "manifest": str(manifest_path),
    }
