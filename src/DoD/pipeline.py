"""Document digestion pipeline."""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from DoD.config import PipelineConfig
from DoD.io.artifacts import write_json
from DoD.normalize.normalize import normalize_to_images
from DoD.ocr.base import TextExtractor
from DoD.ocr.dummy import DummyExtractor
from DoD.ocr.glm_ocr import GlmOcrExtractor
from DoD.ocr.glm_ocr_transformers import GlmOcrTransformersExtractor
from DoD.ocr.ollama_ocr import OllamaOcrExtractor
from DoD.ocr.plain_text import PlainTextExtractor
from DoD.page_table import PageRecord, write_image_page_table, write_page_table
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
    return _write_artifacts(
        cfg, input_path, output_dir, page_records, toc_tree, image_paths
    )


def _resolve_input_path(input_path: str) -> Path:
    path = Path(input_path)
    if not path.is_absolute():
        path = Path(get_original_cwd()) / path
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return path


def _prepare_output_dirs(cfg: PipelineConfig) -> Tuple[Path, Path]:
    output_root: Path
    if HydraConfig.initialized():
        output_root = Path(HydraConfig.get().runtime.output_dir)
    else:
        output_root = Path.cwd()
    output_dir = output_root / cfg.artifacts.output_dir
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
    if backend == "ollama_ocr":
        return OllamaOcrExtractor(
            host=cfg.ocr.ollama_host,
            model=cfg.ocr.ollama_model,
            prompt=cfg.ocr.ollama_prompt,
            timeout=cfg.ocr.ollama_timeout,
            api_path=cfg.ocr.ollama_api_path,
            max_long_edge=cfg.ocr.ollama_max_long_edge,
        )

    raise ValueError(f"Unsupported OCR backend: {cfg.ocr.backend}")


def _generate_toc(
    cfg: PipelineConfig, input_path: Path, page_records: List[PageRecord]
) -> Dict[str, object]:
    if cfg.toc.backend.lower() == "pageindex":
        try:
            raw_config = (
                asdict(cfg.toc)
                if is_dataclass(cfg.toc)
                else OmegaConf.to_container(cfg.toc, resolve=True)
            )
            toc_config = cast(Dict[str, Any], raw_config)
            adapter = PageIndexAdapter(config=toc_config)
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
    image_paths: Optional[List[Path]],
) -> Dict[str, str]:
    page_table_path = output_dir / cfg.artifacts.page_table_filename
    write_page_table(page_table_path, page_records)

    image_page_table_path = output_dir / cfg.artifacts.image_page_table_filename
    if image_paths:
        write_image_page_table(image_page_table_path, image_paths)

    toc_path = output_dir / cfg.artifacts.toc_filename
    write_json(toc_path, toc_tree)

    config_payload = (
        asdict(cfg) if is_dataclass(cfg) else OmegaConf.to_container(cfg, resolve=True)
    )
    manifest = {
        "input_path": str(input_path),
        "page_count": len(page_records),
        "artifacts": {
            "page_table": str(page_table_path),
            "image_page_table": str(image_page_table_path),
            "toc_tree": str(toc_path),
        },
        "config": config_payload,
    }
    manifest_path = output_dir / cfg.artifacts.manifest_filename
    write_json(manifest_path, manifest)

    return {
        "page_table": str(page_table_path),
        "toc_tree": str(toc_path),
        "manifest": str(manifest_path),
    }
