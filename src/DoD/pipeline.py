"""Document digestion pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from DoD.config import PipelineConfig
from DoD.io.artifacts import write_json
from DoD.normalize.normalize import normalize_to_images
from DoD.page_table import PageRecord, write_image_page_table, write_page_table
from DoD.text_extractor.base import TextExtractor
from DoD.text_extractor.dummy import DummyExtractor
from DoD.text_extractor.plain_text import PlainTextExtractor
from DoD.text_extractor.pymupdf import PyMuPDFExtractor
from DoD.text_extractor.pytesseract import PyTesseractExtractor
from DoD.toc.pageindex_adapter import PageIndexAdapter

logger = logging.getLogger(__name__)


def digest_document(cfg: PipelineConfig) -> Dict[str, str]:
    """Run the document digestion pipeline and return artifact paths."""
    profiling: Dict[str, Any] = {}
    profiling_stages: Dict[str, Dict[str, object]] = {}
    profiling["stages"] = profiling_stages
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    input_path = _resolve_input_path(cfg.input_path)
    profiling_stages["resolve_input"] = {
        "seconds": time.perf_counter() - stage_start,
        "skipped": False,
    }

    stage_start = time.perf_counter()
    output_dir, images_dir = _prepare_output_dirs(cfg)
    profiling_stages["prepare_output_dirs"] = {
        "seconds": time.perf_counter() - stage_start,
        "skipped": False,
    }

    stage_start = time.perf_counter()
    extractor = _select_extractor(cfg, input_path)
    profiling_stages["select_extractor"] = {
        "seconds": time.perf_counter() - stage_start,
        "skipped": False,
    }

    logger.info(
        "Pipeline start: input=%s backend=%s batch=%s",
        input_path,
        cfg.text_extractor.backend,
        cfg.text_extractor.batch,
    )

    stage_start = time.perf_counter()
    image_paths = _normalize_if_needed(cfg, input_path, images_dir, extractor)
    normalize_duration = time.perf_counter() - stage_start
    profiling_stages["normalize"] = {
        "seconds": normalize_duration,
        "skipped": image_paths is None,
    }
    if image_paths is None:
        logger.info("Normalize skipped (extractor does not require images).")
    else:
        logger.info(
            "Normalize completed: %s pages in %.2fs",
            len(image_paths),
            normalize_duration,
        )

    stage_start = time.perf_counter()
    page_records = extractor.extract(input_path, image_paths=image_paths)
    extract_duration = time.perf_counter() - stage_start
    profiling_stages["extract_text"] = {"seconds": extract_duration, "skipped": False}
    logger.info("Extracted %s pages in %.2fs.", len(page_records), extract_duration)

    stage_start = time.perf_counter()
    toc_tree = _generate_toc(cfg, input_path, page_records)
    toc_duration = time.perf_counter() - stage_start
    profiling_stages["generate_toc"] = {"seconds": toc_duration, "skipped": False}
    logger.info("TOC generated in %.2fs.", toc_duration)

    stage_start = time.perf_counter()
    artifacts = _write_artifacts(
        cfg,
        input_path,
        output_dir,
        page_records,
        toc_tree,
        image_paths,
        profiling=profiling,
    )
    write_duration = time.perf_counter() - stage_start
    profiling_stages["write_artifacts"] = {"seconds": write_duration, "skipped": False}
    profiling["total_seconds"] = time.perf_counter() - total_start
    profiling["page_count"] = len(page_records)
    logger.info(
        "Artifacts written in %.2fs (total %.2fs).",
        write_duration,
        profiling["total_seconds"],
    )
    _log_profiling_summary(profiling)

    return artifacts


def _log_profiling_summary(profiling: Dict[str, Any]) -> None:
    """Emit a single-line profiling summary at pipeline completion."""
    stages = profiling.get("stages", {})
    stage_parts: List[str] = []
    for name, payload in stages.items():
        if not isinstance(payload, dict):
            continue
        seconds = payload.get("seconds")
        skipped = payload.get("skipped")
        if skipped:
            stage_parts.append(f"{name}=skipped")
        elif isinstance(seconds, (int, float)):
            stage_parts.append(f"{name}={seconds:.2f}s")
    total_seconds = profiling.get("total_seconds")
    page_count = profiling.get("page_count")
    suffix = ""
    if isinstance(total_seconds, (int, float)):
        suffix = f" total={total_seconds:.2f}s"
    if isinstance(page_count, int):
        suffix = f"{suffix} pages={page_count}"
    logger.info("Pipeline finished: %s%s", ", ".join(stage_parts), suffix)


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
    backend = cfg.text_extractor.backend.lower()
    suffix = input_path.suffix.lower()

    if suffix == ".txt":
        return PlainTextExtractor()

    if backend == "dummy":
        return DummyExtractor()

    if backend in {"pymupdf", "pymupdf4llm"}:
        return PyMuPDFExtractor(batch=cfg.text_extractor.batch)

    if backend == "pytesseract":
        return PyTesseractExtractor(batch=cfg.text_extractor.batch)

    removed_backends = {"glm_ocr", "glm_ocr_transformers", "ollama_ocr"}
    if backend in removed_backends:
        raise ValueError(
            "Text extractor backend "
            f"'{cfg.text_extractor.backend}' has been removed. Use 'pymupdf'."
        )

    raise ValueError(
        f"Unsupported text extractor backend: {cfg.text_extractor.backend}"
    )


def _generate_toc(
    cfg: PipelineConfig, input_path: Path, page_records: List[PageRecord]
) -> Dict[str, object]:
    if cfg.toc.backend.lower() != "pageindex":
        raise ValueError(f"Unsupported TOC backend: {cfg.toc.backend}")

    raw_config = (
        asdict(cfg.toc)
        if is_dataclass(cfg.toc)
        else OmegaConf.to_container(cfg.toc, resolve=True)
    )
    toc_config = cast(Dict[str, Any], raw_config)
    adapter = PageIndexAdapter(config=toc_config)

    try:
        return adapter.generate(input_path, page_records=page_records)
    except Exception as exc:  # noqa: BLE001 - strict mode: fail the whole run
        raise RuntimeError(f"PageIndex TOC generation failed: {exc}") from exc


def _write_artifacts(
    cfg: PipelineConfig,
    input_path: Path,
    output_dir: Path,
    page_records: List[PageRecord],
    toc_tree: Dict[str, object],
    image_paths: Optional[List[Path]],
    profiling: Optional[Dict[str, object]] = None,
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
    manifest: Dict[str, object] = {
        "input_path": str(input_path),
        "page_count": len(page_records),
        "artifacts": {
            "page_table": str(page_table_path),
            "image_page_table": str(image_page_table_path),
            "toc_tree": str(toc_path),
        },
        "config": config_payload,
    }
    if profiling is not None:
        manifest["profiling"] = profiling
    manifest_path = output_dir / cfg.artifacts.manifest_filename
    write_json(manifest_path, manifest)

    return {
        "page_table": str(page_table_path),
        "toc_tree": str(toc_path),
        "manifest": str(manifest_path),
    }
