"""MCP wrapper for DoD server endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf

from DoD.config import PipelineConfig

mcp = FastMCP("dod")
BASE_URL = "http://127.0.0.1:8000"


def _load_retrieval_defaults() -> tuple[Optional[int], int]:
    """Load MCP retrieval guardrails from conf/config.yaml."""
    config_path = Path(__file__).resolve().parents[2] / "conf" / "config.yaml"
    cfg = PipelineConfig()
    if config_path.exists():
        raw_cfg = OmegaConf.load(str(config_path))
        data = OmegaConf.to_container(raw_cfg, resolve=False)
        if isinstance(data, dict):
            data.pop("hydra", None)
            merged = OmegaConf.merge(OmegaConf.structured(PipelineConfig()), data)
            cfg_obj = OmegaConf.to_object(merged)
            if isinstance(cfg_obj, PipelineConfig):
                cfg = cfg_obj
    return cfg.retrieval.max_chars_per_page, cfg.retrieval.max_pages_per_call


def _normalize_pages_spec(pages: Optional[str]) -> Optional[str]:
    """Normalize flexible page spec (e.g. '1,3-5') to CSV page ids."""
    if pages is None or not pages.strip():
        return None
    parsed_pages: set[int] = set()
    for token in pages.split(","):
        item = token.strip()
        if not item:
            continue
        if "-" in item:
            parts = [part.strip() for part in item.split("-", maxsplit=1)]
            if len(parts) != 2:
                raise ValueError(f"Invalid page range: {item}")
            start = int(parts[0])
            end = int(parts[1])
            if start <= 0 or end <= 0 or start > end:
                raise ValueError(f"Invalid page range: {item}")
            parsed_pages.update(range(start, end + 1))
        else:
            page = int(item)
            if page <= 0:
                raise ValueError(f"Page ids must be positive integers: {item}")
            parsed_pages.add(page)
    if not parsed_pages:
        return None
    return ",".join(str(page) for page in sorted(parsed_pages))


DEFAULT_MAX_CHARS_PER_PAGE, DEFAULT_MAX_PAGES_PER_CALL = _load_retrieval_defaults()


@mcp.tool()
def get_toc(job_ref: str) -> Dict[str, Any]:
    """Return TOC tree for a completed job."""
    with httpx.Client(timeout=120.0) as client:
        response = client.get(f"{BASE_URL}/v1/docs/{job_ref}/toc")
        response.raise_for_status()
        return response.json()


@mcp.tool()
def list_jobs() -> Dict[str, Any]:
    """Return submitted jobs metadata for job resolution."""
    with httpx.Client(timeout=120.0) as client:
        response = client.get(f"{BASE_URL}/v1/jobs")
        response.raise_for_status()
        return response.json()


@mcp.tool()
def get_page_texts(job_ref: str, pages: Optional[str] = None) -> Dict[str, Any]:
    """Return selected page text rows from page_table artifact."""
    params: Dict[str, Any] = {}
    normalized_pages = _normalize_pages_spec(pages)
    if normalized_pages is not None:
        params["page_ids"] = normalized_pages
    if DEFAULT_MAX_CHARS_PER_PAGE is not None:
        params["max_chars_per_page"] = DEFAULT_MAX_CHARS_PER_PAGE
    params["max_pages_per_call"] = DEFAULT_MAX_PAGES_PER_CALL
    with httpx.Client(timeout=120.0) as client:
        response = client.get(f"{BASE_URL}/v1/docs/{job_ref}/pages/text", params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def get_page_images(
    job_ref: str, pages: Optional[str] = None, mode: str = "path"
) -> Dict[str, Any]:
    """Return selected page image rows from image_page_table artifact."""
    params: Dict[str, Any] = {"mode": mode}
    normalized_pages = _normalize_pages_spec(pages)
    if normalized_pages is not None:
        params["page_ids"] = normalized_pages
    params["max_pages_per_call"] = DEFAULT_MAX_PAGES_PER_CALL
    with httpx.Client(timeout=120.0) as client:
        response = client.get(
            f"{BASE_URL}/v1/docs/{job_ref}/pages/images", params=params
        )
        response.raise_for_status()
        return response.json()


def main() -> None:
    """Run the DoD MCP wrapper server."""
    mcp.run()


if __name__ == "__main__":
    main()
