"""MCP wrapper for DoD server endpoints."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dod")
BASE_URL = "http://127.0.0.1:8000"


@mcp.tool()
def get_toc(job_ref: str) -> Dict[str, Any]:
    """Return TOC tree for a completed job."""
    with httpx.Client(timeout=120.0) as client:
        response = client.get(f"{BASE_URL}/v1/docs/{job_ref}/toc")
        response.raise_for_status()
        return response.json()


@mcp.tool()
def get_page_texts(
    job_ref: str,
    page_ids: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    max_chars_per_page: Optional[int] = None,
) -> Dict[str, Any]:
    """Return selected page text rows from page_table artifact."""
    params: Dict[str, Any] = {}
    if page_ids is not None:
        params["page_ids"] = page_ids
    if start_page is not None:
        params["start_page"] = start_page
    if end_page is not None:
        params["end_page"] = end_page
    if max_chars_per_page is not None:
        params["max_chars_per_page"] = max_chars_per_page
    with httpx.Client(timeout=120.0) as client:
        response = client.get(f"{BASE_URL}/v1/docs/{job_ref}/pages/text", params=params)
        response.raise_for_status()
        return response.json()


@mcp.tool()
def get_page_images(
    job_ref: str,
    page_ids: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    mode: str = "path",
) -> Dict[str, Any]:
    """Return selected page image rows from image_page_table artifact."""
    params: Dict[str, Any] = {"mode": mode}
    if page_ids is not None:
        params["page_ids"] = page_ids
    if start_page is not None:
        params["start_page"] = start_page
    if end_page is not None:
        params["end_page"] = end_page
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
