"""GLM-OCR backend wrapper (SDK)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, cast

from DoD.ocr.base import TextExtractor
from DoD.page_table import PageRecord


class GlmOcrExtractor(TextExtractor):
    """Extract text using GLM-OCR from images."""

    requires_images = True

    def __init__(
        self,
        batch_size: int = 1,
        device: Optional[str] = None,
        api_host: Optional[str] = None,
        api_port: Optional[int] = None,
        maas_enabled: Optional[bool] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the GLM-OCR extractor."""
        self.batch_size = batch_size
        self.device = device
        self.api_host = api_host
        self.api_port = api_port
        self.maas_enabled = maas_enabled
        self.api_key = api_key
        self._parser: Optional[_GlmOcrParser] = None
        self._mode = "sdk"

    def _load_pipeline(self) -> None:
        if self._parser is not None or self._mode == "http":
            return
        GlmOcr = _import_glmocr()
        if GlmOcr is None:
            self._mode = "http"
            return

        kwargs = {}
        if self.api_host is not None:
            kwargs["api_host"] = self.api_host
        if self.api_port is not None:
            kwargs["api_port"] = self.api_port
        if self.maas_enabled is not None:
            kwargs["maas_enabled"] = self.maas_enabled
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key

        try:
            self._parser = cast(_GlmOcrParser, GlmOcr(**kwargs))
        except TypeError:
            # Fallback if SDK doesn't accept kwargs
            self._parser = cast(_GlmOcrParser, GlmOcr())

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract text for each image using GLM-OCR."""
        if not image_paths:
            raise ValueError("GLM-OCR requires image paths for extraction.")

        self._load_pipeline()
        if self._mode == "http":
            return _extract_with_http_batch(
                image_paths,
                api_host=self.api_host,
                api_port=self.api_port,
                api_key=self.api_key,
            )
        if self._parser is None:
            raise RuntimeError("GLM-OCR parser failed to initialize.")
        records: List[PageRecord] = []

        for idx, image_path in enumerate(image_paths, start=1):
            text = _extract_with_sdk(self._parser, image_path)
            records.append(
                PageRecord(
                    page_id=idx,
                    text=text,
                    image_path=str(image_path),
                    metadata={"backend": "glm_ocr"},
                )
            )

        return records


class _GlmOcrParser(Protocol):
    """Minimal protocol for GLM-OCR SDK parser objects."""

    def parse(self, path: str) -> Any: ...


def _extract_with_sdk(parser: _GlmOcrParser, image_path: Path) -> str:
    """Extract text using the GLM-OCR SDK parser."""
    # SDK supports parser.parse() and top-level parse().
    try:
        result = parser.parse(str(image_path))
    except AttributeError:
        try:
            from glmocr import parse  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "glmocr SDK is required for GLM-OCR. Install from "
                "https://github.com/zai-org/GLM-OCR."
            ) from exc
        parse_fn: Callable[[str], Any] = parse
        result = parse_fn(str(image_path))

    if hasattr(result, "markdown_result"):
        payload = getattr(result, "markdown_result")
        if payload:
            return str(payload).strip()
    if hasattr(result, "json_result"):
        payload = getattr(result, "json_result")
        if isinstance(payload, dict) and "text" in payload:
            return str(payload["text"]).strip()
        return str(payload).strip()

    return str(result).strip()


def _extract_with_http_batch(
    image_paths: List[Path],
    api_host: Optional[str],
    api_port: Optional[int],
    api_key: Optional[str],
) -> List[PageRecord]:
    """Extract text by calling a GLM-OCR server at /glmocr/parse."""
    if not api_host or not api_port:
        raise RuntimeError(
            "GLM-OCR HTTP mode requires ocr.glmocr_api_host and ocr.glmocr_api_port."
        )
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for GLM-OCR HTTP mode.") from exc

    url = f"http://{api_host}:{api_port}/glmocr/parse"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    records: List[PageRecord] = []
    for idx, image_path in enumerate(image_paths, start=1):
        payload = {"images": [str(image_path)]}
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        text = ""
        if isinstance(data, dict):
            if "markdown_result" in data and data["markdown_result"]:
                text = str(data["markdown_result"]).strip()
            elif "json_result" in data:
                text = str(data["json_result"]).strip()
        records.append(
            PageRecord(
                page_id=idx,
                text=text,
                image_path=str(image_path),
                metadata={"backend": "glm_ocr_http"},
            )
        )
    return records


def _try_add_glmocr_source_to_path() -> None:
    """Try to prepend a GLM-OCR source checkout to sys.path."""
    home = Path.home()
    candidates = []
    cache_root = home / ".cache" / "uv" / "git-v0" / "checkouts"
    if cache_root.exists():
        for path in cache_root.rglob("glmocr/parser_result"):
            candidates.append(path.parent)
        for path in cache_root.rglob("glmocr/pipeline"):
            candidates.append(path.parent)
    for repo_root in candidates:
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            break


def _import_glmocr():
    """Return GlmOcr class if available, otherwise None."""
    try:
        from glmocr.api import GlmOcr  # type: ignore[import-not-found]

        return GlmOcr
    except Exception:
        _try_add_glmocr_source_to_path()
        try:
            from glmocr.api import GlmOcr  # type: ignore[import-not-found]

            return GlmOcr
        except Exception:
            return None
