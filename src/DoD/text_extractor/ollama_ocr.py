"""Ollama OCR backend using the local Ollama HTTP API."""

from __future__ import annotations

import base64
import concurrent.futures
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from DoD.page_table import PageRecord
from DoD.text_extractor.base import TextExtractor


class OllamaOcrExtractor(TextExtractor):
    """Extract text using an Ollama-hosted vision model."""

    requires_images = True

    def __init__(
        self,
        host: str,
        model: str,
        prompt: str,
        timeout: int = 300,
        api_path: str = "/api/chat",
        max_long_edge: Optional[int] = 1600,
        concurrent_requests: int = 1,
    ) -> None:
        """Initialize the Ollama OCR extractor."""
        self.host = host.rstrip("/")
        self.model = model
        self.prompt = prompt
        self.timeout = timeout
        self.api_path = api_path
        self.max_long_edge = max_long_edge
        self.concurrent_requests = max(1, int(concurrent_requests))

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract text for each image using Ollama."""
        if not image_paths:
            raise ValueError("Ollama OCR requires image paths for extraction.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for Ollama OCR.") from exc

        total = len(image_paths)
        if self.concurrent_requests <= 1 or total <= 1:
            records: List[PageRecord] = []
            for idx, image_path in self._progress(
                enumerate(image_paths, start=1), total=total, desc="OCR (Ollama)"
            ):
                records.append(self._extract_one(requests, idx, image_path))
            return records

        records_by_index: List[Optional[PageRecord]] = [None] * total
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.concurrent_requests, total)
        ) as executor:
            futures = [
                executor.submit(self._extract_one, requests, idx, image_path)
                for idx, image_path in enumerate(image_paths, start=1)
            ]
            for future in self._progress(
                concurrent.futures.as_completed(futures),
                total=total,
                desc="OCR (Ollama)",
            ):
                record = future.result()
                records_by_index[record.page_id - 1] = record

        return [record for record in records_by_index if record is not None]

    def _extract_one(self, requests_module, idx: int, image_path: Path) -> PageRecord:
        url = f"{self.host}{self.api_path}"
        image_b64 = _encode_image(image_path, max_long_edge=self.max_long_edge)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.prompt, "images": [image_b64]}
            ],
            "stream": False,
        }
        response = requests_module.post(url, json=payload, timeout=self.timeout)
        if response.status_code == 404 and self.api_path == "/api/chat":
            url = f"{self.host}/api/generate"
            payload = {
                "model": self.model,
                "prompt": self.prompt,
                "images": [image_b64],
                "stream": False,
            }
            response = requests_module.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(
                f"Ollama OCR request failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        text = ""
        if isinstance(data, dict):
            message = data.get("message") or {}
            if isinstance(message, dict):
                text = str(message.get("content", "")).strip()
            elif "response" in data:
                text = str(data.get("response", "")).strip()
        return PageRecord(
            page_id=idx,
            text=text,
            image_path=str(image_path),
            metadata={"backend": "ollama_ocr", "model": self.model},
        )


def _encode_image(path: Path, max_long_edge: Optional[int] = None) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for Ollama OCR.") from exc

    image = Image.open(path).convert("RGB")
    if max_long_edge:
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size)

    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
