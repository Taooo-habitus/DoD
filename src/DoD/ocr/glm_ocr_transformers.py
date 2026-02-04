"""GLM-OCR backend using Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from DoD.ocr.base import TextExtractor
from DoD.page_table import PageRecord


class GlmOcrTransformersExtractor(TextExtractor):
    """Extract text using GLM-OCR via Transformers."""

    requires_images = True

    def __init__(
        self,
        model_name: str,
        prompt: str,
        device: Optional[str] = None,
        max_new_tokens: int = 4096,
    ) -> None:
        """Initialize the Transformers-backed GLM-OCR extractor."""
        self.model_name = model_name
        self.prompt = prompt
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for GLM-OCR Transformers backend. "
                "Install from https://github.com/huggingface/transformers."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        device_map = self.device or "auto"
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )

    def extract(
        self, input_path: Path, image_paths: Optional[List[Path]] = None
    ) -> List[PageRecord]:
        """Extract text for each image using Transformers."""
        if not image_paths:
            raise ValueError("GLM-OCR requires image paths for extraction.")

        self._load()
        assert self._model is not None
        assert self._processor is not None

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for GLM-OCR Transformers backend."
            ) from exc

        records: List[PageRecord] = []
        for idx, image_path in self._progress(
            enumerate(image_paths, start=1),
            total=len(image_paths),
            desc="OCR (Transformers)",
        ):
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(text=[text], images=[image], return_tensors="pt")
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self._model.device)
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
            response_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            response = self._processor.batch_decode(
                response_ids, skip_special_tokens=True
            )[0].strip()
            records.append(
                PageRecord(
                    page_id=idx,
                    text=response,
                    image_path=str(image_path),
                    metadata={"backend": "glm_ocr_transformers"},
                )
            )

        return records
