"""Configuration models for the document digestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalizeConfig:
    """Settings for document normalization into images."""

    dpi: int = 200
    image_format: str = "png"
    max_pages: Optional[int] = None


@dataclass
class OcrConfig:
    """Settings for OCR / text extraction."""

    backend: str = "glm_ocr"
    batch_size: int = 1
    language_hint: Optional[str] = None
    device: Optional[str] = None
    glmocr_model: str = "zai-org/GLM-OCR"
    glmocr_prompt: str = "Text Recognition:"
    glmocr_max_new_tokens: int = 4096
    glmocr_api_host: Optional[str] = None
    glmocr_api_port: Optional[int] = None
    glmocr_maas_enabled: Optional[bool] = None
    glmocr_api_key: Optional[str] = None
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "glm-ocr:q8_0"
    ollama_prompt: str = "Recognize the text in the image and output in Markdown."
    ollama_timeout: int = 300
    ollama_api_path: str = "/api/chat"
    ollama_max_long_edge: Optional[int] = 1600
    ollama_concurrent_requests: int = 1


@dataclass
class PageTableConfig:
    """Settings for page table generation."""

    include_metadata: bool = True


@dataclass
class TocConfig:
    """Settings for TOC generation."""

    backend: str = "pageindex"
    model: str = "gpt-4o-2024-11-20"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 10
    max_token_num_each_node: int = 20000
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "yes"
    if_add_doc_description: str = "no"
    if_add_node_text: str = "no"


@dataclass
class ArtifactConfig:
    """Output artifact filenames."""

    output_dir: str = "artifacts"
    page_table_filename: str = "page_table.jsonl"
    image_page_table_filename: str = "image_page_table.jsonl"
    toc_filename: str = "toc_tree.json"
    manifest_filename: str = "manifest.json"


@dataclass
class PipelineConfig:
    """Top-level pipeline config."""

    input_path: str = ""
    run_mode: str = "dev"
    logging_level: str = "INFO"
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    page_table: PageTableConfig = field(default_factory=PageTableConfig)
    toc: TocConfig = field(default_factory=TocConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
