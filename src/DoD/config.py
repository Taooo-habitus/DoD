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
class TextExtractorConfig:
    """Settings for text extraction."""

    backend: str = "pymupdf"
    batch: int = 1


@dataclass
class PageTableConfig:
    """Settings for page table generation."""

    include_metadata: bool = True


@dataclass
class TocConfig:
    """Settings for TOC generation."""

    backend: str = "pageindex"
    model: str = "gpt-4o-2024-11-20"
    concurrent_requests: int = 4
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
    text_extractor: TextExtractorConfig = field(default_factory=TextExtractorConfig)
    page_table: PageTableConfig = field(default_factory=PageTableConfig)
    toc: TocConfig = field(default_factory=TocConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
