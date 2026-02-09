"""Configuration for PageIndex indexing behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PageIndexConfig:
    """Configuration values for PageIndex."""

    model: str = "gpt-4o-2024-11-20"
    concurrent_requests: int = 4
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 10
    max_token_num_each_node: int = 20000
    max_fix_attempts: int = 2
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "yes"
    if_add_doc_description: str = "no"
    if_add_node_text: str = "no"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict representation of the config."""
        return {
            "model": self.model,
            "concurrent_requests": self.concurrent_requests,
            "toc_check_page_num": self.toc_check_page_num,
            "max_page_num_each_node": self.max_page_num_each_node,
            "max_token_num_each_node": self.max_token_num_each_node,
            "max_fix_attempts": self.max_fix_attempts,
            "if_add_node_id": self.if_add_node_id,
            "if_add_node_summary": self.if_add_node_summary,
            "if_add_doc_description": self.if_add_doc_description,
            "if_add_node_text": self.if_add_node_text,
            "api_key": self.api_key,
            "api_base_url": self.api_base_url,
        }


DEFAULT_CONFIG = PageIndexConfig().to_dict()
