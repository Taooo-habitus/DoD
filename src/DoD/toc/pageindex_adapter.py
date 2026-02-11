"""Adapter for PageIndex-based TOC generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from DoD.page_table import PageRecord


class PageIndexAdapter:
    """Generate TOC tree via PageIndex."""

    def __init__(self, config: Dict[str, Any]):
        """Create the adapter with resolved config."""
        self.config = config

    def generate(
        self, input_path: Path, page_records: Optional[List[PageRecord]] = None
    ) -> Dict[str, Any]:
        """Generate a TOC tree for the given input."""
        if page_records:
            return self._generate_from_page_records(input_path, page_records)
        return self._generate_from_pdf(input_path)

    def _generate_from_pdf(self, input_path: Path) -> Dict[str, Any]:
        try:
            from DoD.pageindex import page_index
        except ImportError as exc:
            raise RuntimeError(
                "DoD.pageindex is required for TOC generation. Ensure PageIndex "
                "code is available in src/DoD/pageindex."
            ) from exc

        return page_index(
            doc=str(input_path),
            model=self.config["model"],
            concurrent_requests=self.config.get("concurrent_requests", 4),
            toc_check_page_num=self.config["toc_check_page_num"],
            max_page_num_each_node=self.config["max_page_num_each_node"],
            max_token_num_each_node=self.config["max_token_num_each_node"],
            max_fix_attempts=self.config.get("max_fix_attempts", 2),
            if_add_node_id=self.config["if_add_node_id"],
            if_add_node_summary=self.config["if_add_node_summary"],
            if_add_doc_description=self.config["if_add_doc_description"],
            if_add_node_text=self.config["if_add_node_text"],
            api_key=self.config.get("api_key"),
            api_base_url=self.config.get("api_base_url"),
        )

    def _generate_from_page_records(
        self, input_path: Path, page_records: List[PageRecord]
    ) -> Dict[str, Any]:
        try:
            from DoD.pageindex import page_index_from_page_list
            from DoD.pageindex.utils import count_tokens
        except ImportError as exc:
            raise RuntimeError(
                "DoD.pageindex is required for PageIndex TOC generation."
            ) from exc

        page_list = [
            (record.text, count_tokens(record.text, model=self.config["model"]))
            for record in page_records
        ]
        return page_index_from_page_list(
            page_list,
            doc_name=input_path.stem,
            model=self.config["model"],
            concurrent_requests=self.config.get("concurrent_requests", 4),
            toc_check_page_num=self.config["toc_check_page_num"],
            max_page_num_each_node=self.config["max_page_num_each_node"],
            max_token_num_each_node=self.config["max_token_num_each_node"],
            max_fix_attempts=self.config.get("max_fix_attempts", 2),
            if_add_node_id=self.config["if_add_node_id"],
            if_add_node_summary=self.config["if_add_node_summary"],
            if_add_doc_description=self.config["if_add_doc_description"],
            if_add_node_text=self.config["if_add_node_text"],
            api_key=self.config.get("api_key"),
            api_base_url=self.config.get("api_base_url"),
        )


def fallback_toc(total_pages: int) -> Dict[str, Any]:
    """Fallback TOC when PageIndex is unavailable."""
    return {
        "doc_name": "unknown",
        "structure": [
            {
                "title": "Document",
                "start_index": 1,
                "end_index": total_pages,
                "nodes": [],
            }
        ],
    }
