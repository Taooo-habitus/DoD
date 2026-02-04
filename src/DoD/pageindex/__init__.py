"""Refactored PageIndex package."""

from DoD.pageindex.page_index import (
    page_index,
    page_index_from_page_list,
    page_index_main,
)
from DoD.pageindex.page_index_md import md_to_tree

__all__ = ["page_index", "page_index_main", "page_index_from_page_list", "md_to_tree"]
