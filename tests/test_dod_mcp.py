"""Tests for MCP page spec normalization."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp")

from scripts.dod_mcp import _normalize_pages_spec


def test_normalize_pages_spec_mixed_ranges() -> None:
    """Mixed ranges and page ids should expand and sort."""
    assert _normalize_pages_spec("110, 111, 89-91") == "89,90,91,110,111"


def test_normalize_pages_spec_deduplicates() -> None:
    """Overlapping inputs should deduplicate page ids."""
    assert _normalize_pages_spec("1-3,2,3,5") == "1,2,3,5"


def test_normalize_pages_spec_invalid_range() -> None:
    """Descending ranges should be rejected."""
    with pytest.raises(ValueError):
        _normalize_pages_spec("10-5")
