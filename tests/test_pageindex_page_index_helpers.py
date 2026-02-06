"""Tests for helper logic in pageindex.page_index."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

pi = importlib.import_module("DoD.pageindex.page_index")


class _Logger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


def test_token_prefix_helpers() -> None:
    """Prefix and range token helpers should compute inclusive sums."""
    page_list = [("a", 2), ("b", 3), ("c", 5)]
    prefix = pi._build_token_prefix(page_list)
    assert prefix == [0, 2, 5, 10]
    assert pi._range_token_sum(prefix, 1, 1) == 2
    assert pi._range_token_sum(prefix, 2, 3) == 8


def test_ensure_list_and_ensure_opt() -> None:
    """Validation helpers should enforce basic contracts."""
    assert pi._ensure_list([{"a": 1}], "ctx") == [{"a": 1}]
    with pytest.raises(ValueError):
        pi._ensure_list({"a": 1}, "ctx")

    opt = pi._ensure_opt(None)
    assert hasattr(opt, "model")
    assert pi._ensure_opt(SimpleNamespace(x=1)).x == 1


def test_offset_and_pair_helpers() -> None:
    """Page matching and offset helpers should behave predictably."""
    toc_page = [{"title": "A", "page": 10}, {"title": "B", "page": 12}]
    toc_phy = [
        {"title": "A", "physical_index": 15},
        {"title": "B", "physical_index": 17},
    ]
    pairs = pi.extract_matching_page_pairs(toc_page, toc_phy, start_page_index=1)
    assert len(pairs) == 2
    assert pi.calculate_page_offset(pairs) == 5
    assert pi.calculate_page_offset([]) is None


def test_add_page_offset_to_toc_json() -> None:
    """Offset application should write physical_index and remove page."""
    data = [{"title": "A", "page": 3}, {"title": "B", "page": None}]
    out = pi.add_page_offset_to_toc_json(data, offset=2)
    assert out[0]["physical_index"] == 5
    assert "page" not in out[0]


def test_page_grouping_and_cleanup_helpers() -> None:
    """Grouping and cleanup helpers should transform content safely."""
    grouped = pi.page_list_to_group_text(
        page_contents=["A", "B", "C"],
        token_lengths=[2, 2, 2],
        max_tokens=3,
        overlap_page=1,
    )
    assert len(grouped) >= 2
    assert all(isinstance(part, str) for part in grouped)

    text = (
        "<physical_index_1>aaa<physical_index_1><physical_index_2>bbb<physical_index_2>"
    )
    cleaned = pi.remove_first_physical_index_section(text)
    assert "<physical_index_1>" not in cleaned


def test_remove_page_number_recursive() -> None:
    """remove_page_number should strip page_number from nested nodes."""
    data = [
        {"title": "A", "page_number": 1, "nodes": [{"title": "B", "page_number": 2}]}
    ]
    out = pi.remove_page_number(data)
    assert "page_number" not in out[0]
    assert "page_number" not in out[0]["nodes"][0]


def test_validate_and_truncate_physical_indices() -> None:
    """Out-of-range physical indices should be nulled."""
    logger = _Logger()
    items = [{"title": "A", "physical_index": 2}, {"title": "B", "physical_index": 9}]
    out = pi.validate_and_truncate_physical_indices(
        items, page_list_length=5, start_index=1, logger=logger
    )
    assert out[0]["physical_index"] == 2
    assert out[1]["physical_index"] is None
    assert any("Removed physical_index" in str(m) for m in logger.messages)


def test_find_toc_pages_async_with_mocked_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """find_toc_pages should preserve contiguous-TOC stopping logic."""
    page_list = [("p1", 1), ("p2", 1), ("p3", 1), ("p4", 1)]
    opt = SimpleNamespace(model="m", toc_check_page_num=4, concurrent_requests=2)

    async def fake_detect(indices, _page_list, _model, _concurrent_requests):
        table = {0: "yes", 1: "yes", 2: "no", 3: "no"}
        return {i: table[i] for i in indices}

    monkeypatch.setattr(pi, "_detect_toc_for_indices", fake_detect)
    out = asyncio.run(pi.find_toc_pages(0, page_list, opt))
    assert out == [0, 1]
