"""Tests for pure/helper utilities in pageindex.utils."""

from __future__ import annotations

from typing import Any, Dict, cast

import pytest

from DoD.pageindex import utils


def test_json_helpers_parse_fenced_and_cleanup() -> None:
    """JSON helpers should parse fenced content and normalize common issues."""
    fenced = '```json\n{"a": 1, "b": None,}\n```'
    content = utils.get_json_content(fenced)
    parsed = utils.extract_json(content)
    assert parsed == {"a": 1, "b": None}


def test_write_node_and_tree_flattening_helpers() -> None:
    """Tree helpers should assign ids and flatten consistently."""
    tree: Dict[str, Any] = {"title": "A", "nodes": [{"title": "B", "nodes": []}]}
    utils.write_node_id(tree)
    assert tree["node_id"] == "0000"
    first_node = cast(list[dict[str, Any]], tree["nodes"])[0]
    assert first_node["node_id"] == "0001"

    flattened = utils.get_nodes(tree)
    assert all("nodes" not in item for item in flattened)
    assert len(flattened) == 2

    full_flat = utils.structure_to_list(tree)
    assert len(full_flat) == 2

    leaves = utils.get_leaf_nodes(tree)
    assert len(leaves) == 1
    assert leaves[0]["title"] == "B"
    assert utils.is_leaf_node(tree, "0001") is True
    assert utils.is_leaf_node(tree, "0000") is False


def test_start_index_and_filename_helpers() -> None:
    """Tag parsing and filename sanitation should work for common cases."""
    text = "x <start_index_2> y <start_index_10> z"
    assert utils.get_first_start_page_from_text(text) == 2
    assert utils.get_last_start_page_from_text(text) == 10
    assert utils.sanitize_filename("a/b/c") == "a-b-c"


def test_list_to_tree_and_preface_helpers() -> None:
    """List/tree conversion and preface insertion should be deterministic."""
    items = [
        {"structure": "1", "title": "A", "start_index": 1, "end_index": 2},
        {"structure": "1.1", "title": "B", "start_index": 2, "end_index": 2},
    ]
    tree = utils.list_to_tree(items)
    assert tree[0]["title"] == "A"
    assert tree[0]["nodes"][0]["title"] == "B"

    toc = [{"structure": "1", "title": "Intro", "physical_index": 3}]
    out = utils.add_preface_if_needed(toc)
    assert out[0]["title"] == "Preface"
    assert out[1]["title"] == "Intro"


def test_pdf_page_text_helpers() -> None:
    """Helpers that join page text should preserve ordering and labels."""
    pages = [("A", 1), ("B", 1), ("C", 1)]
    assert utils.get_text_of_pdf_pages(pages, 1, 2) == "AB"
    labeled = utils.get_text_of_pdf_pages_with_labels(pages, 2, 3)
    assert "<physical_index_2>" in labeled
    assert "<physical_index_3>" in labeled


def test_post_processing_and_cleaning_helpers() -> None:
    """Post-processing should compute ranges and clean fields."""
    flat = [
        {"structure": "1", "title": "A", "physical_index": 1, "appear_start": "yes"},
        {"structure": "2", "title": "B", "physical_index": 3, "appear_start": "no"},
    ]
    tree = utils.post_processing(flat, end_physical_index=5)
    assert tree[0]["start_index"] == 1
    assert tree[0]["end_index"] == 3

    cleaned = utils.clean_structure_post(
        [{"title": "A", "page_number": 1, "start_index": 1, "end_index": 2}]
    )
    assert "page_number" not in cleaned[0]
    assert "start_index" not in cleaned[0]
    assert "end_index" not in cleaned[0]

    removed = utils.remove_fields({"x": 1, "text": "abc", "n": {"text": "q"}}, ["text"])
    assert removed == {"x": 1, "n": {}}

    stripped = utils.remove_structure_text([{"title": "A", "text": "x", "nodes": []}])
    assert "text" not in stripped[0]


def test_index_and_page_conversion_helpers() -> None:
    """Converters should normalize physical_index/page fields where possible."""
    items = [
        {"physical_index": "<physical_index_12>"},
        {"physical_index": "physical_index_7"},
    ]
    assert utils.convert_physical_index_to_int(items) == [
        {"physical_index": 12},
        {"physical_index": 7},
    ]
    assert utils.convert_physical_index_to_int("<physical_index_9>") == 9
    assert utils.convert_physical_index_to_int("not_an_index") is None

    pages = [{"page": "3"}, {"page": "x"}]
    out = utils.convert_page_to_int(pages)
    assert out[0]["page"] == 3
    assert out[1]["page"] == "x"


def test_add_node_text_helpers() -> None:
    """Node text population helpers should attach range content."""
    pages = [("A", 1), ("B", 1), ("C", 1)]
    structure = [{"title": "A", "start_index": 1, "end_index": 2, "nodes": []}]
    utils.add_node_text(structure, pages)
    assert structure[0]["text"] == "AB"

    structure2 = [{"title": "A", "start_index": 2, "end_index": 3, "nodes": []}]
    utils.add_node_text_with_labels(structure2, pages)
    text = cast(str, structure2[0]["text"])
    assert "<physical_index_2>" in text


def test_structure_format_and_loader_helpers() -> None:
    """Formatting and config loader helpers should enforce expected behavior."""
    data = {"b": 2, "a": 1}
    assert list(utils.reorder_dict(data, ["a", "b"]).keys()) == ["a", "b"]

    structure = {"title": "A", "nodes": [], "start_index": 1}
    formatted = utils.format_structure(
        structure, order=["title", "start_index", "nodes"]
    )
    assert list(formatted.keys()) == ["title", "start_index"]

    loader = utils.ConfigLoader(default_dict={"x": 1, "y": 2})
    cfg = loader.load({"x": 10})
    assert cfg.x == 10
    assert cfg.y == 2
    with pytest.raises(ValueError, match="Unknown config keys"):
        loader.load({"z": 1})
    with pytest.raises(TypeError):
        loader.load(user_opt=123)  # type: ignore[arg-type]


def test_get_pdf_name_for_path() -> None:
    """String PDF path should return basename."""
    assert utils.get_pdf_name("/tmp/my-doc.pdf") == "my-doc.pdf"


def test_get_last_node() -> None:
    """Helper should return last item from node list."""
    nodes = [{"title": "A"}, {"title": "B"}]
    assert utils.get_last_node(nodes)["title"] == "B"
