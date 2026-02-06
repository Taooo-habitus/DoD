"""Pipeline tests using the plain text extractor."""

from pathlib import Path

import pytest

import DoD.pipeline as pipeline
from DoD.config import PipelineConfig
from DoD.pipeline import digest_document


def test_plain_text_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the pipeline using the plain text extractor."""
    text_path = tmp_path / "sample.txt"
    text_path.write_text("Hello world", encoding="utf-8")

    class StubAdapter:
        def __init__(self, config):
            self.config = config

        def generate(self, *_args, **_kwargs):
            return {
                "doc_name": "sample",
                "structure": [
                    {"title": "Document", "start_index": 1, "end_index": 1, "nodes": []}
                ],
            }

    monkeypatch.setattr(pipeline, "PageIndexAdapter", StubAdapter)

    cfg = PipelineConfig(input_path=str(text_path))
    cfg.text_extractor.backend = "plain_text"
    cfg.toc.backend = "pageindex"
    cfg.artifacts.output_dir = str(tmp_path / "artifacts")

    outputs = digest_document(cfg)

    assert Path(outputs["page_table"]).exists()
    assert Path(outputs["toc_tree"]).exists()
    assert Path(outputs["manifest"]).exists()
