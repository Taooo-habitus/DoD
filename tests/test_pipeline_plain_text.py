"""Pipeline tests using the plain text extractor."""

from pathlib import Path

from DoD.config import PipelineConfig
from DoD.pipeline import digest_document


def test_plain_text_pipeline(tmp_path: Path) -> None:
    """Run the pipeline using the plain text extractor."""
    text_path = tmp_path / "sample.txt"
    text_path.write_text("Hello world", encoding="utf-8")

    cfg = PipelineConfig(input_path=str(text_path))
    cfg.text_extractor.backend = "plain_text"
    cfg.toc.backend = "fallback"
    cfg.artifacts.output_dir = str(tmp_path / "artifacts")

    outputs = digest_document(cfg)

    assert Path(outputs["page_table"]).exists()
    assert Path(outputs["toc_tree"]).exists()
    assert Path(outputs["manifest"]).exists()
