"""CLI entrypoint for the digestion pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from DoD.config import PipelineConfig
from DoD.pipeline import digest_document

cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "conf")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: PipelineConfig) -> None:
    """Run the document digestion pipeline CLI."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.info("Starting document digestion.")
    artifacts = digest_document(cfg)
    logging.info("Config: %s", OmegaConf.to_container(cfg, resolve=True))
    logging.info("Output files: %s", artifacts)


if __name__ == "__main__":
    main()
