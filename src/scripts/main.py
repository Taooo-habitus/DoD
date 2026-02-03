"""Main entrypoint for the project."""

import hydra
from omegaconf import DictConfig
import logging


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run the main application with Hydra config."""
    logging.basicConfig(level=getattr(logging, cfg.logging_level.upper(), logging.INFO))
    logging.info("Hello from your new Hydra-powered Python template!")


if __name__ == "__main__":
    main()
