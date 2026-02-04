"""Main entrypoint for the project."""

from __future__ import annotations

import os
import shlex
import sys

from DoD.cli.main import main


if __name__ == "__main__":
    # Hydra override parsing fails on empty argv tokens; sanitize and allow env overrides.
    sys.argv = [arg for arg in sys.argv if arg and arg.strip()]
    extra_overrides = os.getenv("DOD_OVERRIDES")
    if extra_overrides:
        sys.argv.extend(shlex.split(extra_overrides))
    main()
