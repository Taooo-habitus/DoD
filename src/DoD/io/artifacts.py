"""Artifact IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    """Write JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
