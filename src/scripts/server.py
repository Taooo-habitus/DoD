"""Server entrypoint for DoD FastAPI app."""

from __future__ import annotations

import os

import uvicorn


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


if __name__ == "__main__":
    host = os.getenv("DOD_SERVER_HOST", "0.0.0.0")
    port = _env_int("DOD_SERVER_PORT", 8000)
    reload_flag = os.getenv("DOD_SERVER_RELOAD", "0") in {"1", "true", "True"}
    uvicorn.run("DoD.server.app:app", host=host, port=port, reload=reload_flag)
