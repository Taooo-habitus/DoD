"""Tests for request-scoped OpenAI-compatible config resolution."""

from __future__ import annotations

import asyncio
from typing import Optional

from DoD.pageindex import utils


def test_request_openai_config_overrides_and_restores_defaults() -> None:
    """Request-scoped config should override only inside the context."""
    original_key = utils.DEFAULT_API_KEY
    original_base = utils.DEFAULT_BASE_URL
    try:
        utils.set_openai_config(
            api_key="global-key", base_url="https://global.example/v1"
        )
        assert utils._resolve_api_key(None) == "global-key"
        assert utils._resolve_base_url(None) == "https://global.example/v1"

        with utils.request_openai_config(
            api_key="request-key", base_url="https://request.example/v1/"
        ):
            assert utils._resolve_api_key(None) == "request-key"
            assert utils._resolve_base_url(None) == "https://request.example/v1"

        assert utils._resolve_api_key(None) == "global-key"
        assert utils._resolve_base_url(None) == "https://global.example/v1"
    finally:
        utils.set_openai_config(api_key=original_key, base_url=original_base)


def test_request_openai_config_is_isolated_across_async_tasks() -> None:
    """Concurrent async tasks should keep request-scoped credentials isolated."""

    async def _worker(name: str) -> tuple[str, Optional[str]]:
        with utils.request_openai_config(
            api_key=f"{name}-key", base_url=f"https://{name}.example/v1/"
        ):
            # Yield control to ensure context isolation is tested under scheduling.
            await asyncio.sleep(0)
            return utils._resolve_api_key(None), utils._resolve_base_url(None)

    async def _run_workers():
        return await asyncio.gather(_worker("alpha"), _worker("beta"))

    alpha, beta = asyncio.run(_run_workers())

    assert alpha == ("alpha-key", "https://alpha.example/v1")
    assert beta == ("beta-key", "https://beta.example/v1")
