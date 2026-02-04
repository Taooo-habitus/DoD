"""Tests for PageIndex OpenAI-compatible configuration."""

import pytest

openai = pytest.importorskip("openai")

from DoD.pageindex import utils  # noqa: E402


def test_set_openai_config_overrides_defaults() -> None:
    """Override PageIndex OpenAI-compatible defaults."""
    utils.set_openai_config(api_key="test-key", base_url="https://example.com/v1")
    assert utils.DEFAULT_API_KEY == "test-key"
    assert utils.DEFAULT_BASE_URL == "https://example.com/v1"
