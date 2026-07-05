"""Tests for the BFF's Langfuse tracing gating logic (offline)."""

from __future__ import annotations

from config import LangfuseConfig
from openai import AsyncOpenAI
import pytest
import tracing


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    """Reset the module-global enabled flag around every test."""
    tracing._ENABLED = False
    yield
    tracing._ENABLED = False


def test_disabled_uses_plain_client() -> None:
    assert tracing.init_tracing(LangfuseConfig(enabled=False), "pk", "sk") is False
    assert tracing.tracing_enabled() is False
    assert tracing.async_openai_client_class() is AsyncOpenAI
    tracing.flush()  # must not raise


def test_enabled_but_missing_keys_uses_plain_client() -> None:
    assert tracing.init_tracing(LangfuseConfig(enabled=True), "", "") is False
    assert tracing.async_openai_client_class() is AsyncOpenAI


def test_enabled_flag_flips_on() -> None:
    cfg = LangfuseConfig(enabled=True, environment="test")
    assert tracing.init_tracing(cfg, "pk-lf-x", "sk-lf-x") is True
    assert tracing.tracing_enabled() is True
    # Do NOT call async_openai_client_class() here: importing the Langfuse
    # ``openai`` integration patches the openai module process-wide, which would
    # leak into other tests. The disabled cases cover the plain-client path.
