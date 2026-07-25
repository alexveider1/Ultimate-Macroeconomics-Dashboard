"""Tests for the ingestion job's Langfuse tracing gating logic (offline)."""

from __future__ import annotations

from openai import OpenAI
import pytest
from src.config import LangfuseConfig
from src.core import tracing


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    """Reset the module-global enabled flag around every test."""
    tracing._ENABLED = False
    yield
    tracing._ENABLED = False


def test_disabled_uses_plain_client() -> None:
    assert tracing.init_tracing(LangfuseConfig(enabled=False), "pk", "sk") is False
    assert tracing.tracing_enabled() is False
    # Plain OpenAI class when tracing is off.
    assert tracing.openai_client_class() is OpenAI
    tracing.flush()  # must not raise


def test_enabled_but_missing_keys_uses_plain_client() -> None:
    assert tracing.init_tracing(LangfuseConfig(enabled=True), "", "") is False
    assert tracing.openai_client_class() is OpenAI


def test_enabled_flag_flips_on() -> None:
    cfg = LangfuseConfig(enabled=True, environment="test")
    assert tracing.init_tracing(cfg, "pk-lf-x", "sk-lf-x") is True
    assert tracing.tracing_enabled() is True
    # We deliberately do NOT call openai_client_class() here: the Langfuse
    # ``openai`` integration patches the openai module process-wide on import,
    # which would leak into other tests. The disabled cases above already assert
    # the plain-client path.
