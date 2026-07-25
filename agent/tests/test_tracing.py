"""Tests for the Langfuse tracing bootstrap gating logic.

These stay offline: the disabled / missing-key branches never construct a
Langfuse client, and the one enabled case builds the client lazily (no network
until a flush, which the tests never trigger).
"""

from __future__ import annotations

from agent.config import LangfuseConfig
import pytest

from agent import tracing


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    """Reset the module-global enabled flag around every test."""
    tracing._ENABLED = False
    yield
    tracing._ENABLED = False


def test_disabled_config_is_noop() -> None:
    assert tracing.init_tracing(LangfuseConfig(enabled=False), "pk", "sk") is False
    assert tracing.tracing_enabled() is False
    assert tracing.get_callback_handler() is None
    # Flushing while disabled must not raise.
    tracing.flush()


def test_enabled_but_missing_keys_is_noop() -> None:
    cfg = LangfuseConfig(enabled=True)
    assert tracing.init_tracing(cfg, public_key="", secret_key="") is False
    assert tracing.tracing_enabled() is False
    assert tracing.get_callback_handler() is None


def test_enabled_with_keys_builds_handler() -> None:
    cfg = LangfuseConfig(enabled=True, host="http://langfuse_web:3000", environment="test")
    assert tracing.init_tracing(cfg, public_key="pk-lf-x", secret_key="sk-lf-x") is True
    assert tracing.tracing_enabled() is True
    handler = tracing.get_callback_handler()
    assert handler is not None
    assert type(handler).__name__ == "LangchainCallbackHandler"
