"""Langfuse tracing bootstrap for the BFF service.

Traces the OpenAI **embedding** calls the news semantic-search path makes, so
their token usage and cost show up in Langfuse. Gated by the ``langfuse`` block
in ``config.yaml`` (``enabled``) plus the ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` secrets — a no-op when either is missing.

Flat-layout / duplicated per service by design (no shared package), matching the
repo's one-container-one-copy convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import LangfuseConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_ENABLED: bool = False


def init_tracing(
    cfg: LangfuseConfig,
    public_key: str,
    secret_key: str,
    release: str | None = None,
) -> bool:
    """Initialise the process-wide Langfuse client from config + secrets."""
    global _ENABLED
    if not cfg.enabled:
        logger.info("Langfuse tracing disabled (langfuse.enabled=false).")
        return False
    if not (public_key and secret_key):
        logger.warning("Langfuse enabled but LANGFUSE_PUBLIC_KEY/SECRET_KEY missing; tracing off.")
        return False
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse package not installed; tracing off.")
        return False

    try:
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=cfg.host,
            environment=cfg.environment,
            sample_rate=cfg.sample_rate,
            release=release,
            tracing_enabled=True,
        )
    except Exception:
        logger.exception("Failed to initialise Langfuse client; tracing off.")
        return False

    _ENABLED = True
    logger.info("Langfuse tracing enabled (host=%s, environment=%s).", cfg.host, cfg.environment)
    return True


def tracing_enabled() -> bool:
    """Return whether Langfuse tracing was successfully initialised."""
    return _ENABLED


def async_openai_client_class() -> "type[AsyncOpenAI]":
    """Return the ``AsyncOpenAI`` class to instantiate for news embeddings.

    The Langfuse-instrumented ``AsyncOpenAI`` when tracing is on (the Langfuse
    ``openai`` integration patches the module in place), the plain client
    otherwise.
    """
    from openai import AsyncOpenAI

    if not _ENABLED:
        return AsyncOpenAI
    try:
        from langfuse.openai import AsyncOpenAI as TracedAsyncOpenAI

        return TracedAsyncOpenAI
    except Exception:
        logger.exception("Failed to import Langfuse OpenAI wrapper; using plain client.")
        return AsyncOpenAI


def flush() -> None:
    """Flush any queued events (call on shutdown so nothing is lost)."""
    if not _ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed.")
