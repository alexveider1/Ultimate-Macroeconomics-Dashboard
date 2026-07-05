"""Langfuse tracing bootstrap for the ingestion (downloader_general) job.

Traces the OpenAI **embedding** calls the RAG downloaders make (news, Actually
Relevant, World Bank articles) so their token usage and cost are visible in
Langfuse. Gated by the ``langfuse`` block in ``config.yaml`` (``enabled``) plus
the ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` secrets — when either is
missing every helper here is a no-op and the plain OpenAI client is used.

Duplicated per service by design (no shared package), matching the repo's
one-container-one-copy convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import LangfuseConfig

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_ENABLED: bool = False


def init_tracing(
    cfg: LangfuseConfig,
    public_key: str,
    secret_key: str,
    release: str | None = None,
) -> bool:
    """Initialise the process-wide Langfuse client from config + secrets.

    Returns ``True`` when tracing is live. Safe to call once at startup.
    """
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


def openai_client_class() -> "type[OpenAI]":
    """Return the OpenAI client class the embed downloaders should instantiate.

    The Langfuse-wrapped ``OpenAI`` (a drop-in subclass) when tracing is on, so
    embedding calls are traced; the plain client otherwise.
    """
    from openai import OpenAI

    if not _ENABLED:
        return OpenAI
    try:
        from langfuse.openai import OpenAI as TracedOpenAI

        return TracedOpenAI
    except Exception:
        logger.exception("Failed to import Langfuse OpenAI wrapper; using plain client.")
        return OpenAI


def flush() -> None:
    """Flush any queued events (call before the process exits)."""
    if not _ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed.")
