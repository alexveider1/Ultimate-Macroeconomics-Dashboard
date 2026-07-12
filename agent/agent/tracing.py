"""Langfuse tracing bootstrap for the agent service.

Wires the LangGraph run, the raw-OpenAI vision call and the RAG-embedding call
into a self-hosted Langfuse instance, so every worker, tool, LLM and embedding
call is traced with its tokens, latency and cost. All of it is gated by the
``langfuse`` block in ``config.yaml`` (``enabled``) and the presence of the
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` secrets — when either is
missing the module degrades to a no-op (:func:`get_callback_handler` returns
``None`` and nothing is emitted). Core request handling never depends on
Langfuse being reachable: the SDK batches and drops on failure off the hot path.

Duplicated per service by design (no shared package), matching the repo's
one-container-one-copy convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent.config import LangfuseConfig

if TYPE_CHECKING:
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

_ENABLED: bool = False


def init_tracing(
    cfg: LangfuseConfig,
    public_key: str,
    secret_key: str,
    release: str | None = None,
) -> bool:
    """Initialise the process-wide Langfuse client from config + secrets.

    Returns ``True`` when tracing is live. Idempotent-ish: only the first call
    with valid credentials flips the module into the enabled state.

    Args:
        cfg: The parsed ``langfuse`` config block (enabled / host / env / rate).
        public_key: ``LANGFUSE_PUBLIC_KEY`` (empty when unset).
        secret_key: ``LANGFUSE_SECRET_KEY`` (empty when unset).
        release: Optional release/version tag applied to every trace.
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
    logger.info(
        "Langfuse tracing enabled (host=%s, environment=%s, sample_rate=%s).",
        cfg.host,
        cfg.environment,
        cfg.sample_rate,
    )
    return True


def tracing_enabled() -> bool:
    """Return whether Langfuse tracing was successfully initialised."""
    return _ENABLED


def get_callback_handler() -> "CallbackHandler | None":
    """Return a fresh LangChain ``CallbackHandler`` bound to the global client.

    Returns ``None`` when tracing is disabled, so callers can attach it
    unconditionally with a simple ``if handler is not None`` guard.
    """
    if not _ENABLED:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        logger.exception("Failed to build Langfuse CallbackHandler.")
        return None


def flush() -> None:
    """Flush any queued events (call on shutdown so nothing is lost)."""
    if not _ENABLED:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed.")
