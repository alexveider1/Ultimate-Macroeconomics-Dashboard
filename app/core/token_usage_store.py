"""Persistent token usage accounting via SQLAlchemy ORM.

The agent returns prompt/completion/total counts in the SSE ``final`` event for
each ``/chat/stream`` request and in the body of ``/plots/interpret`` responses.
Per-session totals are kept in-memory by :mod:`core.token_usage`; this module
also persists each record into Postgres so historical trends survive process
restarts and are visible across sessions.

The table is created on first use with ``CREATE TABLE IF NOT EXISTS`` (no
``downloader_general`` involvement — this is an app-side concern). All writes
use the superuser role so the agent's read-only role isn't touched.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import DateTime, Engine, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from core.config import load_config
from core.settings import get_settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")


class Base(DeclarativeBase):
    """Declarative base scoped to the app's persistence layer."""


class TokenUsageRecord(Base):
    """One LLM-call usage record written by the dashboard.

    Args:
        ts: UTC timestamp of when the call completed.
        source: Logical origin of the call (``chat`` or ``plot_interpret``).
        model: The OpenAI-compatible model id reported by the agent.
        prompt_tokens: Input/prompt token count.
        completion_tokens: Output/completion token count.
        total_tokens: Sum reported by the provider (may differ from prompt + completion).
    """

    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    source: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False, index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


def _sql_uri() -> str:
    """Build the superuser Postgres URI from ``config.yaml`` + secrets."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH.resolve()}")
    pg = load_config(CONFIG_PATH).postgres
    settings = get_settings()
    return (
        f"postgresql+psycopg2://"
        f"{settings.postgres_user}:{settings.postgres_password}"
        f"@{pg.host}:{pg.port}/{settings.postgres_db}"
    )


@lru_cache(maxsize=1)
def _get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine with the table bootstrapped."""
    engine = create_engine(_sql_uri(), pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    return engine


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=_get_engine(), expire_on_commit=False)


def record_persistent(source: str, usage: dict[str, Any] | None) -> None:
    """Persist one LLM-call usage record to Postgres.

    Args:
        source: Logical origin label (``chat``, ``plot_interpret``, ...).
        usage: The ``usage`` dict from the agent response. Pass ``None`` or an
            empty dict to skip silently.

    Failures are logged but never raised, so a degraded DB doesn't break the
    streaming chat path.
    """
    if not isinstance(usage, dict):
        return
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    if prompt == 0 and completion == 0 and total == 0:
        return
    model = str(usage.get("model") or "unknown").strip() or "unknown"

    try:
        session_factory = _get_session_factory()
        with session_factory() as session, session.begin():
            session.add(
                TokenUsageRecord(
                    source=source,
                    model=model,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                )
            )
    except Exception:
        logger.warning("Failed to persist token usage record", exc_info=True)


def get_totals() -> dict[str, int]:
    """Return aggregate prompt/completion/total tokens across all rows."""
    try:
        with _get_session_factory()() as session:
            row = session.execute(
                select(
                    func.coalesce(func.sum(TokenUsageRecord.prompt_tokens), 0),
                    func.coalesce(func.sum(TokenUsageRecord.completion_tokens), 0),
                    func.coalesce(func.sum(TokenUsageRecord.total_tokens), 0),
                )
            ).one()
    except Exception:
        logger.warning("Failed to load token usage totals", exc_info=True)
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(row[0]),
        "completion_tokens": int(row[1]),
        "total_tokens": int(row[2]),
    }


def get_aggregates_by_model() -> pl.DataFrame:
    """Return per-model totals as a Polars DataFrame."""
    try:
        with _get_session_factory()() as session:
            rows = session.execute(
                select(
                    TokenUsageRecord.model,
                    func.count(TokenUsageRecord.id),
                    func.sum(TokenUsageRecord.prompt_tokens),
                    func.sum(TokenUsageRecord.completion_tokens),
                    func.sum(TokenUsageRecord.total_tokens),
                )
                .group_by(TokenUsageRecord.model)
                .order_by(func.sum(TokenUsageRecord.total_tokens).desc())
            ).all()
    except Exception:
        logger.warning("Failed to load token usage per-model aggregate", exc_info=True)
        return pl.DataFrame()

    return pl.DataFrame(
        {
            "model": [r[0] for r in rows],
            "calls": [int(r[1] or 0) for r in rows],
            "prompt_tokens": [int(r[2] or 0) for r in rows],
            "completion_tokens": [int(r[3] or 0) for r in rows],
            "total_tokens": [int(r[4] or 0) for r in rows],
        }
    )


def get_aggregates_by_day() -> pl.DataFrame:
    """Return daily totals (UTC) as a Polars DataFrame, oldest first."""
    try:
        with _get_session_factory()() as session:
            rows = session.execute(
                select(
                    func.date(TokenUsageRecord.ts),
                    func.sum(TokenUsageRecord.prompt_tokens),
                    func.sum(TokenUsageRecord.completion_tokens),
                    func.sum(TokenUsageRecord.total_tokens),
                )
                .group_by(func.date(TokenUsageRecord.ts))
                .order_by(func.date(TokenUsageRecord.ts))
            ).all()
    except Exception:
        logger.warning("Failed to load token usage daily aggregate", exc_info=True)
        return pl.DataFrame()

    days: list[date] = []
    for r in rows:
        raw_day = r[0]
        days.append(raw_day if isinstance(raw_day, date) else date.fromisoformat(str(raw_day)))
    return pl.DataFrame(
        {
            "day": days,
            "prompt_tokens": [int(r[1] or 0) for r in rows],
            "completion_tokens": [int(r[2] or 0) for r in rows],
            "total_tokens": [int(r[3] or 0) for r in rows],
        }
    )


def get_aggregates_by_source() -> pl.DataFrame:
    """Return per-source totals (chat / plot_interpret / ...)."""
    try:
        with _get_session_factory()() as session:
            rows = session.execute(
                select(
                    TokenUsageRecord.source,
                    func.count(TokenUsageRecord.id),
                    func.sum(TokenUsageRecord.total_tokens),
                ).group_by(TokenUsageRecord.source)
            ).all()
    except Exception:
        logger.warning("Failed to load token usage per-source aggregate", exc_info=True)
        return pl.DataFrame()

    return pl.DataFrame(
        {
            "source": [r[0] for r in rows],
            "calls": [int(r[1] or 0) for r in rows],
            "total_tokens": [int(r[2] or 0) for r in rows],
        }
    )
