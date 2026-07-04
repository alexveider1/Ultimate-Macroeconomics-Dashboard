"""Postgres engine + session plumbing for the BFF (read-only LLM role).

The engine is built once at startup (``build_engine``) and stashed on
``app.state``; routers pull a short-lived :class:`~sqlalchemy.orm.Session` via
the :func:`get_session` FastAPI dependency. All access is ORM ``select()`` on
the ``Mapped`` models in :mod:`schema` — no raw ``text()`` — matching the repo
convention for every service except the agent's dynamic SQL worker.
"""

from collections.abc import Iterator

from config import PostgresConfig
from fastapi import Request
from settings import Settings
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker


def build_sql_uri(postgres: PostgresConfig, settings: Settings) -> str:
    """Return the SQLAlchemy URI for the read-only LLM role.

    ``POSTGRES_DB`` (env) is the source of truth for the database name;
    ``config.yaml``'s ``postgres.database`` is the fallback when it is unset.
    """
    database = settings.postgres_db or postgres.database
    return (
        f"postgresql+psycopg2://"
        f"{settings.postgres_llm_user}:{settings.postgres_llm_password}"
        f"@{postgres.host}:{postgres.port}/{database}"
    )


def build_engine(sql_uri: str) -> Engine:
    """Create a pooled engine and verify connectivity with a probe query.

    Raises:
        RuntimeError: When the URI is empty or the connect probe fails.
    """
    if not sql_uri:
        raise RuntimeError("No PostgreSQL connection URI was configured.")

    engine = create_engine(
        sql_uri,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"connect_timeout": 5},
    )
    try:
        with Session(engine) as session:
            session.execute(select(1))
    except Exception as exc:
        engine.dispose()
        raise RuntimeError(f"Could not connect to PostgreSQL: {exc}") from exc
    return engine


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped ORM session."""
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as session:
        yield session
