"""FastAPI service: on-demand single-indicator ingestion from the World Bank.

The agent's ``downloader_agent`` worker POSTs to ``/ingest`` whenever the LLM
decides an indicator the user is asking about is missing from Postgres. The
endpoint short-circuits when the indicator is already present (returns
``status="already_downloaded"``), otherwise it fetches and stores it via
:mod:`client_wb` on a worker thread so the event loop stays free.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from client_wb import fetch_and_store_indicator
from config import load_config
from schema import Base, IngestRequest, IngestResponse, MacroIndicator
from settings import get_settings

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)
SETTINGS = get_settings()

_PG = CONFIG.postgres
_PG_DATABASE = SETTINGS.postgres_db or _PG.database
SQL_URI = (
    f"postgresql+psycopg2://"
    f"{SETTINGS.postgres_user}:{SETTINGS.postgres_password}"
    f"@{_PG.host}:{_PG.port}/{_PG_DATABASE}"
)


def _create_engine(sql_uri: str):
    """Create a SQLAlchemy engine and immediately verify connectivity.

    Args:
        sql_uri: Standard SQLAlchemy Postgres URI (must be non-empty).

    Returns:
        Tuple of ``(engine, sql_uri)`` so callers don't lose the URI.

    Raises:
        RuntimeError: When the URI is missing or the connect probe fails.
    """
    if not sql_uri:
        raise RuntimeError("No PostgreSQL connection sql_uri were configured.")

    engine = create_engine(
        sql_uri,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    try:
        with Session(engine) as session:
            session.execute(select(1))
    except Exception as exc:
        engine.dispose()
        raise RuntimeError(
            f"Could not connect to PostgreSQL using configured sql_uri: {exc}"
        ) from exc

    return engine, sql_uri


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bootstrap the engine + session factory on startup, dispose on shutdown."""
    engine, sql_uri = _create_engine(SQL_URI)
    Base.metadata.create_all(bind=engine)

    app.state.engine = engine
    app.state.sql_uri = sql_uri
    app.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    yield

    engine.dispose()


app = FastAPI(
    title="Macroeconomics Data Ingestion Service",
    description="Fetches macroeconomic indicators from the World Bank and stores them in a database",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Macroeconomics Data Ingestion Service!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/indicators")
def list_indicators() -> dict[str, list[str]]:
    """Return every distinct ``indicator_id`` currently stored in Postgres."""
    session_factory: sessionmaker[Session] = app.state.session_factory
    with session_factory() as session:
        rows = session.execute(
            select(MacroIndicator.indicator_id).distinct().order_by(MacroIndicator.indicator_id)
        ).all()
    return {"indicators": [row[0] for row in rows]}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_indicator(payload: IngestRequest):
    """Ingest a single World Bank indicator into Postgres.

    Short-circuits when at least one row for ``(indicator_id, db_id)``
    already exists. Otherwise hands off to :func:`client_wb.fetch_and_store_indicator`
    on the threadpool so the event loop stays free.

    Args:
        payload: ``IngestRequest`` with the WB indicator and database ids.

    Raises:
        HTTPException: 404 when the indicator can't be fetched from WB,
            500 for any other unexpected error.
    """
    session_factory: sessionmaker[Session] = app.state.session_factory
    try:
        with session_factory() as session:
            existing = session.execute(
                select(MacroIndicator)
                .where(
                    MacroIndicator.indicator_id == payload.indicator_id,
                    MacroIndicator.db_id == payload.db_id,
                )
                .limit(1)
            ).scalar()

        if existing is not None:
            return IngestResponse(
                indicator_id=payload.indicator_id,
                db_id=payload.db_id,
                rows_inserted=0,
                status="already_downloaded",
            )

        rows_inserted = await run_in_threadpool(
            fetch_and_store_indicator,
            payload.indicator_id,
            payload.db_id,
            app.state.sql_uri,
        )

        return IngestResponse(
            indicator_id=payload.indicator_id,
            db_id=payload.db_id,
            rows_inserted=rows_inserted,
            status="success",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
