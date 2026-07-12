"""FastAPI service: on-demand single-unit ingestion from five sources.

The agent's ``downloader_agent`` worker POSTs to ``/ingest`` whenever the LLM
decides data the user is asking about is missing from Postgres. ``source``
selects the path:

* ``worldbank`` — one indicator via :mod:`client_wb`.
* ``yahoo`` — one ticker via :mod:`client_yahoo`.
* ``binance`` — one spot pair via :mod:`client_binance`.
* ``fred`` — one US-state indicator (50 states + DC) via :mod:`client_fred`.
* ``eurostat`` — one EU NUTS-2 dataset via :mod:`client_eurostat`.

Each path short-circuits when the data is already present (returns
``status="already_downloaded"``), otherwise it fetches and stores it on a worker
thread so the event loop stays free.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from client_binance import fetch_and_store_binance
from client_eurostat import fetch_and_store_eurostat
from client_fred import fetch_and_store_fred
from client_wb import fetch_and_store_indicator
from client_yahoo import fetch_and_store_yahoo
from config import load_config
from fastapi import FastAPI, HTTPException
from schema import (
    Base,
    BinanceMetadata,
    EurostatIndicator,
    IngestRequest,
    IngestResponse,
    MacroIndicator,
    StateIndicator,
    YahooMetadata,
)
from settings import get_settings
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

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
    description=(
        "On-demand ingestion of a single unit of data from the World Bank "
        "(indicator), Yahoo Finance (ticker), Binance (spot pair), FRED "
        "(US-state indicator), or Eurostat (EU NUTS-2 dataset)."
    ),
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


def _already_present(session_factory: sessionmaker[Session], model, **filters) -> bool:
    """Return ``True`` if at least one row of ``model`` matches ``filters``."""
    with session_factory() as session:
        existing = session.execute(select(model).filter_by(**filters).limit(1)).scalar()
    return existing is not None


async def _ingest_worldbank(payload: IngestRequest) -> IngestResponse:
    """World Bank path: short-circuit on ``(indicator_id, db_id)`` then fetch."""
    if not payload.indicator_id or payload.db_id is None:
        raise HTTPException(status_code=400, detail="worldbank requires indicator_id and db_id")
    indicator_id, db_id = payload.indicator_id, payload.db_id

    if _already_present(
        app.state.session_factory, MacroIndicator, indicator_id=indicator_id, db_id=db_id
    ):
        return IngestResponse(
            source="worldbank",
            identifier=indicator_id,
            db_id=db_id,
            rows_inserted=0,
            status="already_downloaded",
        )

    rows_inserted = await fetch_and_store_indicator(indicator_id, db_id, app.state.sql_uri)
    return IngestResponse(
        source="worldbank",
        identifier=indicator_id,
        db_id=db_id,
        rows_inserted=rows_inserted,
        status="success",
    )


async def _ingest_yahoo(payload: IngestRequest) -> IngestResponse:
    """Yahoo path: short-circuit on ``ticker`` in yahoo_metadata then fetch."""
    if not payload.ticker:
        raise HTTPException(status_code=400, detail="yahoo requires ticker")
    ticker = payload.ticker.strip()

    if _already_present(app.state.session_factory, YahooMetadata, ticker=ticker):
        return IngestResponse(
            source="yahoo", identifier=ticker, rows_inserted=0, status="already_downloaded"
        )

    rows_inserted = await fetch_and_store_yahoo(ticker, app.state.sql_uri)
    return IngestResponse(
        source="yahoo", identifier=ticker, rows_inserted=rows_inserted, status="success"
    )


async def _ingest_binance(payload: IngestRequest) -> IngestResponse:
    """Binance path: short-circuit on ``symbol`` in binance_metadata then fetch."""
    if not payload.symbol:
        raise HTTPException(status_code=400, detail="binance requires symbol")
    symbol = payload.symbol.strip().upper()

    if _already_present(app.state.session_factory, BinanceMetadata, symbol=symbol):
        return IngestResponse(
            source="binance", identifier=symbol, rows_inserted=0, status="already_downloaded"
        )

    rows_inserted = await fetch_and_store_binance(symbol, app.state.sql_uri)
    return IngestResponse(
        source="binance", identifier=symbol, rows_inserted=rows_inserted, status="success"
    )


async def _ingest_fred(payload: IngestRequest) -> IngestResponse:
    """FRED path: short-circuit on the resolved slug in state_indicators then fetch.

    The slug is the upper-cased ``series_id``; an ``example_series_id`` match also
    short-circuits so asking for a series backing a pre-loaded indicator (e.g.
    ``CAUR`` → ``unemployment_rate``) doesn't re-download it.
    """
    if not payload.series_id:
        raise HTTPException(status_code=400, detail="fred requires series_id")
    series_id = payload.series_id.strip()
    slug = series_id.upper()

    if _already_present(
        app.state.session_factory, StateIndicator, indicator_id=slug
    ) or _already_present(app.state.session_factory, StateIndicator, example_series_id=series_id):
        return IngestResponse(
            source="fred", identifier=slug, rows_inserted=0, status="already_downloaded"
        )

    rows_inserted = await fetch_and_store_fred(series_id, app.state.sql_uri, SETTINGS.fred_api_key)
    return IngestResponse(
        source="fred", identifier=slug, rows_inserted=rows_inserted, status="success"
    )


async def _ingest_eurostat(payload: IngestRequest) -> IngestResponse:
    """Eurostat path: short-circuit on the dataset slug in eurostat_indicators then fetch.

    The slug is the lower-cased dataset code; if it (or an already-loaded config
    indicator sharing the same ``dataset``) is present, the fetch is skipped.
    """
    if not payload.dataset:
        raise HTTPException(status_code=400, detail="eurostat requires dataset")
    dataset = payload.dataset.strip()
    slug = dataset.lower()

    if _already_present(
        app.state.session_factory, EurostatIndicator, indicator_id=slug
    ) or _already_present(app.state.session_factory, EurostatIndicator, dataset=dataset):
        return IngestResponse(
            source="eurostat", identifier=slug, rows_inserted=0, status="already_downloaded"
        )

    rows_inserted = await fetch_and_store_eurostat(dataset, payload.filters, app.state.sql_uri)
    return IngestResponse(
        source="eurostat", identifier=slug, rows_inserted=rows_inserted, status="success"
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_data(payload: IngestRequest):
    """Ingest a single unit of data from the requested ``source``.

    Dispatches on ``payload.source`` to the World Bank / Yahoo / Binance path,
    each of which short-circuits when the data is already present and otherwise
    fetches it (offloading the blocking DB write to a worker thread).

    Args:
        payload: ``IngestRequest`` carrying the source and its id field(s).

    Raises:
        HTTPException: 404 when the data can't be fetched from the source,
            400 for a bad request, 500 for any other unexpected error.
    """
    handlers = {
        "worldbank": _ingest_worldbank,
        "yahoo": _ingest_yahoo,
        "binance": _ingest_binance,
        "fred": _ingest_fred,
        "eurostat": _ingest_eurostat,
    }
    handler = handlers.get(payload.source)
    if handler is None:  # pragma: no cover — guarded by the request validator
        raise HTTPException(status_code=400, detail=f"Unknown source: {payload.source}")

    try:
        return await handler(payload)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
