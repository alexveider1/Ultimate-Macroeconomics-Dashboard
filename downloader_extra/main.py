import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import create_engine, text

from dotenv import load_dotenv

from schema import Base, IngestRequest, IngestResponse
from client_wb import fetch_and_store_indicator


CONFIG_PATH = "config.yaml"
ENV_FILE_PATH = ".env"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)
load_dotenv(ENV_FILE_PATH)

_PG = CONFIG.get("postgres", {})
SQL_URI = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{_PG.get('host')}:{_PG.get('port')}/{_PG.get('database')}"
)


def _create_engine(sql_uri: str):
    if not sql_uri:
        raise RuntimeError("No PostgreSQL connection sql_uri were configured.")

    last_error: Exception | None = None
    engine = create_engine(
        sql_uri,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine, sql_uri
    except Exception as exc:
        engine.dispose()
        last_error = exc

    if last_error is not None:
        raise RuntimeError(
            f"Could not connect to PostgreSQL using configured sql_uri: {last_error}"
        ) from last_error

    raise RuntimeError("No PostgreSQL connection sql_uri were usable.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine, sql_uri = _create_engine(SQL_URI)
    Base.metadata.create_all(bind=engine)

    app.state.engine = engine
    app.state.sql_uri = sql_uri

    yield

    engine.dispose()


app = FastAPI(
    title="Macroeconomics Data Ingestion Service",
    description="Fetches macroeconomic indicators from the World Bank and stores them in a database",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Macroeconomics Data Ingestion Service!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/indicators")
def list_indicators() -> dict[str, list[str]]:
    with app.state.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT indicator_id FROM indicators ORDER BY indicator_id")
        ).fetchall()

    return {"indicators": [row[0] for row in rows]}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_indicator(payload: IngestRequest):
    try:
        with app.state.engine.connect() as conn:
            existing = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM indicators
                    WHERE indicator_id = :indicator_id AND db_id = :db_id
                    LIMIT 1
                    """
                ),
                {"indicator_id": payload.indicator_id, "db_id": payload.db_id},
            ).scalar()

        if existing:
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
