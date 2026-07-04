"""BFF (backend-for-frontend) FastAPI app.

One read-only origin the frontend talks to: typed ORM reads of the Postgres
tables ``downloader_general`` populates (World Bank / Yahoo / Binance / FRED /
Eurostat), semantic search + browse over the Qdrant news corpus, and thin
proxies to the existing forecaster / clustering / agent services. It is purely
additive — the Streamlit app keeps its own ``connectorx`` read path until the
Phase-6 frontend cutover.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

import clients
from config import load_config
from db import build_engine, build_sql_uri
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
from routers import agent, cluster, crypto, eurostat, forecast, fred, news, worldbank, yahoo
from schema import Base
from settings import get_settings
from sqlalchemy.orm import sessionmaker
import vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")
CONFIG = load_config(CONFIG_PATH)
SETTINGS = get_settings()

_HTTPX_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)


def _cors_origins() -> list[str]:
    """Return the CORS allow-list (``BFF_CORS_ORIGINS`` env, comma-separated)."""
    raw = os.getenv("BFF_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build shared clients on startup; dispose them on shutdown."""
    engine = build_engine(build_sql_uri(CONFIG.postgres, SETTINGS))
    # No-op against the live tables; materialises them for an isolated test DB.
    Base.metadata.create_all(bind=engine)
    app.state.engine = engine
    app.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    app.state.qdrant = vector.build_qdrant_client(
        CONFIG.qdrant.host, CONFIG.qdrant.port, SETTINGS.qdrant_api_key
    )
    app.state.openai = vector.build_openai_client(
        SETTINGS.openai_api_key, CONFIG.shared.openai_base_url
    )
    app.state.embedding_model = CONFIG.shared.openai_embedding_model
    app.state.news_search_enabled = bool(SETTINGS.openai_api_key)

    app.state.http_client = httpx.AsyncClient(limits=_HTTPX_LIMITS)
    app.state.forecaster_url = clients.resolve_base_url(
        "FORECASTER_BASE_URL", f"http://forecaster:{CONFIG.forecaster.port}"
    )
    app.state.clustering_url = clients.resolve_base_url(
        "CLUSTERING_BASE_URL", f"http://clustering:{CONFIG.clustering.port}"
    )
    app.state.agent_url = clients.resolve_base_url(
        "AGENT_BASE_URL", f"http://agent:{CONFIG.agent.port}"
    )

    try:
        yield
    finally:
        engine.dispose()
        await app.state.http_client.aclose()
        await app.state.qdrant.close()
        await app.state.openai.close()


app = FastAPI(
    title="Macroeconomics Dashboard BFF",
    description=(
        "Read-only backend-for-frontend: typed ORM reads of the macro data "
        "(World Bank, Yahoo Finance, Binance, FRED, Eurostat), Qdrant news "
        "search, and proxies to the forecaster / clustering / agent services."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for module in (worldbank, yahoo, crypto, fred, eurostat, news, forecast, cluster, agent):
    app.include_router(module.router)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Macroeconomics Dashboard BFF!"}


@app.get("/health")
def health() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}
