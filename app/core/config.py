"""Typed view over ``config.yaml`` for the Streamlit dashboard.

Covers the slice the app reads: Postgres + Qdrant connection details and the
per-service ports the Monitoring page probes. Every section carries a default so
a partial or missing config still yields a usable object (the Monitoring page is
fail-soft). Unknown sections (``shared``, ``services``, …) are ignored.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class PostgresConfig(BaseModel):
    """The ``postgres`` section. ``database`` falls back to ``POSTGRES_DB``."""

    host: str = "db"
    port: int = 5432
    database: str | None = None


class QdrantConfig(BaseModel):
    """The ``qdrant`` section."""

    host: str = "vector_db"
    port: int = 6333


class PortConfig(BaseModel):
    """A bare ``{port: int}`` service block."""

    port: int


class AppConfig(BaseModel):
    """The portion of ``config.yaml`` the dashboard reads."""

    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    app: PortConfig = PortConfig(port=8501)
    agent: PortConfig = PortConfig(port=8000)
    forecaster: PortConfig = PortConfig(port=8001)
    clustering: PortConfig = PortConfig(port=8002)
    downloader_extra: PortConfig = PortConfig(port=8003)
    python_sandbox: PortConfig = PortConfig(port=8004)


def load_config(path: Path) -> AppConfig:
    """Parse and validate ``config.yaml`` into an :class:`AppConfig`."""
    return AppConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
