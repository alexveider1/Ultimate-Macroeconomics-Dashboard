"""Typed view over ``config.yaml`` for the ingestion (downloader_general) job.

Replaces the bespoke ``_require(...)`` nested-key walker in ``main.py`` with a
validated model: every required key is declared once and a malformed config
fails at load with a precise Pydantic error instead of a generic ``KeyError``.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel


class SharedConfig(BaseModel):
    """The ``shared`` section consumed by the ingestion job."""

    env_file: str
    database_schema: str
    world_bank_download_config: str
    news_download_config: str
    yahoo_download_config: str
    binance_download_config: str
    openai_base_url: str
    openai_embedding_model: str
    openai_embedding_model_max_tokens: int
    openai_embedding_model_dimensions: int


class PostgresConfig(BaseModel):
    """The ``postgres`` section. ``database`` falls back to ``POSTGRES_DB``."""

    host: str = "db"
    port: int = 5432
    database: str | None = None


class QdrantConfig(BaseModel):
    """The ``qdrant`` section."""

    host: str = "vector_db"
    port: int = 6333


class DownloaderGeneralSection(BaseModel):
    """The ``downloader_general`` section (the news-repo URL)."""

    repo_url: str


class DownloaderGeneralConfig(BaseModel):
    """The portion of ``config.yaml`` the ingestion job reads."""

    shared: SharedConfig
    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    downloader_general: DownloaderGeneralSection


def load_config(path: Path) -> DownloaderGeneralConfig:
    """Parse and validate ``config.yaml`` into a :class:`DownloaderGeneralConfig`."""
    return DownloaderGeneralConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
