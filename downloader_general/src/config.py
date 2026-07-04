"""Typed view over ``config.yaml`` for the ingestion (downloader_general) job.

Replaces the bespoke ``_require(...)`` nested-key walker in ``main.py`` with a
validated model: every required key is declared once and a malformed config
fails at load with a precise Pydantic error instead of a generic ``KeyError``.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class SharedConfig(BaseModel):
    """The ``shared`` section consumed by the ingestion job."""

    env_file: str
    database_schema: str
    world_bank_download_config: str
    news_download_config: str
    yahoo_download_config: str
    binance_download_config: str
    fred_download_config: str
    eurostat_download_config: str
    actually_relevant_download_config: str
    world_bank_articles_download_config: str
    nuts_geojson: str
    eurostat_nuts_level: int = 2
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


class SchedulerSourceConfig(BaseModel):
    """Per-source scheduling knobs (``enabled`` + its own ``interval_minutes``)."""

    enabled: bool = True
    interval_minutes: float = 10080.0  # weekly by default


class SchedulerConfig(BaseModel):
    """The ``scheduler`` block driving the post-ingest incremental updater.

    ``enabled`` is the master switch. When on, ``downloader_general`` stays alive
    after the initial ingest and refreshes each source in ``sources`` on its own
    interval (append-only). ``run_on_start`` false waits one interval before a
    source's first update (the initial ingest just populated everything). An
    empty/omitted ``sources`` map disables the scheduler in practice.
    """

    enabled: bool = True
    run_on_start: bool = False
    sources: dict[str, SchedulerSourceConfig] = {}


class DownloaderGeneralConfig(BaseModel):
    """The portion of ``config.yaml`` the ingestion job reads."""

    shared: SharedConfig
    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    downloader_general: DownloaderGeneralSection
    scheduler: SchedulerConfig = SchedulerConfig()


def load_config(path: Path) -> DownloaderGeneralConfig:
    """Parse and validate ``config.yaml`` into a :class:`DownloaderGeneralConfig`."""
    return DownloaderGeneralConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
