"""Validation tests for the ingestion job's typed ``config.yaml`` view."""

from pydantic import ValidationError
import pytest

from src.config import DownloaderGeneralConfig

VALID: dict = {
    "shared": {
        "env_file": "_container_data/.env",
        "database_schema": "database_schema.yaml",
        "world_bank_download_config": "_configs/world_bank_download_config.json",
        "news_download_config": "_configs/news_download_config.json",
        "yahoo_download_config": "_configs/yahoo_download_config.json",
        "binance_download_config": "_configs/binance_download_config.json",
        "fred_download_config": "_configs/fred_download_config.json",
        "eurostat_download_config": "_configs/eurostat_download_config.json",
        "actually_relevant_download_config": "_configs/actually_relevant_download_config.json",
        "world_bank_articles_download_config": "_configs/world_bank_articles_download_config.json",
        "nuts_geojson": "_configs/nuts_level2_2021.geojson",
        "eurostat_nuts_level": 2,
        "openai_base_url": "https://api.openai.com/v1",
        "openai_embedding_model": "text-embedding-3-small",
        "openai_embedding_model_max_tokens": 8192,
        "openai_embedding_model_dimensions": 1536,
    },
    "postgres": {"host": "db", "port": 5432},
    "qdrant": {"host": "vector_db", "port": 6333},
    "downloader_general": {"repo_url": "https://example.com/news.git"},
    "agent": {"port": 8000},  # foreign section ignored
}


def test_valid_config_parses() -> None:
    cfg = DownloaderGeneralConfig.model_validate(VALID)
    assert cfg.shared.openai_embedding_model_max_tokens == 8192
    assert cfg.postgres.host == "db"
    assert cfg.qdrant.port == 6333
    assert cfg.downloader_general.repo_url == "https://example.com/news.git"


def test_missing_required_shared_key_raises() -> None:
    bad = {**VALID, "shared": {k: v for k, v in VALID["shared"].items() if k != "env_file"}}
    with pytest.raises(ValidationError):
        DownloaderGeneralConfig.model_validate(bad)


def test_missing_repo_url_raises() -> None:
    bad = {k: v for k, v in VALID.items() if k != "downloader_general"}
    with pytest.raises(ValidationError):
        DownloaderGeneralConfig.model_validate(bad)


def test_scheduler_defaults_when_absent() -> None:
    cfg = DownloaderGeneralConfig.model_validate(VALID)
    assert cfg.scheduler.enabled is True
    assert cfg.scheduler.run_on_start is False
    assert cfg.scheduler.sources == {}


def test_scheduler_block_parses() -> None:
    with_scheduler = {
        **VALID,
        "scheduler": {
            "enabled": True,
            "run_on_start": False,
            "sources": {
                "yahoo": {"enabled": True, "interval_minutes": 1440},
                "fred": {"enabled": False, "interval_minutes": 10080},
            },
        },
    }
    cfg = DownloaderGeneralConfig.model_validate(with_scheduler)
    assert cfg.scheduler.sources["yahoo"].enabled is True
    assert cfg.scheduler.sources["yahoo"].interval_minutes == 1440
    assert cfg.scheduler.sources["fred"].enabled is False
