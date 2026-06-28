"""Validation tests for the ingestion job's typed ``config.yaml`` view."""

import pytest
from pydantic import ValidationError

from src.config import DownloaderGeneralConfig

VALID: dict = {
    "shared": {
        "env_file": "_container_data/.env",
        "database_schema": "database_schema.yaml",
        "world_bank_download_config": "_configs/world_bank_download_config.json",
        "news_download_config": "_configs/news_download_config.json",
        "yahoo_download_config": "_configs/yahoo_download_config.json",
        "binance_download_config": "_configs/binance_download_config.json",
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
