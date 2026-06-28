"""Validation tests for downloader_extra's typed ``config.yaml`` view."""

import pytest
from pydantic import ValidationError

from config import DownloaderExtraConfig


def test_valid_config_parses() -> None:
    cfg = DownloaderExtraConfig.model_validate(
        {
            "postgres": {"host": "db", "port": 5432},
            "downloader_extra": {"port": 8003},
            "app": {"port": 8501},  # foreign section ignored
        }
    )
    assert cfg.postgres.host == "db"
    assert cfg.postgres.port == 5432
    assert cfg.downloader_extra.port == 8003


def test_defaults_when_sections_absent() -> None:
    cfg = DownloaderExtraConfig.model_validate({})
    assert cfg.postgres.host == "db"
    assert cfg.postgres.database is None
    assert cfg.downloader_extra.port == 8003


def test_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        DownloaderExtraConfig.model_validate({"postgres": {"port": "nope"}})
