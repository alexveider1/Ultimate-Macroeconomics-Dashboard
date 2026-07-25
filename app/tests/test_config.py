"""Validation tests for the dashboard's typed ``config.yaml`` view."""

from core.config import AppConfig
from pydantic import ValidationError
import pytest


def test_valid_config_parses() -> None:
    cfg = AppConfig.model_validate(
        {
            "postgres": {"host": "db", "port": 5432},
            "qdrant": {"host": "vector_db", "port": 6333},
            "forecaster": {"port": 8001},
            "app": {"port": 8501},
            "shared": {"openai_llm_model": "gpt-5.4"},  # foreign section ignored
        }
    )
    assert cfg.postgres.host == "db"
    assert cfg.qdrant.port == 6333
    assert cfg.forecaster.port == 8001
    assert cfg.app.port == 8501


def test_service_port_defaults() -> None:
    cfg = AppConfig.model_validate({})
    assert cfg.agent.port == 8000
    assert cfg.python_sandbox.port == 8004
    assert cfg.postgres.host == "db"
    assert cfg.postgres.database is None


def test_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"forecaster": {"port": "nope"}})
