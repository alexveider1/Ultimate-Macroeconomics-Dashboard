"""Validation tests for the BFF's typed config view."""

from config import BffConfig
from pydantic import ValidationError
import pytest


def test_full_config_parses() -> None:
    cfg = BffConfig.model_validate(
        {
            "postgres": {"host": "db", "port": 5432},
            "qdrant": {"host": "vector_db", "port": 6333},
            "shared": {"openai_embedding_model": "m", "openai_base_url": "u"},
            "bff": {"port": 8005},
            "agent": {"port": 8000},
            "forecaster": {"port": 8001},
            "clustering": {"port": 8002},
        }
    )
    assert cfg.bff.port == 8005
    assert cfg.forecaster.port == 8001
    assert cfg.shared.openai_embedding_model == "m"


def test_defaults_when_sections_absent() -> None:
    cfg = BffConfig.model_validate({})
    assert cfg.bff.port == 8005
    assert cfg.postgres.host == "db"
    assert cfg.qdrant.port == 6333
    assert cfg.agent.port == 8000


def test_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        BffConfig.model_validate({"bff": {"port": "not-an-int"}})
