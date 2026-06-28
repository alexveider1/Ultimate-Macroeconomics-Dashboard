"""Validation tests for the agent's typed ``config.yaml`` view."""

import pytest
from pydantic import ValidationError

from agent.config import AgentConfig, SharedConfig

VALID: dict = {
    "shared": {
        "openai_base_url": "https://api.openai.com/v1",
        "openai_llm_model": "gpt-5.4",
        "openai_llm_model_fast": "gpt-5.4-mini",
        "openai_embedding_model": "text-embedding-3-small",
    },
    "postgres": {"host": "db", "port": 5432},
    "qdrant": {"host": "vector_db", "port": 6333},
    "python_sandbox": {"port": 8004},
    "downloader_extra": {"port": 8003},
    # Sections owned by other services must be ignored, not rejected.
    "app": {"port": 8501},
    "forecaster": {"port": 8001, "ARIMA_AVAILABLE": True},
}


def test_valid_config_parses() -> None:
    cfg = AgentConfig.model_validate(VALID)
    assert cfg.shared.openai_llm_model == "gpt-5.4"
    assert cfg.shared.openai_llm_model_fast == "gpt-5.4-mini"
    assert cfg.postgres.host == "db"
    assert cfg.qdrant.port == 6333
    assert cfg.python_sandbox.port == 8004
    assert cfg.downloader_extra.port == 8003


def test_fast_model_defaults_to_strong_when_missing() -> None:
    shared = SharedConfig(openai_llm_model="gpt-5.4")
    assert shared.openai_llm_model_fast == "gpt-5.4"


def test_missing_required_section_raises() -> None:
    bad = {k: v for k, v in VALID.items() if k != "python_sandbox"}
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(bad)


def test_missing_required_shared_model_raises() -> None:
    with pytest.raises(ValidationError):
        SharedConfig()  # ty: ignore[missing-argument]
