"""Tests for the ingestion job's secrets model (optional, graceful defaults)."""

import pytest
from src.settings import Settings

_SECRET_VARS = [
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_LLM_USER",
    "POSTGRES_LLM_PASSWORD",
    "OPENAI_API_KEY",
    "QDRANT__SERVICE__API_KEY",
    "QDRANT__API_KEY",
    "QDRANT_API_KEY",
]


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SECRET_VARS:
        monkeypatch.delenv(var, raising=False)


def test_missing_secrets_default_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.postgres_user == ""
    assert settings.postgres_db is None
    assert settings.openai_api_key == ""
    assert settings.qdrant_api_key == ""


def test_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("POSTGRES_USER", "main")
    monkeypatch.setenv("POSTGRES_LLM_USER", "llm")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.postgres_user == "main"
    assert settings.postgres_llm_user == "llm"
    assert settings.openai_api_key == "sk-test"


def test_qdrant_alias_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("QDRANT_API_KEY", "legacy-key")
    assert Settings(_env_file=None).qdrant_api_key == "legacy-key"
