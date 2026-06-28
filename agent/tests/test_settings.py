"""Tests for the agent's least-privilege secrets model."""

import pytest
from pydantic import ValidationError

from agent.settings import Settings

_SECRET_VARS = [
    "OPENAI_API_KEY",
    "QDRANT__SERVICE__API_KEY",
    "QDRANT__API_KEY",
    "QDRANT_API_KEY",
    "POSTGRES_LLM_USER",
    "POSTGRES_LLM_PASSWORD",
    "POSTGRES_DB",
]


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SECRET_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("POSTGRES_LLM_USER", "llm")
    monkeypatch.setenv("POSTGRES_LLM_PASSWORD", "pw")
    monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "qk")
    settings = Settings(_env_file=None)  # ty: ignore[missing-argument]
    assert settings.openai_api_key == "sk-test"
    assert settings.postgres_llm_user == "llm"
    assert settings.postgres_llm_password == "pw"
    assert settings.qdrant_api_key == "qk"
    assert settings.postgres_db is None


def test_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # ty: ignore[missing-argument]


def test_qdrant_alias_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("POSTGRES_LLM_USER", "llm")
    monkeypatch.setenv("POSTGRES_LLM_PASSWORD", "pw")
    monkeypatch.setenv("QDRANT_API_KEY", "legacy-key")
    settings = Settings(_env_file=None)  # ty: ignore[missing-argument]
    assert settings.qdrant_api_key == "legacy-key"
