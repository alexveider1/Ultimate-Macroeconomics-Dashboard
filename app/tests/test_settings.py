"""Tests for the dashboard's optional secrets model."""

import pytest

from core.settings import Settings

_SECRET_VARS = [
    "POSTGRES_LLM_USER",
    "POSTGRES_LLM_PASSWORD",
    "POSTGRES_DB",
    "QDRANT_API_KEY",
    "QDRANT__API_KEY",
    "QDRANT__SERVICE__API_KEY",
    "OPENAI_API_KEY",
]


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SECRET_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    settings = Settings(_env_file=None)
    assert settings.postgres_llm_user == ""
    assert settings.postgres_db is None
    assert settings.qdrant_api_key == ""
    assert settings.openai_api_key == ""


def test_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("POSTGRES_LLM_USER", "llm")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.postgres_llm_user == "llm"
    assert settings.openai_api_key == "sk-test"


def test_qdrant_alias_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "svc")
    assert Settings(_env_file=None).qdrant_api_key == "svc"
    # QDRANT_API_KEY is first in the alias list, so it wins when both are set.
    monkeypatch.setenv("QDRANT_API_KEY", "primary")
    assert Settings(_env_file=None).qdrant_api_key == "primary"
