"""Tests for the BFF's typed secrets loader."""

import pytest
from settings import Settings


def test_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_LLM_USER", "llm_reader")
    monkeypatch.setenv("POSTGRES_LLM_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "macro")
    monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "qkey")
    monkeypatch.setenv("OPENAI_API_KEY", "okey")

    settings = Settings()  # ty: ignore[missing-argument]

    assert settings.postgres_llm_user == "llm_reader"
    assert settings.postgres_llm_password == "secret"
    assert settings.postgres_db == "macro"
    assert settings.qdrant_api_key == "qkey"
    assert settings.openai_api_key == "okey"


def test_optional_secrets_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_LLM_USER", "llm_reader")
    monkeypatch.setenv("POSTGRES_LLM_PASSWORD", "secret")
    monkeypatch.delenv("QDRANT__SERVICE__API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)

    settings = Settings()  # ty: ignore[missing-argument]

    assert settings.qdrant_api_key == ""
    assert settings.openai_api_key == ""
    assert settings.postgres_db is None
