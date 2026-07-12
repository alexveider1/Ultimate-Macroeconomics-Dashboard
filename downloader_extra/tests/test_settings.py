"""Tests for downloader_extra's least-privilege secrets model."""

from pydantic import ValidationError
import pytest
from settings import Settings

_SECRET_VARS = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SECRET_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("POSTGRES_USER", "main")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    settings = Settings(_env_file=None)  # ty: ignore[missing-argument]
    assert settings.postgres_user == "main"
    assert settings.postgres_password == "pw"
    assert settings.postgres_db is None


def test_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # ty: ignore[missing-argument]
