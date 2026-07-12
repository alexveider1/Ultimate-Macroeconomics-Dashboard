"""Typed secrets for the downloader_extra service.

Writes new indicators into Postgres as the superuser role, so it needs the three
Postgres credentials plus the FRED API key (for on-demand state-indicator
ingestion) — nothing OpenAI- or Qdrant-related. The container is given exactly
this set in ``docker-compose.yaml``. ``fred_api_key`` defaults to empty so the
World Bank / Yahoo / Binance paths still work when it is unset.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Postgres superuser credentials + FRED key used for ingestion writes."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = Field(validation_alias=AliasChoices("POSTGRES_USER"))
    postgres_password: str = Field(validation_alias=AliasChoices("POSTGRES_PASSWORD"))
    postgres_db: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DB"))
    fred_api_key: str = Field(default="", validation_alias=AliasChoices("FRED_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, validated on first call."""
    # Fields are populated from the environment, not constructor args.
    return Settings()  # ty: ignore[missing-argument]
