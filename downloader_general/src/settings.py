"""Typed secrets for the ingestion (downloader_general) container.

This one-shot job bootstraps the read-only LLM Postgres role and runs the World
Bank / Yahoo / Binance / FRED / news ingestion, so it legitimately needs the
superuser creds, the LLM-role creds, the OpenAI key (news embeddings), the FRED
API key (state indicators) and the Qdrant key. Every field
is optional with a safe default so a missing secret degrades gracefully — the
relevant connection check fails and that source is skipped — rather than
aborting the whole run, matching the service's existing behaviour.
"""

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All secrets the ingestion job may use, each optional with a default."""

    model_config = SettingsConfigDict(extra="ignore")

    postgres_user: str = Field(default="", validation_alias=AliasChoices("POSTGRES_USER"))
    postgres_password: str = Field(default="", validation_alias=AliasChoices("POSTGRES_PASSWORD"))
    postgres_db: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DB"))
    postgres_llm_user: str = Field(default="", validation_alias=AliasChoices("POSTGRES_LLM_USER"))
    postgres_llm_password: str = Field(
        default="", validation_alias=AliasChoices("POSTGRES_LLM_PASSWORD")
    )
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))
    fred_api_key: str = Field(default="", validation_alias=AliasChoices("FRED_API_KEY"))
    qdrant_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "QDRANT__SERVICE__API_KEY", "QDRANT__API_KEY", "QDRANT_API_KEY"
        ),
    )
    langfuse_public_key: str = Field(
        default="", validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY")
    )
    langfuse_secret_key: str = Field(
        default="", validation_alias=AliasChoices("LANGFUSE_SECRET_KEY")
    )


def load_settings(env_file: str | os.PathLike[str]) -> Settings:
    """Build :class:`Settings` from process env plus the given ``.env`` file.

    Process environment variables (injected by Compose) take precedence over the
    file, mirroring the previous ``load_dotenv`` + ``os.getenv`` flow. A missing
    file is ignored.
    """
    return Settings(_env_file=env_file)
