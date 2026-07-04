"""Typed secrets for the BFF service.

The BFF only ever *reads* Postgres, so it uses the read-only ``POSTGRES_LLM``
role (never the superuser) — exactly like the ``agent`` and the Streamlit app's
page reads. It also needs the Qdrant API key (news search / browse) and the
OpenAI-compatible API key (to embed news-search queries). The container is given
exactly this set in ``docker-compose.yaml``; ``openai_api_key`` /
``qdrant_api_key`` default to empty so the pure-Postgres endpoints still work
when they are unset.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read-only Postgres role + Qdrant key + OpenAI key used by the BFF."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_llm_user: str = Field(validation_alias=AliasChoices("POSTGRES_LLM_USER"))
    postgres_llm_password: str = Field(validation_alias=AliasChoices("POSTGRES_LLM_PASSWORD"))
    postgres_db: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DB"))
    qdrant_api_key: str = Field(
        default="", validation_alias=AliasChoices("QDRANT__SERVICE__API_KEY")
    )
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, validated on first call."""
    # Fields are populated from the environment, not constructor args.
    return Settings()  # ty: ignore[missing-argument]
