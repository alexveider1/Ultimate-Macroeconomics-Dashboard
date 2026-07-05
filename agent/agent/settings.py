"""Typed secrets for the agent service.

Only the secrets the agent actually uses are declared here; the container is
given exactly this set in ``docker-compose.yaml`` (least privilege — no
Postgres superuser password, for example). Values come from the process
environment (injected by Compose) with a local ``.env`` fallback for dev.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets required by the agent (read-only Postgres role + API keys)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(validation_alias=AliasChoices("OPENAI_API_KEY"))
    qdrant_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "QDRANT__SERVICE__API_KEY", "QDRANT__API_KEY", "QDRANT_API_KEY"
        ),
    )
    postgres_llm_user: str = Field(validation_alias=AliasChoices("POSTGRES_LLM_USER"))
    postgres_llm_password: str = Field(validation_alias=AliasChoices("POSTGRES_LLM_PASSWORD"))
    postgres_db: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DB"))
    langfuse_public_key: str = Field(
        default="", validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY")
    )
    langfuse_secret_key: str = Field(
        default="", validation_alias=AliasChoices("LANGFUSE_SECRET_KEY")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, validated on first call."""
    # Fields are populated from the environment, not constructor args.
    return Settings()  # ty: ignore[missing-argument]
