"""Typed secrets for the Streamlit dashboard.

The app reads with the LLM (read-only) Postgres role for page queries, talks to
Qdrant for the news / RAG pages, and calls the OpenAI-compatible Whisper endpoint
to transcribe voice / audio attachments in the AI chat — so it needs the
read-only Postgres creds, the Qdrant key, and the OpenAI key (but never the
superuser role: token/cost accounting lives in Langfuse, not a Postgres table).
Every field is optional so the app degrades gracefully (pages surface a
connection error) instead of failing to import when a secret is absent.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets the dashboard uses: the read-only Postgres role, Qdrant + OpenAI keys."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_llm_user: str = Field(default="", validation_alias=AliasChoices("POSTGRES_LLM_USER"))
    postgres_llm_password: str = Field(
        default="", validation_alias=AliasChoices("POSTGRES_LLM_PASSWORD")
    )
    postgres_db: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DB"))
    qdrant_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "QDRANT_API_KEY", "QDRANT__API_KEY", "QDRANT__SERVICE__API_KEY"
        ),
    )
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, read from env + ``.env``."""
    return Settings()
