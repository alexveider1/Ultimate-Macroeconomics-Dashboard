"""Typed secrets for the docling service.

PDF conversion offloads its OCR/VLM inference to a cloud OpenAI-compatible
endpoint (``docling.vlm`` in ``config.yaml``), authenticated with the shared
``OPENAI_API_KEY`` — the same key the agent LLM and the chat's Whisper voice
input use. The field is optional so the service still imports (and Office-format
conversion still works) when the key is absent.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets the docling service uses: the OpenAI-compatible OCR endpoint key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, read from env + ``.env``."""
    return Settings()
