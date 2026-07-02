"""Typed view over ``config.yaml`` for the agent service.

The bind-mounted ``config.yaml`` is the single source of truth for ports,
hostnames and LLM settings. This module parses the slice the agent actually
reads into Pydantic models so access is attribute-based and validated at
startup instead of silent ``dict.get(...)`` chains. Unknown keys (sections
owned by other services) are ignored.
"""

from pathlib import Path

from pydantic import BaseModel, model_validator
import yaml


class SharedConfig(BaseModel):
    """The ``shared`` section: OpenAI endpoint + model names."""

    openai_base_url: str = "https://api.openai.com/v1"
    openai_llm_model: str
    openai_llm_model_fast: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    @model_validator(mode="after")
    def _default_fast_model(self) -> "SharedConfig":
        """Fall back to the strong model when no fast model is configured."""
        if not self.openai_llm_model_fast:
            self.openai_llm_model_fast = self.openai_llm_model
        return self


class PostgresConfig(BaseModel):
    """The ``postgres`` section. ``database`` falls back to ``POSTGRES_DB``."""

    host: str = "db"
    port: int = 5432
    database: str | None = None


class QdrantConfig(BaseModel):
    """The ``qdrant`` section."""

    host: str = "vector_db"
    port: int = 6333


class PortConfig(BaseModel):
    """A bare ``{port: int}`` service block."""

    port: int


class AgentConfig(BaseModel):
    """The portion of ``config.yaml`` the agent service reads."""

    shared: SharedConfig
    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    python_sandbox: PortConfig
    downloader_extra: PortConfig


def load_config(path: Path) -> AgentConfig:
    """Parse and validate ``config.yaml`` into an :class:`AgentConfig`."""
    return AgentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
