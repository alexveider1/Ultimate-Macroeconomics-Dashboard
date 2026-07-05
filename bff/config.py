"""Typed view over the slice of ``config.yaml`` the BFF service reads.

The BFF is a read-only backend-for-frontend: it serves ORM reads of the
Postgres tables that ``downloader_general`` populates, a semantic search over
the Qdrant news corpus, and thin proxies to the forecaster / clustering / agent
services. It therefore needs the ``postgres`` + ``qdrant`` connection blocks,
the shared OpenAI embedding settings (to embed news-search queries), its own
port, and the ports of the three services it proxies (to build their in-network
base URLs).
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class PostgresConfig(BaseModel):
    """The ``postgres`` section. ``database`` falls back to ``POSTGRES_DB``."""

    host: str = "db"
    port: int = 5432
    database: str | None = None


class QdrantConfig(BaseModel):
    """The ``qdrant`` section (news / RAG vector store)."""

    host: str = "vector_db"
    port: int = 6333


class SharedConfig(BaseModel):
    """The subset of the ``shared`` block used for news-query embeddings."""

    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "openai/text-embedding-3-small"


class PortConfig(BaseModel):
    """A bare ``{port: int}`` service block."""

    port: int


class BffPortConfig(BaseModel):
    """The ``bff`` service block (defaults to 8005)."""

    port: int = 8005


class LangfuseConfig(BaseModel):
    """The ``langfuse`` section — news-embedding tracing knobs.

    Keys are secrets (read via :class:`Settings`); only these non-secret knobs
    live in ``config.yaml``. Absent/``enabled: false`` makes tracing a no-op.
    """

    enabled: bool = False
    host: str = "http://langfuse_web:3000"
    environment: str = "dev"
    sample_rate: float = 1.0


class BffConfig(BaseModel):
    """The portion of ``config.yaml`` the BFF service reads."""

    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    shared: SharedConfig = SharedConfig()
    bff: BffPortConfig = BffPortConfig()
    agent: PortConfig = PortConfig(port=8000)
    forecaster: PortConfig = PortConfig(port=8001)
    clustering: PortConfig = PortConfig(port=8002)
    langfuse: LangfuseConfig = LangfuseConfig()


def load_config(path: Path) -> BffConfig:
    """Parse and validate ``config.yaml`` into a :class:`BffConfig`."""
    return BffConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
