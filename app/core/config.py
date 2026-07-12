"""Typed view over ``config.yaml`` for the Streamlit dashboard.

Covers the slice the app reads: Postgres + Qdrant connection details, the
per-service ports, and the AI chat's multimodal endpoints (``whisper`` for voice
transcription, ``docling`` for document conversion). Every section carries a
default so a partial or missing config still yields a usable object. Unknown
sections (``shared``, ``services``, …) are ignored.
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
    """The ``qdrant`` section."""

    host: str = "vector_db"
    port: int = 6333


class PortConfig(BaseModel):
    """A bare ``{port: int}`` service block."""

    port: int


class TritonConfig(BaseModel):
    """The ``triton`` block — only the HTTP port is read here."""

    host: str = "triton"
    http_port: int = 8000


class DoclingConfig(BaseModel):
    """The ``docling`` block — the document→Markdown service the chat calls for
    file uploads (only host + port are read; ``vlm`` etc. are the service's own)."""

    host: str = "docling"
    port: int = 8006


class WhisperConfig(BaseModel):
    """The ``whisper`` block — the OpenAI-compatible transcription endpoint the
    chat's voice / audio input uses (the API key is the shared OpenAI key)."""

    enabled: bool = True
    base_url: str = "https://api.openai.com/v1"
    model: str = "whisper-1"


class AppConfig(BaseModel):
    """The portion of ``config.yaml`` the dashboard reads."""

    postgres: PostgresConfig = PostgresConfig()
    qdrant: QdrantConfig = QdrantConfig()
    app: PortConfig = PortConfig(port=8501)
    agent: PortConfig = PortConfig(port=8000)
    forecaster: PortConfig = PortConfig(port=8001)
    clustering: PortConfig = PortConfig(port=8002)
    downloader_extra: PortConfig = PortConfig(port=8003)
    python_sandbox: PortConfig = PortConfig(port=8004)
    triton: TritonConfig = TritonConfig()
    docling: DoclingConfig = DoclingConfig()
    whisper: WhisperConfig = WhisperConfig()


def load_config(path: Path) -> AppConfig:
    """Parse and validate ``config.yaml`` into an :class:`AppConfig`."""
    return AppConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
