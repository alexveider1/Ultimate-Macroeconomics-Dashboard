"""Typed view over the ``docling`` + ``triton`` sections of ``config.yaml``.

The docling service converts uploaded documents (PDF via a remote VLM, Office
formats via native backends) to Markdown. Its only external dependency is the
Triton OpenAI-compatible endpoint that serves the docling VLM, so this config
carries the docling port + the Triton host/OpenAI-port/model name.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class DoclingSection(BaseModel):
    """The ``docling`` block: port + per-request convert timeout."""

    port: int = 8006
    convert_timeout_seconds: int = 120


class TritonSection(BaseModel):
    """The ``triton`` block: the OpenAI-compatible endpoint + VLM model name.

    ``vlm_model`` is the **Triton model name** (the ``model_repository`` dir),
    which is what the OpenAI ``model`` field must carry — not the HF repo id.
    """

    host: str = "triton"
    openai_port: int = 9000
    vlm_model: str = "granite_docling"


class DoclingConfig(BaseModel):
    """The portion of ``config.yaml`` the docling service reads."""

    docling: DoclingSection = DoclingSection()
    triton: TritonSection = TritonSection()


def load_config(path: Path) -> DoclingConfig:
    """Parse and validate ``config.yaml`` into a :class:`DoclingConfig`."""
    return DoclingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
