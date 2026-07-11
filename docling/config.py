"""Typed view over the ``docling`` section of ``config.yaml``.

The docling service converts uploaded documents (PDF via a remote VLM, Office
formats via native backends) to Markdown. PDF inference is offloaded to a cloud
OpenAI-compatible OCR/VLM endpoint, so this config carries the docling port +
that endpoint's base URL + model name (the API key is a secret in ``settings``).
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class VlmSection(BaseModel):
    """The ``docling.vlm`` block: the cloud OpenAI-compatible OCR endpoint.

    ``base_url`` is the OpenAI-compatible ``/v1`` base; the converter appends
    ``/chat/completions``. ``model`` is the OpenAI ``model`` field the endpoint
    expects (e.g. the served granite-docling model id).
    """

    base_url: str = "https://your-cloud-vllm-host/v1"
    model: str = "ibm-granite/granite-docling-258M"


class DoclingSection(BaseModel):
    """The ``docling`` block: port, per-request convert timeout, VLM endpoint."""

    port: int = 8006
    convert_timeout_seconds: int = 120
    vlm: VlmSection = VlmSection()


class DoclingConfig(BaseModel):
    """The portion of ``config.yaml`` the docling service reads."""

    docling: DoclingSection = DoclingSection()


def load_config(path: Path) -> DoclingConfig:
    """Parse and validate ``config.yaml`` into a :class:`DoclingConfig`."""
    return DoclingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
