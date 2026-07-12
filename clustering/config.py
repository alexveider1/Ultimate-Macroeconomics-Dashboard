"""Typed view over the ``clustering`` + ``triton`` sections of ``config.yaml``.

The clustering adapter operates purely on the tabular payload it receives, so it
reads no secrets — only its own port and the Triton endpoint it forwards to.
Parsing through a model keeps startup validation consistent with the other
services.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class ClusteringSection(BaseModel):
    """The ``clustering`` block."""

    port: int = 8002


class TritonSection(BaseModel):
    """The ``triton`` block: gRPC/HTTP endpoint the adapter forwards to."""

    host: str = "triton"
    grpc_port: int = 8001
    http_port: int = 8000


class ClusteringConfig(BaseModel):
    """The portion of ``config.yaml`` the clustering adapter reads."""

    clustering: ClusteringSection = ClusteringSection()
    triton: TritonSection = TritonSection()


def load_config(path: Path) -> ClusteringConfig:
    """Parse and validate ``config.yaml`` into a :class:`ClusteringConfig`."""
    return ClusteringConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
