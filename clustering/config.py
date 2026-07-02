"""Typed view over the ``clustering`` section of ``config.yaml``.

The clustering service operates purely on the tabular payload it receives, so
it reads no secrets and only a port from config. Parsing through a model keeps
startup validation consistent with the other services.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class ClusteringSection(BaseModel):
    """The ``clustering`` block."""

    port: int = 8002


class ClusteringConfig(BaseModel):
    """The portion of ``config.yaml`` the clustering service reads."""

    clustering: ClusteringSection = ClusteringSection()


def load_config(path: Path) -> ClusteringConfig:
    """Parse and validate ``config.yaml`` into a :class:`ClusteringConfig`."""
    return ClusteringConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
