"""Typed view over the ``python_sandbox`` section of ``config.yaml``.

The sandbox executor reads no secrets and only a port from config; parsing
through a model keeps startup validation consistent across services.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class PythonSandboxSection(BaseModel):
    """The ``python_sandbox`` block."""

    port: int = 8004


class PythonSandboxConfig(BaseModel):
    """The portion of ``config.yaml`` the sandbox service reads."""

    python_sandbox: PythonSandboxSection = PythonSandboxSection()


def load_config(path: Path) -> PythonSandboxConfig:
    """Parse and validate ``config.yaml`` into a :class:`PythonSandboxConfig`."""
    return PythonSandboxConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
