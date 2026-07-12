"""Typed view over the ``forecaster`` + ``triton`` sections of ``config.yaml``.

Replaces the previous ``yaml.safe_load(...).get("forecaster", {})`` dict access
with a validated model so the heavy-dependency toggles, the Chronos model name,
and the Triton endpoint are checked at startup instead of silently defaulting.
"""

from pathlib import Path

from pydantic import BaseModel
import yaml


class ForecasterSection(BaseModel):
    """The ``forecaster`` block: port + heavy-dependency family toggles."""

    port: int = 8001
    ARIMA_AVAILABLE: bool = False
    PROPHET_AVAILABLE: bool = False
    CHRONOS_AVAILABLE: bool = False
    CHRONOS_MODEL: str | None = None


class TritonSection(BaseModel):
    """The ``triton`` block: gRPC/HTTP endpoint the adapter forwards to."""

    host: str = "triton"
    grpc_port: int = 8001
    http_port: int = 8000


class ForecasterConfig(BaseModel):
    """The portion of ``config.yaml`` the forecaster adapter reads."""

    forecaster: ForecasterSection = ForecasterSection()
    triton: TritonSection = TritonSection()


def load_config(path: Path) -> ForecasterConfig:
    """Parse and validate ``config.yaml`` into a :class:`ForecasterConfig`."""
    return ForecasterConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
