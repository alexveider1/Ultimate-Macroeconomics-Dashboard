"""Typed view over the ``forecaster`` section of ``config.yaml``.

Replaces the previous ``yaml.safe_load(...).get("forecaster", {})`` dict access
with a validated model so the heavy-dependency toggles and the Chronos model
name are checked at startup instead of silently defaulting.
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


class ForecasterConfig(BaseModel):
    """The portion of ``config.yaml`` the forecaster service reads."""

    forecaster: ForecasterSection = ForecasterSection()


def load_config(path: Path) -> ForecasterConfig:
    """Parse and validate ``config.yaml`` into a :class:`ForecasterConfig`."""
    return ForecasterConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
