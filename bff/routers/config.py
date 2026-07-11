"""Config endpoints: theme palettes + the dashboard indicator config.

Serves the frontend's runtime configuration. The theme file is re-read on every
request (it is tiny, and reading it live lets a palette edit take effect without
a BFF restart — the "swap ``active:`` and re-fetch" flow). The dashboard config
is the same ``world_bank_download_config.json`` that drives the Streamlit
indicator pages, so the React ``<IndicatorPage>`` stays config-driven too.
"""

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from models import ActiveThemeOut, ThemesConfigOut
from utils import resolve_data_file
import yaml

router = APIRouter(prefix="/config", tags=["config"])

_UI_THEMES_FILE = "ui_themes.yaml"
_DASHBOARD_FILE = "_configs/world_bank_download_config.json"


def _load_ui_themes() -> dict[str, Any]:
    """Load + minimally validate ``ui_themes.yaml`` (``active`` + ``themes`` keys)."""
    path = resolve_data_file(_UI_THEMES_FILE)
    if path is None:
        raise HTTPException(status_code=503, detail=f"{_UI_THEMES_FILE} is not available")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "active" not in data or "themes" not in data:
        raise HTTPException(
            status_code=500, detail=f"{_UI_THEMES_FILE} must contain 'active' and 'themes' keys"
        )
    return data


@router.get("/themes", response_model=ThemesConfigOut)
def get_themes() -> ThemesConfigOut:
    """Return every theme + the active one's name (drives the runtime switcher)."""
    data = _load_ui_themes()
    return ThemesConfigOut(active=str(data["active"]), themes=dict(data["themes"]))


@router.get("/theme", response_model=ActiveThemeOut)
def get_active_theme() -> ActiveThemeOut:
    """Return only the active theme's tokens (small initial payload)."""
    data = _load_ui_themes()
    active = str(data["active"])
    themes = data["themes"]
    if active not in themes:
        raise HTTPException(status_code=500, detail=f"Active theme '{active}' is not defined")
    return ActiveThemeOut(name=active, theme=dict(themes[active]))


@router.get("/dashboard", response_model=dict[str, Any])
def get_dashboard_config() -> dict[str, Any]:
    """Return the World Bank page→section→indicator mapping as raw JSON."""
    path = resolve_data_file(_DASHBOARD_FILE)
    if path is None:
        raise HTTPException(status_code=503, detail=f"{_DASHBOARD_FILE} is not available")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Malformed dashboard config: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HTTPException(status_code=500, detail="Dashboard config must be a JSON object")
    return loaded
