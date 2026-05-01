"""Single entry-point for dashboard theming.

Loads `themes.yaml` (master config), exposes color tokens for use across pages,
registers a Plotly template named "app" derived from the active theme, and syncs
the active theme's colors into `app/.streamlit/config.toml` so Streamlit picks
them up on the next server restart.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yaml


THEMES_FILENAME = "themes.yaml"
STREAMLIT_CONFIG_RELATIVE = Path(".streamlit") / "config.toml"
PLOTLY_TEMPLATE_NAME = "app"


def _candidate_themes_paths() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        Path.cwd() / THEMES_FILENAME,
        Path.cwd().parent / "_container_data" / THEMES_FILENAME,
        here.parent.parent.parent / "_container_data" / THEMES_FILENAME,
        Path("/app") / THEMES_FILENAME,
    ]


def _resolve_themes_path() -> Path:
    for candidate in _candidate_themes_paths():
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {THEMES_FILENAME}. Tried: "
        + ", ".join(str(p) for p in _candidate_themes_paths())
    )


def _candidate_streamlit_config_paths() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        Path.cwd() / STREAMLIT_CONFIG_RELATIVE,
        here.parent.parent / STREAMLIT_CONFIG_RELATIVE,
        Path("/app") / STREAMLIT_CONFIG_RELATIVE,
    ]


def _resolve_streamlit_config_path() -> Path:
    for candidate in _candidate_streamlit_config_paths():
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate Streamlit config.toml. Tried: "
        + ", ".join(str(p) for p in _candidate_streamlit_config_paths())
    )


@st.cache_data(show_spinner=False)
def load_themes() -> dict:
    path = _resolve_themes_path()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "themes" not in data or "active" not in data:
        raise ValueError(
            f"{path} must contain top-level 'active' and 'themes' keys."
        )
    return data


def _invalidate_cache() -> None:
    load_themes.clear()


def list_theme_names() -> list[str]:
    return list(load_themes()["themes"].keys())


def get_active_theme_name() -> str:
    return str(load_themes()["active"])


def get_active_theme() -> dict:
    data = load_themes()
    name = data["active"]
    themes = data["themes"]
    if name not in themes:
        raise KeyError(
            f"Active theme '{name}' is not defined. Available: {list(themes.keys())}"
        )
    return themes[name]


def get_color(token: str) -> str:
    """Return a hex color for a semantic token in the active theme.

    Raises KeyError on unknown token — fail loud, no silent fallback.
    """
    semantic = get_active_theme().get("semantic") or {}
    if token not in semantic:
        raise KeyError(
            f"Unknown color token '{token}'. Defined in active theme: "
            f"{sorted(semantic.keys())}"
        )
    return str(semantic[token])


def get_colorway() -> list[str]:
    return list(get_active_theme().get("plotly", {}).get("colorway") or [])


def register_plotly_template() -> None:
    """Build a Plotly template from the active theme and set it as default."""
    plotly_cfg: dict[str, Any] = get_active_theme().get("plotly") or {}
    base_name = str(plotly_cfg.get("template_base", "plotly"))
    if base_name not in pio.templates:
        raise ValueError(f"Unknown base Plotly template: {base_name}")
    base_template = pio.templates[base_name]

    template = go.layout.Template(base_template.to_plotly_json())
    layout_overrides: dict[str, Any] = {}
    if "paper_bgcolor" in plotly_cfg:
        layout_overrides["paper_bgcolor"] = plotly_cfg["paper_bgcolor"]
    if "plot_bgcolor" in plotly_cfg:
        layout_overrides["plot_bgcolor"] = plotly_cfg["plot_bgcolor"]
    if "font_color" in plotly_cfg:
        layout_overrides["font"] = {"color": plotly_cfg["font_color"]}
    if "colorway" in plotly_cfg:
        layout_overrides["colorway"] = list(plotly_cfg["colorway"])

    template.layout.update(layout_overrides)
    pio.templates[PLOTLY_TEMPLATE_NAME] = template
    pio.templates.default = PLOTLY_TEMPLATE_NAME


def _format_streamlit_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return f"\"{value}\""


def _sync_streamlit_config(theme_name: str) -> Path:
    """Rewrite the [theme] block of .streamlit/config.toml to match the named theme."""
    data = load_themes()
    if theme_name not in data["themes"]:
        raise KeyError(f"Unknown theme: {theme_name}")
    streamlit_cfg = data["themes"][theme_name].get("streamlit") or {}

    config_path = _resolve_streamlit_config_path()
    text = config_path.read_text(encoding="utf-8")

    new_block_lines = ["[theme]"]
    for key, value in streamlit_cfg.items():
        new_block_lines.append(f"{key} = {_format_streamlit_value(value)}")
    new_block = "\n".join(new_block_lines)

    pattern = re.compile(
        r"^\[theme\][^\[]*", flags=re.MULTILINE
    )
    if pattern.search(text):
        replaced = pattern.sub(new_block + "\n\n", text, count=1)
    else:
        replaced = text.rstrip() + "\n\n" + new_block + "\n"

    config_path.write_text(replaced, encoding="utf-8")
    return config_path


def set_active_theme(name: str) -> Path:
    """Persist the active theme name and sync Streamlit config.

    Returns the path to the rewritten Streamlit config so callers can show it.
    Caller is responsible for prompting a server restart — Plotly template
    re-registers on next app boot, and Streamlit re-reads config.toml on boot.
    """
    data = load_themes()
    if name not in data["themes"]:
        raise KeyError(
            f"Unknown theme '{name}'. Available: {list(data['themes'].keys())}"
        )

    themes_path = _resolve_themes_path()
    data["active"] = name
    with open(themes_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    _invalidate_cache()
    return _sync_streamlit_config(name)
