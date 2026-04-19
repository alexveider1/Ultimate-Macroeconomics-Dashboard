import os
import json
from functools import lru_cache
from string import Template
from typing import Any


_ASSETS_DIR = "assets"
_PLOT_MARKUP_TEMPLATES_PATH = os.path.join(_ASSETS_DIR, "plot_markup_templates.json")


@lru_cache(maxsize=1)
def _load_plot_markup_templates() -> dict[str, str]:
    with open(_PLOT_MARKUP_TEMPLATES_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def get_markup_template(name: str) -> str:
    templates = _load_plot_markup_templates()
    if name not in templates:
        raise KeyError(f"Unknown markup template: {name}")
    return templates[name]


def render_markup_template(name: str, **substitutions: Any) -> str:
    return Template(get_markup_template(name)).substitute(substitutions)
