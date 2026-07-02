"""Loader for the named HTML/markdown snippets used to render chart cards.

The snippets live in ``app/assets/plot_markup_templates.json`` and are
``string.Template`` strings with named ``${placeholder}`` substitutions
(themed colour tokens, captions, ...). Each page calls
:func:`render_markup_template` to inject runtime values.
"""

from functools import lru_cache
import json
from pathlib import Path
from string import Template
from typing import Any

_ASSETS_DIR = Path("assets")
_PLOT_MARKUP_TEMPLATES_PATH = _ASSETS_DIR / "plot_markup_templates.json"


@lru_cache(maxsize=1)
def _load_plot_markup_templates() -> dict[str, str]:
    """Read the JSON file once and cache the ``name -> template`` mapping."""
    payload = json.loads(_PLOT_MARKUP_TEMPLATES_PATH.read_text(encoding="utf-8"))
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def get_markup_template(name: str) -> str:
    """Return the raw template string registered under ``name``.

    Args:
        name: Template id (e.g. ``card_with_title``).

    Returns:
        Raw template string.

    Raises:
        KeyError: When the template id is unknown.
    """
    templates = _load_plot_markup_templates()
    if name not in templates:
        raise KeyError(f"Unknown markup template: {name}")
    return templates[name]


def render_markup_template(name: str, **substitutions: Any) -> str:
    """Substitute the keyword arguments into the template and return the result.

    Args:
        name: Template id.
        **substitutions: Values for every ``${placeholder}`` in the template.

    Returns:
        The fully rendered string.
    """
    return Template(get_markup_template(name)).substitute(substitutions)
