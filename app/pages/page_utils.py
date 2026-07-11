"""Shared rendering helpers used by every indicator-style dashboard page.

A page that displays a set of World Bank indicators only needs to call
:func:`render_page_from_config` with its title and the list of section
keys it wants from ``_configs/world_bank_download_config.json``. This
module handles loading the config, rendering the country multi-select,
and instantiating one :class:`core.plotting.GraphBox` per indicator.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from core.app_logging import log_page_render
from core.plotting import GraphBox
from core.postgres_client import (
    get_world_bank_country_codes,
    get_world_bank_country_mapping,
)

logger = logging.getLogger(__name__)

INDICATOR_CONFIG = Path("_configs/world_bank_download_config.json")
DEFAULT_COUNTRY_ALIASES = {
    "USA": "United States",
    "CHN": "China",
    "DEU": "Germany",
}
MAX_COUNTRY_SELECTION = 10
SHARED_COUNTRIES_STATE_KEY = "wb_selected_countries"


def get_shared_selected_countries() -> list[str]:
    """Return the country ISO codes shared across every World Bank page.

    Pages read this in ``after_graphs_renderer`` callbacks instead of building
    their own per-page session keys, so the user's choice carries across the
    whole Dashboard section.
    """
    return list(st.session_state.get(SHARED_COUNTRIES_STATE_KEY, []) or [])


@st.cache_data(show_spinner=False)
def _load_indicator_config() -> dict[str, list[dict[str, Any]]]:
    """Load and cache the WB indicator config; return an empty dict on failure."""
    try:
        loaded = json.loads(INDICATOR_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load %s: %s", INDICATOR_CONFIG, exc)
        return {}

    if isinstance(loaded, dict):
        return loaded

    return {}


def load_dashboard_config() -> dict[str, list[dict[str, Any]]]:
    """Public alias used by the custom-plot page; returns the cached config dict."""
    return _load_indicator_config()


def _collect_items(
    config_data: dict[str, list[dict[str, Any]]], section_keys: list[str]
) -> list[dict[str, Any]]:
    """Flatten the indicators from ``section_keys`` into one list, in order.

    Args:
        config_data: Loaded indicator config (section -> list of item dicts).
        section_keys: Section names to pull from, in display order.

    Returns:
        List of ``{"id", "name", ...}`` dicts ready to feed ``GraphBox``.
    """
    items: list[dict[str, Any]] = []
    for section in section_keys:
        for item in config_data.get(section, []):
            if "id" in item and "name" in item:
                items.append(item)
    return items


def _resolve_default_countries(available_countries: list[str]) -> list[str]:
    """Pick the subset of :data:`DEFAULT_COUNTRY_ALIASES` actually present in the DB."""
    if not available_countries:
        return []

    normalized_to_original = {
        str(code).strip().upper(): str(code).strip() for code in available_countries
    }

    defaults: list[str] = []
    for iso_code in DEFAULT_COUNTRY_ALIASES:
        key = iso_code.strip().upper()
        if key in normalized_to_original:
            defaults.append(normalized_to_original[key])
    return defaults


def render_country_selector(page_title: str) -> list[str]:
    """Render the country multiselect shared across every World Bank page.

    All pages persist the selection under the single session-state key
    :data:`SHARED_COUNTRIES_STATE_KEY`, so switching between Dashboard pages
    keeps the same countries selected. Labels are pretty-printed
    (``"Germany (DEU)"``) when the country lookup table is available.

    Args:
        page_title: Ignored beyond logging — kept for backwards compatibility
            with callers that still pass it.

    Returns:
        List of selected ISO codes (capped at
        :data:`MAX_COUNTRY_SELECTION`).
    """
    del page_title  # the selection is shared across pages — no per-page key
    country_options = get_world_bank_country_codes()
    country_mapping_df = get_world_bank_country_mapping()
    label_by_iso: dict[str, str] = {}
    if not country_mapping_df.is_empty() and {"id", "value"}.issubset(
        set(country_mapping_df.columns)
    ):
        for row in country_mapping_df.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            name = str(row.get("value", "")).strip()
            if iso and name:
                label_by_iso[iso] = f"{name} ({iso})"

    sorted_options = sorted(country_options)
    default_countries = _resolve_default_countries(country_options)

    selected_countries = st.multiselect(
        "Countries for time trends",
        options=sorted_options,
        default=st.session_state.get(SHARED_COUNTRIES_STATE_KEY, default_countries),
        max_selections=MAX_COUNTRY_SELECTION,
        format_func=lambda iso: label_by_iso.get(str(iso).upper(), str(iso).upper()),
        help=(
            "Shared across all Dashboard pages. Applies to time trends and to "
            "reference lines in distribution plots; distribution shape still "
            "uses global data."
        ),
        key=SHARED_COUNTRIES_STATE_KEY,
    )

    if len(selected_countries) > MAX_COUNTRY_SELECTION:
        selected_countries = selected_countries[:MAX_COUNTRY_SELECTION]
    return selected_countries


def render_page_from_config(
    page_title: str,
    section_keys: list[str],
    caption: str | None = None,
    before_graphs_renderer: Callable[[], None] | None = None,
    after_graphs_renderer: Callable[[], None] | None = None,
) -> None:
    """Render a standard indicator-style page end-to-end.

    Steps: log the render, write the title and optional caption, pull
    the relevant indicators from the config, render the country selector,
    optionally call ``before_graphs_renderer``, render one ``GraphBox``
    per indicator, and optionally call ``after_graphs_renderer``.

    Args:
        page_title: Page heading; also used to namespace session state.
        section_keys: Section names to pull indicators from.
        caption: Optional one-line description shown under the title.
        before_graphs_renderer: Optional callback rendered after the
            country selector and before the indicator cards.
        after_graphs_renderer: Optional callback rendered after the
            indicator cards.
    """
    log_page_render(page_title)
    st.title(page_title)
    if caption:
        st.caption(caption)

    items = _collect_items(_load_indicator_config(), section_keys)

    if not items:
        st.warning("No indicators found for this page in config.json")
        return

    selected_countries = render_country_selector(page_title)

    if before_graphs_renderer:
        before_graphs_renderer()

    for item in items:
        GraphBox(
            item_config=item,
            selected_countries=selected_countries,
        ).render_streamlit_ui()

    if after_graphs_renderer:
        after_graphs_renderer()
