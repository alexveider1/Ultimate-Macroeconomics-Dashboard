import json
from typing import Any, Callable

import streamlit as st

from core.app_logging import log_page_render
from core.plotting import GraphBox
from core.postgres_client import (
    get_world_bank_country_codes,
    get_world_bank_country_mapping,
)

INDICATOR_CONFIG = "_configs/world_bank_download_config.json"
DEFAULT_COUNTRY_ALIASES = {
    "USA": "United States",
    "CHN": "China",
    "DEU": "Germany",
}
MAX_COUNTRY_SELECTION = 10


@st.cache_data(show_spinner=False)
def _load_indicator_config() -> dict[str, list[dict[str, Any]]]:
    try:
        with open(INDICATOR_CONFIG, "r", encoding="utf-8") as file:
            loaded = json.load(file)
    except Exception:
        return {}

    if isinstance(loaded, dict):
        return loaded

    return {}


def load_dashboard_config() -> dict[str, list[dict[str, Any]]]:
    return _load_indicator_config()


def _collect_items(
    config_data: dict[str, list[dict[str, Any]]], section_keys: list[str]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for section in section_keys:
        for item in config_data.get(section, []):
            if "id" in item and "name" in item:
                items.append(item)
    return items


def _resolve_default_countries(available_countries: list[str]) -> list[str]:
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
    state_key = f"{page_title}_countries"

    selected_countries = st.multiselect(
        "Countries for time trends",
        options=sorted_options,
        default=st.session_state.get(state_key, default_countries),
        max_selections=MAX_COUNTRY_SELECTION,
        format_func=lambda iso: label_by_iso.get(str(iso).upper(), str(iso).upper()),
        help=(
            "Applies to time trends and to reference lines in distribution plots. "
            "Distribution shape still uses global data."
        ),
        key=state_key,
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
