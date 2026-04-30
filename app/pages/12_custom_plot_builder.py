import streamlit as st

from core.app_logging import log_page_render
from core.plotting import GraphBox
from pages.page_utils import load_dashboard_config, render_country_selector


log_page_render("Custom Plot Constructor")
st.title("Custom Plot Constructor")
st.caption(
    "Each item renders as a two-panel dashboard: map on the left, and a selector-driven chart (time trend or distribution) on the right."
)

config_data = load_dashboard_config()
sections = list(config_data.keys())

selected_countries = render_country_selector("custom_plot_builder")

selected_section = st.selectbox(
    "Select category",
    options=sections,
    index=0 if sections else None,
)

available_items = config_data.get(selected_section, [])

if not available_items:
    st.warning("No indicators available for the selected category.")
    st.stop()

indicator_indices = list(range(len(available_items)))


def _format_indicator(index: int) -> str:
    item = available_items[index]
    item_id = str(item.get("id", "")).strip()
    item_name = str(item.get("name", "")).strip()

    if item_id and item_name:
        return f"{item_name} ({item_id})"
    if item_name:
        return item_name
    if item_id:
        return item_id
    return f"Indicator {index + 1}"


selected_indicator_index = st.selectbox(
    "Select indicator",
    options=indicator_indices,
    index=0,
    format_func=_format_indicator,
)

selected_item = available_items[selected_indicator_index]

GraphBox(
    item_config=selected_item,
    selected_countries=selected_countries,
).render_streamlit_ui()
