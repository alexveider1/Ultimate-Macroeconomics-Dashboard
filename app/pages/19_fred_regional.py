"""US state indicators (FRED) — the first Regional Statistics page.

Renders the state-level indicators ingested by ``downloader_general`` from the
FRED / GeoFRED regional API: an indicator picker (grouped by category), a
choropleth of the 50 states + DC for a chosen year, top/bottom state rankings,
snapshot metric tiles, and a multi-state time-trend chart. Everything is driven
from the cached Polars frames in ``core.postgres_client`` and the indicator
catalogue in the ``state_indicators`` table, so adding an indicator to
``fred_download_config.json`` surfaces it here automatically after the next
ingestion.
"""

import polars as pl
import streamlit as st

from core.app_logging import log_page_render
from core.plotting import (
    build_region_ranking_bar,
    build_region_trend_lines,
    build_us_state_choropleth,
)
from core.postgres_client import (
    get_fred_indicator,
    get_fred_indicators_catalog,
    get_fred_states,
)

FRED_KEY_PREFIX = "fred_regional"
DEFAULT_INDICATOR = "unemployment_rate"
DEFAULT_TREND_STATES = ["CA", "TX", "NY", "FL"]
MAX_TREND_STATES = 10


def _indicator_selector(catalog: pl.DataFrame) -> dict:
    """Render the category-grouped indicator selectbox; return the chosen row dict.

    Options are the indicator slugs, ordered by ``(category, name)`` and labelled
    ``"Category — Name"`` so the single control reads like a grouped menu.
    """
    ordered = catalog.sort(["category", "name"])
    slugs = ordered["indicator_id"].to_list()
    label_by_slug = {
        row["indicator_id"]: f"{row['category']} — {row['name']}" for row in ordered.to_dicts()
    }
    default_index = slugs.index(DEFAULT_INDICATOR) if DEFAULT_INDICATOR in slugs else 0

    selected = st.selectbox(
        "Indicator",
        options=slugs,
        index=default_index,
        format_func=lambda slug: label_by_slug.get(slug, slug),
        key=f"{FRED_KEY_PREFIX}_indicator",
    )
    return ordered.filter(pl.col("indicator_id") == selected).to_dicts()[0]


def _render_snapshot(cross: pl.DataFrame, units: str) -> None:
    """Show median / highest / lowest metric tiles for the selected year."""
    if cross.is_empty():
        return
    median = float(cross["value"].cast(pl.Float64).median() or 0.0)
    hi = cross.sort("value", descending=True).row(0, named=True)
    lo = cross.sort("value", descending=False).row(0, named=True)

    unit_suffix = f" {units}" if units and len(units) <= 12 else ""
    col_median, col_high, col_low = st.columns(3)
    col_median.metric("Median state", f"{median:,.2f}{unit_suffix}")
    col_high.metric("Highest", f"{hi['value']:,.2f}", help=hi["name"])
    col_high.caption(hi["name"])
    col_low.metric("Lowest", f"{lo['value']:,.2f}", help=lo["name"])
    col_low.caption(lo["name"])


def _render_map_and_rankings(
    indicator: dict, values_df: pl.DataFrame, states_df: pl.DataFrame
) -> None:
    """Render the year slider, choropleth, snapshot tiles and top/bottom rankings."""
    name = indicator["name"]
    units = indicator.get("units") or "Value"

    years = sorted({int(y) for y in values_df["year"].to_list()})
    if not years:
        st.info("No values available for this indicator.")
        return

    if len(years) > 1:
        selected_year = st.slider(
            "Year",
            min_value=years[0],
            max_value=years[-1],
            value=years[-1],
            key=f"{FRED_KEY_PREFIX}_year",
        )
    else:
        selected_year = years[0]
        st.caption(f"Only one year available: {selected_year}")

    cross = (
        values_df.filter(pl.col("year") == selected_year)
        .join(states_df.select(["id", "name", "region"]), left_on="state", right_on="id")
        .drop_nulls(subset=["value"])
    )
    if cross.is_empty():
        st.info(f"No state data for {name} in {selected_year}.")
        return

    _render_snapshot(cross, units)

    map_fig = build_us_state_choropleth(
        cross,
        state_col="state",
        val_col="value",
        title=f"{name} by State — {selected_year}",
        value_label=units,
        name_col="name",
        hover_context=f"{name} ({selected_year})",
    )
    st.plotly_chart(map_fig, width="stretch", key=f"{FRED_KEY_PREFIX}_map")

    left_col, right_col = st.columns(2)
    with left_col:
        top_fig = build_region_ranking_bar(
            cross,
            region_col="state",
            val_col="value",
            title=f"Top 10 states — {selected_year}",
            value_label=units,
            label_col="name",
            top_n=10,
            ascending=False,
        )
        st.plotly_chart(top_fig, width="stretch", key=f"{FRED_KEY_PREFIX}_top")
    with right_col:
        bottom_fig = build_region_ranking_bar(
            cross,
            region_col="state",
            val_col="value",
            title=f"Bottom 10 states — {selected_year}",
            value_label=units,
            label_col="name",
            top_n=10,
            ascending=True,
        )
        st.plotly_chart(bottom_fig, width="stretch", key=f"{FRED_KEY_PREFIX}_bottom")


def _render_trends(indicator: dict, values_df: pl.DataFrame, states_df: pl.DataFrame) -> None:
    """Render the multi-state time-trend chart with a state multiselect."""
    name = indicator["name"]
    units = indicator.get("units") or "Value"

    st.markdown("### State time trends")
    available = sorted(set(values_df["state"].to_list()))
    if not available:
        return
    name_by_state = {row["id"]: row["name"] for row in states_df.to_dicts()}
    default_states = [s for s in DEFAULT_TREND_STATES if s in available] or available[:3]

    selected_states = st.multiselect(
        f"States (max {MAX_TREND_STATES})",
        options=available,
        default=default_states,
        max_selections=MAX_TREND_STATES,
        format_func=lambda code: f"{name_by_state.get(code, code)} ({code})",
        key=f"{FRED_KEY_PREFIX}_trend_states",
    )
    if not selected_states:
        st.info("Select at least one state to display the trend chart.")
        return

    trend_fig = build_region_trend_lines(
        values_df,
        region_col="state",
        year_col="year",
        val_col="value",
        regions=selected_states,
        title=f"{name} over time",
        value_label=units,
        label_by_region=name_by_state,
    )
    st.plotly_chart(trend_fig, width="stretch", key=f"{FRED_KEY_PREFIX}_trend")


def _render_about(indicator: dict) -> None:
    """Render the indicator metadata (units, frequency, coverage, notes) in an expander."""
    with st.expander("About this indicator"):
        st.markdown(
            f"**{indicator['name']}** &nbsp;·&nbsp; category: *{indicator['category']}*\n\n"
            f"- **Units:** {indicator.get('units') or 'n/a'}\n"
            f"- **Native frequency:** {indicator.get('frequency') or 'n/a'} "
            f"({indicator.get('seasonal_adjustment') or 'NSA'}); shown as annual values\n"
            f"- **Coverage:** {indicator.get('min_date') or '?'} → "
            f"{indicator.get('max_date') or '?'}\n"
            f"- **FRED series group:** {indicator.get('series_group') or 'n/a'} "
            f"(example series `{indicator.get('example_series_id') or 'n/a'}`)\n"
        )
        if indicator.get("notes"):
            st.caption(indicator["notes"])


def render_fred_regional() -> None:
    """Page entry-point: indicator picker, choropleth, rankings, and trends."""
    log_page_render("US Regional Statistics (FRED)")
    st.title("United States — Regional Statistics (FRED)")
    st.caption(
        "State-level indicators from the Federal Reserve (FRED) regional data: "
        "unemployment, GDP, income, housing, sector employment and more, across "
        "the 50 states and DC. Annual values aggregated from the native FRED frequency."
    )

    with st.container(border=True):
        catalog = get_fred_indicators_catalog()
        states_df = get_fred_states()

        if catalog.is_empty() or states_df.is_empty():
            st.info(
                "No FRED regional data available. It is ingested on a clean boot of the "
                "`downloader_general` container (needs a `FRED_API_KEY`)."
            )
            return

        indicator = _indicator_selector(catalog)
        values_df = get_fred_indicator(indicator["indicator_id"])

        if values_df.is_empty():
            st.info(f"No values stored for {indicator['name']}.")
            return

        _render_map_and_rankings(indicator, values_df, states_df)
        _render_trends(indicator, values_df, states_df)
        _render_about(indicator)


render_fred_regional()
