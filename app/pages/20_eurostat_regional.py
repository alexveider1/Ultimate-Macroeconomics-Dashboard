"""EU NUTS-2 regional indicators (Eurostat) — the second Regional Statistics page.

Renders the region-level indicators ingested by ``downloader_general`` from the
Eurostat dissemination (JSON-stat) API: a category-grouped indicator picker, a
NUTS-2 choropleth of Europe (drawn from the bundled GISCO GeoJSON), top/bottom
region rankings, snapshot metric tiles, and a multi-region time-trend chart.
Everything is driven from the cached Polars frames in ``core.postgres_client`` and
the indicator catalogue in the ``eurostat_indicators`` table, so adding an
indicator to ``eurostat_download_config.json`` surfaces it here automatically
after the next ingestion.
"""

from core.app_logging import log_page_render
from core.assets import load_nuts_geojson
from core.plotting import (
    build_nuts_choropleth,
    build_region_ranking_bar,
    build_region_trend_lines,
)
from core.postgres_client import (
    get_eurostat_indicator,
    get_eurostat_indicators_catalog,
    get_eurostat_regions,
)
import polars as pl
import streamlit as st

EUROSTAT_KEY_PREFIX = "eurostat_regional"
DEFAULT_INDICATOR = "gdp_per_capita_pps"
# A spread of large, recognisable NUTS-2 regions used as the default trend lines.
DEFAULT_TREND_REGIONS = ["DE21", "FR10", "ES30", "ITC4", "PL91"]
MAX_TREND_REGIONS = 10


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
        key=f"{EUROSTAT_KEY_PREFIX}_indicator",
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
    col_median.metric("Median region", f"{median:,.2f}{unit_suffix}")
    col_high.metric("Highest", f"{hi['value']:,.2f}", help=hi["name"])
    col_high.caption(f"{hi['name']} ({hi['country_name']})")
    col_low.metric("Lowest", f"{lo['value']:,.2f}", help=lo["name"])
    col_low.caption(f"{lo['name']} ({lo['country_name']})")


def _render_map_and_rankings(
    indicator: dict, values_df: pl.DataFrame, regions_df: pl.DataFrame
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
            key=f"{EUROSTAT_KEY_PREFIX}_year",
        )
    else:
        selected_year = years[0]
        st.caption(f"Only one year available: {selected_year}")

    cross = (
        values_df.filter(pl.col("year") == selected_year)
        .join(
            regions_df.select(["id", "name", "country_code", "country_name"]),
            left_on="region",
            right_on="id",
        )
        .drop_nulls(subset=["value"])
    )
    if cross.is_empty():
        st.info(f"No regional data for {name} in {selected_year}.")
        return

    _render_snapshot(cross, units)

    map_fig = build_nuts_choropleth(
        cross,
        region_col="region",
        val_col="value",
        geojson=load_nuts_geojson(),
        title=f"{name} by NUTS-2 region — {selected_year}",
        value_label=units,
        name_col="name",
        hover_context=f"{name} ({selected_year})",
    )
    st.plotly_chart(map_fig, width="stretch", key=f"{EUROSTAT_KEY_PREFIX}_map")

    left_col, right_col = st.columns(2)
    with left_col:
        top_fig = build_region_ranking_bar(
            cross,
            region_col="region",
            val_col="value",
            title=f"Top 10 regions — {selected_year}",
            value_label=units,
            label_col="name",
            top_n=10,
            ascending=False,
        )
        st.plotly_chart(top_fig, width="stretch", key=f"{EUROSTAT_KEY_PREFIX}_top")
    with right_col:
        bottom_fig = build_region_ranking_bar(
            cross,
            region_col="region",
            val_col="value",
            title=f"Bottom 10 regions — {selected_year}",
            value_label=units,
            label_col="name",
            top_n=10,
            ascending=True,
        )
        st.plotly_chart(bottom_fig, width="stretch", key=f"{EUROSTAT_KEY_PREFIX}_bottom")


def _render_trends(indicator: dict, values_df: pl.DataFrame, regions_df: pl.DataFrame) -> None:
    """Render the multi-region time-trend chart with a region multiselect."""
    name = indicator["name"]
    units = indicator.get("units") or "Value"

    st.markdown("### Regional time trends")
    available = sorted(set(values_df["region"].to_list()))
    if not available:
        return
    name_by_region = {row["id"]: row["name"] for row in regions_df.to_dicts()}
    country_by_region = {row["id"]: row["country_name"] for row in regions_df.to_dicts()}
    default_regions = [r for r in DEFAULT_TREND_REGIONS if r in available] or available[:3]

    def _fmt(code: str) -> str:
        country = country_by_region.get(code)
        label = name_by_region.get(code, code)
        return f"{label} ({country})" if country else f"{label} ({code})"

    selected_regions = st.multiselect(
        f"Regions (max {MAX_TREND_REGIONS})",
        options=available,
        default=default_regions,
        max_selections=MAX_TREND_REGIONS,
        format_func=_fmt,
        key=f"{EUROSTAT_KEY_PREFIX}_trend_regions",
    )
    if not selected_regions:
        st.info("Select at least one region to display the trend chart.")
        return

    trend_fig = build_region_trend_lines(
        values_df,
        region_col="region",
        year_col="year",
        val_col="value",
        regions=selected_regions,
        title=f"{name} over time",
        value_label=units,
        label_by_region=name_by_region,
    )
    st.plotly_chart(trend_fig, width="stretch", key=f"{EUROSTAT_KEY_PREFIX}_trend")


def _render_about(indicator: dict) -> None:
    """Render the indicator metadata (dataset, filters, units, coverage) in an expander."""
    with st.expander("About this indicator"):
        st.markdown(
            f"**{indicator['name']}** &nbsp;·&nbsp; category: *{indicator['category']}*\n\n"
            f"- **Units:** {indicator.get('units') or 'n/a'}\n"
            f"- **Frequency:** {indicator.get('frequency') or 'Annual'}\n"
            f"- **Coverage:** {indicator.get('min_year') or '?'} → "
            f"{indicator.get('max_year') or '?'}\n"
            f"- **Eurostat dataset:** `{indicator.get('dataset') or 'n/a'}` "
            f"(filters: `{indicator.get('filters') or '{}'}`)\n"
        )
        if indicator.get("notes"):
            st.caption(indicator["notes"])


def render_eurostat_regional() -> None:
    """Page entry-point: indicator picker, choropleth, rankings, and trends."""
    log_page_render("EU Regional Statistics (Eurostat)")
    st.title("European Union — Regional Statistics (Eurostat)")
    st.caption(
        "NUTS-2 region-level indicators from Eurostat: GDP, unemployment, "
        "population, life expectancy, R&D and more, across the EU, EFTA and "
        "candidate-country regions. Annual values. Boundaries © EuroGeographics "
        "(GISCO); data © Eurostat."
    )

    with st.container(border=True):
        catalog = get_eurostat_indicators_catalog()
        regions_df = get_eurostat_regions()

        if catalog.is_empty() or regions_df.is_empty():
            st.info(
                "No Eurostat regional data available. It is ingested on a clean boot of "
                "the `downloader_general` container (keyless — no API key required)."
            )
            return

        indicator = _indicator_selector(catalog)
        values_df = get_eurostat_indicator(indicator["indicator_id"])

        if values_df.is_empty():
            st.info(f"No values stored for {indicator['name']}.")
            return

        _render_map_and_rankings(indicator, values_df, regions_df)
        _render_trends(indicator, values_df, regions_df)
        _render_about(indicator)


render_eurostat_regional()
