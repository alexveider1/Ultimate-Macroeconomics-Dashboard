"""Trade and external sector page — imports/exports, balance, composition.

Around the standard cards this page adds an imports × exports scatter
(highlighting net importers vs exporters) and a deeper export-composition
view broken down by category.
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
)
from core.theming import get_color, get_diverging_colorscale
from pages.page_utils import get_shared_selected_countries, render_page_from_config

IMPORT_INDICATOR_ID = "NE.IMP.GNFS.ZS"
EXPORT_INDICATOR_ID = "NE.EXP.GNFS.ZS"


def _render_import_export_scatter() -> None:
    """Render the imports × exports (%-of-GDP) scatter block at the top of the page."""
    st.subheader("Imports vs Exports Scatter")
    st.caption(
        "Compares imports and exports of goods and services as percent of GDP for "
        "the same year and country."
    )

    imports_df = fetch_indicator_slice(IMPORT_INDICATOR_ID, value_col="imports_pct_gdp")
    exports_df = fetch_indicator_slice(EXPORT_INDICATOR_ID, value_col="exports_pct_gdp")

    if imports_df.is_empty() or exports_df.is_empty():
        st.info("Import-export scatterplot is unavailable because source data is empty.")
        st.divider()
        return

    joined_df = imports_df.join(exports_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping import and export values were found.")
        st.divider()
        return

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        country_map = country_map.select(
            [
                pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Utf8).alias("country_name"),
            ]
        )
        joined_df = joined_df.join(country_map, on="economy", how="left")

    joined_df = joined_df.with_columns(
        [
            pl.col("country_name").fill_null(pl.col("economy")).alias("country_name"),
            (pl.col("exports_pct_gdp") - pl.col("imports_pct_gdp")).alias("net_external_pct_gdp"),
        ]
    )

    year_options = joined_df.select("year").unique().sort("year").get_column("year").to_list()
    if not year_options:
        st.info("Import-export scatterplot is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Scatter year",
        options=year_options,
        value=year_options[-1],
        key="trade_scatter_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No import-export observations are available for this year.")
        st.divider()
        return

    plot_df = year_df.to_pandas()

    axis_min = float(
        min(
            plot_df["imports_pct_gdp"].min(),
            plot_df["exports_pct_gdp"].min(),
        )
    )
    axis_max = float(
        max(
            plot_df["imports_pct_gdp"].max(),
            plot_df["exports_pct_gdp"].max(),
        )
    )
    if axis_max <= axis_min:
        axis_max = axis_min + 1.0

    net_values = plot_df["net_external_pct_gdp"].to_numpy()
    net_abs_max = float(max(abs(net_values.min()), abs(net_values.max()), 1.0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["imports_pct_gdp"],
            y=plot_df["exports_pct_gdp"],
            mode="markers",
            name="Countries",
            marker={
                "color": net_values,
                "colorscale": get_diverging_colorscale(),
                "cmin": -net_abs_max,
                "cmid": 0.0,
                "cmax": net_abs_max,
                "showscale": True,
                "colorbar": {"title": "Exports - Imports (% of GDP)"},
                "size": 9,
                "opacity": 0.82,
                "line": {"width": 0.5},
            },
            text=plot_df["country_name"],
            customdata=plot_df[
                ["economy", "imports_pct_gdp", "exports_pct_gdp", "net_external_pct_gdp"]
            ].to_numpy(),
            hovertemplate=(
                "<b>%{text}</b> (%{customdata[0]})<br>"
                "Imports: %{customdata[1]:.2f}%<br>"
                "Exports: %{customdata[2]:.2f}%<br>"
                "Net: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="y = x",
            line={"color": get_color("reference_line"), "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"Imports vs Exports (% of GDP) in {selected_year}",
        xaxis_title="Imports (% of GDP)",
        yaxis_title="Exports (% of GDP)",
        xaxis_range=[axis_min, axis_max],
        yaxis_range=[axis_min, axis_max],
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Points above the dashed y=x line indicate stronger exports than imports "
        "relative to GDP; below the line indicates the opposite."
    )
    st.divider()


PAGE_TITLE = "Trade and External sector"
MANUFACTURES_INDICATOR_ID = "TX.VAL.MANF.ZS.UN"
FOOD_INDICATOR_ID = "TX.VAL.FOOD.ZS.UN"
FUEL_INDICATOR_ID = "TX.VAL.FUEL.ZS.UN"
GDP_INDICATOR_ID = "NY.GDP.MKTP.CD"
TOP_N_EXPORTERS = 20


def _render_export_composition_deep_dive() -> None:
    """Render the export-composition breakdown chart at the bottom of the page."""
    st.divider()
    st.subheader("Export Composition — Top-20 by total exports")
    st.caption(
        "Merchandise-export breakdown into Manufactures, Food, Fuel, and Other. "
        "Bars are ranked by total merchandise exports (export share of GDP × "
        "GDP) in the latest available year. Selected countries are appended in "
        "the highlight colour even if outside the top-20."
    )

    manf = fetch_indicator_slice(MANUFACTURES_INDICATOR_ID, value_col="manf_pct")
    food = fetch_indicator_slice(FOOD_INDICATOR_ID, value_col="food_pct")
    fuel = fetch_indicator_slice(FUEL_INDICATOR_ID, value_col="fuel_pct")
    exp_gdp = fetch_indicator_slice(EXPORT_INDICATOR_ID, value_col="exp_gdp")
    gdp = fetch_indicator_slice(GDP_INDICATOR_ID, value_col="gdp")

    if any(df.is_empty() for df in (manf, food, fuel, exp_gdp, gdp)):
        st.info("Export composition is unavailable - source data missing.")
        return

    joined = (
        manf.join(food, on=["year", "economy"])
        .join(fuel, on=["year", "economy"])
        .join(exp_gdp, on=["year", "economy"])
        .join(gdp, on=["year", "economy"])
    )
    if joined.is_empty():
        st.info("No overlapping export composition observations.")
        return

    latest_year = int(joined.select(pl.col("year").max()).item())
    snapshot = joined.filter(pl.col("year") == latest_year).with_columns(
        (pl.col("exp_gdp") / 100.0 * pl.col("gdp")).alias("exports_total"),
    )
    snapshot = snapshot.with_columns(
        (100.0 - pl.col("manf_pct") - pl.col("food_pct") - pl.col("fuel_pct"))
        .clip(0.0, 100.0)
        .alias("other_pct"),
    )

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        country_map = country_map.select(
            [
                pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Utf8).alias("country_name"),
            ]
        )
        snapshot = snapshot.join(country_map, on="economy", how="left")
    snapshot = snapshot.with_columns(
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name"),
    )

    selected_iso_codes = [
        str(c).strip().upper() for c in get_shared_selected_countries() if str(c).strip()
    ]

    top = snapshot.sort("exports_total", descending=True).head(TOP_N_EXPORTERS)
    top_set = set(top.get_column("economy").to_list())
    extra_selected = snapshot.filter(
        pl.col("economy").is_in(selected_iso_codes) & ~pl.col("economy").is_in(list(top_set))
    )

    combined = pl.concat([top, extra_selected], how="vertical_relaxed").sort(
        "exports_total", descending=False
    )
    if combined.is_empty():
        st.info("No countries to plot for export composition.")
        return

    plot_df = combined.to_pandas()
    fig = go.Figure()
    palette = [
        get_color("sector_manufacturing"),
        get_color("sector_agriculture"),
        get_color("negative"),
        get_color("reference_line"),
    ]
    for column, label, color in [
        ("manf_pct", "Manufactures", palette[0]),
        ("food_pct", "Food", palette[1]),
        ("fuel_pct", "Fuel", palette[2]),
        ("other_pct", "Other", palette[3]),
    ]:
        fig.add_trace(
            go.Bar(
                y=plot_df["country_name"],
                x=plot_df[column],
                orientation="h",
                name=label,
                marker={"color": color},
                customdata=plot_df["economy"],
                hovertemplate=(
                    "<b>%{y}</b> (%{customdata})<br>"
                    f"{label}: %{{x:.1f}}% of merchandise exports<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        title=f"Merchandise-export composition ({latest_year})",
        xaxis_title="% of merchandise exports",
        yaxis_title="",
        height=max(450, 26 * combined.height + 100),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.08},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Bars are ranked by total merchandise exports descending (largest at "
        "top of the chart). 'Other' is residual after manufactures + food + "
        "fuel — includes raw materials and unclassified categories."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Trade and External sector"],
    caption=(
        "Analyze external-sector dynamics through trade, openness, and balance "
        "signals across countries and over time."
    ),
    before_graphs_renderer=_render_import_export_scatter,
    after_graphs_renderer=_render_export_composition_deep_dive,
)
