"""Environment and sustainability page — emissions, energy mix, air quality.

Custom blocks: a renewables-share × PM2.5 scatter highlighting selected
countries and an animated Environmental Kuznets Curve (income vs. CO₂
per capita over time, faceted by region).
"""

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_country_regions,
)
from core.theming import get_color, get_colorway
from pages.page_utils import get_shared_selected_countries, render_page_from_config

PAGE_TITLE = "Environment and Sustainability"
RENEWABLE_INDICATOR_ID = "EG.FEC.RNEW.ZS"
PM25_INDICATOR_ID = "EN.ATM.PM25.MC.M3"


def _render_renewable_vs_pm25_overview() -> None:
    """Render the renewables-share × PM2.5 scatter at the top of the page."""
    st.subheader("Renewable Energy vs PM2.5 Air Pollution")
    st.caption(
        "Compares the share of renewables in final energy consumption with "
        "average ambient PM2.5 pollution. The expected pattern is a negative "
        "slope — cleaner energy mixes coincide with cleaner air — but the "
        "relationship is noisy because of geography, density, and industry mix."
    )

    renewable_df = fetch_indicator_slice(RENEWABLE_INDICATOR_ID, value_col="renewable_pct")
    pm25_df = fetch_indicator_slice(PM25_INDICATOR_ID, value_col="pm25_ugm3")

    if renewable_df.is_empty() or pm25_df.is_empty():
        st.info("Renewable / PM2.5 scatter is unavailable because source data is empty.")
        st.divider()
        return

    joined_df = renewable_df.join(pm25_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping renewable and PM2.5 values were found.")
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
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name"),
    )

    year_options = joined_df.select("year").unique().sort("year").get_column("year").to_list()
    if not year_options:
        st.info("Renewable / PM2.5 scatter is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Scatter year",
        options=year_options,
        value=year_options[-1],
        key="env_renewable_pm25_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No renewable / PM2.5 observations are available for this year.")
        st.divider()
        return

    selected_countries = {
        str(code).strip().upper() for code in get_shared_selected_countries() if str(code).strip()
    }
    year_df = year_df.with_columns(
        pl.when(pl.col("economy").is_in(list(selected_countries)))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Other"))
        .alias("country_group")
    )

    plot_df = year_df.to_pandas()

    fig = go.Figure()
    group_colors = {
        "Other": get_color("reference_line"),
        "Selected": get_colorway()[0],
    }
    for group_name in ("Other", "Selected"):
        group_rows = plot_df[plot_df["country_group"] == group_name]
        if group_rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group_rows["renewable_pct"],
                y=group_rows["pm25_ugm3"],
                mode="markers",
                name=group_name,
                marker={
                    "color": group_colors[group_name],
                    "size": 9,
                    "opacity": 0.78,
                    "line": {"width": 0.5},
                },
                text=group_rows["country_name"],
                customdata=group_rows[["economy"]].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b> (%{customdata[0]})<br>"
                    "Renewable: %{x:.2f}%<br>"
                    "PM2.5: %{y:.2f} µg/m³<extra></extra>"
                ),
            )
        )

    selected_df = plot_df[plot_df["country_group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["renewable_pct"],
                y=selected_df["pm25_ugm3"],
                mode="text",
                text=selected_df["economy"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Renewable Energy vs PM2.5 ({selected_year})",
        xaxis_title="Renewable energy (% of final consumption)",
        yaxis_title="PM2.5 (µg/m³, lower is better)",
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Lower Y is healthier air. Selected countries from the multiselect "
        "above are highlighted and labelled."
    )
    st.divider()


GHG_PC_INDICATOR_ID = "EN.GHG.ALL.PC.CE.AR5"
GHG_TOTAL_INDICATOR_ID = "EN.GHG.ALL.MT.CE.AR5"
GDP_PC_INDICATOR_ID = "NY.GDP.PCAP.CD"


def _render_kuznets_curve_deep_dive() -> None:
    """Render the animated Environmental Kuznets Curve at the bottom of the page."""
    st.divider()
    st.subheader("Environmental Kuznets Curve (animation)")
    st.caption(
        "X = GDP per capita (current US$, log). Y = total greenhouse-gas "
        "emissions per capita (t CO2e/capita). Bubble size = total emissions "
        "(Mt). Animation frame = year. Watch the inverted-U / sideways drift "
        "as economies decouple growth from emissions intensity."
    )

    ghg_pc = fetch_indicator_slice(GHG_PC_INDICATOR_ID, value_col="ghg_pc")
    ghg_total = fetch_indicator_slice(GHG_TOTAL_INDICATOR_ID, value_col="ghg_total")
    gdp_pc = fetch_indicator_slice(GDP_PC_INDICATOR_ID, value_col="gdp_pc")
    regions_df = get_world_bank_country_regions()

    if any(df.is_empty() for df in (ghg_pc, ghg_total, gdp_pc)) or regions_df.is_empty():
        st.info("Kuznets-curve animation is unavailable - source data missing.")
        return

    regions_df = regions_df.select(
        [
            pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
            pl.col("value").cast(pl.Utf8).alias("country_name"),
            pl.col("region").cast(pl.Utf8).alias("region"),
        ]
    )

    joined = (
        ghg_pc.join(ghg_total, on=["year", "economy"])
        .join(gdp_pc, on=["year", "economy"])
        .join(regions_df, on="economy", how="inner")
        .filter((pl.col("ghg_pc") > 0) & (pl.col("ghg_total") > 0) & (pl.col("gdp_pc") > 0))
        .sort(["year", "economy"])
    )
    if joined.is_empty():
        st.info("No overlapping observations for the Kuznets curve.")
        return

    plot_df = joined.to_pandas().sort_values(["year", "country_name"])
    regions = sorted(plot_df["region"].dropna().unique().tolist())
    years = sorted(plot_df["year"].dropna().unique().tolist())
    palette = get_colorway()
    region_colors = {
        region: palette[idx % len(palette)] if palette else None
        for idx, region in enumerate(regions)
    }
    ghg_total_max = float(plot_df["ghg_total"].max() or 1.0)
    sizeref = 2.0 * ghg_total_max / (55.0 * 55.0)

    def _build_year_traces(year_value: int) -> list[go.Scatter]:
        """Return one scatter trace per region for the given year (one animation frame)."""
        year_df = plot_df[plot_df["year"] == year_value]
        traces: list[go.Scatter] = []
        for region in regions:
            sub = year_df[year_df["region"] == region]
            traces.append(
                go.Scatter(
                    x=sub["gdp_pc"],
                    y=sub["ghg_pc"],
                    mode="markers",
                    name=region,
                    marker={
                        "size": sub["ghg_total"],
                        "sizemode": "area",
                        "sizeref": sizeref,
                        "sizemin": 4,
                        "color": region_colors[region],
                        "opacity": 0.7,
                        "line": {"width": 0.5},
                    },
                    text=sub["country_name"],
                    customdata=sub[
                        ["economy", "gdp_pc", "ghg_pc", "ghg_total", "region"]
                    ].to_numpy(),
                    hovertemplate=(
                        "<b>%{text}</b> (%{customdata[0]})<br>"
                        "Region: %{customdata[4]}<br>"
                        "GDP per capita: %{customdata[1]:,.0f}<br>"
                        "GHG per capita: %{customdata[2]:.2f}<br>"
                        "Total emissions: %{customdata[3]:,.1f} Mt<extra></extra>"
                    ),
                )
            )
        return traces

    initial_traces = _build_year_traces(years[0])
    frames = [
        go.Frame(data=_build_year_traces(year_value), name=str(year_value)) for year_value in years
    ]

    x_min = max(100.0, float(plot_df["gdp_pc"].min()) * 0.8)
    x_max = float(plot_df["gdp_pc"].max()) * 1.2

    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title="Income vs Emissions intensity",
        xaxis_title="GDP per capita (US$, log)",
        yaxis_title="GHG emissions per capita (t CO2e)",
        xaxis_type="log",
        xaxis_range=[math.log10(x_min), math.log10(x_max)],
        yaxis_range=[0.0, float(plot_df["ghg_pc"].quantile(0.99)) * 1.1],
        margin={"l": 60, "r": 40, "t": 60, "b": 110},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "pad": {"r": 10, "t": 70},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
                "len": 0.9,
                "pad": {"b": 10, "t": 50},
                "currentvalue": {"prefix": "Year: "},
                "steps": [
                    {
                        "args": [
                            [str(year_value)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(year_value),
                        "method": "animate",
                    }
                    for year_value in years
                ],
            }
        ],
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Y-axis clipped at the 99th percentile so a few extreme outliers don't "
        "compress the rest of the chart. The classic Kuznets-curve hypothesis "
        "is that emissions per capita rise with income, peak, then decline as "
        "economies decarbonise."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Environment and ecology"],
    caption=(
        "Track environmental pressures and sustainability signals that interact "
        "with growth, risk, and long-term development."
    ),
    before_graphs_renderer=_render_renewable_vs_pm25_overview,
    after_graphs_renderer=_render_kuznets_curve_deep_dive,
)
