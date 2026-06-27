"""Demography page — population, growth, labour force, age structure.

Adds two custom blocks around the standard cards: a population-bubble
explorer (size = population, hover = growth / labour force) and an
age-structure pyramid showing 0–14 / 15–64 / 65+ shares over time.
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
)
from core.theming import get_color, get_colorway
from pages.page_utils import get_shared_selected_countries, render_page_from_config

PAGE_TITLE = "Demography"
AGE_0_14_INDICATOR_ID = "SP.POP.0014.TO.ZS"
AGE_15_64_INDICATOR_ID = "SP.POP.1564.TO.ZS"
AGE_65_INDICATOR_ID = "SP.POP.65UP.TO.ZS"


POPULATION_INDICATOR_ID = "SP.POP.TOTL"
POPULATION_GROWTH_INDICATOR_ID = "SP.POP.GROW"
LABOR_FORCE_INDICATOR_ID = "SL.TLF.TOTL.IN"
MALE_POPULATION_INDICATOR_ID = "SP.POP.TOTL.MA.IN"
FEMALE_POPULATION_INDICATOR_ID = "SP.POP.TOTL.FE.IN"


def _render_demography_bubble() -> None:
    """Render the population × growth bubble explorer at the top of the page."""
    st.subheader("Population Bubble Explorer")
    st.caption(
        "Bubble size reflects total population. Hover includes population growth, "
        "labor force, and male/female population totals."
    )

    total_population_df = fetch_indicator_slice(
        POPULATION_INDICATOR_ID,
        value_col="total_population",
    )
    population_growth_df = fetch_indicator_slice(
        POPULATION_GROWTH_INDICATOR_ID,
        value_col="population_growth",
    )
    labor_force_df = fetch_indicator_slice(
        LABOR_FORCE_INDICATOR_ID,
        value_col="labor_force",
    )
    male_population_df = fetch_indicator_slice(
        MALE_POPULATION_INDICATOR_ID,
        value_col="male_population",
    )
    female_population_df = fetch_indicator_slice(
        FEMALE_POPULATION_INDICATOR_ID,
        value_col="female_population",
    )

    joined_df = (
        total_population_df.join(population_growth_df, on=["year", "economy"], how="inner")
        .join(labor_force_df, on=["year", "economy"], how="inner")
        .join(male_population_df, on=["year", "economy"], how="inner")
        .join(female_population_df, on=["year", "economy"], how="inner")
    )

    if joined_df.is_empty():
        st.info("Demography bubble chart is unavailable because source data is incomplete.")
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
        pl.coalesce([pl.col("country_name"), pl.col("economy")]).alias("country_name")
    )

    year_options = joined_df.select("year").unique().sort("year").get_column("year").to_list()
    if not year_options:
        st.info("Demography bubble chart is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Bubble chart year",
        options=year_options,
        value=year_options[-1],
        key="demography_bubble_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year)).sort(
        "total_population", descending=True
    )
    if year_df.is_empty():
        st.info("No demography observations are available for this year.")
        st.divider()
        return

    plot_df = year_df.to_pandas()
    pop_max = float(plot_df["total_population"].max() or 1.0)
    sizeref = 2.0 * pop_max / (55.0 * 55.0)
    palette = get_colorway()

    fig = go.Figure()
    for index, country_name in enumerate(plot_df["country_name"].tolist()):
        row = plot_df.iloc[index]
        fig.add_trace(
            go.Scatter(
                x=[row["population_growth"]],
                y=[row["labor_force"]],
                mode="markers",
                name=str(country_name),
                marker={
                    "size": [float(row["total_population"])],
                    "sizemode": "area",
                    "sizeref": sizeref,
                    "sizemin": 4,
                    "color": palette[index % len(palette)] if palette else None,
                    "opacity": 0.78,
                    "line": {"width": 0.5},
                },
                text=[country_name],
                customdata=[
                    [
                        row["economy"],
                        row["male_population"],
                        row["female_population"],
                        row["total_population"],
                    ]
                ],
                hovertemplate=(
                    "<b>%{text}</b> (%{customdata[0]})<br>"
                    "Population growth: %{x:.2f}%<br>"
                    "Labor force: %{y:,.0f}<br>"
                    "Male population: %{customdata[1]:,.0f}<br>"
                    "Female population: %{customdata[2]:,.0f}<br>"
                    "Total population: %{customdata[3]:,.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"Population Growth vs Labor Force ({selected_year})",
        xaxis_title="Population growth (% annual)",
        yaxis_title="Labor force (people)",
        showlegend=False,
    )
    fig.update_yaxes(tickformat=",.2s")
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.divider()


def _render_age_structure_deep_dive() -> None:
    """Render the age-structure (0-14 / 15-64 / 65+) block at the bottom of the page."""
    st.divider()
    st.subheader("Age Structure (ternary plot)")
    st.caption(
        "Every country plotted on a triangle whose corners are the three age "
        "shares — Young (0-14), Working (15-64), Elderly (65+). Latest "
        "available year. Selected countries are highlighted and labelled."
    )

    young = fetch_indicator_slice(AGE_0_14_INDICATOR_ID, value_col="young")
    working = fetch_indicator_slice(AGE_15_64_INDICATOR_ID, value_col="working")
    elderly = fetch_indicator_slice(AGE_65_INDICATOR_ID, value_col="elderly")

    if young.is_empty() or working.is_empty() or elderly.is_empty():
        st.info("Age structure ternary is unavailable - source data missing.")
        return

    joined = young.join(working, on=["year", "economy"]).join(elderly, on=["year", "economy"])
    if joined.is_empty():
        st.info("No overlapping age-structure observations.")
        return

    latest_year = int(joined.select(pl.col("year").max()).item())
    snapshot = joined.filter(pl.col("year") == latest_year)

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
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name")
    )

    selected_iso_codes = {
        str(c).strip().upper() for c in get_shared_selected_countries() if str(c).strip()
    }
    snapshot = snapshot.with_columns(
        pl.when(pl.col("economy").is_in(list(selected_iso_codes)))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Other"))
        .alias("group")
    )

    plot_df = snapshot.to_pandas()

    fig = go.Figure()
    group_colors = {
        "Other": get_color("reference_line"),
        "Selected": get_colorway()[0],
    }
    for group_name in ("Other", "Selected"):
        group_rows = plot_df[plot_df["group"] == group_name]
        if group_rows.empty:
            continue
        fig.add_trace(
            go.Scatterternary(
                a=group_rows["young"],
                b=group_rows["working"],
                c=group_rows["elderly"],
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
                    "Young: %{a:.1f}%<br>"
                    "Working: %{b:.1f}%<br>"
                    "Elderly: %{c:.1f}%<extra></extra>"
                ),
            )
        )

    selected_df = plot_df[plot_df["group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatterternary(
                a=selected_df["young"],
                b=selected_df["working"],
                c=selected_df["elderly"],
                mode="text",
                text=selected_df["economy"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Age structure ({latest_year})",
        ternary={
            "aaxis": {"title": "Young 0-14 (%)"},
            "baxis": {"title": "Working 15-64 (%)"},
            "caxis": {"title": "Elderly 65+ (%)"},
        },
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Top vertex = young countries (high 0-14 share); bottom-left = working-"
        "age heavy; bottom-right = aged societies. The demographic transition "
        "moves countries clockwise from top down to the right."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Demography"],
    caption=(
        "Explore population size, structure, and demographic dynamics to connect "
        "labor and social trends with macroeconomic outcomes."
    ),
    before_graphs_renderer=_render_demography_bubble,
    after_graphs_renderer=_render_age_structure_deep_dive,
)
