"""Health and well-being page — life expectancy, mortality, health spending.

Custom blocks: the female-vs-male life-expectancy gap bar chart at the
top, and an animated health-transition bubble at the bottom (analogous
to the Hans-Rosling bubble on page 01 but over health metrics).
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

PAGE_TITLE = "Health and wellbeing"
LIFE_EXP_FEMALE_ID = "SP.DYN.LE00.FE.IN"
LIFE_EXP_MALE_ID = "SP.DYN.LE00.MA.IN"
TOP_N_GAPS = 15


def _render_life_expectancy_gap_overview() -> None:
    """Render the female-minus-male life expectancy gap bar chart at the top of the page."""
    st.subheader("Life Expectancy Gender Gap")
    st.caption(
        "Female minus male life expectancy at birth in years. Positive values "
        "mean women outlive men, the typical worldwide pattern."
    )

    female_df = fetch_indicator_slice(LIFE_EXP_FEMALE_ID, value_col="life_exp_female")
    male_df = fetch_indicator_slice(LIFE_EXP_MALE_ID, value_col="life_exp_male")

    if female_df.is_empty() or male_df.is_empty():
        st.info("Life-expectancy gap bar is unavailable because source data is empty.")
        st.divider()
        return

    joined_df = female_df.join(male_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping male/female life-expectancy values were found.")
        st.divider()
        return

    joined_df = joined_df.with_columns(
        (pl.col("life_exp_female") - pl.col("life_exp_male")).alias("gap_years")
    )

    year_options = joined_df.select("year").unique().sort("year").get_column("year").to_list()
    if not year_options:
        st.info("Life-expectancy gap bar is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Snapshot year",
        options=year_options,
        value=year_options[-1],
        key="health_lifeexp_gap_year",
    )

    year_df = joined_df.filter(pl.col("year") == int(selected_year))
    if year_df.is_empty():
        st.info("No life-expectancy gap observations are available for this year.")
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
        year_df = year_df.join(country_map, on="economy", how="left")

    year_df = year_df.with_columns(
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name"),
    )

    selected_iso_codes = [
        str(code).strip().upper() for code in get_shared_selected_countries() if str(code).strip()
    ]

    top_df = year_df.sort("gap_years", descending=True).head(TOP_N_GAPS)
    top_economies = set(top_df.get_column("economy").to_list())

    extra_selected_df = year_df.filter(
        pl.col("economy").is_in(selected_iso_codes) & ~pl.col("economy").is_in(list(top_economies))
    )

    combined_df = pl.concat([top_df, extra_selected_df], how="vertical_relaxed").sort(
        "gap_years", descending=False
    )

    if combined_df.is_empty():
        st.info("No countries to display for this year.")
        st.divider()
        return

    plot_df = combined_df.with_columns(
        pl.when(pl.col("economy").is_in(selected_iso_codes))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Top-15"))
        .alias("bar_group")
    ).to_pandas()

    color_map = {
        "Top-15": get_color("reference_line"),
        "Selected": get_colorway()[0],
    }
    fig = go.Figure()
    for group_name, group_color in color_map.items():
        group_rows = plot_df[plot_df["bar_group"] == group_name]
        if group_rows.empty:
            continue
        fig.add_trace(
            go.Bar(
                y=group_rows["country_name"],
                x=group_rows["gap_years"],
                orientation="h",
                marker={"color": group_color},
                name=group_name,
                hovertemplate=(
                    "<b>%{y}</b><br>Female - Male life expectancy: %{x:.2f} years<extra></extra>"
                ),
            )
        )

    fig.add_vline(
        x=0,
        line={"color": get_color("reference_line"), "dash": "dash", "width": 1.2},
    )
    fig.update_layout(
        title=f"Female minus Male Life Expectancy ({selected_year})",
        xaxis_title="Years",
        yaxis_title="",
        barmode="group",
        bargap=0.25,
        height=max(380, 28 * combined_df.height + 120),
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Top {TOP_N_GAPS} widest gaps shown by default; any countries selected "
        "in the multiselect above are appended in the highlighted colour if not "
        "already in the top set."
    )
    st.divider()


FERTILITY_INDICATOR_ID = "SP.DYN.TFRT.IN"
UNDER5_INDICATOR_ID = "SH.DYN.MORT"
POPULATION_INDICATOR_ID = "SP.POP.TOTL"


def _render_health_transition_deep_dive() -> None:
    """Render the animated health-transition bubble chart at the bottom of the page."""
    st.divider()
    st.subheader("Demographic-Health Transition (animation)")
    st.caption(
        "X = total fertility rate (births per woman). Y = under-5 mortality "
        "(per 1,000 live births, log). Size = total population. Colour = WB "
        "region. Play the animation to see the global health transition: "
        "fertility and child mortality fall together as development progresses."
    )

    fert = fetch_indicator_slice(FERTILITY_INDICATOR_ID, value_col="tfr")
    u5 = fetch_indicator_slice(UNDER5_INDICATOR_ID, value_col="u5m")
    pop = fetch_indicator_slice(POPULATION_INDICATOR_ID, value_col="pop")
    regions_df = get_world_bank_country_regions()

    if any(df.is_empty() for df in (fert, u5, pop)) or regions_df.is_empty():
        st.info("Health transition animation is unavailable.")
        return

    regions_df = regions_df.select(
        [
            pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
            pl.col("value").cast(pl.Utf8).alias("country_name"),
            pl.col("region").cast(pl.Utf8).alias("region"),
        ]
    )

    joined = (
        fert.join(u5, on=["year", "economy"])
        .join(pop, on=["year", "economy"])
        .join(regions_df, on="economy", how="inner")
        .filter((pl.col("tfr") > 0) & (pl.col("u5m") > 0) & (pl.col("pop") > 0))
        .sort(["year", "economy"])
    )
    if joined.is_empty():
        st.info("No overlapping fertility/under-5 mortality observations.")
        return

    plot_df = joined.to_pandas().sort_values(["year", "country_name"])
    regions = sorted(plot_df["region"].dropna().unique().tolist())
    years = sorted(plot_df["year"].dropna().unique().tolist())
    palette = get_colorway()
    region_colors = {
        region: palette[idx % len(palette)] if palette else None
        for idx, region in enumerate(regions)
    }
    pop_max = float(plot_df["pop"].max() or 1.0)
    sizeref = 2.0 * pop_max / (55.0 * 55.0)

    def _build_year_traces(year_value: int) -> list[go.Scatter]:
        """Return one scatter trace per region for the given year (one animation frame)."""
        year_df = plot_df[plot_df["year"] == year_value]
        traces: list[go.Scatter] = []
        for region in regions:
            sub = year_df[year_df["region"] == region]
            traces.append(
                go.Scatter(
                    x=sub["tfr"],
                    y=sub["u5m"],
                    mode="markers",
                    name=region,
                    marker={
                        "size": sub["pop"],
                        "sizemode": "area",
                        "sizeref": sizeref,
                        "sizemin": 4,
                        "color": region_colors[region],
                        "opacity": 0.72,
                        "line": {"width": 0.5},
                    },
                    text=sub["country_name"],
                    customdata=sub[["economy", "tfr", "u5m", "pop", "region"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{text}</b> (%{customdata[0]})<br>"
                        "Region: %{customdata[4]}<br>"
                        "Fertility: %{customdata[1]:.2f}<br>"
                        "Under-5 mortality: %{customdata[2]:.1f}<br>"
                        "Population: %{customdata[3]:,.0f}<extra></extra>"
                    ),
                )
            )
        return traces

    initial_traces = _build_year_traces(years[0])
    frames = [
        go.Frame(data=_build_year_traces(year_value), name=str(year_value)) for year_value in years
    ]

    y_min = max(1.0, float(plot_df["u5m"].min()) * 0.8)
    y_max = float(plot_df["u5m"].max()) * 1.2

    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(
        title="Fertility vs Under-5 Mortality over time",
        xaxis_title="Fertility rate (births per woman)",
        yaxis_title="Under-5 mortality (per 1,000, log)",
        xaxis_range=[
            max(0.5, float(plot_df["tfr"].min()) - 0.3),
            float(plot_df["tfr"].max()) + 0.3,
        ],
        yaxis_type="log",
        yaxis_range=[math.log10(y_min), math.log10(y_max)],
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
        "The classic demographic-and-health transition: a southwestward drift "
        "from the upper-right (high fertility, high child mortality) to the "
        "lower-left over the decades."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Health and wellbeing"],
    caption=(
        "Assess health outcomes and wellbeing indicators that influence human "
        "capital, resilience, and long-run economic performance."
    ),
    before_graphs_renderer=_render_life_expectancy_gap_overview,
    after_graphs_renderer=_render_health_transition_deep_dive,
)
