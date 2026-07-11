"""General-economics indicator page — entry for the "Dashboard" navigation group.

Renders three sections: (1) a global GDP snapshot with summary cards and the
top-10-by-GDP table, (2) the standard set of WB indicators driven by
:func:`pages.page_utils.render_page_from_config`, and (3) a Hans-Rosling-style
animated bubble chart (GDP per capita × life expectancy × population by region).
"""

import math

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.assets import get_markup_template, render_markup_template
from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_country_regions,
)
from core.theming import get_color, get_colorway
from pages.page_utils import render_page_from_config

LIFE_EXP_INDICATOR_ID = "SP.DYN.LE00.IN"
POPULATION_INDICATOR_ID = "SP.POP.TOTL"


GDP_INDICATOR_ID = "NY.GDP.MKTP.CD"
GDP_PER_CAPITA_INDICATOR_ID = "NY.GDP.PCAP.CD"
SUMMARY_CARD_MIN_HEIGHT = 150
TOP10_TABLE_HEIGHT = (2 * SUMMARY_CARD_MIN_HEIGHT) + 90
PAGE_TITLE = "General Economics Indicators"


def _format_large_usd(value: float) -> str:
    """Render a USD amount with ``T/B/M`` suffix and dollar sign."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def _compute_summary(
    indicator_df: pl.DataFrame,
    aggregation: str,
    target_year: int | None = None,
) -> dict[str, float | int | None] | None:
    """Aggregate an indicator for one target year and compute YoY change.

    Args:
        indicator_df: Frame with ``year``/``economy``/``value`` columns.
        aggregation: ``"sum"`` (totals) or ``"mean"`` (averages).
        target_year: Preferred year; falls back to the latest available
            when missing.

    Returns:
        Dict with ``target_year``, ``previous_year``, ``target_value``,
        and ``pct_change`` (or ``None`` when the frame is empty).
    """
    if indicator_df.is_empty():
        return None

    available_years = (
        indicator_df.select(pl.col("year")).unique().sort("year").get_column("year").to_list()
    )
    if not available_years:
        return None

    resolved_target_year = (
        int(target_year) if target_year in available_years else int(available_years[-1])
    )
    previous_candidates = [
        int(year) for year in available_years if int(year) < resolved_target_year
    ]
    previous_year = previous_candidates[-1] if previous_candidates else None

    target_year_df = indicator_df.filter(pl.col("year") == resolved_target_year)
    previous_year_df = (
        indicator_df.filter(pl.col("year") == previous_year)
        if previous_year is not None
        else pl.DataFrame()
    )

    if aggregation == "sum":
        agg_expr = pl.col("value").sum()
    elif aggregation == "mean":
        agg_expr = pl.col("value").mean()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    target_value = float(target_year_df.select(agg_expr).item() or 0.0)
    previous_value = (
        float(previous_year_df.select(agg_expr).item() or 0.0) if previous_year is not None else 0.0
    )

    pct_change = None
    if previous_year is not None and previous_value != 0:
        pct_change = ((target_value - previous_value) / abs(previous_value)) * 100

    return {
        "target_year": resolved_target_year,
        "previous_year": previous_year,
        "target_value": target_value,
        "pct_change": pct_change,
    }


def _render_snapshot_card(
    title: str,
    value: float,
    pct_change: float | None,
    previous_year: int | None,
) -> None:
    """Draw one bordered summary card with value, YoY delta, and trend colour."""
    trend_is_positive = pct_change is None or pct_change >= 0
    trend_color = get_color("positive") if trend_is_positive else get_color("negative")
    delta_prefix = "+" if pct_change is not None and pct_change >= 0 else ""

    if pct_change is None:
        comparison_label = str(previous_year) if previous_year is not None else "previous year"
        delta_text = f"Change vs {comparison_label}: n/a"
    else:
        delta_text = f"{delta_prefix}{pct_change:.2f}% vs {previous_year}"

    with st.container(border=True):
        st.markdown(
            render_markup_template(
                "gdp_snapshot_card",
                min_height=SUMMARY_CARD_MIN_HEIGHT,
                title=title,
                trend_color=trend_color,
                card_title_color=get_color("card_title_color"),
                formatted_value=_format_large_usd(value),
                delta_text=delta_text,
            ),
            unsafe_allow_html=True,
        )


def _build_gdp_share_pie(share_df: pl.DataFrame, title: str) -> go.Figure:
    """Build the "GDP share of world total" pie chart."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=share_df["country"].to_list(),
                values=share_df["value"].to_list(),
                sort=False,
                textinfo="label+percent",
                hovertemplate=get_markup_template("gdp_share_pie_hovertemplate"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return apply_plotly_theme(fig)


def _render_gdp_overview() -> None:
    """Render the GDP snapshot block (summary cards + top-10 table + share pie)."""
    gdp_df = fetch_indicator_slice(GDP_INDICATOR_ID)
    gdp_per_capita_df = fetch_indicator_slice(GDP_PER_CAPITA_INDICATOR_ID)

    if gdp_df.is_empty():
        st.info("GDP summary is unavailable right now.")
        return

    gdp_summary = _compute_summary(gdp_df, aggregation="sum")
    if gdp_summary is None:
        st.info("GDP summary is unavailable right now.")
        return

    target_year = int(gdp_summary["target_year"])
    previous_year = (
        int(gdp_summary["previous_year"]) if gdp_summary["previous_year"] is not None else None
    )
    total_target = float(gdp_summary["target_value"])
    pct_change = float(gdp_summary["pct_change"]) if gdp_summary["pct_change"] is not None else None

    gdp_per_capita_summary = (
        _compute_summary(
            gdp_per_capita_df,
            aggregation="mean",
            target_year=target_year,
        )
        if not gdp_per_capita_df.is_empty()
        else None
    )

    target_year_df = gdp_df.filter(pl.col("year") == target_year)
    if target_year_df.is_empty():
        st.info(f"No GDP data available for {target_year} yet.")
        return

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        country_map = country_map.select(
            [
                pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Utf8).alias("country"),
            ]
        )
        target_year_df = target_year_df.join(country_map, on="economy", how="left")
    else:
        target_year_df = target_year_df.with_columns(pl.col("economy").alias("country"))

    ranked_gdp_df = target_year_df.sort("value", descending=True).with_columns(
        pl.col("country").fill_null(pl.col("economy")).alias("country")
    )

    top10_df = ranked_gdp_df.head(10).select(
        [
            pl.col("country").alias("Country"),
            pl.col("economy").alias("Code"),
            pl.col("value").alias("GDP (current US$)"),
        ]
    )

    pie_df = ranked_gdp_df.head(10).select(
        [
            pl.col("country"),
            pl.col("value"),
        ]
    )
    other_value = total_target - float(pie_df.select(pl.col("value").sum()).item() or 0.0)
    if ranked_gdp_df.height > 10 and other_value > 0:
        pie_df = pl.concat(
            [
                pie_df,
                pl.DataFrame({"country": ["Other"], "value": [other_value]}),
            ],
            how="vertical_relaxed",
        )

    left_col, right_col = st.columns([0.38, 0.62])
    with left_col:
        st.subheader(f"Global GDP Snapshot ({target_year})")
        _render_snapshot_card(
            title="Total GDP",
            value=total_target,
            pct_change=pct_change,
            previous_year=previous_year,
        )
        if gdp_per_capita_summary is not None:
            _render_snapshot_card(
                title="Average GDP per Capita",
                value=float(gdp_per_capita_summary["target_value"]),
                pct_change=(
                    float(gdp_per_capita_summary["pct_change"])
                    if gdp_per_capita_summary["pct_change"] is not None
                    else None
                ),
                previous_year=(
                    int(gdp_per_capita_summary["previous_year"])
                    if gdp_per_capita_summary["previous_year"] is not None
                    else None
                ),
            )
        else:
            st.info("Average GDP per capita is unavailable right now.")

    with right_col:
        st.subheader(f"Top-10 Countries by GDP ({target_year})")
        st.dataframe(
            top10_df,
            width="stretch",
            hide_index=True,
            height=TOP10_TABLE_HEIGHT,
            column_config={
                "GDP (current US$)": st.column_config.NumberColumn(
                    "GDP (current US$)", format="$%.0f"
                )
            },
        )

    st.subheader(f"GDP Share of World Total ({target_year})")
    st.plotly_chart(
        _build_gdp_share_pie(pie_df, title=f"GDP Share of World Total ({target_year})"),
        width="stretch",
    )

    st.divider()


def _render_development_transition_deep_dive() -> None:
    """Render the animated Hans-Rosling bubble chart at the bottom of the page."""
    st.divider()
    st.subheader("Development Transition (Hans-Rosling animation)")
    st.caption(
        "Each bubble is a country. X = GDP per capita (current US$, log scale). "
        "Y = life expectancy at birth. Size = total population. Color = World "
        "Bank region. Use the play button or scrub the year slider to watch the "
        "global development transition unfold."
    )

    gdp_pc = fetch_indicator_slice(GDP_PER_CAPITA_INDICATOR_ID, value_col="gdp_pc")
    life_exp = fetch_indicator_slice(LIFE_EXP_INDICATOR_ID, value_col="life_exp")
    pop = fetch_indicator_slice(POPULATION_INDICATOR_ID, value_col="pop")

    if gdp_pc.is_empty() or life_exp.is_empty() or pop.is_empty():
        st.info("Development transition animation is unavailable - missing source data.")
        return

    regions_df = get_world_bank_country_regions()
    if regions_df.is_empty() or not {"id", "value", "region"}.issubset(set(regions_df.columns)):
        st.info("Country region metadata is unavailable.")
        return

    regions_df = regions_df.select(
        [
            pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
            pl.col("value").cast(pl.Utf8).alias("country_name"),
            pl.col("region").cast(pl.Utf8).alias("region"),
        ]
    )

    joined = (
        gdp_pc.join(life_exp, on=["year", "economy"])
        .join(pop, on=["year", "economy"])
        .join(regions_df, on="economy", how="inner")
        .filter((pl.col("gdp_pc") > 0) & (pl.col("life_exp") > 0) & (pl.col("pop") > 0))
        .sort(["year", "economy"])
    )

    if joined.is_empty():
        st.info("No overlapping observations for the development transition.")
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
        """Return one scatter trace per region for the given year."""
        year_df = plot_df[plot_df["year"] == year_value]
        traces: list[go.Scatter] = []
        for region in regions:
            sub = year_df[year_df["region"] == region]
            traces.append(
                go.Scatter(
                    x=sub["gdp_pc"],
                    y=sub["life_exp"],
                    mode="markers",
                    name=region,
                    marker={
                        "size": sub["pop"],
                        "sizemode": "area",
                        "sizeref": sizeref,
                        "sizemin": 4,
                        "color": region_colors[region],
                        "opacity": 0.7,
                        "line": {"width": 0.5},
                    },
                    text=sub["country_name"],
                    customdata=sub[["economy", "gdp_pc", "life_exp", "pop", "region"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{text}</b> (%{customdata[0]})<br>"
                        "Region: %{customdata[4]}<br>"
                        "GDP per capita: %{customdata[1]:,.0f}<br>"
                        "Life expectancy: %{customdata[2]:.1f}<br>"
                        "Population: %{customdata[3]:,.0f}<extra></extra>"
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
        title="Income vs Longevity over time",
        xaxis_title="GDP per capita (US$, log)",
        yaxis_title="Life expectancy at birth (years)",
        xaxis_type="log",
        xaxis_range=[math.log10(x_min), math.log10(x_max)],
        yaxis_range=[
            float(plot_df["life_exp"].min()) - 2.0,
            float(plot_df["life_exp"].max()) + 2.0,
        ],
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
        "Inspired by Gapminder/Hans-Rosling. The rightward-and-upward drift over "
        "time shows the global development transition: rising incomes alongside "
        "longer lives, with regional clusters diverging then partially converging."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["General Economics Indicators"],
    caption=(
        "Track core macroeconomic and structural indicators across countries with "
        "map, trend, and distribution views."
    ),
    before_graphs_renderer=_render_gdp_overview,
    after_graphs_renderer=_render_development_transition_deep_dive,
)
