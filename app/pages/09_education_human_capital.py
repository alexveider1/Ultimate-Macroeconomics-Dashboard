"""Education and human-capital page — literacy, enrolment, R&D workforce.

Custom blocks: an adult-literacy × researchers-per-million scatter, and
an enrolment-ladder deep dive showing primary → secondary → tertiary
progression across countries.
"""

from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
)
from core.theming import get_color, get_colorway
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from pages.page_utils import get_shared_selected_countries, render_page_from_config

PAGE_TITLE = "Education and Human Capital"
LITERACY_INDICATOR_ID = "SE.ADT.LITR.ZS"
RESEARCHERS_INDICATOR_ID = "SP.POP.SCIE.RD.P6"


def _latest_per_country(df: pl.DataFrame, value_col: str, year_col: str) -> pl.DataFrame:
    """Keep one row per country: the latest year present for ``value_col``."""
    if df.is_empty():
        return df
    return (
        df.sort([pl.col("economy"), pl.col("year")])
        .group_by("economy")
        .agg(
            [
                pl.col("year").last().alias(year_col),
                pl.col(value_col).last().alias(value_col),
            ]
        )
    )


def _render_literacy_vs_researchers_overview() -> None:
    """Render the literacy × researchers-per-million scatter at the top of the page."""
    st.subheader("Literacy vs Research Intensity")
    st.caption(
        "Adult literacy rate versus researchers per million people. Both series "
        "are sparse for many countries, so each axis uses the latest available "
        "year per country (the year is shown in the tooltip)."
    )

    literacy_df = fetch_indicator_slice(LITERACY_INDICATOR_ID, value_col="literacy_pct")
    researchers_df = fetch_indicator_slice(
        RESEARCHERS_INDICATOR_ID, value_col="researchers_per_million"
    )

    if literacy_df.is_empty() or researchers_df.is_empty():
        st.info("Literacy/researchers scatter is unavailable because source data is empty.")
        st.divider()
        return

    literacy_latest = _latest_per_country(
        literacy_df, value_col="literacy_pct", year_col="literacy_year"
    )
    researchers_latest = _latest_per_country(
        researchers_df,
        value_col="researchers_per_million",
        year_col="researchers_year",
    )

    joined_df = literacy_latest.join(researchers_latest, on="economy", how="inner").filter(
        pl.col("researchers_per_million") > 0
    )
    if joined_df.is_empty():
        st.info("No countries have both literacy and researcher observations.")
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

    selected_countries = {
        str(code).strip().upper() for code in get_shared_selected_countries() if str(code).strip()
    }
    joined_df = joined_df.with_columns(
        pl.when(pl.col("economy").is_in(list(selected_countries)))
        .then(pl.lit("Selected"))
        .otherwise(pl.lit("Other"))
        .alias("country_group")
    )

    plot_df = joined_df.to_pandas()

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
                x=group_rows["literacy_pct"],
                y=group_rows["researchers_per_million"],
                mode="markers",
                name=group_name,
                marker={
                    "color": group_colors[group_name],
                    "size": 9,
                    "opacity": 0.78,
                    "line": {"width": 0.5},
                },
                text=group_rows["country_name"],
                customdata=group_rows[["economy", "literacy_year", "researchers_year"]].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b> (%{customdata[0]})<br>"
                    "Literacy: %{x:.2f}% (year %{customdata[1]})<br>"
                    "Researchers/M: %{y:,.0f} (year %{customdata[2]})<extra></extra>"
                ),
            )
        )

    selected_df = plot_df[plot_df["country_group"] == "Selected"]
    if not selected_df.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_df["literacy_pct"],
                y=selected_df["researchers_per_million"],
                mode="text",
                text=selected_df["economy"],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Literacy vs Research Intensity (latest year per country)",
        xaxis_title="Adult literacy rate (%)",
        yaxis_title="Researchers per million (log)",
        yaxis_type="log",
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each point uses the most recent year each country reports for each "
        "indicator (years can differ between axes — see tooltip). Y-axis is "
        "logarithmic."
    )
    st.divider()


PRIMARY_INDICATOR_ID = "SE.PRM.ENRR"
SECONDARY_INDICATOR_ID = "SE.SEC.ENRR"
TERTIARY_INDICATOR_ID = "SE.TER.ENRR"


def _render_enrollment_ladder_deep_dive() -> None:
    """Render the primary → secondary → tertiary enrolment ladder at the bottom of the page."""
    st.divider()
    st.subheader("Enrollment Ladder — Primary / Secondary / Tertiary")
    st.caption(
        "Gross enrollment ratio (%) at each schooling level for the selected "
        "countries — latest available year per country per level. Ratios above "
        "100% reflect over-age enrollment, not full coverage."
    )

    prim = fetch_indicator_slice(PRIMARY_INDICATOR_ID, value_col="primary")
    sec_ = fetch_indicator_slice(SECONDARY_INDICATOR_ID, value_col="secondary")
    ter = fetch_indicator_slice(TERTIARY_INDICATOR_ID, value_col="tertiary")

    if prim.is_empty() and sec_.is_empty() and ter.is_empty():
        st.info("Enrollment ladder data is unavailable.")
        return

    pp = _latest_per_country(prim, value_col="primary", year_col="primary_year")
    ss = _latest_per_country(sec_, value_col="secondary", year_col="secondary_year")
    tt = _latest_per_country(ter, value_col="tertiary", year_col="tertiary_year")

    panel = pp.join(ss, on="economy", how="outer", coalesce=True).join(
        tt, on="economy", how="outer", coalesce=True
    )

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        country_map = country_map.select(
            [
                pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Utf8).alias("country_name"),
            ]
        )
        panel = panel.join(country_map, on="economy", how="left")
    panel = panel.with_columns(
        pl.col("country_name").fill_null(pl.col("economy")).alias("country_name")
    )

    selected_iso_codes = [
        str(c).strip().upper() for c in get_shared_selected_countries() if str(c).strip()
    ]

    if selected_iso_codes:
        subset = panel.filter(pl.col("economy").is_in(selected_iso_codes))
        if subset.is_empty():
            st.info("Selected countries have no enrollment observations.")
            return
        plot_df = subset.sort("country_name").to_pandas()
        title = "Enrollment ratio (latest available year)"
    else:
        means = panel.select(
            [
                pl.col("primary").mean().alias("primary"),
                pl.col("secondary").mean().alias("secondary"),
                pl.col("tertiary").mean().alias("tertiary"),
            ]
        )
        plot_df = means.with_columns(
            [
                pl.lit("Global mean").alias("country_name"),
                pl.lit("---").alias("economy"),
            ]
        ).to_pandas()
        title = "Enrollment ratio (global cross-country mean)"

    fig = go.Figure()
    colors = get_colorway()
    for index, (column, label) in enumerate(
        [("primary", "Primary"), ("secondary", "Secondary"), ("tertiary", "Tertiary")]
    ):
        fig.add_trace(
            go.Bar(
                y=plot_df["country_name"],
                x=plot_df[column].fillna(0),
                orientation="h",
                name=label,
                marker={"color": colors[index % len(colors)] if colors else None},
                customdata=plot_df.get(f"{column}_year")
                if f"{column}_year" in plot_df.columns
                else None,
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}% gross enrollment<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="% (gross enrollment ratio)",
        yaxis_title="",
        height=max(380, 28 * len(plot_df) * 3 + 100),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
        legend={"orientation": "h", "y": -0.15},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Gross enrollment ratio = total enrolled at the level, regardless of "
        "age, divided by official school-age population for that level. Values "
        ">100% reflect over-age and repeat enrollments."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Education and Human Capital"],
    caption=(
        "Analyze schooling, skills, and human capital indicators linked to "
        "productivity, inclusion, and labor market quality."
    ),
    before_graphs_renderer=_render_literacy_vs_researchers_overview,
    after_graphs_renderer=_render_enrollment_ladder_deep_dive,
)
