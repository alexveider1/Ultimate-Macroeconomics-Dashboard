import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.assets import get_markup_template, render_markup_template
from core.plotting import apply_plotly_theme
from core.theming import get_color
from core.postgres_client import (
    get_world_bank_country_codes,
    get_world_bank_country_mapping,
    get_world_bank_indicator,
)
from pages.page_utils import render_page_from_config


GDP_INDICATOR_ID = "NY.GDP.MKTP.CD"
GDP_PER_CAPITA_INDICATOR_ID = "NY.GDP.PCAP.CD"
SUMMARY_CARD_MIN_HEIGHT = 150
TOP10_TABLE_HEIGHT = (2 * SUMMARY_CARD_MIN_HEIGHT) + 90
PAGE_TITLE = "General Economics Indicators"


def _format_large_usd(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def _prepare_indicator_slice(
    indicator_id: str, country_code: str | list[str] = "ALL"
) -> pl.DataFrame:
    indicator_df = get_world_bank_indicator(indicator_id, country_code=country_code)
    if indicator_df.is_empty() or not {"year", "economy", "value"}.issubset(
        set(indicator_df.columns)
    ):
        return pl.DataFrame()

    return (
        indicator_df.with_columns(
            [
                pl.col("year").cast(pl.Int64, strict=False).alias("year"),
                pl.col("economy").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Float64, strict=False).alias("value"),
            ]
        )
        .filter(
            pl.col("year").is_not_null()
            & pl.col("economy").is_not_null()
            & pl.col("value").is_not_null()
        )
        .sort(["year", "economy"])
    )


def _compute_summary(
    indicator_df: pl.DataFrame,
    aggregation: str,
    target_year: int | None = None,
) -> dict[str, float | int | None] | None:
    if indicator_df.is_empty():
        return None

    available_years = (
        indicator_df.select(pl.col("year"))
        .unique()
        .sort("year")
        .get_column("year")
        .to_list()
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
        float(previous_year_df.select(agg_expr).item() or 0.0)
        if previous_year is not None
        else 0.0
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
    trend_is_positive = pct_change is None or pct_change >= 0
    trend_color = get_color("positive") if trend_is_positive else get_color("negative")
    delta_prefix = "+" if pct_change is not None and pct_change >= 0 else ""

    if pct_change is None:
        comparison_label = (
            str(previous_year) if previous_year is not None else "previous year"
        )
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
                formatted_value=_format_large_usd(value),
                delta_text=delta_text,
            ),
            unsafe_allow_html=True,
        )


def _build_gdp_share_pie(share_df: pl.DataFrame, title: str) -> go.Figure:
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
    gdp_df = _prepare_indicator_slice(GDP_INDICATOR_ID)
    gdp_per_capita_df = _prepare_indicator_slice(GDP_PER_CAPITA_INDICATOR_ID)

    if gdp_df.is_empty():
        st.info("GDP summary is unavailable right now.")
        return

    gdp_summary = _compute_summary(gdp_df, aggregation="sum")
    if gdp_summary is None:
        st.info("GDP summary is unavailable right now.")
        return

    target_year = int(gdp_summary["target_year"])
    previous_year = (
        int(gdp_summary["previous_year"])
        if gdp_summary["previous_year"] is not None
        else None
    )
    total_target = float(gdp_summary["target_value"])
    pct_change = (
        float(gdp_summary["pct_change"])
        if gdp_summary["pct_change"] is not None
        else None
    )

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
    if not country_map.is_empty() and {"id", "value"}.issubset(
        set(country_map.columns)
    ):
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
    other_value = total_target - float(
        pie_df.select(pl.col("value").sum()).item() or 0.0
    )
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


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["General Economics Indicators"],
    caption=(
        "Track core macroeconomic and structural indicators across countries with "
        "map, trend, and distribution views."
    ),
    before_graphs_renderer=_render_gdp_overview,
)
