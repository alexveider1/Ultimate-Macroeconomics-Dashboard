import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_indicator,
)
from pages.page_utils import render_page_from_config


IMPORT_INDICATOR_ID = "NE.IMP.GNFS.ZS"
EXPORT_INDICATOR_ID = "NE.EXP.GNFS.ZS"


def _prepare_indicator_slice(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    required_cols = {"year", "economy", "value"}
    if df.is_empty() or not required_cols.issubset(set(df.columns)):
        return pl.DataFrame()

    return (
        df.select(
            [
                pl.col("year").cast(pl.Int64, strict=False).alias("year"),
                pl.col("economy").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Float64, strict=False).alias(value_col),
            ]
        )
        .filter(
            pl.col("year").is_not_null()
            & pl.col("economy").is_not_null()
            & pl.col(value_col).is_not_null()
        )
        .group_by(["year", "economy"])
        .agg(pl.col(value_col).mean().alias(value_col))
        .sort(["year", "economy"])
    )


def _render_import_export_scatter() -> None:
    st.subheader("Imports vs Exports Scatter")
    st.caption(
        "Compares imports and exports of goods and services as percent of GDP for "
        "the same year and country."
    )

    imports_df = _prepare_indicator_slice(
        get_world_bank_indicator(IMPORT_INDICATOR_ID, country_code="ALL"),
        value_col="imports_pct_gdp",
    )
    exports_df = _prepare_indicator_slice(
        get_world_bank_indicator(EXPORT_INDICATOR_ID, country_code="ALL"),
        value_col="exports_pct_gdp",
    )

    if imports_df.is_empty() or exports_df.is_empty():
        st.info(
            "Import-export scatterplot is unavailable because source data is empty."
        )
        st.divider()
        return

    joined_df = imports_df.join(exports_df, on=["year", "economy"], how="inner")
    if joined_df.is_empty():
        st.info("No overlapping import and export values were found.")
        st.divider()
        return

    country_map = get_world_bank_country_mapping()
    if not country_map.is_empty() and {"id", "value"}.issubset(
        set(country_map.columns)
    ):
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
            (pl.col("exports_pct_gdp") - pl.col("imports_pct_gdp")).alias(
                "net_external_pct_gdp"
            ),
        ]
    )

    year_options = (
        joined_df.select("year").unique().sort("year").get_column("year").to_list()
    )
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

    fig = px.scatter(
        plot_df,
        x="imports_pct_gdp",
        y="exports_pct_gdp",
        color="net_external_pct_gdp",
        color_continuous_scale="RdBu",
        hover_name="country_name",
        hover_data={
            "economy": True,
            "imports_pct_gdp": ":.2f",
            "exports_pct_gdp": ":.2f",
            "net_external_pct_gdp": ":.2f",
        },
        labels={
            "imports_pct_gdp": "Imports (% of GDP)",
            "exports_pct_gdp": "Exports (% of GDP)",
            "net_external_pct_gdp": "Exports - Imports (% of GDP)",
        },
        title=f"Imports vs Exports (% of GDP) in {selected_year}",
    )

    fig.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="y = x",
            line={"color": "#64748b", "dash": "dash", "width": 1.5},
            hoverinfo="skip",
        )
    )

    fig.update_traces(marker={"size": 9, "opacity": 0.82, "line": {"width": 0.5}})
    fig.update_xaxes(range=[axis_min, axis_max])
    fig.update_yaxes(range=[axis_min, axis_max])
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Points above the dashed y=x line indicate stronger exports than imports "
        "relative to GDP; below the line indicates the opposite."
    )
    st.divider()


render_page_from_config(
    page_title="Trade and External sector",
    section_keys=["Trade and External sector"],
    caption=(
        "Analyze external-sector dynamics through trade, openness, and balance "
        "signals across countries and over time."
    ),
    before_graphs_renderer=_render_import_export_scatter,
)
