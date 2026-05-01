import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.app_logging import log_page_render
from core.plotting import apply_plotly_theme
from core.theming import get_color
from core.postgres_client import (
    get_world_bank_country_codes,
    get_world_bank_country_mapping,
    get_world_bank_indicator,
)
from pages.page_utils import render_page_from_config


ECONOMY_STRUCTURE_INDICATORS = [
    ("Agriculture", "NV.AGR.TOTL.ZS", "sector_agriculture"),
    ("Manufacturing", "NV.IND.MANF.ZS", "sector_manufacturing"),
    ("Services", "NV.SRV.TOTL.ZS", "sector_services"),
]
PAGE_TITLE = "Economy Structure"


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


def _build_country_labels() -> tuple[list[str], dict[str, str], dict[str, str]]:
    country_options = sorted(get_world_bank_country_codes())
    country_mapping_df = get_world_bank_country_mapping()
    label_by_iso: dict[str, str] = {}
    name_by_iso: dict[str, str] = {}
    if not country_mapping_df.is_empty() and {"id", "value"}.issubset(
        set(country_mapping_df.columns)
    ):
        for row in country_mapping_df.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            name = str(row.get("value", "")).strip()
            if iso and name:
                label_by_iso[iso] = f"{name} ({iso})"
                name_by_iso[iso] = name
    return country_options, label_by_iso, name_by_iso


def _resolve_default_structure_country(country_options: list[str]) -> str | None:
    if not country_options:
        return None

    selected_trend_countries = st.session_state.get(f"{PAGE_TITLE}_countries", [])
    if selected_trend_countries:
        first_selected = str(selected_trend_countries[0]).strip().upper()
        if first_selected in country_options:
            return first_selected

    if "USA" in country_options:
        return "USA"

    return country_options[0]


def _build_economy_structure_data(
    country_code: str,
) -> tuple[pl.DataFrame, int | None]:
    latest_common_year_df: pl.DataFrame | None = None

    for sector_name, indicator_id, _ in ECONOMY_STRUCTURE_INDICATORS:
        sector_df = _prepare_indicator_slice(indicator_id, country_code=country_code)
        if sector_df.is_empty():
            return pl.DataFrame(), None

        sector_year_df = sector_df.select(
            [
                pl.col("year"),
                pl.col("value").alias(sector_name),
            ]
        )

        if latest_common_year_df is None:
            latest_common_year_df = sector_year_df
        else:
            latest_common_year_df = latest_common_year_df.join(
                sector_year_df,
                on="year",
                how="inner",
            )

    if latest_common_year_df is None or latest_common_year_df.is_empty():
        return pl.DataFrame(), None

    latest_row = latest_common_year_df.sort("year").tail(1)
    latest_year = int(latest_row.get_column("year")[0])
    structure_df = pl.DataFrame(
        {
            "sector": [item[0] for item in ECONOMY_STRUCTURE_INDICATORS],
            "indicator_id": [item[1] for item in ECONOMY_STRUCTURE_INDICATORS],
            "color": [get_color(item[2]) for item in ECONOMY_STRUCTURE_INDICATORS],
            "value": [
                float(latest_row.get_column(item[0])[0])
                for item in ECONOMY_STRUCTURE_INDICATORS
            ],
        }
    ).filter(pl.col("value").is_not_null() & (pl.col("value") >= 0))

    return structure_df, latest_year


def _build_economy_structure_pie(
    structure_df: pl.DataFrame,
    country_name: str,
    year: int,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=structure_df["sector"].to_list(),
                values=structure_df["value"].to_list(),
                sort=False,
                hole=0.35,
                marker=dict(colors=structure_df["color"].to_list()),
                texttemplate="%{label}<br>%{value:.1f}%",
                textinfo="text",
                customdata=structure_df["indicator_id"].to_list(),
                hovertemplate=(
                    "%{label}<br>%{value:.2f}% of GDP"
                    "<br>Indicator: %{customdata}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Economic Structure of {country_name} ({year})",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return apply_plotly_theme(fig)


def _render_economy_structure_section() -> None:
    st.subheader("Economy Structure")

    country_options, label_by_iso, name_by_iso = _build_country_labels()
    if not country_options:
        st.info("Country selector is unavailable right now.")
        return

    default_country = _resolve_default_structure_country(country_options)
    if default_country is None:
        st.info("Country selector is unavailable right now.")
        return

    selected_country = st.selectbox(
        "Country for economy structure",
        options=country_options,
        index=country_options.index(default_country),
        format_func=lambda iso: label_by_iso.get(str(iso).upper(), str(iso).upper()),
        key="economy_structure_country",
    )

    structure_df, latest_year = _build_economy_structure_data(selected_country)
    if structure_df.is_empty() or latest_year is None:
        st.info(
            "No common year with all three structure indicators is available for the selected country."
        )
        return

    country_name = name_by_iso.get(selected_country, selected_country)
    st.plotly_chart(
        _build_economy_structure_pie(
            structure_df,
            country_name=country_name,
            year=latest_year,
        ),
        width="stretch",
    )
    st.caption(
        "Uses the latest year where agriculture, manufacturing, and services values are all available. "
        "The pie is normalized across these three indicators, while slice labels show each indicator's original value as a percent of GDP."
    )

    st.divider()


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Structure"],
    caption=(
        "Explore the structure of national economies: agriculture, manufacturing, "
        "and services as a share of GDP."
    ),
    before_graphs_renderer=_render_economy_structure_section,
)
