"""Governance and institutions page — World Governance Indicators (WGI).

Renders a 6-axis radar comparing selected countries on Rule of Law,
Control of Corruption, Government Effectiveness, Voice and
Accountability, Regulatory Quality and Political Stability, plus a
country × dimension heatmap for the latest year.
"""

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.page_helpers import fetch_indicator_slice
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
)
from core.theming import get_color, get_colorway, get_diverging_colorscale
from pages.page_utils import get_shared_selected_countries, render_page_from_config

PAGE_TITLE = "Governance and Institutions"

WGI_DIMENSIONS: list[tuple[str, str]] = [
    ("GOV_WGI_RL.EST", "Rule of Law"),
    ("GOV_WGI_CC.EST", "Control of Corruption"),
    ("GOV_WGI_GE.EST", "Government Effectiveness"),
    ("GOV_WGI_VA.EST", "Voice and Accountability"),
    ("GOV_WGI_RQ.EST", "Regulatory Quality"),
    ("GOV_WGI_PV.EST", "Political Stability"),
]


def _build_wgi_panel() -> pl.DataFrame:
    """Join every WGI dimension into one ``year × economy`` panel.

    Returns:
        Polars frame with one column per dimension; empty when no
        dimensions are loaded.
    """
    panel: pl.DataFrame | None = None
    for indicator_id, label in WGI_DIMENSIONS:
        slice_df = fetch_indicator_slice(indicator_id, value_col=label)
        if slice_df.is_empty():
            continue
        panel = (
            slice_df
            if panel is None
            else panel.join(slice_df, on=["year", "economy"], how="full", coalesce=True)
        )
    return panel if panel is not None else pl.DataFrame()


def _render_wgi_radar_overview() -> None:
    """Render the WGI radar comparing selected countries on six axes."""
    st.subheader("Governance Radar")
    st.caption(
        "Compares World Governance Indicators across countries. Each axis is a "
        "governance dimension (range roughly -2.5 to +2.5; higher is better). "
        "Pick countries in the multiselect above to overlay their polygons."
    )

    panel_df = _build_wgi_panel()
    dimension_cols = [label for _, label in WGI_DIMENSIONS if label in panel_df.columns]

    if panel_df.is_empty() or not dimension_cols:
        st.info("WGI radar is unavailable because source data is empty.")
        st.divider()
        return

    year_options = panel_df.select("year").unique().sort("year").get_column("year").to_list()
    if not year_options:
        st.info("WGI radar is unavailable because years are missing.")
        st.divider()
        return

    selected_year = st.select_slider(
        "Radar year",
        options=year_options,
        value=year_options[-1],
        key="governance_radar_year",
    )

    year_panel = panel_df.filter(pl.col("year") == int(selected_year))
    if year_panel.is_empty():
        st.info("No WGI observations are available for this year.")
        st.divider()
        return

    country_map = get_world_bank_country_mapping()
    name_by_iso: dict[str, str] = {}
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        for row in country_map.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            name = str(row.get("value", "")).strip()
            if iso and name:
                name_by_iso[iso] = name

    selected_iso_codes = [
        str(code).strip().upper() for code in get_shared_selected_countries() if str(code).strip()
    ]

    axis_categories = dimension_cols + [dimension_cols[0]]
    fig = go.Figure()
    palette = get_colorway()

    if selected_iso_codes:
        countries_df = year_panel.filter(pl.col("economy").is_in(selected_iso_codes))
        if countries_df.is_empty():
            st.info(
                "Selected countries do not have WGI observations for this year. "
                "Showing the global mean instead."
            )
            selected_iso_codes = []
        else:
            for index, row in enumerate(countries_df.to_dicts()):
                iso = str(row.get("economy", "")).strip().upper()
                display_name = name_by_iso.get(iso, iso)
                values = [row.get(label) for label in dimension_cols]
                if any(v is None for v in values):
                    continue
                values_loop = [float(v) for v in values] + [float(values[0])]
                color = palette[index % len(palette)] if palette else None
                fig.add_trace(
                    go.Scatterpolar(
                        r=values_loop,
                        theta=axis_categories,
                        fill="toself",
                        name=f"{display_name} ({iso})",
                        line={"color": color} if color else None,
                        opacity=0.75,
                    )
                )

    if not selected_iso_codes:
        mean_values: list[float] = []
        for label in dimension_cols:
            mean_value = year_panel.select(pl.col(label).mean()).item()
            mean_values.append(float(mean_value) if mean_value is not None else 0.0)
        mean_values_loop = mean_values + [mean_values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=mean_values_loop,
                theta=axis_categories,
                fill="toself",
                name=f"Global mean ({selected_year})",
                line={"color": get_color("reference_line")},
                opacity=0.6,
            )
        )

    value_min, value_max = -2.5, 2.5
    fig.update_layout(
        title=f"Governance Radar ({selected_year})",
        polar={
            "radialaxis": {
                "visible": True,
                "range": [value_min, value_max],
                "tickvals": [-2.0, -1.0, 0.0, 1.0, 2.0],
                "gridcolor": get_color("reference_line"),
            },
            "angularaxis": {"gridcolor": get_color("reference_line")},
        },
        showlegend=True,
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "WGI scores roughly span -2.5 (weakest) to +2.5 (strongest). With no "
        "countries selected, the chart shows the global cross-country mean for "
        "the chosen year."
    )
    st.divider()


HEATMAP_TOP_N = 25
HEATMAP_BOTTOM_N = 15


def _render_wgi_heatmap_deep_dive() -> None:
    """Render the country × dimension WGI heatmap at the bottom of the page."""
    st.divider()
    st.subheader("Governance Heatmap")
    st.caption(
        "Pick a governance dimension and see how it has evolved by country since "
        f"1996. Rows are the global top-{HEATMAP_TOP_N} and bottom-"
        f"{HEATMAP_BOTTOM_N} by mean score across the whole period."
    )

    dim_labels = [label for _, label in WGI_DIMENSIONS]
    label_to_id = {label: indicator_id for indicator_id, label in WGI_DIMENSIONS}

    selected_dim = st.selectbox(
        "Governance dimension",
        options=dim_labels,
        index=0,
        key="governance_heatmap_dim",
    )
    indicator_id = label_to_id[selected_dim]

    series = fetch_indicator_slice(indicator_id, value_col="wgi")
    if series.is_empty():
        st.info("Heatmap data is unavailable.")
        return

    means = series.group_by("economy").agg(pl.col("wgi").mean().alias("wgi_mean"))
    top = means.sort("wgi_mean", descending=True).head(HEATMAP_TOP_N)
    bottom = means.sort("wgi_mean").head(HEATMAP_BOTTOM_N)
    keepers = pl.concat([top, bottom], how="vertical_relaxed").sort("wgi_mean", descending=True)
    keep_economies = keepers.get_column("economy").to_list()

    panel = series.filter(pl.col("economy").is_in(keep_economies))
    years = sorted(int(y) for y in panel.select("year").unique().to_series().to_list())

    country_map = get_world_bank_country_mapping()
    name_by_iso: dict[str, str] = {}
    if not country_map.is_empty() and {"id", "value"}.issubset(set(country_map.columns)):
        for row in country_map.to_dicts():
            iso = str(row.get("id", "")).strip().upper()
            nm = str(row.get("value", "")).strip()
            if iso and nm:
                name_by_iso[iso] = nm

    pivot = panel.pivot(values="wgi", index="economy", on="year", aggregate_function="mean")
    rows_by_economy = {row["economy"]: row for row in pivot.to_dicts()}

    z = []
    y_labels = []
    for econ in keep_economies:
        row = rows_by_economy.get(econ)
        if row is None:
            continue
        z.append([row.get(str(year)) for year in years])
        country_name = name_by_iso.get(econ, econ)
        y_labels.append(f"{country_name} ({econ})")

    if not z:
        st.info("Could not build heatmap matrix.")
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=years,
            y=y_labels,
            zmin=-2.5,
            zmax=2.5,
            colorscale=get_diverging_colorscale(),
            colorbar={"title": "Score"},
            hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{selected_dim}",
        xaxis_title="Year",
        yaxis_title="",
        height=max(450, 18 * len(y_labels) + 100),
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    fig.update_yaxes(autorange="reversed")
    fig = apply_plotly_theme(fig)

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "WGI scores roughly span -2.5 (worst) to +2.5 (best). The gap between "
        "the top block and bottom block visualises governance inequality across "
        "the world over time."
    )


render_page_from_config(
    page_title=PAGE_TITLE,
    section_keys=["Governance and Institutions"],
    caption=(
        "Assess institutional quality and governance through the World Bank's "
        "Worldwide Governance Indicators, covering rule of law, accountability, "
        "control of corruption, and political stability."
    ),
    before_graphs_renderer=_render_wgi_radar_overview,
    after_graphs_renderer=_render_wgi_heatmap_deep_dive,
)
