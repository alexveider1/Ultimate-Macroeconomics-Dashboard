from __future__ import annotations

import plotly.express as px
import polars as pl
import streamlit as st

from core.app_logging import log_page_render
from core.api_client import cluster_dataframe
from core.plotting import apply_plotly_theme
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_indicator,
    get_world_bank_indicator_name,
)
from pages.page_utils import load_dashboard_config


RESULT_STATE_KEY = "clustering_sandbox_result"
MAX_INDICATORS = 8
FEATURE_MODE_ABSOLUTE = "absolute"
FEATURE_MODE_CHANGE = "relative_change"
VIZ_X_COL = "__viz_x"
VIZ_Y_COL = "__viz_y"


@st.cache_data(show_spinner=False)
def _indicator_years(indicator_id: str) -> list[int]:
    df = get_world_bank_indicator(indicator_id, country_code="ALL")
    if df.is_empty() or "year" not in df.columns or "value" not in df.columns:
        return []

    years = (
        df.filter(pl.col("value").is_not_null() & pl.col("year").is_not_null())
        .select(pl.col("year").cast(pl.Int64))
        .unique()
        .sort("year")
        .get_column("year")
        .to_list()
    )
    return [int(year) for year in years]


@st.cache_data(show_spinner=False)
def _indicator_year_slice(indicator_id: str, year: int) -> pl.DataFrame:
    df = get_world_bank_indicator(indicator_id, country_code="ALL")
    if df.is_empty() or not {"year", "economy", "value"}.issubset(df.columns):
        return pl.DataFrame()

    return (
        df.filter(
            (pl.col("year").cast(pl.Int64) == int(year))
            & pl.col("economy").is_not_null()
        )
        .select(
            [
                pl.col("economy").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                pl.col("value").cast(pl.Float64).alias(indicator_id),
            ]
        )
        .group_by("economy")
        .agg(pl.col(indicator_id).mean().alias(indicator_id))
        .sort("economy")
    )


def _indicator_feature_slice(
    indicator_id: str,
    year: int,
    feature_mode: str,
) -> pl.DataFrame:
    current_df = _indicator_year_slice(indicator_id, year)
    if feature_mode == FEATURE_MODE_ABSOLUTE or current_df.is_empty():
        return current_df

    previous_df = _indicator_year_slice(indicator_id, year - 1)
    if previous_df.is_empty():
        return current_df.with_columns(
            pl.lit(None).cast(pl.Float64).alias(indicator_id)
        )

    joined_df = current_df.rename({indicator_id: "current_value"}).join(
        previous_df.rename({indicator_id: "previous_value"}),
        on="economy",
        how="left",
    )

    return joined_df.with_columns(
        pl.when(
            pl.col("current_value").is_null()
            | pl.col("previous_value").is_null()
            | (pl.col("previous_value") == 0)
        )
        .then(None)
        .otherwise(
            (pl.col("current_value") - pl.col("previous_value"))
            / pl.col("previous_value")
        )
        .cast(pl.Float64)
        .alias(indicator_id)
    ).select(["economy", indicator_id])


def _feature_label(feature_name: str, feature_mode: str, base_label: str) -> str:
    if feature_mode == FEATURE_MODE_CHANGE:
        return f"{base_label} (YoY change)"
    return f"{base_label} (absolute)"


def _normalize_features(
    df: pl.DataFrame,
    feature_columns: list[str],
    mode: str,
) -> pl.DataFrame:
    if mode == "none":
        return df

    normalized = df
    for col_name in feature_columns:
        series = normalized[col_name]
        if mode == "zscore":
            mean_val = float(series.mean()) if series.len() > 0 else 0.0
            std_val = float(series.std()) if series.len() > 1 else 0.0
            if std_val == 0.0:
                normalized = normalized.with_columns(pl.lit(0.0).alias(col_name))
            else:
                normalized = normalized.with_columns(
                    ((pl.col(col_name) - mean_val) / std_val).alias(col_name)
                )
        elif mode == "minmax":
            min_val = float(series.min()) if series.len() > 0 else 0.0
            max_val = float(series.max()) if series.len() > 0 else 0.0
            width = max_val - min_val
            if width == 0.0:
                normalized = normalized.with_columns(pl.lit(0.0).alias(col_name))
            else:
                normalized = normalized.with_columns(
                    ((pl.col(col_name) - min_val) / width).alias(col_name)
                )

    return normalized


def _apply_missing_strategy(
    df: pl.DataFrame,
    feature_columns: list[str],
    strategy: str,
) -> pl.DataFrame:
    if strategy == "drop":
        return df.drop_nulls(subset=feature_columns)

    filled = df
    for col_name in feature_columns:
        if strategy == "mean":
            fill_val = float(filled[col_name].mean()) if filled.height else 0.0
        else:
            fill_val = float(filled[col_name].median()) if filled.height else 0.0
        filled = filled.with_columns(
            pl.col(col_name).fill_null(fill_val).alias(col_name)
        )
    return filled


def _render_visuals(
    result_df: pl.DataFrame,
    visualization_columns: list[str],
    visualization_labels: list[str],
    visualization_mode: str,
) -> None:
    if result_df.is_empty():
        st.info("No clustering result to visualize.")
        return

    if len(visualization_columns) < 2:
        st.warning(
            "Clustering response does not contain valid visualization coordinates."
        )
        return

    scatter_x, scatter_y = visualization_columns[0], visualization_columns[1]
    if scatter_x not in result_df.columns or scatter_y not in result_df.columns:
        st.warning("Clustering response is missing scatterplot coordinate columns.")
        return

    x_axis_title = (
        visualization_labels[0] if len(visualization_labels) > 0 else scatter_x
    )
    y_axis_title = (
        visualization_labels[1] if len(visualization_labels) > 1 else scatter_y
    )

    plot_df = result_df.to_pandas()
    plot_df["cluster_label"] = plot_df["cluster"].astype(str)
    if "country_name" not in plot_df.columns:
        plot_df["country_name"] = plot_df["economy"]

    st.subheader("Cluster Visualization")
    left, right = st.columns([0.55, 0.45])

    with left:
        scatter_fig = px.scatter(
            plot_df,
            x=scatter_x,
            y=scatter_y,
            color="cluster_label",
            hover_name="country_name",
            hover_data={"economy": True, "cluster_label": True},
            title=(
                "Feature Space"
                if visualization_mode == "feature_space"
                else "t-SNE Projection"
            ),
        )
        scatter_fig.update_layout(
            xaxis_title=x_axis_title,
            yaxis_title=y_axis_title,
        )
        scatter_fig = apply_plotly_theme(scatter_fig)
        st.plotly_chart(scatter_fig, width="stretch")

    with right:
        map_fig = px.choropleth(
            plot_df,
            locations="economy",
            locationmode="ISO-3",
            color="cluster_label",
            hover_name="country_name",
            hover_data={"economy": True, "cluster_label": True},
            title="Cluster Map",
            projection="natural earth",
        )
        map_fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        map_fig = apply_plotly_theme(map_fig)
        st.plotly_chart(map_fig, width="stretch")


def render_page() -> None:
    log_page_render("Clustering Sandbox")
    st.title("Clustering Sandbox")
    st.caption(
        "Build country clusters from World Bank indicators for a selected year using k-means or DBSCAN."
    )

    config_data = load_dashboard_config()
    if not config_data:
        st.error("config.json is missing or empty, so indicators cannot be loaded.")
        return

    sections = list(config_data.keys())
    selected_section = st.selectbox(
        "Indicator category",
        options=sections,
        index=0 if sections else None,
    )

    section_items = config_data.get(selected_section, [])
    if not section_items:
        st.warning("No indicators available in this category.")
        return

    indicator_ids = [str(item["id"]) for item in section_items]
    fallback_by_id = {
        str(item["id"]): str(item.get("name", item["id"])) for item in section_items
    }
    label_by_id = {
        indicator_id: (
            get_world_bank_indicator_name(indicator_id, preferred_database_id="2")
            or fallback_by_id.get(indicator_id, indicator_id)
        )
        for indicator_id in indicator_ids
    }

    default_count = min(3, len(indicator_ids))
    selected_indicators = st.multiselect(
        "Indicators",
        options=indicator_ids,
        default=indicator_ids[:default_count],
        format_func=lambda x: f"{label_by_id.get(x, x)} ({x})",
        max_selections=MAX_INDICATORS,
        help="Select at least two indicators to define the clustering feature space.",
    )

    if len(selected_indicators) < 2:
        st.info("Select at least two indicators to continue.")
        return

    st.caption(
        "Feature mode lets each indicator be used as absolute value or year-over-year change ratio."
    )
    feature_mode_by_indicator: dict[str, str] = {}
    for indicator_id in selected_indicators:
        base_label = label_by_id.get(indicator_id, indicator_id)
        feature_mode_by_indicator[indicator_id] = st.selectbox(
            f"Feature mode: {base_label}",
            options=[FEATURE_MODE_ABSOLUTE, FEATURE_MODE_CHANGE],
            format_func=lambda mode: (
                "Absolute value"
                if mode == FEATURE_MODE_ABSOLUTE
                else "Year-over-year change (current - previous) / previous"
            ),
            key=f"cluster_feature_mode_{indicator_id}",
        )

    feature_label_by_id = {
        indicator_id: _feature_label(
            feature_name=indicator_id,
            feature_mode=feature_mode_by_indicator[indicator_id],
            base_label=label_by_id.get(indicator_id, indicator_id),
        )
        for indicator_id in selected_indicators
    }

    common_years: set[int] | None = None
    for indicator_id in selected_indicators:
        years = set(_indicator_years(indicator_id))
        if feature_mode_by_indicator[indicator_id] == FEATURE_MODE_CHANGE:
            years = {year for year in years if (year - 1) in years}
        common_years = (
            years if common_years is None else common_years.intersection(years)
        )

    year_options = sorted(common_years or [])
    if not year_options:
        st.warning(
            "No common years with non-null values were found for the selected indicators."
        )
        return

    selected_year = st.select_slider(
        "Year",
        options=year_options,
        value=year_options[-1],
    )

    prep_col, method_col = st.columns(2)
    with prep_col:
        missing_strategy = st.selectbox(
            "Missing value strategy",
            options=["drop", "mean", "median"],
            index=0,
            help="drop removes incomplete countries; mean/median impute per feature.",
        )
        normalization = st.selectbox(
            "Normalization",
            options=["none", "zscore", "minmax"],
            index=0,
        )

    with method_col:
        method = st.radio(
            "Algorithm",
            options=["kmeans", "dbscan"],
            horizontal=True,
        )
        if method == "kmeans":
            k = st.slider("k clusters", min_value=2, max_value=12, value=4)
            n_init = st.slider("n_init", min_value=5, max_value=50, value=10)
            random_state = st.number_input(
                "random_state", min_value=0, max_value=9999, value=42
            )
            eps = 0.5
            min_samples = 5
        else:
            eps = st.slider("eps", min_value=0.05, max_value=3.0, value=0.5, step=0.05)
            min_samples = st.slider("min_samples", min_value=2, max_value=25, value=5)
            k = 3
            n_init = 10
            random_state = 42

    run_button = st.button("Run clustering", type="primary", width="stretch")

    if run_button:
        with st.spinner("Preparing feature matrix and running clustering API..."):
            feature_df: pl.DataFrame | None = None
            for indicator_id in selected_indicators:
                slice_df = _indicator_feature_slice(
                    indicator_id=indicator_id,
                    year=selected_year,
                    feature_mode=feature_mode_by_indicator[indicator_id],
                )
                if slice_df.is_empty():
                    continue
                if feature_df is None:
                    feature_df = slice_df
                else:
                    feature_df = feature_df.join(slice_df, on="economy", how="full")
                    if "economy_right" in feature_df.columns:
                        feature_df = feature_df.with_columns(
                            pl.coalesce(
                                [pl.col("economy"), pl.col("economy_right")]
                            ).alias("economy")
                        ).drop("economy_right")

            if feature_df is None or feature_df.is_empty():
                st.error("No data available to cluster for the selected setup.")
                return

            feature_df = feature_df.with_columns(pl.col("economy").cast(pl.Utf8))
            prepared_df = _apply_missing_strategy(
                feature_df,
                selected_indicators,
                missing_strategy,
            )

            if prepared_df.is_empty() or prepared_df.height < 3:
                st.error(
                    "Too few rows remain after preprocessing. Try fewer indicators or use imputation."
                )
                return

            transformed_df = _normalize_features(
                prepared_df,
                selected_indicators,
                normalization,
            )

            api_rows = transformed_df.select(
                ["economy", *selected_indicators]
            ).to_dicts()

            try:
                api_result = cluster_dataframe(
                    dataframe=api_rows,
                    method=method,
                    feature_columns=selected_indicators,
                    k=int(k),
                    n_init=int(n_init),
                    random_state=int(random_state),
                    eps=float(eps),
                    min_samples=int(min_samples),
                )
            except Exception as exc:
                st.error(
                    f"Clustering request failed: {exc}. Set CLUSTERING_BASE_URL if your service uses a custom host/port."
                )
                return

            response_rows = api_result.get("dataframe", [])
            if not response_rows:
                st.error("Clustering API returned an empty response.")
                return

            visualization_columns = api_result.get(
                "visualization_columns", [VIZ_X_COL, VIZ_Y_COL]
            )
            if not isinstance(visualization_columns, list):
                visualization_columns = [VIZ_X_COL, VIZ_Y_COL]
            visualization_columns = [str(col) for col in visualization_columns]
            if len(visualization_columns) < 2:
                visualization_columns = [VIZ_X_COL, VIZ_Y_COL]

            visualization_labels = api_result.get("visualization_labels", [])
            if not isinstance(visualization_labels, list):
                visualization_labels = []
            visualization_labels = [str(label) for label in visualization_labels]

            cluster_df = pl.DataFrame(response_rows)
            cluster_df = cluster_df.with_columns(
                pl.col("economy").cast(pl.Utf8).str.to_uppercase().alias("economy")
            )

            country_map = get_world_bank_country_mapping()
            if not country_map.is_empty() and {"id", "value"}.issubset(
                set(country_map.columns)
            ):
                mapping_df = country_map.select(
                    [
                        pl.col("id").cast(pl.Utf8).str.to_uppercase().alias("economy"),
                        pl.col("value").cast(pl.Utf8).alias("country_name"),
                    ]
                )
                cluster_df = cluster_df.join(mapping_df, on="economy", how="left")
            else:
                cluster_df = cluster_df.with_columns(
                    pl.col("economy").alias("country_name")
                )

            st.session_state[RESULT_STATE_KEY] = {
                "result_df": cluster_df,
                "feature_columns": selected_indicators,
                "feature_label_by_id": feature_label_by_id,
                "feature_mode_by_indicator": feature_mode_by_indicator,
                "visualization_mode": str(
                    api_result.get(
                        "visualization_mode",
                        "feature_space" if len(selected_indicators) == 2 else "tsne",
                    )
                ),
                "visualization_columns": visualization_columns,
                "visualization_labels": visualization_labels,
                "method": method,
                "year": selected_year,
                "n_rows": cluster_df.height,
                "n_clusters": cluster_df.select(pl.col("cluster").n_unique()).item(),
            }

    state = st.session_state.get(RESULT_STATE_KEY)
    if not state:
        st.info("Choose your setup and click 'Run clustering' to generate results.")
        return

    result_df = state["result_df"]
    st.subheader("Run Summary")
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Method", str(state["method"]).upper())
    metric_col_2.metric("Year", str(state["year"]))
    metric_col_3.metric("Countries clustered", int(state["n_rows"]))
    metric_col_4.metric("Distinct labels", int(state["n_clusters"]))

    if "cluster" in result_df.columns:
        counts_df = (
            result_df.group_by("cluster")
            .len()
            .sort("cluster")
            .rename({"len": "countries"})
        )
        st.dataframe(counts_df, width="stretch")

    feature_columns = list(state.get("feature_columns", []))
    feature_label_by_id = dict(
        state.get("feature_label_by_id", {col: col for col in feature_columns})
    )
    feature_mode_by_indicator = dict(
        state.get(
            "feature_mode_by_indicator",
            {col: FEATURE_MODE_ABSOLUTE for col in feature_columns},
        )
    )

    default_viz_columns = [VIZ_X_COL, VIZ_Y_COL]
    if len(feature_columns) >= 2:
        default_viz_columns = [feature_columns[0], feature_columns[1]]

    _render_visuals(
        result_df=result_df,
        visualization_columns=list(
            state.get("visualization_columns", default_viz_columns)
        ),
        visualization_labels=list(state.get("visualization_labels", [])),
        visualization_mode=str(state.get("visualization_mode", "feature_space")),
    )

    mode_pairs = [
        {
            "feature": feature_label_by_id.get(indicator_id, indicator_id),
            "mode": (
                "YoY change"
                if feature_mode_by_indicator.get(indicator_id) == FEATURE_MODE_CHANGE
                else "Absolute"
            ),
        }
        for indicator_id in feature_columns
    ]
    st.dataframe(pl.DataFrame(mode_pairs), width="stretch")

    ordered_cols = ["country_name", "economy", "cluster", *feature_columns]
    final_cols = [col for col in ordered_cols if col in result_df.columns]
    st.subheader("Clustered Dataset")
    st.dataframe(
        result_df.select(final_cols).sort(["cluster", "country_name"]),
        width="stretch",
    )


render_page()
