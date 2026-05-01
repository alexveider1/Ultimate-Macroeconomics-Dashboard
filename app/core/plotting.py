import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import base64
import hashlib
import yaml
import json
from plotly.colors import hex_to_rgb
from typing import Optional, Dict, Any
from datetime import datetime

from core.assets import get_markup_template, render_markup_template
from core.api_client import (
    forecast_timeseries,
    interpret_plot_image,
)
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_indicator,
    get_world_bank_indicator_name,
    get_world_bank_metadata,
)
from core.theming import PLOTLY_TEMPLATE_NAME, get_color, get_colorway

CONFIG = yaml.safe_load(open("config.yaml"))
FORECASTER_BASE_URL = f"http://forecaster:{CONFIG.get('forecaster').get('port')}"


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE_NAME)
    fig.for_each_xaxis(lambda axis: axis.update(showgrid=False))
    fig.for_each_yaxis(lambda axis: axis.update(showgrid=False))
    return fig


def _apply_plotly_template(fig: go.Figure) -> go.Figure:
    return apply_plotly_theme(fig)


def build_line_plot(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    title: str = "",
    forecast_df: Optional[pl.DataFrame] = None,
    forecast_lower_col: Optional[str] = None,
    forecast_upper_col: Optional[str] = None,
    hover_context: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()

    def _rgba(color: str, alpha: float) -> str:
        if color.startswith("#"):
            red, green, blue = hex_to_rgb(color)
            return f"rgba({red}, {green}, {blue}, {alpha})"
        return color

    def _prepare_line_df(local_df: pl.DataFrame) -> pl.DataFrame:
        if (
            local_df.is_empty()
            or x_col not in local_df.columns
            or y_col not in local_df.columns
        ):
            return pl.DataFrame()
        return (
            local_df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
            .sort(x_col)
            .unique(subset=[x_col], keep="last", maintain_order=True)
            .sort(x_col)
        )

    def _build_hovertemplate(include_ci: bool = False) -> str:
        value_label = "Forecast" if include_ci else "Value"
        ci_suffix = get_markup_template("line_plot_ci_suffix") if include_ci else ""
        if group_col:
            series_label = "Country" if hover_context else "Series"
            return render_markup_template(
                "line_plot_group_hovertemplate",
                series_label=series_label,
                value_label=value_label,
                ci_suffix=ci_suffix,
            )
        return render_markup_template(
            "line_plot_single_hovertemplate",
            value_label=value_label,
            ci_suffix=ci_suffix,
        )

    def _build_unified_hover_title() -> Optional[str]:
        if hover_context:
            return render_markup_template(
                "line_plot_unified_hover_title",
                hover_context=hover_context,
            )
        return None

    series_names: list[str] = []
    if group_col and group_col in df.columns:
        series_names.extend(
            [
                str(val)
                for val in df[group_col]
                .drop_nulls()
                .unique(maintain_order=True)
                .to_list()
            ]
        )
    elif not df.is_empty():
        series_names.append("Historical")

    if forecast_df is not None and not forecast_df.is_empty():
        if group_col and group_col in forecast_df.columns:
            for value in (
                forecast_df[group_col]
                .drop_nulls()
                .unique(maintain_order=True)
                .to_list()
            ):
                series_name = str(value)
                if series_name not in series_names:
                    series_names.append(series_name)
        elif "Historical" not in series_names:
            series_names.append("Historical")

    palette = get_colorway()
    series_colors = {
        name: palette[index % len(palette)] for index, name in enumerate(series_names)
    }

    if df.is_empty():
        fig.add_annotation(text="No historical data.", showarrow=False)
        fig.update_layout(title=title)
        return _apply_plotly_template(fig)

    if group_col and group_col in df.columns:
        for (group_val,), group_df in df.partition_by(group_col, as_dict=True).items():
            prepared_group_df = _prepare_line_df(group_df)
            if prepared_group_df.is_empty():
                continue
            fig.add_trace(
                go.Scatter(
                    x=prepared_group_df[x_col],
                    y=prepared_group_df[y_col],
                    mode="lines",
                    name=str(group_val),
                    line=dict(color=series_colors.get(str(group_val))),
                    legendgroup=str(group_val),
                    hovertemplate=_build_hovertemplate(),
                )
            )
    else:
        prepared_df = _prepare_line_df(df)
        if prepared_df.is_empty():
            fig.add_annotation(text="No historical data.", showarrow=False)
            fig.update_layout(title=title)
            return _apply_plotly_template(fig)
        fig.add_trace(
            go.Scatter(
                x=prepared_df[x_col],
                y=prepared_df[y_col],
                mode="lines",
                name="Historical",
                line=dict(color=series_colors.get("Historical")),
                legendgroup="Historical",
                hovertemplate=_build_hovertemplate(),
            )
        )

    if forecast_df is not None and not forecast_df.is_empty():
        if group_col and group_col in forecast_df.columns:
            for (group_val,), f_df in forecast_df.partition_by(
                group_col, as_dict=True
            ).items():
                prepared_forecast_df = _prepare_line_df(f_df)
                if prepared_forecast_df.is_empty():
                    continue
                series_name = str(group_val)
                trace_color = series_colors.get(series_name)
                has_ci = bool(
                    forecast_lower_col
                    and forecast_upper_col
                    and forecast_lower_col in prepared_forecast_df.columns
                    and forecast_upper_col in prepared_forecast_df.columns
                )
                if has_ci:
                    fig.add_trace(
                        go.Scatter(
                            x=prepared_forecast_df[x_col],
                            y=prepared_forecast_df[forecast_upper_col],
                            mode="lines",
                            line=dict(color=trace_color, width=0),
                            legendgroup=series_name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=prepared_forecast_df[x_col],
                            y=prepared_forecast_df[forecast_lower_col],
                            mode="lines",
                            line=dict(color=trace_color, width=0),
                            fill="tonexty",
                            fillcolor=_rgba(trace_color, 0.18)
                            if trace_color
                            else "rgba(99, 110, 250, 0.18)",
                            legendgroup=series_name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=prepared_forecast_df[x_col],
                        y=prepared_forecast_df[y_col],
                        mode="lines",
                        name=f"{series_name} (Forecast)",
                        line=dict(color=trace_color, width=2, dash="dash"),
                        legendgroup=series_name,
                        customdata=(
                            prepared_forecast_df.select(
                                [forecast_lower_col, forecast_upper_col]
                            ).to_numpy()
                            if has_ci
                            else None
                        ),
                        hovertemplate=_build_hovertemplate(include_ci=has_ci),
                    )
                )
        else:
            prepared_forecast_df = _prepare_line_df(forecast_df)
            has_ci = bool(
                forecast_lower_col
                and forecast_upper_col
                and forecast_lower_col in prepared_forecast_df.columns
                and forecast_upper_col in prepared_forecast_df.columns
            )
            trace_color = series_colors.get("Historical")
            if has_ci:
                fig.add_trace(
                    go.Scatter(
                        x=prepared_forecast_df[x_col],
                        y=prepared_forecast_df[forecast_upper_col],
                        mode="lines",
                        line=dict(color=trace_color, width=0),
                        legendgroup="Historical",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=prepared_forecast_df[x_col],
                        y=prepared_forecast_df[forecast_lower_col],
                        mode="lines",
                        line=dict(color=trace_color, width=0),
                        fill="tonexty",
                        fillcolor=_rgba(trace_color, 0.18)
                        if trace_color
                        else "rgba(99, 110, 250, 0.18)",
                        legendgroup="Historical",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=prepared_forecast_df[x_col],
                    y=prepared_forecast_df[y_col],
                    mode="lines",
                    name="Forecast",
                    line=dict(color=trace_color, width=2, dash="dash"),
                    legendgroup="Historical",
                    customdata=(
                        prepared_forecast_df.select(
                            [forecast_lower_col, forecast_upper_col]
                        ).to_numpy()
                        if has_ci
                        else None
                    ),
                    hovertemplate=_build_hovertemplate(include_ci=has_ci),
                )
            )

    fig.update_layout(
        title=title, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20)
    )
    unified_hover_title = _build_unified_hover_title()
    if unified_hover_title:
        fig.update_xaxes(unifiedhovertitle_text=unified_hover_title)
    return _apply_plotly_template(fig)


def build_distribution_plot(
    df: pl.DataFrame,
    val_col: str,
    group_col: Optional[str] = None,
    title: str = "",
    plot_type: str = "histplot",
    orientation: str = "vertical",
    reference_lines: Optional[list[dict[str, Any]]] = None,
) -> go.Figure:
    fig = go.Figure()
    is_normalized_hist = plot_type == "normalized_histplot"
    is_histplot = plot_type in {"histplot", "normalized_histplot"}
    is_vertical = orientation != "horizontal"

    if df.is_empty() or val_col not in df.columns:
        fig.add_annotation(text="No data available for distribution.", showarrow=False)
        fig.update_layout(title=title)
        return _apply_plotly_template(fig)

    def _add_distribution_trace(
        local_df: pl.DataFrame,
        trace_name: str,
        nbins: int,
        opacity: Optional[float] = None,
    ) -> None:
        values = local_df[val_col]

        if plot_type == "violinplot":
            trace_kwargs: Dict[str, Any] = {
                "name": trace_name,
                "box_visible": True,
                "meanline_visible": True,
                "orientation": "v" if is_vertical else "h",
            }
            if is_vertical:
                trace_kwargs["y"] = values
            else:
                trace_kwargs["x"] = values
            fig.add_trace(go.Violin(**trace_kwargs))
            return

        if plot_type == "boxplot":
            trace_kwargs = {
                "name": trace_name,
                "orientation": "v" if is_vertical else "h",
            }
            if is_vertical:
                trace_kwargs["y"] = values
            else:
                trace_kwargs["x"] = values
            fig.add_trace(go.Box(**trace_kwargs))
            return

        trace_kwargs = {
            "name": trace_name,
            "histnorm": "probability density" if is_normalized_hist else None,
        }
        if opacity is not None:
            trace_kwargs["opacity"] = opacity
        if is_vertical:
            trace_kwargs["x"] = values
            trace_kwargs["nbinsx"] = nbins
        else:
            trace_kwargs["y"] = values
            trace_kwargs["nbinsy"] = nbins
        fig.add_trace(go.Histogram(**trace_kwargs))

    if group_col and group_col in df.columns:
        for (group_val,), group_df in df.partition_by(group_col, as_dict=True).items():
            _add_distribution_trace(
                group_df,
                trace_name=str(group_val),
                nbins=20,
                opacity=0.65 if is_histplot else None,
            )
        if is_histplot:
            fig.update_layout(barmode="overlay")
    else:
        _add_distribution_trace(df, trace_name="Distribution", nbins=30)

    if is_histplot:
        numeric_axis = "x" if is_vertical else "y"
        distribution_axis_title = "Density" if is_normalized_hist else "Count"
        xaxis_title = val_col if numeric_axis == "x" else distribution_axis_title
        yaxis_title = distribution_axis_title if numeric_axis == "x" else val_col
    else:
        numeric_axis = "y" if is_vertical else "x"
        category_axis_title = group_col if group_col and group_col in df.columns else ""
        xaxis_title = category_axis_title if numeric_axis == "y" else val_col
        yaxis_title = val_col if numeric_axis == "y" else category_axis_title

    fig.update_layout(
        title=title,
        hovermode=numeric_axis,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    if reference_lines:
        line_palette = get_colorway()
        for idx, line_info in enumerate(reference_lines):
            try:
                level = float(line_info.get("value"))
            except (TypeError, ValueError):
                continue
            label = str(line_info.get("label") or f"Series {idx + 1}")
            line_color = line_palette[idx % len(line_palette)]
            if numeric_axis == "x":
                fig.add_vline(
                    x=level,
                    line_width=1,
                    line_dash="dash",
                    line_color=line_color,
                    annotation_text=label,
                    annotation_position="top right",
                    annotation_font_size=10,
                )
            else:
                fig.add_hline(
                    y=level,
                    line_width=1,
                    line_dash="dash",
                    line_color=line_color,
                    annotation_text=label,
                    annotation_position="top right",
                    annotation_font_size=10,
                )

    return _apply_plotly_template(fig)


def build_map_plot(
    df: pl.DataFrame,
    iso_col: str,
    val_col: str,
    title: str = "",
    text_col: Optional[str] = None,
    hover_context: Optional[str] = None,
    value_label: str = "Value",
) -> go.Figure:
    fig = go.Figure()

    if df.is_empty():
        fig.add_annotation(text="No data available for map.", showarrow=False)
        fig.update_layout(title=title)
        return _apply_plotly_template(fig)

    locations = [str(code).upper() for code in df[iso_col].to_list()]
    z_values = df[val_col].to_list()
    hover_text = (
        df[text_col].to_list() if text_col and text_col in df.columns else locations
    )
    hovertemplate = render_markup_template(
        "map_hovertemplate",
        value_label=value_label,
    )
    if hover_context:
        hovertemplate = render_markup_template(
            "map_hovertemplate_with_context",
            hover_context=hover_context,
            value_label=value_label,
        )

    fig.add_trace(
        go.Choropleth(
            locations=locations,
            z=z_values,
            text=hover_text,
            hovertemplate=hovertemplate,
            locationmode="ISO-3",
            autocolorscale=False,
            colorbar_title=value_label,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor=get_color("map_coastline"),
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return _apply_plotly_template(fig)


class GraphBox:
    def __init__(
        self,
        item_config: Dict[str, Any],
        selected_countries: Optional[list[str]] = None,
    ):
        self.config = item_config
        self.item_id = self.config["id"]
        self.name = self.config["name"]
        self.selected_countries = selected_countries or []
        resolved_name = get_world_bank_indicator_name(
            self.item_id,
            preferred_database_id="2",
        )
        if resolved_name:
            self.name = resolved_name

        self.key_prefix = f"world_bank_{self.item_id}"

    def _get_schema_mapping(self) -> Dict[str, str]:
        return {
            "x": "year",
            "y": "value",
            "group": "economy",
        }

    def _fetch_data(self) -> pl.DataFrame:
        return get_world_bank_indicator(self.item_id, country_code="ALL")

    def _prepare_time_trend_df(self, historical_df: pl.DataFrame) -> pl.DataFrame:
        schema = self._get_schema_mapping()
        if historical_df.is_empty() or schema["y"] not in historical_df.columns:
            return pl.DataFrame()

        cleaned = historical_df.filter(pl.col(schema["y"]).is_not_null())
        if self.selected_countries:
            normalized = [str(c).upper() for c in self.selected_countries]
            cleaned = cleaned.filter(
                pl.col(schema["group"])
                .cast(pl.Utf8)
                .str.to_uppercase()
                .is_in(normalized)
            )

        return cleaned.sort([schema["x"], schema["group"]])

    def _build_forecast_input(
        self,
        historical_df: pl.DataFrame,
    ) -> tuple[list[str], list[float]]:
        schema = self._get_schema_mapping()
        if historical_df.is_empty() or schema["x"] not in historical_df.columns:
            return [], []

        series_df = (
            historical_df.filter(pl.col(schema["y"]).is_not_null())
            .sort(schema["x"])
            .unique(subset=[schema["x"]], keep="last", maintain_order=True)
            .sort(schema["x"])
        )

        dates = [f"{int(year)}-01-01" for year in series_df[schema["x"]].to_list()]
        values = [float(v) for v in series_df[schema["y"]].to_list()]

        return dates, values

    def _format_forecast_response(
        self,
        points: list[dict[str, Any]],
        group_value: str,
    ) -> pl.DataFrame:
        schema = self._get_schema_mapping()
        forecast_df = pl.DataFrame(points)

        return forecast_df.with_columns(
            pl.col("ds")
            .str.strptime(pl.Datetime, strict=False)
            .dt.year()
            .alias(schema["x"]),
            pl.col("yhat").alias(schema["y"]),
            pl.col("yhat_lower").alias(f"{schema['y']}_lower"),
            pl.col("yhat_upper").alias(f"{schema['y']}_upper"),
            pl.lit(group_value).alias(schema["group"]),
        ).select(
            [
                schema["x"],
                schema["y"],
                f"{schema['y']}_lower",
                f"{schema['y']}_upper",
                schema["group"],
            ]
        )

    def _fetch_forecast(
        self,
        historical_df: pl.DataFrame,
        lookback: int,
        steps: int,
        alpha: float,
        model_type: str,
    ) -> pl.DataFrame:
        schema = self._get_schema_mapping()
        if historical_df.is_empty():
            return pl.DataFrame()

        group_values: list[str] = []
        if schema["group"] in historical_df.columns:
            group_values = [
                str(value)
                for value in historical_df[schema["group"]]
                .drop_nulls()
                .unique(maintain_order=True)
                .to_list()
            ]

        if group_values and len(group_values) > 20:
            st.warning(
                "Forecasting is limited to 20 series at a time. Narrow the country selection and rerun the model."
            )
            return pl.DataFrame()

        series_frames: list[pl.DataFrame] = []
        insufficient_series: list[str] = []

        if group_values:
            grouped_series = historical_df.partition_by(schema["group"], as_dict=True)
            for (group_value,), group_df in grouped_series.items():
                dates, values = self._build_forecast_input(group_df)
                if len(values) < 6:
                    insufficient_series.append(str(group_value))
                    continue

                try:
                    response = forecast_timeseries(
                        base_url=FORECASTER_BASE_URL,
                        dates=dates,
                        values=values,
                        n_prev=lookback,
                        n_predict=steps,
                        alpha=alpha,
                        model_type=model_type,
                    )
                except Exception as exc:
                    st.warning(f"Forecast service is unavailable: {exc}")
                    return pl.DataFrame()

                points = response.get("forecast", [])
                if points:
                    series_frames.append(
                        self._format_forecast_response(points, str(group_value))
                    )
        else:
            dates, values = self._build_forecast_input(historical_df)
            if len(values) < 6:
                st.warning("Not enough historical points to run forecasting.")
                return pl.DataFrame()

            try:
                response = forecast_timeseries(
                    base_url=FORECASTER_BASE_URL,
                    dates=dates,
                    values=values,
                    n_prev=lookback,
                    n_predict=steps,
                    alpha=alpha,
                    model_type=model_type,
                )
            except Exception as exc:
                st.warning(f"Forecast service is unavailable: {exc}")
                return pl.DataFrame()

            points = response.get("forecast", [])
            if points:
                series_frames.append(self._format_forecast_response(points, "Forecast"))

        if insufficient_series:
            st.info(
                "Skipped series with fewer than 6 historical points: "
                + ", ".join(insufficient_series)
            )

        if not series_frames:
            return pl.DataFrame()

        return pl.concat(series_frames, how="vertical_relaxed")

    def _get_metadata(self) -> dict[str, Any]:
        meta_df = get_world_bank_metadata(self.item_id)
        if meta_df.is_empty():
            return {}
        return meta_df.to_dicts()[0]

    def _build_hover_context(self, metadata: dict[str, Any]) -> str:
        indicator_name = self.name
        units = str(metadata.get("units") or "").strip()
        if units:
            return render_markup_template(
                "indicator_hover_context_with_units",
                indicator_name=indicator_name,
                units=units,
            )
        return render_markup_template(
            "indicator_hover_context",
            indicator_name=indicator_name,
        )

    def _apply_log_to_columns(
        self,
        df: pl.DataFrame,
        value_columns: list[str],
    ) -> tuple[pl.DataFrame, int]:
        valid_columns = [col for col in value_columns if col in df.columns]
        if df.is_empty() or not valid_columns:
            return df, 0

        positive_mask = pl.all_horizontal(
            [
                pl.col(col).is_not_null() & (pl.col(col).cast(pl.Float64) > 0)
                for col in valid_columns
            ]
        )
        filtered_df = df.filter(positive_mask)
        dropped_rows = df.height - filtered_df.height

        transformed_df = filtered_df.with_columns(
            [pl.col(col).cast(pl.Float64).log().alias(col) for col in valid_columns]
        )
        return transformed_df, dropped_rows

    def _render_metadata_markdown(self, metadata: dict[str, Any]) -> None:
        if not metadata:
            st.info("No metadata found for this identifier.")
            return

        ordered_fields = [
            "indicator_name",
            "units",
            "source",
            "development_relevance",
            "limitations_and_exceptions",
            "Statisticalconceptandmethodology",
        ]

        def _format_label(field: str) -> str:
            if field == "Statisticalconceptandmethodology":
                return "Statistical concept and methodology"
            return field.replace("_", " ").strip().title()

        st.markdown("## Metadata Overview")
        st.markdown("---")
        for field in ordered_fields:
            value = metadata.get(field)
            if value is None:
                continue
            text_value = str(value).strip()
            if not text_value:
                continue
            st.markdown(f"### {_format_label(field)}")
            st.markdown(text_value)
            st.markdown("")

        extra_keys = [
            key
            for key in metadata.keys()
            if key not in ordered_fields and str(metadata.get(key, "")).strip()
        ]
        for key in extra_keys:
            st.markdown(f"### {_format_label(key)}")
            st.markdown(str(metadata[key]).strip())
            st.markdown("")

    def _right_plot_signature(self, figure: go.Figure) -> str:
        try:
            raw_json = figure.to_plotly_json()
            serialized = json.dumps(raw_json, sort_keys=True, default=str)
        except Exception:
            serialized = f"{self.key_prefix}:{datetime.now().isoformat()}"
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_plot_description(
        self,
        figure: go.Figure,
        mode: str,
        chart_context: str,
    ) -> str:
        fig_signature = self._right_plot_signature(figure)
        cache_key = f"{self.key_prefix}_plot_description_cache"
        mode_cache_key = f"{mode}:{fig_signature}"

        cached = st.session_state.get(cache_key, {})
        if isinstance(cached, dict) and mode_cache_key in cached:
            return str(cached[mode_cache_key])

        image_bytes = pio.to_image(figure, format="png")
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        response = interpret_plot_image(
            image_base64=image_base64,
            mode=mode,
            chart_context=chart_context,
        )
        description = str(response.get("description", "")).strip()
        if not description:
            description = "No interpretation returned."

        if not isinstance(cached, dict):
            cached = {}
        cached[mode_cache_key] = description
        st.session_state[cache_key] = cached
        return description

    @st.fragment
    def render_streamlit_ui(self):
        schema = self._get_schema_mapping()
        distribution_type = "histplot"
        distribution_orientation = "vertical"
        use_log_key = f"{self.key_prefix}_use_log_transform"
        use_log = bool(st.session_state.get(use_log_key, False))
        selected_model = "prophet"
        alpha_value = 0.05
        points_to_use = 50
        points_to_predict = 10
        run_model_clicked = False

        with st.container(border=True):
            col_title, col_settings = st.columns([0.85, 0.15])
            with col_title:
                st.markdown(f"**{self.name}**")

            with col_settings:
                with st.popover("⚙️"):
                    st.markdown("**Layout**")
                    right_plot = st.selectbox(
                        "Right-side chart",
                        ["time trend", "distribution"],
                        index=0,
                        key=f"{self.key_prefix}_right_plot",
                    )

                    st.divider()
                    if right_plot == "time trend":
                        st.markdown("**Time Series Forecasting**")
                        with st.form(
                            key=f"{self.key_prefix}_forecast_form",
                            border=False,
                        ):
                            selected_model = st.selectbox(
                                "Model",
                                ["prophet", "arima", "chronos"],
                                index=0,
                                key=f"{self.key_prefix}_model",
                            )
                            alpha_value = st.slider(
                                "Alpha",
                                min_value=0.01,
                                max_value=0.2,
                                value=0.05,
                                step=0.01,
                                key=f"{self.key_prefix}_alpha",
                            )
                            points_to_use = st.number_input(
                                "Points to use",
                                min_value=6,
                                max_value=500,
                                value=50,
                                key=f"{self.key_prefix}_lookback",
                            )
                            points_to_predict = st.number_input(
                                "Points to predict",
                                min_value=1,
                                max_value=int(points_to_use),
                                value=min(10, int(points_to_use)),
                                key=f"{self.key_prefix}_predict",
                            )
                            run_model_clicked = st.form_submit_button(
                                "Run model",
                                type="primary",
                                width="stretch",
                            )
                    else:
                        st.markdown("**Distribution**")
                        distribution_type = st.selectbox(
                            "Distribution plot type",
                            [
                                "histplot",
                                "normalized_histplot",
                                "violinplot",
                                "boxplot",
                            ],
                            index=0,
                            key=f"{self.key_prefix}_distribution_type",
                        )
                        distribution_orientation = st.selectbox(
                            "Distribution orientation",
                            ["vertical", "horizontal"],
                            index=0,
                            key=f"{self.key_prefix}_distribution_orientation",
                        )

            df_hist_raw = self._fetch_data()
            df_hist_non_null = df_hist_raw.filter(pl.col(schema["y"]).is_not_null())
            with_year = df_hist_non_null.with_columns(
                pl.col(schema["x"]).cast(pl.Int64).alias("__year")
            )

            available_years = with_year["__year"].drop_nulls().unique().sort().to_list()
            year_key = f"{self.key_prefix}_selected_year"
            if available_years:
                min_year = int(available_years[0])
                max_year = int(available_years[-1])
                default_year = datetime.now().year - 1
                if default_year < min_year:
                    default_year = min_year
                if default_year > max_year:
                    default_year = max_year

                if year_key not in st.session_state:
                    st.session_state[year_key] = default_year

                selected_year = int(st.session_state.get(year_key, default_year))
                if selected_year < min_year:
                    selected_year = min_year
                    st.session_state[year_key] = selected_year
                if selected_year > max_year:
                    selected_year = max_year
                    st.session_state[year_key] = selected_year

                df_year = with_year.filter(pl.col("__year") == selected_year).drop(
                    "__year"
                )
            else:
                selected_year = datetime.now().year - 1
                min_year = selected_year
                max_year = selected_year
                df_year = pl.DataFrame()

            df_time_trend = self._prepare_time_trend_df(df_hist_raw)

            df_forecast = None
            forecast_data_key = f"{self.key_prefix}_forecast_df"
            forecast_params_key = f"{self.key_prefix}_forecast_params"
            if right_plot == "time trend":
                current_forecast_params = {
                    "model": selected_model,
                    "alpha": float(alpha_value),
                    "points_to_use": int(points_to_use),
                    "points_to_predict": int(points_to_predict),
                    "countries": tuple(sorted(str(c) for c in self.selected_countries)),
                }
                previous_params = st.session_state.get(forecast_params_key)
                if previous_params != current_forecast_params and not run_model_clicked:
                    st.session_state.pop(forecast_data_key, None)

                if run_model_clicked:
                    with st.spinner("Generating forecast..."):
                        fresh_forecast = self._fetch_forecast(
                            df_time_trend,
                            int(points_to_use),
                            int(points_to_predict),
                            float(alpha_value),
                            selected_model,
                        )
                    st.session_state[forecast_params_key] = current_forecast_params
                    st.session_state[forecast_data_key] = (
                        fresh_forecast.to_dicts()
                        if fresh_forecast is not None and not fresh_forecast.is_empty()
                        else []
                    )

                stored_forecast = st.session_state.get(forecast_data_key, [])
                if stored_forecast:
                    df_forecast = pl.DataFrame(stored_forecast)

            plotted_time_trend_df = df_time_trend
            plotted_forecast_df = df_forecast
            dropped_time_trend_points = 0
            dropped_forecast_points = 0
            if use_log:
                plotted_time_trend_df, dropped_time_trend_points = (
                    self._apply_log_to_columns(df_time_trend, [schema["y"]])
                )

                if df_forecast is not None and not df_forecast.is_empty():
                    plotted_forecast_df, dropped_forecast_points = (
                        self._apply_log_to_columns(
                            df_forecast,
                            [
                                schema["y"],
                                f"{schema['y']}_lower",
                                f"{schema['y']}_upper",
                            ],
                        )
                    )

            map_title = "Map"
            metadata_for_hover = self._get_metadata()
            hover_context = self._build_hover_context(metadata_for_hover)
            map_value_label = "Value"
            dropped_map_points = 0
            country_lookup = get_world_bank_country_mapping()
            map_df = (
                df_year.filter(
                    pl.col(schema["group"])
                    .cast(pl.Utf8)
                    .str.to_uppercase()
                    .str.len_chars()
                    == 3
                )
                .group_by(schema["group"])
                .agg(pl.col(schema["y"]).last().alias(schema["y"]))
            )
            if not country_lookup.is_empty() and {"id", "value"}.issubset(
                set(country_lookup.columns)
            ):
                map_df = map_df.join(
                    country_lookup.rename(
                        {"id": schema["group"], "value": "country_name"}
                    ),
                    on=schema["group"],
                    how="left",
                ).with_columns(
                    pl.coalesce(
                        [pl.col("country_name"), pl.col(schema["group"])]
                    ).alias("country_name")
                )
            if use_log:
                map_df, dropped_map_points = self._apply_log_to_columns(
                    map_df, [schema["y"]]
                )
                map_title = "map (ln)"
                map_value_label = "ln(Value)"

            map_fig = build_map_plot(
                map_df,
                iso_col=schema["group"],
                val_col=schema["y"],
                title=map_title,
                text_col="country_name",
                hover_context=hover_context,
                value_label=map_value_label,
            )

            dropped_distribution_points = 0
            dropped_reference_points = 0

            if right_plot == "time trend":
                right_fig = build_line_plot(
                    plotted_time_trend_df,
                    x_col=schema["x"],
                    y_col=schema["y"],
                    group_col=schema["group"],
                    title=("Time trend (ln)" if use_log else "Time trend"),
                    forecast_df=plotted_forecast_df,
                    forecast_lower_col=f"{schema['y']}_lower",
                    forecast_upper_col=f"{schema['y']}_upper",
                    hover_context=hover_context,
                )
            else:
                distribution_df = df_year
                if use_log:
                    distribution_df, dropped_distribution_points = (
                        self._apply_log_to_columns(df_year, [schema["y"]])
                    )

                distribution_reference_lines = None
                if not df_year.is_empty():
                    selected_country_codes = [
                        str(country).upper() for country in self.selected_countries
                    ]
                    if selected_country_codes:
                        selected_levels_df = (
                            df_year.filter(
                                pl.col(schema["group"])
                                .cast(pl.Utf8)
                                .str.to_uppercase()
                                .is_in(selected_country_codes)
                            )
                            .group_by(schema["group"])
                            .agg(pl.col(schema["y"]).last().alias(schema["y"]))
                        )

                        country_lookup = get_world_bank_country_mapping()
                        if (
                            not selected_levels_df.is_empty()
                            and not country_lookup.is_empty()
                            and {"id", "value"}.issubset(set(country_lookup.columns))
                        ):
                            selected_levels_df = selected_levels_df.join(
                                country_lookup.rename(
                                    {"id": schema["group"], "value": "country_name"}
                                ),
                                on=schema["group"],
                                how="left",
                            )

                        if not selected_levels_df.is_empty():
                            if use_log:
                                selected_levels_df, dropped_reference_points = (
                                    self._apply_log_to_columns(
                                        selected_levels_df, [schema["y"]]
                                    )
                                )

                            selected_levels_df = selected_levels_df.with_columns(
                                pl.coalesce(
                                    [
                                        pl.col("country_name")
                                        if "country_name" in selected_levels_df.columns
                                        else pl.lit(None),
                                        pl.col(schema["group"]).cast(pl.Utf8),
                                    ]
                                ).alias("country_label")
                            )
                            distribution_reference_lines = [
                                {
                                    "label": str(row.get("country_label", "")),
                                    "value": row.get(schema["y"]),
                                }
                                for row in selected_levels_df.to_dicts()
                                if row.get(schema["y"]) is not None
                            ]

                right_fig = build_distribution_plot(
                    distribution_df,
                    val_col=schema["y"],
                    group_col=None,
                    title=("Distribution (ln)" if use_log else "Distribution"),
                    plot_type=distribution_type,
                    orientation=distribution_orientation,
                    reference_lines=distribution_reference_lines,
                )

            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.plotly_chart(
                    map_fig,
                    width="stretch",
                    key=f"{self.key_prefix}_left_map_chart",
                )
                st.toggle(
                    "Apply log transformation",
                    value=use_log,
                    key=use_log_key,
                    help=(
                        "Applies natural logarithm (ln) to map values and to the "
                        "selected right-side chart. Non-positive values are skipped."
                    ),
                )
            with right_col:
                st.plotly_chart(
                    right_fig,
                    width="stretch",
                    key=f"{self.key_prefix}_right_selected_chart",
                )

                strict_desc_toggle = st.toggle(
                    "Plot description",
                    value=False,
                    key=f"{self.key_prefix}_strict_plot_description",
                    help=(
                        "Describes only visible line behavior over time without causal explanations."
                    ),
                )
                creative_desc_toggle = st.toggle(
                    "Creative plot description",
                    value=False,
                    key=f"{self.key_prefix}_creative_plot_description",
                    help=(
                        "Describes patterns and also suggests plausible reasons behind the changes."
                    ),
                )

                context_label = (
                    f"{self.name} | right chart: distribution | year: {selected_year} | type: {distribution_type} | orientation: {distribution_orientation}"
                    if right_plot == "distribution"
                    else f"{self.name} | right chart: time trend | full history"
                )

                if strict_desc_toggle:
                    st.markdown("**No-hallucinations description**")
                    with st.spinner("Generating strict plot description..."):
                        try:
                            strict_text = self._get_plot_description(
                                figure=right_fig,
                                mode="no_hallucinations",
                                chart_context=context_label,
                            )
                            st.write(strict_text)
                        except Exception as exc:
                            st.error(f"Plot description failed: {exc}")

                if creative_desc_toggle:
                    st.markdown("**Creative description**")
                    with st.spinner("Generating creative plot description..."):
                        try:
                            creative_text = self._get_plot_description(
                                figure=right_fig,
                                mode="creative",
                                chart_context=context_label,
                            )
                            st.write(creative_text)
                        except Exception as exc:
                            st.error(f"Creative plot description failed: {exc}")

            if use_log:
                dropped_notes: list[str] = []
                if dropped_map_points:
                    dropped_notes.append(f"map: {dropped_map_points}")
                if right_plot == "time trend":
                    if dropped_time_trend_points:
                        dropped_notes.append(f"time trend: {dropped_time_trend_points}")
                    if dropped_forecast_points:
                        dropped_notes.append(f"forecast: {dropped_forecast_points}")
                else:
                    if dropped_distribution_points:
                        dropped_notes.append(
                            f"distribution: {dropped_distribution_points}"
                        )
                    if dropped_reference_points:
                        dropped_notes.append(
                            f"distribution reference lines: {dropped_reference_points}"
                        )

                if dropped_notes:
                    st.caption(
                        "ln transform skipped non-positive values in "
                        + ", ".join(dropped_notes)
                        + "."
                    )

            if available_years:
                st.slider(
                    "Year filter",
                    min_value=min_year,
                    max_value=max_year,
                    key=year_key,
                    help="Default is current year minus one. Applies to the map and distribution chart only.",
                )
            else:
                st.info("No year data available for this indicator.")

            show_meta = st.toggle("ℹ️ Metadata", key=f"{self.key_prefix}_toggle_meta")

            if show_meta:
                with st.expander("Database Details", expanded=True):
                    metadata = metadata_for_hover
                    self._render_metadata_markdown(metadata)
