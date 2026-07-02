"""Plotly chart builders and the :class:`GraphBox` widget used by every page.

Top-level functions (:func:`build_line_plot`, :func:`build_distribution_plot`,
:func:`build_map_plot`) are pure: they take a Polars frame and return a
``go.Figure`` already wrapped in the project's ``"app"`` Plotly template.
:class:`GraphBox` composes them into a Streamlit fragment that handles
metadata, year filtering, forecast calls, log-transforms, and the optional
LLM-generated plot descriptions.
"""

import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from plotly.colors import hex_to_rgb
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import streamlit as st

from core.api_client import (
    forecast_timeseries,
    interpret_plot_image,
)
from core.assets import get_markup_template, render_markup_template
from core.config import load_config
from core.postgres_client import (
    get_world_bank_country_mapping,
    get_world_bank_indicator,
    get_world_bank_indicator_name,
    get_world_bank_metadata,
)
from core.theming import (
    PLOTLY_TEMPLATE_NAME,
    get_color,
    get_colorway,
    get_confidence_band_alpha,
    get_diverging_colorscale,
    get_sequential_colorscale,
)
from core.token_usage import record_usage
from core.token_usage_store import record_persistent

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)
FORECASTER_BASE_URL = f"http://forecaster:{CONFIG.forecaster.port}"


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply the registered ``"app"`` template and disable axis grids.

    Args:
        fig: Figure to mutate in place.

    Returns:
        The same figure (returned for chaining).
    """
    fig.update_layout(template=PLOTLY_TEMPLATE_NAME)
    fig.for_each_xaxis(lambda axis: axis.update(showgrid=False))
    fig.for_each_yaxis(lambda axis: axis.update(showgrid=False))
    return fig


def _apply_plotly_template(fig: go.Figure) -> go.Figure:
    """Internal alias kept for readability inside the chart builders."""
    return apply_plotly_theme(fig)


def build_candlestick_plot(
    df: pl.DataFrame,
    date_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    title: str = "",
) -> go.Figure:
    """Build a themed candlestick chart from one series' OHLC history.

    Args:
        df: Source frame with at least the OHLC + date columns.
        date_col / open_col / high_col / low_col / close_col: Column names.
        title: Chart title.

    Returns:
        A themed ``go.Figure``; shows a placeholder annotation when the frame is
        empty or missing a required column.
    """
    fig = go.Figure()
    required_cols = [date_col, open_col, high_col, low_col, close_col]

    if df.is_empty() or any(col not in df.columns for col in required_cols):
        fig.add_annotation(text="No OHLC data available for candlestick chart.", showarrow=False)
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    prepared_df = (
        df.select(required_cols)
        .drop_nulls(required_cols)
        .sort(date_col)
        .unique(subset=[date_col], keep="last", maintain_order=True)
        .sort(date_col)
    )
    if prepared_df.is_empty():
        fig.add_annotation(text="No OHLC data available for candlestick chart.", showarrow=False)
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    fig.add_trace(
        go.Candlestick(
            x=prepared_df[date_col].to_list(),
            open=prepared_df[open_col].to_list(),
            high=prepared_df[high_col].to_list(),
            low=prepared_df[low_col].to_list(),
            close=prepared_df[close_col].to_list(),
            name="OHLC",
            increasing_line_color=get_color("positive"),
            decreasing_line_color=get_color("negative"),
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        hovermode="x",
    )
    return apply_plotly_theme(fig)


def build_correlation_heatmap(
    df: pl.DataFrame,
    date_col: str,
    series_col: str,
    value_col: str,
    title: str = "",
    label_map: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Build a Pearson-correlation heatmap of every series pair in ``df``.

    Pivots ``df`` to one column per ``series_col`` value and correlates each
    pair on their overlapping observations.

    Args:
        df: Long-format frame (one row per date/series).
        date_col: Date column used as the pivot index.
        series_col: Column identifying each series (one heatmap axis tick each).
        value_col: Numeric column to correlate (e.g. daily returns).
        title: Chart title.
        label_map: Optional ``series → display label`` map used in hover text.

    Returns:
        A themed ``go.Figure``; placeholder annotation when fewer than two
        series are available.
    """
    fig = go.Figure()

    if df.is_empty() or any(col not in df.columns for col in [date_col, series_col, value_col]):
        fig.add_annotation(text="No data available for correlation heatmap.", showarrow=False)
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    pivot = (
        df.select([date_col, series_col, value_col])
        .pivot(values=value_col, index=date_col, on=series_col, aggregate_function="last")
        .sort(date_col)
    )
    series = [col for col in pivot.columns if col != date_col]
    if len(series) < 2:
        fig.add_annotation(
            text="Need at least two series to compute correlations.", showarrow=False
        )
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    def _label(name: str) -> str:
        if not label_map:
            return name
        display = label_map.get(name, name)
        return display if display and display != name else name

    corr_rows: list[list[float]] = []
    customdata_rows: list[list[list[str]]] = []
    for row_series in series:
        row_vals: list[float] = []
        row_customdata: list[list[str]] = []
        for col_series in series:
            pair_df = pivot.select(
                [
                    pl.col(row_series).cast(pl.Float64).alias("x"),
                    pl.col(col_series).cast(pl.Float64).alias("y"),
                ]
            ).drop_nulls()
            row_customdata.append([_label(row_series), _label(col_series)])
            if pair_df.height < 2:
                row_vals.append(0.0)
                continue
            corr_val = pair_df.select(pl.corr("x", "y")).item()
            row_vals.append(float(corr_val) if corr_val is not None else 0.0)
        corr_rows.append(row_vals)
        customdata_rows.append(row_customdata)

    fig.add_trace(
        go.Heatmap(
            z=corr_rows,
            x=series,
            y=series,
            customdata=customdata_rows,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=get_diverging_colorscale(),
            colorbar_title="Corr",
            hovertemplate=get_markup_template("correlation_heatmap_hovertemplate"),
        )
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
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
    show_markers: bool = False,
) -> go.Figure:
    """Build a Plotly line chart for one or more time series.

    Args:
        df: Historical data. Must contain ``x_col`` and ``y_col``.
        x_col: Name of the column holding the x-axis values (typically year).
        y_col: Name of the column holding the y-axis values.
        group_col: When supplied, draws one line per distinct value.
        title: Chart title.
        forecast_df: Optional forecast frame appended as dashed lines. When
            both ``forecast_lower_col`` and ``forecast_upper_col`` are present,
            a shaded confidence band is added around each forecast trace.
        forecast_lower_col: Column name of the lower CI bound in ``forecast_df``.
        forecast_upper_col: Column name of the upper CI bound in ``forecast_df``.
        hover_context: Optional descriptive string injected into the unified
            hover title — used to surface units/source.
        show_markers: When ``True``, every historical and forecast point is
            rendered with a circle marker on top of the line.

    Returns:
        Themed ``go.Figure`` ready to pass to ``st.plotly_chart``.
    """
    fig = go.Figure()
    line_mode = "lines+markers" if show_markers else "lines"

    def _rgba(color: str, alpha: float) -> str:
        """Convert a ``#rrggbb`` hex colour into ``rgba(...)`` with given alpha."""
        if color.startswith("#"):
            red, green, blue = hex_to_rgb(color)
            return f"rgba({red}, {green}, {blue}, {alpha})"
        return color

    def _prepare_line_df(local_df: pl.DataFrame) -> pl.DataFrame:
        """Drop rows missing x/y, deduplicate per x (keep last), and sort by x."""
        if local_df.is_empty() or x_col not in local_df.columns or y_col not in local_df.columns:
            return pl.DataFrame()
        return (
            local_df.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())
            .sort(x_col)
            .unique(subset=[x_col], keep="last", maintain_order=True)
            .sort(x_col)
        )

    def _build_hovertemplate(include_ci: bool = False) -> str:
        """Resolve the right hover template (single vs. grouped, with/without CI)."""
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
        """Render the unified-hover title only when context (units/source) is set."""
        if hover_context:
            return render_markup_template(
                "line_plot_unified_hover_title",
                hover_context=hover_context,
            )
        return None

    series_names: list[str] = []
    if group_col and group_col in df.columns:
        series_names.extend(
            [str(val) for val in df[group_col].drop_nulls().unique(maintain_order=True).to_list()]
        )
    elif not df.is_empty():
        series_names.append("Historical")

    if forecast_df is not None and not forecast_df.is_empty():
        if group_col and group_col in forecast_df.columns:
            for value in forecast_df[group_col].drop_nulls().unique(maintain_order=True).to_list():
                series_name = str(value)
                if series_name not in series_names:
                    series_names.append(series_name)
        elif "Historical" not in series_names:
            series_names.append("Historical")

    palette = get_colorway()
    series_colors = {name: palette[index % len(palette)] for index, name in enumerate(series_names)}

    last_historical_by_series: dict[str, tuple[Any, Any]] = {}
    if not df.is_empty():
        if group_col and group_col in df.columns:
            for (group_val,), group_df in df.partition_by(group_col, as_dict=True).items():
                prepared = _prepare_line_df(group_df)
                if prepared.is_empty():
                    continue
                last_historical_by_series[str(group_val)] = (
                    prepared[x_col].to_list()[-1],
                    prepared[y_col].to_list()[-1],
                )
        else:
            prepared_all = _prepare_line_df(df)
            if not prepared_all.is_empty():
                last_historical_by_series["Historical"] = (
                    prepared_all[x_col].to_list()[-1],
                    prepared_all[y_col].to_list()[-1],
                )

    def _add_forecast_connector(
        series_key: str, trace_color: Optional[str], prepared_forecast: pl.DataFrame
    ) -> None:
        """Bridge the last actual point to the first predicted point with a dashed segment.

        The connector carries no confidence band: the CI shading starts at the
        first forecast point so the join with history reads cleanly.
        """
        if prepared_forecast.is_empty() or series_key not in last_historical_by_series:
            return
        last_x, last_y = last_historical_by_series[series_key]
        first_x = prepared_forecast[x_col].to_list()[0]
        first_y = prepared_forecast[y_col].to_list()[0]
        fig.add_trace(
            go.Scatter(
                x=[last_x, first_x],
                y=[last_y, first_y],
                mode="lines",
                line=dict(color=trace_color, width=2, dash="dash"),
                legendgroup=series_key,
                showlegend=False,
                hoverinfo="skip",
            )
        )

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
                    mode=line_mode,
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
                mode=line_mode,
                name="Historical",
                line=dict(color=series_colors.get("Historical")),
                legendgroup="Historical",
                hovertemplate=_build_hovertemplate(),
            )
        )

    if forecast_df is not None and not forecast_df.is_empty():
        if group_col and group_col in forecast_df.columns:
            for (group_val,), f_df in forecast_df.partition_by(group_col, as_dict=True).items():
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
                            fillcolor=_rgba(
                                trace_color or get_colorway()[0],
                                get_confidence_band_alpha(),
                            ),
                            legendgroup=series_name,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                _add_forecast_connector(series_name, trace_color, prepared_forecast_df)
                fig.add_trace(
                    go.Scatter(
                        x=prepared_forecast_df[x_col],
                        y=prepared_forecast_df[y_col],
                        mode=line_mode,
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
            _add_forecast_connector("Historical", trace_color, prepared_forecast_df)
            fig.add_trace(
                go.Scatter(
                    x=prepared_forecast_df[x_col],
                    y=prepared_forecast_df[y_col],
                    mode=line_mode,
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

    fig.update_layout(title=title, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
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
    """Build a distribution chart (histogram / density / violin / box).

    Args:
        df: Input data.
        val_col: Numeric column to summarise.
        group_col: When supplied, draws one overlaid trace per group.
        title: Chart title.
        plot_type: One of ``"histplot"``, ``"normalized_histplot"``,
            ``"violinplot"``, or ``"boxplot"``.
        orientation: ``"vertical"`` (default) or ``"horizontal"``.
        reference_lines: Optional list of ``{"label", "value"}`` dicts drawn
            as dashed reference lines (used to highlight selected countries
            against the global distribution).

    Returns:
        Themed ``go.Figure`` ready to render.
    """
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
        """Append one trace whose kind matches the outer ``plot_type`` switch."""
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
            trace_kwargs: Dict[str, Any] = {
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
            raw_value = line_info.get("value")
            if raw_value is None:
                continue
            try:
                level = float(raw_value)
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
    """Build a choropleth world map keyed by ISO-3 country codes.

    Args:
        df: Source frame.
        iso_col: Column holding ISO-3 codes (upper-cased internally).
        val_col: Column holding the metric to colour by.
        title: Chart title.
        text_col: Optional column used for the hover label (defaults to ISO code).
        hover_context: Optional descriptive string injected into the hover.
        value_label: Legend / hover label for the value scale.

    Returns:
        Themed ``go.Figure`` ready to render.
    """
    fig = go.Figure()

    if df.is_empty():
        fig.add_annotation(text="No data available for map.", showarrow=False)
        fig.update_layout(title=title)
        return _apply_plotly_template(fig)

    locations = [str(code).upper() for code in df[iso_col].to_list()]
    z_values = df[val_col].to_list()
    hover_text = df[text_col].to_list() if text_col and text_col in df.columns else locations
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
            colorscale=get_sequential_colorscale(),
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


def build_us_state_choropleth(
    df: pl.DataFrame,
    state_col: str,
    val_col: str,
    title: str = "",
    value_label: str = "Value",
    name_col: Optional[str] = None,
    hover_context: Optional[str] = None,
    reverse_scale: bool = False,
) -> go.Figure:
    """Build a choropleth of the 50 US states + DC keyed by USPS abbreviation.

    Mirrors :func:`build_map_plot` but uses ``locationmode="USA-states"`` and the
    ``"usa"`` geo scope so the two-letter state codes render as a US map.

    Args:
        df: Source frame.
        state_col: Column holding 2-letter state abbreviations (upper-cased).
        val_col: Column holding the metric to colour by.
        title: Chart title.
        value_label: Colorbar / hover label for the value scale.
        name_col: Optional column used for the hover label (defaults to abbrev).
        hover_context: Optional descriptive string injected into the hover.
        reverse_scale: Flip the sequential colour scale (useful when lower is
            "better", e.g. unemployment).

    Returns:
        Themed ``go.Figure`` ready to render.
    """
    fig = go.Figure()

    if df.is_empty():
        fig.add_annotation(text="No data available for map.", showarrow=False)
        fig.update_layout(title=title)
        return _apply_plotly_template(fig)

    locations = [str(code).upper() for code in df[state_col].to_list()]
    z_values = df[val_col].to_list()
    hover_text = df[name_col].to_list() if name_col and name_col in df.columns else locations
    context_line = f"<br>{hover_context}" if hover_context else ""
    hovertemplate = f"<b>%{{text}}</b>{context_line}<br>{value_label}: %{{z:,.2f}}<extra></extra>"

    fig.add_trace(
        go.Choropleth(
            locations=locations,
            z=z_values,
            text=hover_text,
            hovertemplate=hovertemplate,
            locationmode="USA-states",
            autocolorscale=False,
            colorscale=get_sequential_colorscale(reverse=reverse_scale),
            marker_line_color=get_color("map_coastline"),
            marker_line_width=0.5,
            colorbar_title=value_label,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        geo=dict(
            scope="usa",
            showframe=False,
            showlakes=False,
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return _apply_plotly_template(fig)


def build_region_ranking_bar(
    df: pl.DataFrame,
    region_col: str,
    val_col: str,
    title: str = "",
    value_label: str = "Value",
    label_col: Optional[str] = None,
    top_n: int = 10,
    ascending: bool = False,
) -> go.Figure:
    """Build a horizontal bar chart of the top/bottom-N regions by a metric.

    Region-agnostic (keyed on ``region_col``) so it is reused by both the FRED
    US-state page and future Eurostat regional pages.

    Args:
        df: Source frame (one row per region).
        region_col: Column holding the region code.
        val_col: Column holding the metric.
        title: Chart title.
        value_label: Axis / hover label for the value.
        label_col: Optional column for the tick label (defaults to region code).
        top_n: Number of regions to show.
        ascending: When ``False`` (default) show the highest values; when
            ``True`` show the lowest.

    Returns:
        Themed ``go.Figure`` with the largest bar at the top.
    """
    fig = go.Figure()

    if df.is_empty():
        fig.add_annotation(text="No data available.", showarrow=False)
        fig.update_layout(title=title)
        return _apply_plotly_template(fig)

    label = label_col if label_col and label_col in df.columns else region_col
    # Pick the extreme N, then order so the most extreme sits at the top.
    picked = df.sort(val_col, descending=not ascending).head(top_n)
    picked = picked.sort(val_col, descending=ascending)

    fig.add_trace(
        go.Bar(
            x=picked[val_col].to_list(),
            y=picked[label].to_list(),
            orientation="h",
            marker_color=get_colorway()[0],
            hovertemplate=f"<b>%{{y}}</b><br>{value_label}: %{{x:,.2f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=value_label,
        yaxis_title=None,
        margin=dict(l=0, r=10, t=50, b=0),
    )
    return _apply_plotly_template(fig)


def build_region_trend_lines(
    df: pl.DataFrame,
    region_col: str,
    year_col: str,
    val_col: str,
    regions: list[str],
    title: str = "",
    value_label: str = "Value",
    label_by_region: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """Build a multi-region time-trend line chart (one line per region).

    Region-agnostic so both FRED and future Eurostat pages reuse it.

    Args:
        df: Long frame with region / year / value columns.
        region_col: Column holding the region code.
        year_col: Column holding the year (x axis).
        val_col: Column holding the metric (y axis).
        regions: Region codes to plot, in the desired colour order.
        title: Chart title.
        value_label: Y-axis label.
        label_by_region: Optional mapping of region code to display name.

    Returns:
        Themed ``go.Figure``.
    """
    fig = go.Figure()
    colorway = get_colorway()

    for index, region in enumerate(regions):
        sub = df.filter(pl.col(region_col) == region).sort(year_col)
        if sub.is_empty():
            continue
        display_name = label_by_region.get(region, region) if label_by_region else region
        fig.add_trace(
            go.Scatter(
                x=sub[year_col].to_list(),
                y=sub[val_col].to_list(),
                mode="lines",
                name=display_name,
                line=dict(color=colorway[index % len(colorway)], width=2),
                hovertemplate=f"<b>{display_name}</b><br>%{{x}}: %{{y:,.2f}}<extra></extra>",
            )
        )

    if not fig.data:
        fig.add_annotation(text="No data available for the selected regions.", showarrow=False)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title=value_label,
        margin=dict(l=0, r=10, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return _apply_plotly_template(fig)


class GraphBox:
    def __init__(
        self,
        item_config: Dict[str, Any],
        selected_countries: Optional[list[str]] = None,
    ):
        """Stateful renderer for a World Bank indicator card.

        Composes the left-side world map and the right-side
        (time-trend or distribution) chart, with optional log-transform,
        on-demand forecasting, year filter, metadata expander, and the
        two LLM-driven plot description toggles. Each instance is bound
        to a single indicator id; pages instantiate one per chart.

        Args:
            item_config: Indicator descriptor with at least ``id`` and
                ``name`` keys (typically loaded from the page's chart
                config block).
            selected_countries: ISO codes selected globally on the page;
                used as forecast inputs and as reference lines on the
                distribution plot.
        """
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
        """Return the canonical ``{role: column_name}`` mapping used by the helpers."""
        return {
            "x": "year",
            "y": "value",
            "group": "economy",
        }

    def _fetch_data(self) -> pl.DataFrame:
        """Pull every year/economy row for this indicator from Postgres."""
        return get_world_bank_indicator(self.item_id, country_code="ALL")

    def _prepare_time_trend_df(self, historical_df: pl.DataFrame) -> pl.DataFrame:
        """Return the time-trend slice: rows with non-null y, optionally filtered to selected countries, sorted."""
        schema = self._get_schema_mapping()
        if historical_df.is_empty() or schema["y"] not in historical_df.columns:
            return pl.DataFrame()

        cleaned = historical_df.filter(pl.col(schema["y"]).is_not_null())
        if self.selected_countries:
            normalized = [str(c).upper() for c in self.selected_countries]
            cleaned = cleaned.filter(
                pl.col(schema["group"]).cast(pl.Utf8).str.to_uppercase().is_in(normalized)
            )

        return cleaned.sort([schema["x"], schema["group"]])

    def _build_forecast_input(
        self,
        historical_df: pl.DataFrame,
    ) -> tuple[list[str], list[float]]:
        """Convert a per-country historical slice into ``(dates, values)`` lists.

        Args:
            historical_df: Sub-frame for one country (already filtered).

        Returns:
            Tuple of ISO date strings and floats, suitable for the
            forecaster's JSON payload.
        """
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
        """Reshape the forecaster's ``ds/yhat/...`` payload into the page's schema.

        Args:
            points: Raw rows from the forecast service.
            group_value: Country code (or ``"Forecast"`` when ungrouped) to
                stamp into the group column so the trace gets the right colour.

        Returns:
            Polars frame with columns ``year`` / ``value`` /
            ``value_lower`` / ``value_upper`` / ``economy``.
        """
        schema = self._get_schema_mapping()
        forecast_df = pl.DataFrame(points)

        return forecast_df.with_columns(
            pl.col("ds").str.strptime(pl.Datetime, strict=False).dt.year().alias(schema["x"]),
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
        model_params: Optional[dict[str, Any]] = None,
    ) -> pl.DataFrame:
        """Call the forecaster service for each country in scope, concatenated.

        Args:
            historical_df: Time-trend frame (one row per year × economy).
            lookback: Number of historical points the model is allowed to use.
            steps: Forecast horizon.
            alpha: Confidence-interval alpha (e.g. 0.05 -> 95% CI).
            model_type: One of the model ids from the forecaster's
                ``Literal``. See :func:`core.api_client.forecast_timeseries`.
            model_params: Model-specific hyperparameters (forwarded to the
                forecaster service; unknown keys are ignored downstream).

        Returns:
            Concatenated forecast frame across countries, or an empty
            frame when the inputs are too short or the service errors.
            Side-effect: writes Streamlit warnings/info messages.
        """
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
                        model_params=model_params or {},
                    )
                except Exception as exc:
                    st.warning(f"Forecast service is unavailable: {exc}")
                    return pl.DataFrame()

                points = response.get("forecast", [])
                if points:
                    series_frames.append(self._format_forecast_response(points, str(group_value)))
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
                    model_params=model_params or {},
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
        """Return the first metadata row for this indicator (or empty dict)."""
        meta_df = get_world_bank_metadata(self.item_id)
        if meta_df.is_empty():
            return {}
        return meta_df.to_dicts()[0]

    def _build_hover_context(self, metadata: dict[str, Any]) -> str:
        """Build the indicator-name (and units, when present) hover banner."""
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
        """Replace each value column with its natural log; drop non-positive rows.

        Args:
            df: Input frame.
            value_columns: Columns to log-transform; missing columns are ignored.

        Returns:
            Tuple of ``(transformed_df, dropped_row_count)``. ``dropped_row_count``
            is surfaced to the user as a caption.
        """
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
        """Stream the indicator metadata into the page as titled markdown sections."""
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
            """Humanise a metadata key for display (special-case the WB camel-case one)."""
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
        """Hash the figure's JSON so plot-description caching survives reruns."""
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
        """Render the figure to PNG and ask the agent's vision endpoint to describe it.

        Results are cached per ``(mode, figure_hash)`` in ``st.session_state``
        so toggling the description checkbox on/off does not re-call the LLM.

        Args:
            figure: The figure to describe (right-side chart).
            mode: ``"no_hallucinations"`` for a strict factual reading or
                ``"creative"`` for an interpretive narrative.
            chart_context: Free-text context appended to the prompt
                (indicator name, year, chart type).

        Returns:
            The model's description (or a placeholder string when empty).
        """
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
        usage_payload = response.get("usage")
        record_usage(usage_payload)
        record_persistent("plot_interpret", usage_payload)
        description = str(response.get("description", "")).strip()
        if not description:
            description = "No interpretation returned."

        if not isinstance(cached, dict):
            cached = {}
        cached[mode_cache_key] = description
        st.session_state[cache_key] = cached
        return description

    def _render_model_param_inputs(self, selected_model: str) -> dict[str, Any]:
        """Render the hyperparameter widgets for ``selected_model`` and collect them.

        Each model exposes a small, opinionated set of knobs (e.g. ``(p, d, q)``
        for ARIMA, ``(P, D, Q, s)`` plus the non-seasonal triple for SARIMA,
        ``window`` for the moving-average baseline, lag/tree settings for
        XGBoost). The returned dict is forwarded verbatim to the forecaster
        service as ``model_params``; models that don't expect a given key just
        ignore it.
        """
        params: dict[str, Any] = {}
        if selected_model == "arima":
            params["p"] = int(
                st.number_input("AR order (p)", 0, 10, 1, key=f"{self.key_prefix}_arima_p")
            )
            params["d"] = int(
                st.number_input("Diff order (d)", 0, 3, 1, key=f"{self.key_prefix}_arima_d")
            )
            params["q"] = int(
                st.number_input("MA order (q)", 0, 10, 1, key=f"{self.key_prefix}_arima_q")
            )
        elif selected_model == "sarima":
            params["p"] = int(
                st.number_input("AR order (p)", 0, 10, 1, key=f"{self.key_prefix}_sarima_p")
            )
            params["d"] = int(
                st.number_input("Diff order (d)", 0, 3, 1, key=f"{self.key_prefix}_sarima_d")
            )
            params["q"] = int(
                st.number_input("MA order (q)", 0, 10, 1, key=f"{self.key_prefix}_sarima_q")
            )
            params["P"] = int(
                st.number_input("Seasonal AR (P)", 0, 5, 0, key=f"{self.key_prefix}_sarima_P")
            )
            params["D"] = int(
                st.number_input("Seasonal diff (D)", 0, 2, 0, key=f"{self.key_prefix}_sarima_D")
            )
            params["Q"] = int(
                st.number_input("Seasonal MA (Q)", 0, 5, 0, key=f"{self.key_prefix}_sarima_Q")
            )
            params["s"] = int(
                st.number_input(
                    "Seasonal period (s)", 1, 365, 12, key=f"{self.key_prefix}_sarima_s"
                )
            )
        elif selected_model == "moving_average":
            params["window"] = int(
                st.number_input("Window", 1, 100, 5, key=f"{self.key_prefix}_ma_window")
            )
        elif selected_model == "xgboost":
            params["lags"] = int(
                st.number_input("Lags", 1, 60, 5, key=f"{self.key_prefix}_xgb_lags")
            )
            params["n_estimators"] = int(
                st.number_input(
                    "Estimators", 50, 1000, 200, step=50, key=f"{self.key_prefix}_xgb_n_estimators"
                )
            )
            params["max_depth"] = int(
                st.number_input("Max depth", 1, 12, 3, key=f"{self.key_prefix}_xgb_max_depth")
            )
            params["learning_rate"] = float(
                st.number_input(
                    "Learning rate",
                    min_value=0.005,
                    max_value=0.5,
                    value=0.05,
                    step=0.005,
                    format="%.3f",
                    key=f"{self.key_prefix}_xgb_lr",
                )
            )
        return params

    def _render_header_and_settings(
        self,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        """Render the title bar and settings popover; return the chosen settings.

        Mutates nothing on ``self``; the returned dict carries every choice
        the rest of ``render_streamlit_ui`` needs (right_plot, forecast
        params, distribution params, run-button click, marker toggle).
        """
        settings: dict[str, Any] = dict(defaults)

        col_title, col_settings = st.columns([0.85, 0.15])
        with col_title:
            st.markdown(f"**{self.name}**")

        with col_settings:
            with st.popover("⚙️"):
                st.markdown("**Layout**")
                settings["right_plot"] = st.selectbox(
                    "Right-side chart",
                    ["time trend", "distribution"],
                    index=0,
                    key=f"{self.key_prefix}_right_plot",
                )

                if settings["right_plot"] == "time trend":
                    settings["show_markers"] = st.toggle(
                        "Highlight points",
                        value=False,
                        key=f"{self.key_prefix}_show_markers",
                        help=(
                            "Adds a circle marker on every historical and forecast point. "
                            "Off by default."
                        ),
                    )
                else:
                    settings["show_markers"] = False

                st.divider()
                if settings["right_plot"] == "time trend":
                    st.markdown("**Time Series Forecasting**")
                    settings["selected_model"] = st.selectbox(
                        "Model",
                        [
                            "prophet",
                            "auto_arima",
                            "arima",
                            "sarima",
                            "moving_average",
                            "xgboost",
                            "chronos",
                        ],
                        index=0,
                        key=f"{self.key_prefix}_model",
                        help=(
                            "auto_arima tunes (p, d, q) automatically; arima/sarima take "
                            "the orders from the inputs below."
                        ),
                    )
                    with st.form(
                        key=f"{self.key_prefix}_forecast_form",
                        border=False,
                    ):
                        settings["model_params"] = self._render_model_param_inputs(
                            settings["selected_model"],
                        )
                        settings["alpha_value"] = st.slider(
                            "Alpha",
                            min_value=0.01,
                            max_value=0.2,
                            value=0.05,
                            step=0.01,
                            key=f"{self.key_prefix}_alpha",
                        )
                        settings["points_to_use"] = st.number_input(
                            "Points to use",
                            min_value=6,
                            max_value=500,
                            value=50,
                            key=f"{self.key_prefix}_lookback",
                        )
                        settings["points_to_predict"] = st.number_input(
                            "Points to predict",
                            min_value=1,
                            max_value=int(settings["points_to_use"]),
                            value=min(10, int(settings["points_to_use"])),
                            key=f"{self.key_prefix}_predict",
                        )
                        settings["run_model_clicked"] = st.form_submit_button(
                            "Run model",
                            type="primary",
                            width="stretch",
                        )
                else:
                    st.markdown("**Distribution**")
                    settings["distribution_type"] = st.selectbox(
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
                    settings["distribution_orientation"] = st.selectbox(
                        "Distribution orientation",
                        ["vertical", "horizontal"],
                        index=0,
                        key=f"{self.key_prefix}_distribution_orientation",
                    )

        return settings

    def _render_dropped_log_notes(
        self,
        right_plot: str,
        dropped_map_points: int,
        dropped_time_trend_points: int,
        dropped_forecast_points: int,
        dropped_distribution_points: int,
        dropped_reference_points: int,
    ) -> None:
        """Emit a single caption summarising every row dropped by the log transform."""
        notes: list[str] = []
        if dropped_map_points:
            notes.append(f"map: {dropped_map_points}")
        if right_plot == "time trend":
            if dropped_time_trend_points:
                notes.append(f"time trend: {dropped_time_trend_points}")
            if dropped_forecast_points:
                notes.append(f"forecast: {dropped_forecast_points}")
        else:
            if dropped_distribution_points:
                notes.append(f"distribution: {dropped_distribution_points}")
            if dropped_reference_points:
                notes.append(f"distribution reference lines: {dropped_reference_points}")

        if notes:
            st.caption("ln transform skipped non-positive values in " + ", ".join(notes) + ".")

    @st.fragment
    def render_streamlit_ui(self):
        """Render the full indicator card and re-execute as a Streamlit fragment.

        This is the only public entry-point. Pages call it inside a column
        layout; everything (fetching, forecasting, log-transform, plotting,
        plot description, metadata expander) happens inside one fragment
        so toggling a control re-renders this card only.
        """
        schema = self._get_schema_mapping()
        use_log_key = f"{self.key_prefix}_use_log_transform"
        use_log = bool(st.session_state.get(use_log_key, False))

        with st.container(border=True):
            settings = self._render_header_and_settings(
                defaults={
                    "right_plot": "time trend",
                    "distribution_type": "histplot",
                    "distribution_orientation": "vertical",
                    "selected_model": "prophet",
                    "alpha_value": 0.05,
                    "points_to_use": 50,
                    "points_to_predict": 10,
                    "run_model_clicked": False,
                    "show_markers": False,
                    "model_params": {},
                },
            )
            right_plot = settings["right_plot"]
            distribution_type = settings["distribution_type"]
            distribution_orientation = settings["distribution_orientation"]
            selected_model = settings["selected_model"]
            alpha_value = settings["alpha_value"]
            points_to_use = settings["points_to_use"]
            points_to_predict = settings["points_to_predict"]
            run_model_clicked = settings["run_model_clicked"]
            show_markers = bool(settings.get("show_markers", False))
            model_params: dict[str, Any] = dict(settings.get("model_params") or {})

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

                df_year = with_year.filter(pl.col("__year") == selected_year).drop("__year")
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
                    "model_params": tuple(sorted(model_params.items())),
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
                            model_params=model_params,
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
                plotted_time_trend_df, dropped_time_trend_points = self._apply_log_to_columns(
                    df_time_trend, [schema["y"]]
                )

                if df_forecast is not None and not df_forecast.is_empty():
                    plotted_forecast_df, dropped_forecast_points = self._apply_log_to_columns(
                        df_forecast,
                        [
                            schema["y"],
                            f"{schema['y']}_lower",
                            f"{schema['y']}_upper",
                        ],
                    )

            map_title = "Map"
            metadata_for_hover = self._get_metadata()
            hover_context = self._build_hover_context(metadata_for_hover)
            map_value_label = "Value"
            dropped_map_points = 0
            country_lookup = get_world_bank_country_mapping()
            map_df = (
                df_year.filter(
                    pl.col(schema["group"]).cast(pl.Utf8).str.to_uppercase().str.len_chars() == 3
                )
                .group_by(schema["group"])
                .agg(pl.col(schema["y"]).last().alias(schema["y"]))
            )
            if not country_lookup.is_empty() and {"id", "value"}.issubset(
                set(country_lookup.columns)
            ):
                map_df = map_df.join(
                    country_lookup.rename({"id": schema["group"], "value": "country_name"}),
                    on=schema["group"],
                    how="left",
                ).with_columns(
                    pl.coalesce([pl.col("country_name"), pl.col(schema["group"])]).alias(
                        "country_name"
                    )
                )
            if use_log:
                map_df, dropped_map_points = self._apply_log_to_columns(map_df, [schema["y"]])
                map_title = "Map (ln)"
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
                    show_markers=show_markers,
                )
            else:
                distribution_df = df_year
                if use_log:
                    distribution_df, dropped_distribution_points = self._apply_log_to_columns(
                        df_year, [schema["y"]]
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
                                    self._apply_log_to_columns(selected_levels_df, [schema["y"]])
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
                self._render_dropped_log_notes(
                    right_plot=right_plot,
                    dropped_map_points=dropped_map_points,
                    dropped_time_trend_points=dropped_time_trend_points,
                    dropped_forecast_points=dropped_forecast_points,
                    dropped_distribution_points=dropped_distribution_points,
                    dropped_reference_points=dropped_reference_points,
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
