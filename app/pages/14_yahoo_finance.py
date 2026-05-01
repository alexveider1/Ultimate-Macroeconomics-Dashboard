from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from core.assets import get_markup_template
from core.app_logging import log_page_render
from core.plotting import apply_plotly_theme, build_line_plot
from core.theming import get_color
from core.postgres_client import get_all_yahoo_historical_prices, get_all_yahoo_metadata


YAHOO_KEY_PREFIX = "yahoo_finance_yahoo_market_overview"
DEFAULT_COMPANY_TICKERS = ["META", "AAPL", "AMZN", "GOOGL", "MSFT"]
PREFERRED_INDEX_GROUPS = [
    (
        {"^GSPC", "SPX", "SP500", "S&P500"},
        ["sp500", "s&p 500", "s and p 500", "standard & poor"],
    ),
    (
        {"^DJI", "DJIA", "DOW"},
        ["dow jones", "djia", "dow industrial"],
    ),
    (
        {"^IXIC", "NASDAQ", "NDX"},
        ["nasdaq", "nasdaq composite"],
    ),
]


def build_yahoo_candlestick_plot(
    df: pl.DataFrame,
    date_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    required_cols = [date_col, open_col, high_col, low_col, close_col]

    if df.is_empty() or any(col not in df.columns for col in required_cols):
        fig.add_annotation(
            text="No OHLC data available for candlestick chart.", showarrow=False
        )
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
        fig.add_annotation(
            text="No OHLC data available for candlestick chart.", showarrow=False
        )
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


def build_yahoo_treemap_plot(
    df: pl.DataFrame,
    labels_col: str,
    parents_col: str,
    values_col: str,
    title: str = "",
    hover_col: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()

    if df.is_empty() or values_col not in df.columns:
        fig.add_annotation(text="No data available for treemap.", showarrow=False)
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    customdata = (
        df[hover_col].to_list() if hover_col and hover_col in df.columns else None
    )
    trace = go.Treemap(
        labels=df[labels_col].to_list(),
        parents=df[parents_col].to_list(),
        values=df[values_col].to_list(),
        branchvalues="remainder",
    )
    if customdata is not None:
        trace.customdata = customdata
        trace.hovertemplate = get_markup_template("treemap_hovertemplate")

    fig.add_trace(trace)
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return apply_plotly_theme(fig)


def build_yahoo_correlation_heatmap(
    df: pl.DataFrame,
    date_col: str,
    ticker_col: str,
    value_col: str,
    title: str = "",
    ticker_name_map: Optional[dict[str, str]] = None,
) -> go.Figure:
    fig = go.Figure()

    if df.is_empty() or any(
        col not in df.columns for col in [date_col, ticker_col, value_col]
    ):
        fig.add_annotation(
            text="No data available for correlation heatmap.", showarrow=False
        )
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    pivot = (
        df.select([date_col, ticker_col, value_col])
        .pivot(
            values=value_col,
            index=date_col,
            columns=ticker_col,
            aggregate_function="last",
        )
        .sort(date_col)
    )
    tickers = [col for col in pivot.columns if col != date_col]
    if len(tickers) < 2:
        fig.add_annotation(
            text="Need at least two tickers to compute correlations.", showarrow=False
        )
        fig.update_layout(title=title)
        return apply_plotly_theme(fig)

    corr_rows: list[list[float]] = []
    customdata_rows: list[list[list[str]]] = []

    def _format_ticker_label(ticker: str) -> str:
        if not ticker_name_map:
            return ticker
        company_name = ticker_name_map.get(ticker, ticker)
        if not company_name or company_name == ticker:
            return ticker
        return f"{ticker} - {company_name}"

    for row_ticker in tickers:
        row_vals: list[float] = []
        row_customdata: list[list[str]] = []
        for col_ticker in tickers:
            pair_df = pivot.select(
                [
                    pl.col(row_ticker).cast(pl.Float64).alias("x"),
                    pl.col(col_ticker).cast(pl.Float64).alias("y"),
                ]
            ).drop_nulls()

            row_customdata.append(
                [
                    _format_ticker_label(row_ticker),
                    _format_ticker_label(col_ticker),
                ]
            )

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
            x=tickers,
            y=tickers,
            customdata=customdata_rows,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            colorbar_title="Corr",
            hovertemplate=get_markup_template("correlation_heatmap_hovertemplate"),
        )
    )
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return apply_plotly_theme(fig)


def _build_combined_label_map(df: pl.DataFrame) -> dict[str, str]:
    label_df = df.select(
        [
            pl.col("ticker"),
            pl.coalesce([pl.col("asset_name"), pl.col("ticker")]).alias("asset_name"),
        ]
    ).unique(subset=["ticker"], keep="last")
    return dict(
        zip(
            label_df["ticker"].to_list(),
            [
                ticker if name == ticker else f"{ticker} - {name}"
                for ticker, name in zip(
                    label_df["ticker"].to_list(),
                    label_df["asset_name"].to_list(),
                )
            ],
        )
    )


def _build_asset_name_map(df: pl.DataFrame) -> dict[str, str]:
    label_df = df.select(
        [
            pl.col("ticker"),
            pl.coalesce([pl.col("asset_name"), pl.col("ticker")]).alias("asset_name"),
        ]
    ).unique(subset=["ticker"], keep="last")
    return dict(zip(label_df["ticker"].to_list(), label_df["asset_name"].to_list()))


def _default_index_selection(
    available_index_tickers: list[str],
    index_label_map: dict[str, str],
) -> list[str]:
    selected: list[str] = []
    for ticker_aliases, label_tokens in PREFERRED_INDEX_GROUPS:
        matched_ticker = None
        for ticker in available_index_tickers:
            ticker_upper = str(ticker).upper()
            label_lower = index_label_map.get(ticker, ticker).lower()
            if ticker_upper in ticker_aliases or any(
                token in label_lower for token in label_tokens
            ):
                matched_ticker = ticker
                break
        if matched_ticker and matched_ticker not in selected:
            selected.append(matched_ticker)

    if not selected:
        return available_index_tickers[:3]
    return selected


def render_yahoo_finance_dashboard() -> None:
    log_page_render("Yahoo Finance Dashboard")
    st.title("Yahoo Finance Dashboard")
    st.caption(
        "Yahoo-specific charts and controls live on this page: company trends, candlestick, treemap, correlation heatmap, and index trends."
    )

    with st.container(border=True):
        st.markdown("**Yahoo Finance Market Overview**")

        hist_df = get_all_yahoo_historical_prices()
        meta_df = get_all_yahoo_metadata()

        if hist_df.is_empty():
            st.info("No Yahoo finance data available.")
            return

        if meta_df.is_empty() or "ticker" not in meta_df.columns:
            st.info(
                "Yahoo metadata is missing. Unable to separate companies and indices."
            )
            return

        meta_min = meta_df.select(
            [
                pl.col("ticker"),
                pl.col("category"),
                pl.col("sector"),
                pl.col("asset_name"),
            ]
        )
        hist_enriched = hist_df.join(meta_min, on="ticker", how="left").with_columns(
            pl.col("category").cast(pl.Utf8).fill_null("Unknown").alias("category"),
            pl.col("sector").cast(pl.Utf8).fill_null("Unknown").alias("sector"),
            pl.coalesce([pl.col("asset_name"), pl.col("ticker")]).alias("asset_name"),
        )

        companies_all = hist_enriched.filter(
            pl.col("category").str.to_lowercase() == "companies"
        ).with_columns(pl.col("date").dt.year().alias("__year"))
        indices_all = hist_enriched.filter(
            pl.col("category").str.to_lowercase() == "indices"
        )

        if companies_all.is_empty():
            st.info("No records found for category 'Companies' in yahoo metadata.")
            return

        available_company_tickers = (
            companies_all.select(pl.col("ticker").cast(pl.Utf8))
            .drop_nulls()
            .unique()
            .sort("ticker")["ticker"]
            .to_list()
        )
        if not available_company_tickers:
            st.info("No company tickers are available in yahoo_historical_prices.")
            return

        company_label_map = _build_combined_label_map(companies_all)
        default_company_selection = [
            ticker
            for ticker in DEFAULT_COMPANY_TICKERS
            if ticker in available_company_tickers
        ]
        if not default_company_selection:
            default_company_selection = available_company_tickers[:6]

        st.markdown("### Yahoo Companies Trend (All History)")
        selected_companies = st.multiselect(
            "Select companies (max 20)",
            options=available_company_tickers,
            default=default_company_selection,
            max_selections=20,
            format_func=lambda ticker: company_label_map.get(ticker, ticker),
            key=f"{YAHOO_KEY_PREFIX}_company_trend_selection",
        )

        if selected_companies:
            company_trend_df = (
                companies_all.filter(pl.col("ticker").is_in(selected_companies))
                .sort(["date", "ticker"])
                .group_by(["date", "ticker", "asset_name"])
                .agg(pl.col("close").last().alias("close"))
                .with_columns(
                    pl.when(pl.col("asset_name") == pl.col("ticker"))
                    .then(pl.col("ticker"))
                    .otherwise(
                        pl.concat_str(
                            [pl.col("ticker"), pl.lit(" - "), pl.col("asset_name")]
                        )
                    )
                    .alias("company_label")
                )
            )
            company_trend_fig = build_line_plot(
                company_trend_df,
                x_col="date",
                y_col="close",
                group_col="company_label",
                title="Selected Companies Close Trend",
            )
            st.plotly_chart(
                company_trend_fig,
                width="stretch",
                key=f"{YAHOO_KEY_PREFIX}_company_trend_line",
            )
        else:
            st.info("Select at least one company to display the trend chart.")

        candlestick_default_ticker = (
            "NVDA"
            if "NVDA" in available_company_tickers
            else default_company_selection[0]
            if default_company_selection
            else available_company_tickers[0]
        )
        candlestick_default_index = (
            available_company_tickers.index(candlestick_default_ticker)
            if candlestick_default_ticker in available_company_tickers
            else 0
        )

        st.markdown("### Yahoo Company Candlestick (All History)")
        selected_candlestick_ticker = st.selectbox(
            "Select company for candlestick",
            options=available_company_tickers,
            index=candlestick_default_index,
            format_func=lambda ticker: company_label_map.get(ticker, ticker),
            key=f"{YAHOO_KEY_PREFIX}_company_candlestick_selection",
        )

        candlestick_df = companies_all.filter(
            pl.col("ticker") == selected_candlestick_ticker
        ).sort("date")
        candlestick_fig = build_yahoo_candlestick_plot(
            candlestick_df,
            date_col="date",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            title=(
                f"{company_label_map.get(selected_candlestick_ticker, selected_candlestick_ticker)} Candlestick"
            ),
        )
        st.plotly_chart(
            candlestick_fig,
            width="stretch",
            key=f"{YAHOO_KEY_PREFIX}_company_candlestick",
        )

        years = companies_all["__year"].drop_nulls().unique().sort().to_list()
        if not years:
            st.info("No year information found for Yahoo Companies data.")
            return

        min_year = int(years[0])
        max_year = int(years[-1])
        default_year = max(min(datetime.now().year - 1, max_year), min_year)
        year_key = f"{YAHOO_KEY_PREFIX}_selected_year"
        if year_key not in st.session_state:
            st.session_state[year_key] = default_year

        selected_year = st.slider(
            "Year filter",
            min_value=min_year,
            max_value=max_year,
            key=year_key,
            help="Applies to Companies treemap and Companies correlation heatmap only.",
        )

        year_df = companies_all.filter(pl.col("__year") == int(selected_year)).drop(
            "__year"
        )
        if year_df.is_empty():
            st.info("No Companies observations in selected year.")
            return

        latest_df = (
            year_df.sort(["ticker", "date"])
            .group_by("ticker")
            .agg(
                pl.col("close").last().alias("latest_close"),
                pl.col("volume").last().alias("latest_volume"),
            )
            .join(meta_min, on="ticker", how="left")
            .with_columns(
                pl.col("sector").fill_null("Unknown").alias("sector"),
                pl.coalesce([pl.col("asset_name"), pl.col("ticker")]).alias(
                    "asset_name"
                ),
            )
        )

        leaf_df = latest_df.with_columns(
            pl.col("ticker").alias("label"),
            pl.col("sector").alias("parent"),
            pl.when(pl.col("latest_volume").fill_null(0) > 0)
            .then(pl.col("latest_volume").cast(pl.Float64))
            .otherwise(pl.col("latest_close").abs().fill_null(1).cast(pl.Float64))
            .alias("size_value"),
            pl.format(
                get_markup_template("yahoo_leaf_hover_text"),
                pl.col("ticker"),
                pl.coalesce([pl.col("asset_name"), pl.col("ticker")]),
                pl.col("latest_close").round(4).cast(pl.Utf8),
            ).alias("hover_text"),
        ).select(["label", "parent", "size_value", "hover_text"])

        parent_pairs = latest_df.select(
            pl.col("sector").alias("label"),
            pl.lit("Companies").alias("parent"),
        ).unique()
        category_pairs = pl.DataFrame({"label": ["Companies"], "parent": [""]})

        ticker_name_map = _build_asset_name_map(latest_df)
        treemap_df = pl.concat(
            [
                leaf_df,
                parent_pairs.with_columns(
                    pl.lit(0.0).alias("size_value"),
                    pl.lit("Sector group").alias("hover_text"),
                ),
                category_pairs.with_columns(
                    pl.lit(0.0).alias("size_value"),
                    pl.lit("Category group").alias("hover_text"),
                ),
            ],
            how="vertical_relaxed",
        )

        index_tickers = (
            indices_all.select(pl.col("ticker").cast(pl.Utf8))
            .drop_nulls()
            .unique()
            .to_series()
            .to_list()
        )
        heatmap_source_df = year_df
        if index_tickers:
            heatmap_source_df = heatmap_source_df.filter(
                ~pl.col("ticker").cast(pl.Utf8).is_in(index_tickers)
            )
        heatmap_source_df = heatmap_source_df.filter(
            ~pl.col("ticker").cast(pl.Utf8).str.starts_with("^")
        )

        returns_df = (
            heatmap_source_df.sort(["ticker", "date"])
            .with_columns(pl.col("close").pct_change().over("ticker").alias("ret"))
            .drop_nulls(subset=["ret"])
        )

        treemap_fig = build_yahoo_treemap_plot(
            treemap_df,
            labels_col="label",
            parents_col="parent",
            values_col="size_value",
            title=f"Yahoo Finance Treemap ({selected_year})",
            hover_col="hover_text",
        )
        heatmap_fig = build_yahoo_correlation_heatmap(
            returns_df,
            date_col="date",
            ticker_col="ticker",
            value_col="ret",
            title=f"Yahoo Companies Correlation Heatmap ({selected_year})",
            ticker_name_map=ticker_name_map,
        )

        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.plotly_chart(
                treemap_fig,
                width="stretch",
                key=f"{YAHOO_KEY_PREFIX}_yahoo_treemap",
            )
        with right_col:
            st.plotly_chart(
                heatmap_fig,
                width="stretch",
                key=f"{YAHOO_KEY_PREFIX}_yahoo_corr_heatmap",
            )

        st.markdown("### Yahoo Indices Trend (All History)")
        if indices_all.is_empty():
            st.info("No records found for category 'Indices' in yahoo metadata.")
            return

        index_label_map = _build_asset_name_map(indices_all)
        available_index_tickers = (
            indices_all.select(pl.col("ticker").cast(pl.Utf8))
            .drop_nulls()
            .unique()
            .sort("ticker")["ticker"]
            .to_list()
        )
        default_index_selection = _default_index_selection(
            available_index_tickers,
            index_label_map,
        )

        selected_indices = st.multiselect(
            "Select indices",
            options=available_index_tickers,
            default=default_index_selection,
            format_func=lambda ticker: index_label_map.get(ticker, ticker),
            key=f"{YAHOO_KEY_PREFIX}_indices_trend_selection",
        )

        if selected_indices:
            indices_trend_df = (
                indices_all.filter(pl.col("ticker").is_in(selected_indices))
                .sort(["date", "ticker"])
                .group_by(["date", "ticker", "asset_name"])
                .agg(pl.col("close").last().alias("close"))
                .with_columns(
                    pl.coalesce([pl.col("asset_name"), pl.col("ticker")]).alias(
                        "index_label"
                    )
                )
            )
            idx_line = build_line_plot(
                indices_trend_df,
                x_col="date",
                y_col="close",
                group_col="index_label",
                title="Selected Indices Close Trend (All Years)",
            )
            idx_line.update_traces(fill="tozeroy", opacity=0.45)
            st.plotly_chart(
                idx_line,
                width="stretch",
                key=f"{YAHOO_KEY_PREFIX}_indices_line",
            )
        else:
            st.info("Select at least one index to display the trend chart.")


render_yahoo_finance_dashboard()
