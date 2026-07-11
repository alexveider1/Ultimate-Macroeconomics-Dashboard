"""Crypto (Binance) dashboard — top coins, BTC candlestick, correlation grid.

Renders the 30 most-traded Binance USDT coins ingested by ``downloader_general``:
a market-overview table, a top-coin price-dynamics line chart (log scale so coins
at very different price levels stay comparable), a Bitcoin candlestick, and a
return-correlation heatmap across every coin. Driven entirely from the cached
Polars frames loaded by ``core.postgres_client``.
"""

import polars as pl
import streamlit as st

from core.app_logging import log_page_render
from core.plotting import (
    build_candlestick_plot,
    build_correlation_heatmap,
    build_line_plot,
)
from core.postgres_client import (
    get_all_binance_historical_prices,
    get_all_binance_metadata,
)

CRYPTO_KEY_PREFIX = "binance_crypto"
TOP_N_TREND = 5
BITCOIN_BASE_ASSET = "BTC"


def _available_base_assets(hist_df: pl.DataFrame) -> list[str]:
    """Return the sorted, distinct ``base_asset`` codes present in the history."""
    return (
        hist_df.select(pl.col("base_asset").cast(pl.Utf8))
        .drop_nulls()
        .unique()
        .sort("base_asset")["base_asset"]
        .to_list()
    )


def _render_market_overview(meta_df: pl.DataFrame) -> None:
    """Show the ranked master-data table (rank, price, 24h change, volume)."""
    st.markdown("### Market Overview (ranked by 24h quote volume)")
    overview = meta_df.sort("rank").select(
        [
            pl.col("rank").alias("Rank"),
            pl.col("base_asset").alias("Coin"),
            pl.col("symbol").alias("Pair"),
            pl.col("last_price").alias("Last price"),
            pl.col("price_change_percent_24h").alias("24h %"),
            pl.col("quote_volume_24h").alias("24h volume (USDT)"),
            pl.col("trade_count_24h").alias("24h trades"),
        ]
    )
    st.dataframe(overview, width="stretch", hide_index=True)


def _render_top_trend(hist_df: pl.DataFrame, meta_df: pl.DataFrame) -> None:
    """Render the top-coin close-price trend (log scale) with a coin multiselect."""
    st.markdown("### Top Coins Price Dynamics (All History)")
    st.caption(
        "Close price on a logarithmic axis so coins at different price levels stay readable."
    )

    available = _available_base_assets(hist_df)
    if not available:
        st.info("No crypto history available.")
        return

    top_default = [
        coin
        for coin in meta_df.sort("rank")["base_asset"].cast(pl.Utf8).to_list()
        if coin in available
    ][:TOP_N_TREND] or available[:TOP_N_TREND]

    selected = st.multiselect(
        "Select coins (max 15)",
        options=available,
        default=top_default,
        max_selections=15,
        key=f"{CRYPTO_KEY_PREFIX}_trend_selection",
    )
    if not selected:
        st.info("Select at least one coin to display the trend chart.")
        return

    trend_df = (
        hist_df.filter(pl.col("base_asset").is_in(selected))
        .sort(["date", "base_asset"])
        .group_by(["date", "base_asset"])
        .agg(pl.col("close").last().alias("close"))
        .filter(pl.col("close") > 0)
    )
    fig = build_line_plot(
        trend_df,
        x_col="date",
        y_col="close",
        group_col="base_asset",
        title="Selected Coins Close Trend (log scale)",
    )
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch", key=f"{CRYPTO_KEY_PREFIX}_trend_line")


def _render_bitcoin_candlestick(hist_df: pl.DataFrame) -> None:
    """Render the Bitcoin daily candlestick over its full history."""
    st.markdown("### Bitcoin Candlestick (All History)")
    btc_df = hist_df.filter(pl.col("base_asset") == BITCOIN_BASE_ASSET).sort("date")
    if btc_df.is_empty():
        st.info("No Bitcoin (BTC) history available.")
        return
    fig = build_candlestick_plot(
        btc_df,
        date_col="date",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        title="BTC/USDT Candlestick",
    )
    st.plotly_chart(fig, width="stretch", key=f"{CRYPTO_KEY_PREFIX}_btc_candlestick")


def _render_correlation_heatmap(hist_df: pl.DataFrame) -> None:
    """Render the daily-return correlation heatmap across every coin."""
    st.markdown("### Coin Return Correlation Heatmap (All Coins)")
    st.caption("Pearson correlation of daily returns across the full overlapping history.")
    returns_df = (
        hist_df.sort(["base_asset", "date"])
        .with_columns(pl.col("close").pct_change().over("base_asset").alias("ret"))
        .drop_nulls(subset=["ret"])
    )
    fig = build_correlation_heatmap(
        returns_df,
        date_col="date",
        series_col="base_asset",
        value_col="ret",
        title="Daily Return Correlation",
    )
    st.plotly_chart(fig, width="stretch", key=f"{CRYPTO_KEY_PREFIX}_corr_heatmap")


def render_crypto_dashboard() -> None:
    """Page entry-point: market overview, top-coin trend, BTC candlestick, correlations."""
    log_page_render("Crypto Dashboard")
    st.title("Crypto (Binance) Dashboard")
    st.caption(
        "The 30 most actively traded Binance USDT pairs: market overview, top-coin "
        "price dynamics, a Bitcoin candlestick, and a return-correlation heatmap."
    )

    with st.container(border=True):
        hist_df = get_all_binance_historical_prices()
        meta_df = get_all_binance_metadata()

        if hist_df.is_empty() or meta_df.is_empty():
            st.info(
                "No crypto data available. It is ingested on a clean boot of the "
                "`downloader_general` container."
            )
            return

        _render_market_overview(meta_df)
        _render_top_trend(hist_df, meta_df)

        left_col, right_col = st.columns([1, 1])
        with left_col:
            _render_bitcoin_candlestick(hist_df)
        with right_col:
            _render_correlation_heatmap(hist_df)


render_crypto_dashboard()
