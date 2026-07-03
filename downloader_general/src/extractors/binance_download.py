"""Binance crypto ingestion pipeline.

Selects the most popular spot pairs (ranked by trailing-24h quote volume) from
the documented Binance public REST API, writes their master data to
``binance_metadata``, then pulls each pair's full daily candle history into
``binance_historical_prices``. All HTTP goes through the async
:mod:`src.utils.binance_client`; per-symbol history fetches run concurrently
under a semaphore. No API key is required (public market-data endpoints only).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from src.core.base_downloaders import BaseBinanceDownloader
from src.settings import load_settings
from src.utils import binance_client, wb_client
from src.utils.downloads import (
    _download_config,
    _get_sql_config,
    _test_sql,
    log_progress,
)
from src.utils.schema import (
    bootstrap_schema_group,
    get_table_definition,
    write_polars_to_table,
)

logger = logging.getLogger(__name__)


def _is_leveraged_token(base_asset: str) -> bool:
    """Return ``True`` for Binance leveraged tokens (``…UP``/``…DOWN``/``BULL``/``BEAR``)."""
    upper = base_asset.upper()
    return upper.endswith(("UP", "DOWN")) or "BULL" in upper or "BEAR" in upper


class BinanceDownloader(BaseBinanceDownloader):
    """Concrete Binance downloader writing to Postgres via Polars."""

    SCHEMA_GROUP = "binance"

    def __init__(
        self,
        env_path: str | Path,
        download_config_path: str | Path,
        database_schema: Optional[Dict[str, Any]] = None,
    ):
        """Capture configuration; the Postgres URI is built in ``_initialize_connections``.

        Args:
            env_path: Path to the ``.env`` with Postgres credentials.
            download_config_path: JSON file with the Binance download tunables.
            database_schema: Parsed schema so column lists/dtypes are typed.
        """
        self.env_path = Path(env_path)
        config = _download_config(download_config_path)

        self.base_url = config.get("base_url", binance_client.DEFAULT_BASE_URL)
        self.quote_asset = str(config.get("quote_asset", "USDT")).upper()
        self.top_n = int(config.get("top_n", 30))
        self.kline_interval = str(config.get("kline_interval", "1d"))
        self.max_parallel_symbols = int(config.get("max_parallel_symbols", 6))
        self.exclude_base_assets = {
            str(asset).upper() for asset in config.get("exclude_base_assets", [])
        }

        self.sql_uri: Optional[str] = None
        self.metadata_table_name = "binance_metadata"
        self.historical_data_table_name = "binance_historical_prices"
        self.database_schema = database_schema or {}
        self.successful_connections = False

        self.download_max_retries = 5
        self.download_retry_delay_seconds = 5

    def _table_def(self, table_name: str) -> Dict[str, Any]:
        return get_table_definition(self.database_schema, self.SCHEMA_GROUP, table_name)

    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        secrets = load_settings(self.env_path)
        sql_uri = _get_sql_config(
            username=secrets.postgres_user,
            password=secrets.postgres_password,
            host=host,
            port=port,
            db=db,
        )
        if _test_sql(sql_uri):
            self.sql_uri = sql_uri
            self.successful_connections = True
        else:
            self.sql_uri = None
            self.successful_connections = False
            logger.warning("Connection test to SQL database failed")
        return self.successful_connections

    def _valid_base_assets(self, exchange_info: List[Dict[str, Any]]) -> Dict[str, str]:
        """Map ``symbol → base_asset`` for tradable spot pairs in the target quote asset."""
        valid: Dict[str, str] = {}
        for entry in exchange_info:
            if entry.get("quoteAsset", "").upper() != self.quote_asset:
                continue
            if entry.get("status") != "TRADING" or not entry.get("isSpotTradingAllowed", False):
                continue
            base_asset = str(entry.get("baseAsset", ""))
            if not base_asset or base_asset.upper() in self.exclude_base_assets:
                continue
            if _is_leveraged_token(base_asset):
                continue
            symbol = entry.get("symbol")
            if symbol:
                valid[symbol] = base_asset
        return valid

    async def select_top_symbols(self, client) -> List[Dict[str, Any]]:
        """Return the top-``top_n`` pairs by 24h quote volume, each with its metadata row."""
        exchange_info = await wb_client.call_with_retries(
            operation_name="binance.exchangeInfo",
            request_coro_factory=lambda: binance_client.fetch_exchange_info(client),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        tickers = await wb_client.call_with_retries(
            operation_name="binance.ticker24hr",
            request_coro_factory=lambda: binance_client.fetch_24h_tickers(client),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if not exchange_info or not tickers:
            logger.warning("Binance selection skipped: missing exchangeInfo or 24h tickers")
            return []

        valid = self._valid_base_assets(exchange_info)
        candidates = [ticker for ticker in tickers if ticker.get("symbol") in valid]
        candidates.sort(
            key=lambda t: binance_client._to_float(t.get("quoteVolume")) or 0.0,
            reverse=True,
        )

        rows: List[Dict[str, Any]] = []
        for rank, ticker in enumerate(candidates[: self.top_n], start=1):
            symbol = ticker["symbol"]
            base_asset = valid[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "base_asset": base_asset,
                    "quote_asset": self.quote_asset,
                    "status": "TRADING",
                    "rank": rank,
                    "description": (
                        f"{base_asset}/{self.quote_asset} spot pair on Binance — "
                        f"24h volume rank #{rank}."
                    ),
                    "last_price": binance_client._to_float(ticker.get("lastPrice")),
                    "price_change_percent_24h": binance_client._to_float(
                        ticker.get("priceChangePercent")
                    ),
                    "high_24h": binance_client._to_float(ticker.get("highPrice")),
                    "low_24h": binance_client._to_float(ticker.get("lowPrice")),
                    "quote_volume_24h": binance_client._to_float(ticker.get("quoteVolume")),
                    "trade_count_24h": ticker.get("count"),
                }
            )

        logger.info("Selected %d Binance symbols by 24h quote volume", len(rows))
        return rows

    def download_metadata(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            logger.warning("No Binance metadata rows to write")
            return
        df = pl.DataFrame(rows)
        write_polars_to_table(
            df,
            sql_uri=self.sql_uri,
            table_name=self.metadata_table_name,
            table_def=self._table_def(self.metadata_table_name),
        )
        logger.info("Wrote %d rows to %s", df.height, self.metadata_table_name)

    async def download_historical_data(self, client, symbol: str, base_asset: str) -> None:
        logger.info("Starting download of historical data (symbol=%s)", symbol)
        rows = await wb_client.call_with_retries(
            operation_name=f"binance.klines(symbol={symbol})",
            request_coro_factory=lambda: binance_client.fetch_klines(
                client, symbol, interval=self.kline_interval
            ),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if not rows:
            logger.warning("No historical candles for %s; skipping write", symbol)
            return

        df = pl.DataFrame(rows).with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(base_asset).alias("base_asset"),
        )
        await asyncio.to_thread(
            write_polars_to_table,
            df,
            self.sql_uri,
            self.historical_data_table_name,
            self._table_def(self.historical_data_table_name),
        )
        logger.info("Finished download of historical data (symbol=%s, rows=%d)", symbol, df.height)

    async def _run_async(self) -> None:
        async with binance_client.build_async_client(self.base_url) as client:
            if not await binance_client.healthcheck(client):
                logger.warning("Binance API healthcheck failed; skipping download")
                return

            selected = await self.select_top_symbols(client)
            if not selected:
                return

            # Metadata is the FK target, so it must land before the price rows.
            await asyncio.to_thread(self.download_metadata, selected)

            semaphore = asyncio.Semaphore(self.max_parallel_symbols)

            async def _bounded(symbol: str, base_asset: str) -> None:
                async with semaphore:
                    await self.download_historical_data(client, symbol, base_asset)

            tasks = [
                asyncio.create_task(_bounded(row["symbol"], row["base_asset"])) for row in selected
            ]
            for future in log_progress(
                asyncio.as_completed(tasks),
                label="Downloading Binance history",
                total=len(tasks),
            ):
                try:
                    await future
                except Exception:
                    logger.exception("Binance history download failed for a symbol")

    def run(self) -> None:
        bootstrap_schema_group(self.sql_uri, self.database_schema, self.SCHEMA_GROUP)
        asyncio.run(self._run_async())
