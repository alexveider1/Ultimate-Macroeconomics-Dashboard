"""Yahoo Finance ingestion pipeline.

Per category, pulls metadata (sector / industry / currency / etc.) followed by
the full historical OHLCV series for each ticker via ``yfinance``. Per-ticker
fetches run on a thread pool because each call is independent.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import polars as pl
import yfinance as yf

from src.core.base_downloaders import BaseYahooDownloader
from src.settings import load_settings
from src.utils.downloads import (
    _call_with_retries,
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


class YahooDownloader(BaseYahooDownloader):
    """Concrete Yahoo Finance downloader writing to Postgres via Polars."""

    SCHEMA_GROUP = "yahoo_finance"

    def __init__(
        self,
        env_path: str | Path,
        download_config_path: str | Path,
        database_schema: Optional[Dict[str, Any]] = None,
    ):
        """Capture configuration; Postgres URI is built in ``_initialize_connections``.

        Args:
            env_path: Path to the ``.env`` with Postgres credentials.
            download_config_path: JSON file listing categories → tickers.
            database_schema: Parsed schema so column lists/dtypes are typed.
        """
        self.env_path = Path(env_path)
        self.download_config = _download_config(download_config_path)
        self.sql_uri = None

        self.historical_data_table_name = "yahoo_historical_prices"
        self.metadata_table_name = "yahoo_metadata"

        self.database_schema = database_schema or {}

        self.successful_connections = False

        self.download_max_retries = 3
        self.download_retry_delay_seconds = 5
        # yfinance is per-ticker synchronous; a small pool gives a clear win
        # without overwhelming the Yahoo endpoint.
        self.max_parallel_tickers = 6

    def _table_def(self, table_name: str) -> Dict[str, Any]:
        return get_table_definition(self.database_schema, self.SCHEMA_GROUP, table_name)

    def _normalize_assets(self, category: str, assets: Any) -> Iterable[Dict[str, str]]:
        if isinstance(assets, dict):
            for asset_name, ticker_id in assets.items():
                yield {"id": ticker_id, "name": asset_name}
            return

        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, dict):
                    yield asset
                    continue

                logger.warning(
                    "Skipping unsupported asset entry in category '%s': %r",
                    category,
                    asset,
                )
            return

        logger.warning(
            "Skipping unsupported assets container in category '%s': %r",
            category,
            assets,
        )

    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        secrets = load_settings(self.env_path)
        self.sql_uri = _get_sql_config(
            username=secrets.postgres_user,
            password=secrets.postgres_password,
            host=host,
            port=port,
            db=db,
        )

        if _sql_test := _test_sql(self.sql_uri):
            self.sql_uri = self.sql_uri
        else:
            self.sql_uri = None
            logger.warning("Connection test to SQL database failed")
        self.successful_connections = _sql_test
        return self.successful_connections

    def download_historical_data(self, ticker_id: str, category: str, period: str = "max") -> None:
        logger.info(f"Starting download of historical data (ticker={ticker_id})")

        ticker_obj = yf.Ticker(ticker_id)

        hist_df_pandas = _call_with_retries(
            operation_name=f"yfinance.history(ticker={ticker_id})",
            request_callable=lambda: ticker_obj.history(period=period),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if hist_df_pandas is None:
            logger.warning(f"Skipping historical data for {ticker_id}: all retries failed.")
            return

        if hist_df_pandas.empty:
            logger.warning(f"No historical data found for {ticker_id}.")
            return

        hist_df_pandas = hist_df_pandas.reset_index()
        hist_df_pandas["Date"] = hist_df_pandas["Date"].dt.tz_localize(None)

        df = pl.from_pandas(hist_df_pandas)

        df = df.select(
            [
                pl.col("Date").alias("date"),
                pl.col("Open").alias("open"),
                pl.col("High").alias("high"),
                pl.col("Low").alias("low"),
                pl.col("Close").alias("close"),
                pl.col("Volume").alias("volume"),
            ]
        ).with_columns(
            [
                pl.lit(ticker_id).alias("ticker"),
                pl.lit(category).alias("category"),
            ]
        )

        write_polars_to_table(
            df,
            sql_uri=self.sql_uri,
            table_name=self.historical_data_table_name,
            table_def=self._table_def(self.historical_data_table_name),
        )

        logger.info(f"Finished download of historical data (ticker={ticker_id})")

    def download_metadata(self, ticker_id: str, asset_name: str | None, category: str) -> bool:
        logger.info(f"Starting download of metadata (ticker={ticker_id})")

        ticker_obj = yf.Ticker(ticker_id)

        info_dict: Optional[Dict[str, Any]] = _call_with_retries(
            operation_name=f"yfinance.info(ticker={ticker_id})",
            request_callable=lambda: ticker_obj.info,
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if info_dict is None:
            logger.warning(f"Skipping metadata for {ticker_id}: all retries failed.")
            return False

        dataframe_dict = {
            "ticker": ticker_id,
            "asset_name": asset_name,
            "category": category,
            "short_name": info_dict.get("shortName"),
            "sector": info_dict.get("sector"),
            "industry": info_dict.get("industry"),
            "currency": info_dict.get("currency"),
            "exchange": info_dict.get("exchange"),
            "business_summary": info_dict.get("longBusinessSummary"),
        }

        df = pl.DataFrame([dataframe_dict])

        write_polars_to_table(
            df,
            sql_uri=self.sql_uri,
            table_name=self.metadata_table_name,
            table_def=self._table_def(self.metadata_table_name),
        )

        logger.info(f"Finished download of metadata (ticker={ticker_id})")
        return True

    def _download_asset(self, ticker_id: str, asset_name: str | None, category: str) -> None:
        """Per-ticker work unit: metadata first (FK target), then historical."""
        metadata_ok = self.download_metadata(ticker_id, asset_name, category)
        if not metadata_ok:
            logger.warning(
                f"Skipping historical data for {ticker_id}: metadata write was skipped, "
                f"FK to yahoo_metadata.ticker would fail."
            )
            return
        self.download_historical_data(ticker_id, category, period="max")

    def download_category(self, category: str, assets: Any) -> None:
        logger.info(f"Starting downloads for category: {category}")

        normalized_assets = list(self._normalize_assets(category, assets))
        tasks = [
            (asset.get("id"), asset.get("name")) for asset in normalized_assets if asset.get("id")
        ]
        skipped = len(normalized_assets) - len(tasks)
        if skipped:
            logger.warning(f"Skipping {skipped} asset(s) missing 'id' in category '{category}'")

        with ThreadPoolExecutor(max_workers=self.max_parallel_tickers) as executor:
            futures = {
                executor.submit(self._download_asset, ticker_id, asset_name, category): ticker_id
                for ticker_id, asset_name in tasks
            }
            for future in log_progress(
                as_completed(futures),
                label=f"Downloading {category}",
                total=len(futures),
            ):
                ticker_id = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(
                        "Ticker download failed (ticker=%s, category=%s)",
                        ticker_id,
                        category,
                    )

        logger.info(f"Finished downloads for category: {category}")

    def run(self) -> None:
        bootstrap_schema_group(self.sql_uri, self.database_schema, self.SCHEMA_GROUP)
        for category, assets in self.download_config.items():
            self.download_category(category, assets)
