import sys
import polars as pl
import yfinance as yf

from src.core.base_downloaders import BaseYahooDownloader
from src.utils.downloads import (
    _call_with_retries,
    _download_config,
    _test_sql,
    _get_sql_config,
)
from src.utils.schema import (
    bootstrap_schema_group,
    get_table_definition,
    write_polars_to_table,
)


import os
import time
import logging

from typing import Any, Dict, Iterable, Optional
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)


class YahooDownloader(BaseYahooDownloader):
    """
    Downloader for Yahoo Finance data
    """

    SCHEMA_GROUP = "yahoo_finance"

    def __init__(
        self,
        env_path: str,
        download_config_path: str,
        database_schema: Optional[Dict[str, Any]] = None,
    ):
        self.env_path = env_path
        self.download_config = _download_config(download_config_path)
        self.sql_uri = None

        self.historical_data_table_name = "yahoo_historical_prices"
        self.metadata_table_name = "yahoo_metadata"

        self.database_schema = database_schema or {}

        self.successful_connections = False

        self.download_max_retries = 3
        self.download_retry_delay_seconds = 5

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

    def _initialize_connections(self, host, port, db):
        load_dotenv(self.env_path)
        username = os.getenv("POSTGRES_USERNAME")
        password = os.getenv("POSTGRES_PASSWORD")

        self.sql_uri = _get_sql_config(
            username=username, password=password, host=host, port=port, db=db
        )

        if _sql_test := _test_sql(self.sql_uri):
            self.sql_uri = self.sql_uri
        else:
            self.sql_uri = None
            logger.warning("Connection test to SQL database failed")
        self.successful_connections = _sql_test
        return self.successful_connections

    def download_historical_data(self, ticker_id, category, period="max"):
        logger.info(f"Starting download of historical data (ticker={ticker_id})")

        ticker_obj = yf.Ticker(ticker_id)

        hist_df_pandas = _call_with_retries(
            operation_name=f"yfinance.history(ticker={ticker_id})",
            request_callable=lambda: ticker_obj.history(period=period),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if hist_df_pandas is None:
            logger.warning(
                f"Skipping historical data for {ticker_id}: all retries failed."
            )
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

    def download_metadata(self, ticker_id, asset_name, category) -> bool:
        logger.info(f"Starting download of metadata (ticker={ticker_id})")

        ticker_obj = yf.Ticker(ticker_id)

        info_dict: Optional[Dict[str, Any]] = _call_with_retries(
            operation_name=f"yfinance.info(ticker={ticker_id})",
            request_callable=lambda: ticker_obj.info,
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if info_dict is None:
            logger.warning(
                f"Skipping metadata for {ticker_id}: all retries failed."
            )
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

    def download_category(self, category, assets):
        logger.info(f"Starting downloads for category: {category}")

        normalized_assets = list(self._normalize_assets(category, assets))

        for asset in tqdm(normalized_assets, desc=f"Downloading {category}", dynamic_ncols=True, file=sys.stdout):
            ticker_id = asset.get("id")
            asset_name = asset.get("name")

            if not ticker_id:
                logger.warning(f"Skipping asset missing 'id' in category '{category}'")
                continue

            metadata_ok = self.download_metadata(ticker_id, asset_name, category)
            if not metadata_ok:
                logger.warning(
                    f"Skipping historical data for {ticker_id}: metadata write was skipped, "
                    f"FK to yahoo_metadata.ticker would fail."
                )
                time.sleep(1)
                continue
            self.download_historical_data(ticker_id, category, period="max")

            time.sleep(1)

        logger.info(f"Finished downloads for category: {category}")

    def run(self):
        bootstrap_schema_group(self.sql_uri, self.database_schema, self.SCHEMA_GROUP)
        for category, assets in self.download_config.items():
            self.download_category(category, assets)
