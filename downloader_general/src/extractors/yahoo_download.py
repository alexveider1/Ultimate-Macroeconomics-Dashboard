import polars as pl
import yfinance as yf

from src.core.base_downloaders import BaseYahooDownloader
from src.utils.downloads import (
    _call_with_retries,
    _download_config,
    _test_sql,
    _get_sql_config,
)


import os
import time
import logging

from typing import Any, Dict, Iterable
from dotenv import load_dotenv
from tqdm import tqdm

logger = logging.getLogger(__name__)


class YahooDownloader(BaseYahooDownloader):
    """
    Downloader for Yahoo Finance data
    """

    def __init__(self, env_path: str, download_config_path: str):
        self.env_path = env_path
        self.download_config = _download_config(download_config_path)
        self.sql_uri = None

        self.historical_data_table_name = "yahoo_historical_prices"
        self.metadata_table_name = "yahoo_metadata"

        self._metadata_table_initialized = False
        self._historical_table_initialized = False
        self.successful_connections = False

        self.download_max_retries = 3
        self.download_retry_delay_seconds = 5

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

        table_exists_mode = (
            "replace" if not self._historical_table_initialized else "append"
        )

        df.write_database(
            self.historical_data_table_name,
            connection=self.sql_uri,
            if_table_exists=table_exists_mode,
            engine="sqlalchemy",
        )

        logger.info(f"Finished download of historical data (ticker={ticker_id})")
        self._historical_table_initialized = True

    def download_metadata(self, ticker_id, asset_name, category):
        logger.info(f"Starting download of metadata (ticker={ticker_id})")

        ticker_obj = yf.Ticker(ticker_id)

        info_dict: Dict[str, Any] = _call_with_retries(
            operation_name=f"yfinance.info(ticker={ticker_id})",
            request_callable=lambda: ticker_obj.info,
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

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

        table_exists_mode = (
            "replace" if not self._metadata_table_initialized else "append"
        )

        df.write_database(
            self.metadata_table_name,
            connection=self.sql_uri,
            if_table_exists=table_exists_mode,
            engine="sqlalchemy",
        )

        logger.info(f"Finished download of metadata (ticker={ticker_id})")
        self._metadata_table_initialized = True

    def download_category(self, category, assets):
        logger.info(f"Starting downloads for category: {category}")

        normalized_assets = list(self._normalize_assets(category, assets))

        for asset in tqdm(normalized_assets, desc=f"Downloading {category}"):
            ticker_id = asset.get("id")
            asset_name = asset.get("name")

            if not ticker_id:
                logger.warning(f"Skipping asset missing 'id' in category '{category}'")
                continue

            self.download_metadata(ticker_id, asset_name, category)
            self.download_historical_data(ticker_id, category, period="max")

            time.sleep(1)

        logger.info(f"Finished downloads for category: {category}")

    def run(self):
        for category, assets in self.download_config.items():
            self.download_category(category, assets)
