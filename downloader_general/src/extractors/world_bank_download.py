import os
import sys
import logging
import requests
import polars as pl
import wbgapi as wb

from time import sleep
from typing import Any, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

from src.utils.downloads import (
    _get_sql_config,
    _test_sql,
    _test_world_bank_api,
    _download_config,
    _call_with_retries,
    _polars_from_world_bank_records,
    _download_source_indicators,
)
from src.utils.schema import (
    bootstrap_schema_group,
    get_table_definition,
    write_polars_to_table,
)

from src.core.base_downloaders import BaseWorldBankDownloader

logger = logging.getLogger(__name__)


class WorldBankDownloader(BaseWorldBankDownloader):
    """Downloader for World Bank data"""

    SCHEMA_GROUP = "world_bank"

    def __init__(
        self,
        env_path: str,
        download_config_path: str = None,
        database_schema: Optional[Dict[str, Any]] = None,
    ):
        self.env_path = env_path
        self.download_config = _download_config(download_config_path)
        self.sql_uri = None

        self.database_table_name = "databases"
        self.database_indicators_table_name = "database_indicators"
        self.metadata_table_name = "metadata"
        self.indicators_table_name = "indicators"
        self.countries_table_name = "countries"

        self.database_schema = database_schema or {}

        self.successfull_connections = False
        self.download_max_retries = 3
        self.download_retry_delay_seconds = 5

    def _table_def(self, table_name: str) -> Dict[str, Any]:
        return get_table_definition(self.database_schema, self.SCHEMA_GROUP, table_name)

    def _initialize_connections(self, host, port, db):
        load_dotenv(self.env_path)
        username, password = (
            os.getenv("POSTGRES_USERNAME"),
            os.getenv("POSTGRES_PASSWORD"),
        )
        sql_config = _get_sql_config(
            username=username, password=password, host=host, port=port, db=db
        )
        if _sql_test := _test_sql(sql_config):
            self.sql_uri = sql_config
        else:
            self.sql_uri = None
            logger.warning("Connection test to SQL database failed")
        _world_bank_test = _test_world_bank_api()
        result = _sql_test and _world_bank_test
        self.successfull_connections = result
        return result

    def _fetch_indicator_data_via_api(self, indicator_id: str, db: int) -> Optional[list]:
        """Fallback for sources where wbgapi's /sources/{db}/series/... URL returns empty JSON.
        Uses the standard WB v2 /country/all/indicator/{id}?source={db} endpoint instead."""
        rows = []
        page = 1
        while True:
            resp = requests.get(
                f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}",
                params={"source": db, "format": "json", "per_page": 1000, "page": page},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                break
            meta, records = payload[0], payload[1]
            for r in records:
                iso3 = r.get("countryiso3code", "")
                if len(iso3) != 3:
                    continue
                if r.get("value") is None:
                    continue
                try:
                    year = int(r["date"])
                except (ValueError, TypeError):
                    continue
                rows.append({"economy": iso3, "time": year, "value": r["value"]})
            if page >= int(meta.get("pages", 1)):
                break
            page += 1
        return rows or None

    def _fetch_indicator_metadata_via_api(self, indicator_id: str, db: int) -> Optional[dict]:
        """Fallback metadata for sources where wbgapi's /sources/{db}/series/... URL fails.
        Uses the standard WB v2 /indicator/{id} endpoint instead."""
        resp = requests.get(
            f"https://api.worldbank.org/v2/indicator/{indicator_id}",
            params={"format": "json", "source": db},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            return None
        info = payload[1][0]
        return {
            "indicator_name": info.get("name"),
            "units": info.get("unit") or "",
            "source": info.get("source", {}).get("value"),
            "development_relevance": info.get("sourceNote"),
            "limitations_and_exceptions": None,
            "statistical_concept_and_methodology": None,
        }

    def download_basic_tables(self):
        logger.info("Starting download of World Bank basic tables")
        source_records = _call_with_retries(
            operation_name="source.list",
            request_callable=lambda: list(wb.source.list()),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if source_records is None:
            logger.warning("Skipping basic tables download: source.list failed after all retries")
            return
        df = _polars_from_world_bank_records(source_records)
        write_polars_to_table(
            df,
            sql_uri=self.sql_uri,
            table_name=self.database_table_name,
            table_def=self._table_def(self.database_table_name),
        )
        logger.info("Starting download of World Bank countries table")
        country_records = _call_with_retries(
            operation_name="economy.list",
            request_callable=lambda: list(wb.economy.list(skipAggs=True, db=2, labels=True)),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if country_records is None:
            logger.warning("Skipping countries table download: economy.list failed after all retries")
            return
        df_countries = _polars_from_world_bank_records(country_records)
        write_polars_to_table(
            df_countries,
            sql_uri=self.sql_uri,
            table_name=self.countries_table_name,
            table_def=self._table_def(self.countries_table_name),
        )
        logger.info("Finished downloading World Bank countries table")

        logger.info("Starting download of World Bank source indicators")
        source_ids = df.get_column("id").to_list()
        for source_id in tqdm(source_ids, desc="Downloading source indicators", dynamic_ncols=True, file=sys.stdout):
            _download_source_indicators(
                db_id=source_id,
                sql_uri=self.sql_uri,
                table_name=self.database_indicators_table_name,
                table_def=self._table_def(self.database_indicators_table_name),
                api_max_retries=self.download_max_retries,
                api_retry_delay_seconds=self.download_retry_delay_seconds,
            )
        logger.info("Finished downloading World Bank source indicators")
        logger.info("Finished download of World Bank basic tables")

    def download_db(self, indicator_id, db):
        logger.info(
            f"Starting download of World Bank indicator data (indicator_id={indicator_id}, db={db})"
        )
        data_records = _call_with_retries(
            operation_name=f"data.fetch(indicator_id={indicator_id}, db={db})",
            request_callable=lambda: list(wb.data.fetch(
                indicator_id,
                db=db,
                skipAggs=True,
                economy="all",
                time="all",
                skipBlanks=False,
                numericTimeKeys=True,
            )),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if data_records is None:
            logger.info(
                "Trying fallback API endpoint for indicator data (indicator_id=%s, db=%s)",
                indicator_id, db,
            )
            data_records = _call_with_retries(
                operation_name=f"data.fetch.api(indicator_id={indicator_id}, db={db})",
                request_callable=lambda: self._fetch_indicator_data_via_api(indicator_id, db),
                max_retries=self.download_max_retries,
                retry_delay_seconds=self.download_retry_delay_seconds,
            )

        if data_records is None:
            logger.warning(
                "Skipping indicator data download after all retries failed "
                "(indicator_id=%s, db=%s)", indicator_id, db
            )
            return

        df = _polars_from_world_bank_records(data_records)

        if df.is_empty():
            logger.warning(
                f"No data found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return

        economy_column = "economy"
        year_column = "time"

        df = df.select([
            pl.col(economy_column).alias("economy"),
            pl.col(year_column).alias("year"),
            pl.col("value"),
        ]).with_columns([
            pl.lit(indicator_id).alias("indicator_id"),
            pl.lit(db).alias("db_id"),
        ])
        df = df.drop_nulls(subset=["economy", "year"]).unique(
            subset=["economy", "year", "indicator_id", "db_id"],
            keep="last",
            maintain_order=True,
        )
        if df.is_empty():
            logger.warning(
                f"No PK-valid rows for World Bank indicator after dedup "
                f"(indicator_id={indicator_id}, db={db})"
            )
            return
        write_polars_to_table(
            df,
            sql_uri=self.sql_uri,
            table_name=self.indicators_table_name,
            table_def=self._table_def(self.indicators_table_name),
        )
        logger.info(
            f"Finished download of World Bank indicator data (indicator_id={indicator_id}, db={db})"
        )
        sleep(10)

    def download_metadata(self, indicator_id, db):
        logger.info(
            f"Starting download of World Bank indicator metadata (indicator_id={indicator_id}, db={db})"
        )
        metadata_response = _call_with_retries(
            operation_name=f"series.metadata.get(indicator_id={indicator_id}, db={db})",
            request_callable=lambda: wb.series.metadata.get(indicator_id, db=db),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if metadata_response is not None:
            metadata = metadata_response.metadata
            dataframe_dict = {
                "indicator_id": indicator_id,
                "db_id": db,
                "indicator_name": metadata.get("IndicatorName"),
                "units": metadata.get("Unitofmeasure"),
                "source": metadata.get("Source"),
                "development_relevance": metadata.get("Developmentrelevance"),
                "limitations_and_exceptions": metadata.get("Limitationsandexceptions"),
                "statistical_concept_and_methodology": metadata.get("Statisticalconceptandmethodology"),
            }
        else:
            logger.info(
                "Trying fallback metadata endpoint for indicator (indicator_id=%s, db=%s)",
                indicator_id, db,
            )
            fallback = _call_with_retries(
                operation_name=f"indicator.metadata.get(indicator_id={indicator_id}, db={db})",
                request_callable=lambda: self._fetch_indicator_metadata_via_api(indicator_id, db),
                max_retries=self.download_max_retries,
                retry_delay_seconds=self.download_retry_delay_seconds,
            )
            if fallback is None:
                logger.warning(
                    "Skipping metadata download after all retries failed "
                    "(indicator_id=%s, db=%s)", indicator_id, db,
                )
                return
            dataframe_dict = {"indicator_id": indicator_id, "db_id": db, **fallback}
        df = pl.DataFrame([dataframe_dict])
        if df.is_empty():
            logger.warning(
                f"No metadata found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return
        write_polars_to_table(
            df,
            sql_uri=self.sql_uri,
            table_name=self.metadata_table_name,
            table_def=self._table_def(self.metadata_table_name),
        )
        logger.info(
            f"Finished download of World Bank indicator metadata (indicator_id={indicator_id}, db={db})"
        )
        sleep(10)

    def run(self):
        bootstrap_schema_group(self.sql_uri, self.database_schema, self.SCHEMA_GROUP)
        self.download_basic_tables()
        download_dictionary = {}
        for category in self.download_config:
            for db in self.download_config[category]:
                db_id = db["db"]
                download_dictionary.setdefault(db_id, []).append(db["id"])

        for db_id in download_dictionary:
            logging.info(f"Starting downloads for World Bank database (db_id={db_id})")
            for indicator_id in tqdm(
                download_dictionary[db_id],
                desc=f"Downloading indicators for db_id={db_id}",
                dynamic_ncols=True,
                file=sys.stdout,
            ):
                self.download_metadata(indicator_id, db_id)
                self.download_db(indicator_id, db_id)
            logger.info(f"Finished downloads for World Bank database (db_id={db_id})")
