"""World Bank ingestion pipeline.

Fetches the catalogue of databases and indicators, then walks every indicator
configured under ``world_bank_download_config.json`` concurrently, downloading
both metadata (units, source notes) and data (per economy/year cells). All HTTP
access goes through :mod:`src.utils.wb_client`, an async ``httpx`` wrapper around
the documented World Bank v2 REST API (this replaced the ``wbgapi`` dependency).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import polars as pl

from src.core.base_downloaders import BaseWorldBankDownloader
from src.settings import load_settings
from src.utils import wb_client
from src.utils.downloads import (
    _download_config,
    _download_source_indicators,
    _get_sql_config,
    _polars_from_world_bank_records,
    _test_sql,
    _test_world_bank_api,
    log_progress,
)
from src.utils.incremental import group_max
from src.utils.schema import (
    bootstrap_schema_group,
    get_table_definition,
    write_polars_to_table,
)

logger = logging.getLogger(__name__)


class WorldBankDownloader(BaseWorldBankDownloader):
    """Concrete World Bank downloader that writes to Postgres via Polars."""

    SCHEMA_GROUP = "world_bank"

    def __init__(
        self,
        env_path: str | Path,
        download_config_path: str | Path | None = None,
        database_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture configuration; the Postgres URI is built in ``_initialize_connections``.

        Args:
            env_path: Path to the ``.env`` with Postgres credentials.
            download_config_path: JSON file listing the indicators to fetch.
            database_schema: Parsed schema dict so this downloader can look
                up the canonical column list/dtypes for its tables.
        """
        self.env_path = Path(env_path)
        self.download_config = _download_config(download_config_path)
        self.sql_uri: Optional[str] = None

        self.database_table_name = "databases"
        self.database_indicators_table_name = "database_indicators"
        self.metadata_table_name = "metadata"
        self.indicators_table_name = "indicators"
        self.countries_table_name = "countries"

        self.database_schema = database_schema or {}

        # Retries use exponential backoff with jitter (see wb_client), so 5
        # attempts span ~5s → ~60s and reliably ride out WB rate-limit blips.
        self.download_max_retries = 5
        self.download_retry_delay_seconds = 5
        # Polite ceiling on parallel WB API calls; the API tolerates a handful
        # of concurrent requests but starts rate-limiting beyond that.
        self.max_parallel_indicators = 4

    def _table_def(self, table_name: str) -> Dict[str, Any]:
        return get_table_definition(self.database_schema, self.SCHEMA_GROUP, table_name)

    def _require_sql_uri(self) -> str:
        """Return the initialised Postgres URI or raise if not yet connected."""
        if self.sql_uri is None:
            raise RuntimeError("sql_uri not initialised; call _initialize_connections() first")
        return self.sql_uri

    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        secrets = load_settings(self.env_path)
        sql_config = _get_sql_config(
            username=secrets.postgres_user,
            password=secrets.postgres_password,
            host=host,
            port=port,
            db=db,
        )
        if _sql_test := _test_sql(sql_config):
            self.sql_uri = sql_config
        else:
            self.sql_uri = None
            logger.warning("Connection test to SQL database failed")
        _world_bank_test = _test_world_bank_api()
        return _sql_test and _world_bank_test

    async def download_basic_tables(self, client: httpx.AsyncClient) -> None:
        logger.info("Starting download of World Bank basic tables")
        source_records = await wb_client.call_with_retries(
            operation_name="source.list",
            request_coro_factory=lambda: wb_client.fetch_sources(client),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if source_records is None:
            logger.warning("Skipping basic tables download: source.list failed after all retries")
            return
        df = _polars_from_world_bank_records(source_records)
        await asyncio.to_thread(
            write_polars_to_table,
            df,
            self._require_sql_uri(),
            self.database_table_name,
            self._table_def(self.database_table_name),
        )

        logger.info("Starting download of World Bank countries table")
        country_records = await wb_client.call_with_retries(
            operation_name="economy.list",
            request_coro_factory=lambda: wb_client.fetch_countries(client, skip_aggregates=True),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if country_records is None:
            logger.warning(
                "Skipping countries table download: economy.list failed after all retries"
            )
            return
        df_countries = _polars_from_world_bank_records(country_records)
        await asyncio.to_thread(
            write_polars_to_table,
            df_countries,
            self._require_sql_uri(),
            self.countries_table_name,
            self._table_def(self.countries_table_name),
        )
        logger.info("Finished downloading World Bank countries table")

        logger.info("Starting download of World Bank source indicators")
        source_ids = df.get_column("id").to_list()
        for source_id in log_progress(
            source_ids, label="Downloading source indicators", total=len(source_ids)
        ):
            await _download_source_indicators(
                client=client,
                db_id=source_id,
                sql_uri=self._require_sql_uri(),
                table_name=self.database_indicators_table_name,
                table_def=self._table_def(self.database_indicators_table_name),
                api_max_retries=self.download_max_retries,
                api_retry_delay_seconds=self.download_retry_delay_seconds,
            )
        logger.info("Finished downloading World Bank source indicators")
        logger.info("Finished download of World Bank basic tables")

    async def download_db(self, client: httpx.AsyncClient, indicator_id: str, db: int) -> None:
        logger.info(
            f"Starting download of World Bank indicator data (indicator_id={indicator_id}, db={db})"
        )
        data_records = await wb_client.call_with_retries(
            operation_name=f"data.fetch(indicator_id={indicator_id}, db={db})",
            request_coro_factory=lambda: wb_client.fetch_indicator_data(client, indicator_id, db),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if data_records is None:
            logger.warning(
                "Skipping indicator data download after all retries failed "
                "(indicator_id=%s, db=%s)",
                indicator_id,
                db,
            )
            return

        df = _polars_from_world_bank_records(data_records)

        if df.is_empty():
            logger.warning(
                f"No data found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return

        df = df.select(
            [
                pl.col("economy").alias("economy"),
                pl.col("time").alias("year"),
                pl.col("value"),
            ]
        ).with_columns(
            [
                pl.lit(indicator_id).alias("indicator_id"),
                pl.lit(db).alias("db_id"),
            ]
        )
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
        await asyncio.to_thread(
            write_polars_to_table,
            df,
            self._require_sql_uri(),
            self.indicators_table_name,
            self._table_def(self.indicators_table_name),
        )
        logger.info(
            f"Finished download of World Bank indicator data (indicator_id={indicator_id}, db={db})"
        )

    async def download_metadata(
        self, client: httpx.AsyncClient, indicator_id: str, db: int
    ) -> None:
        logger.info(
            f"Starting download of World Bank indicator metadata (indicator_id={indicator_id}, db={db})"
        )
        metadata_row = await wb_client.call_with_retries(
            operation_name=f"series.metadata.get(indicator_id={indicator_id}, db={db})",
            request_coro_factory=lambda: wb_client.fetch_series_metadata(client, indicator_id, db),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        if metadata_row is None:
            logger.info(
                "Trying fallback metadata endpoint for indicator (indicator_id=%s, db=%s)",
                indicator_id,
                db,
            )
            metadata_row = await wb_client.call_with_retries(
                operation_name=f"indicator.metadata.get(indicator_id={indicator_id}, db={db})",
                request_coro_factory=lambda: wb_client.fetch_indicator_metadata(
                    client, indicator_id, db
                ),
                max_retries=self.download_max_retries,
                retry_delay_seconds=self.download_retry_delay_seconds,
            )

        if metadata_row is None:
            logger.warning(
                "Skipping metadata download after all retries failed (indicator_id=%s, db=%s)",
                indicator_id,
                db,
            )
            return

        dataframe_dict = {"indicator_id": indicator_id, "db_id": db, **metadata_row}
        df = pl.DataFrame([dataframe_dict])
        if df.is_empty():
            logger.warning(
                f"No metadata found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return
        await asyncio.to_thread(
            write_polars_to_table,
            df,
            self._require_sql_uri(),
            self.metadata_table_name,
            self._table_def(self.metadata_table_name),
        )
        logger.info(
            f"Finished download of World Bank indicator metadata (indicator_id={indicator_id}, db={db})"
        )

    async def _download_indicator_pair(
        self, client: httpx.AsyncClient, indicator_id: str, db_id: int
    ) -> None:
        """Run metadata + data download for a single indicator (worker unit).

        Exceptions are caught and logged here so one bad indicator never aborts
        the whole gathered batch.
        """
        try:
            await self.download_metadata(client, indicator_id, db_id)
            await self.download_db(client, indicator_id, db_id)
        except Exception:
            logger.exception(
                "Indicator download failed (indicator_id=%s, db_id=%s)",
                indicator_id,
                db_id,
            )

    async def _run_async(self) -> None:
        bootstrap_schema_group(self._require_sql_uri(), self.database_schema, self.SCHEMA_GROUP)
        download_dictionary: dict[int, list[str]] = {}
        for category in self.download_config:
            for db in self.download_config[category]:
                db_id = db["db"]
                download_dictionary.setdefault(db_id, []).append(db["id"])

        semaphore = asyncio.Semaphore(self.max_parallel_indicators)

        async with wb_client.build_async_client() as client:
            await self.download_basic_tables(client)

            async def _bounded(indicator_id: str, db_id: int) -> None:
                async with semaphore:
                    await self._download_indicator_pair(client, indicator_id, db_id)

            for db_id, indicator_ids in download_dictionary.items():
                logger.info("Starting downloads for World Bank database (db_id=%s)", db_id)
                tasks = [
                    asyncio.create_task(_bounded(indicator_id, db_id))
                    for indicator_id in indicator_ids
                ]
                for coro in log_progress(
                    asyncio.as_completed(tasks),
                    label=f"Downloading indicators for db_id={db_id}",
                    total=len(tasks),
                ):
                    await coro
                logger.info("Finished downloads for World Bank database (db_id=%s)", db_id)

    async def _update_indicator(
        self, client: httpx.AsyncClient, indicator_id: str, db: int, last_year: int
    ) -> None:
        """Append observations for years strictly after ``last_year`` for one indicator."""
        data_records = await wb_client.call_with_retries(
            operation_name=f"data.fetch(indicator_id={indicator_id}, db={db})",
            request_coro_factory=lambda: wb_client.fetch_indicator_data(client, indicator_id, db),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if data_records is None:
            return
        df = _polars_from_world_bank_records(data_records)
        if df.is_empty():
            return
        df = (
            df.select(
                pl.col("economy").alias("economy"),
                pl.col("time").alias("year"),
                pl.col("value"),
            )
            .with_columns(
                pl.lit(indicator_id).alias("indicator_id"),
                pl.lit(db).alias("db_id"),
                pl.col("time").cast(pl.Int64, strict=False).alias("year"),
            )
            .drop_nulls(subset=["economy", "year"])
            .unique(subset=["economy", "year", "indicator_id", "db_id"], keep="last")
            .filter(pl.col("year") > last_year)
        )
        if df.is_empty():
            return
        await asyncio.to_thread(
            write_polars_to_table,
            df,
            self._require_sql_uri(),
            self.indicators_table_name,
            self._table_def(self.indicators_table_name),
        )
        logger.info(
            "World Bank incremental: appended %d rows for indicator_id=%s (db=%s, year>%s)",
            df.height,
            indicator_id,
            db,
            last_year,
        )

    async def _update_async(self, maxima: dict[tuple, int]) -> None:
        semaphore = asyncio.Semaphore(self.max_parallel_indicators)
        async with wb_client.build_async_client() as client:

            async def _bounded(indicator_id: str, db: int, last_year: int) -> None:
                async with semaphore:
                    try:
                        await self._update_indicator(client, indicator_id, db, last_year)
                    except Exception:
                        logger.exception(
                            "World Bank incremental update failed (indicator_id=%s, db=%s)",
                            indicator_id,
                            db,
                        )

            tasks = [
                asyncio.create_task(_bounded(indicator_id, db, last_year))
                for (indicator_id, db), last_year in maxima.items()
            ]
            for coro in log_progress(
                asyncio.as_completed(tasks),
                label="Updating World Bank indicators",
                total=len(tasks),
            ):
                await coro

    def update(self) -> None:
        """Incrementally refresh every already-stored indicator (append new years).

        Reads each ``(indicator_id, db_id)``'s latest stored year from
        ``indicators`` and appends only later years. Falls back to a full
        :meth:`run` when the table doesn't exist yet or is empty.
        """
        if self.sql_uri is None:
            logger.warning("World Bank update skipped: SQL connection not initialised")
            return
        maxima = group_max(
            self._require_sql_uri(), self.indicators_table_name, ["indicator_id", "db_id"], "year"
        )
        if not maxima:
            logger.info("World Bank: no existing indicator data; running full ingest")
            self.run()
            return
        asyncio.run(self._update_async(maxima))

    def run(self) -> None:
        asyncio.run(self._run_async())
