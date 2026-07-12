"""Eurostat EU regional (NUTS-2) indicator ingestion pipeline.

Writes three tables (schema group ``eurostat``), mirroring the FRED US-state trio:

- ``eurostat_regions``           — the NUTS-2 region catalogue (code, name, country,
  parent, level), built from the bundled GISCO GeoJSON so it exactly matches the
  choropleth polygons. Analogous to World Bank ``countries`` / FRED ``states``.
- ``eurostat_indicators``        — one description row per indicator concept
  (name, dataset, filters, units, coverage…). Analogous to FRED ``state_indicators``.
- ``eurostat_indicator_values``  — the long annual panel ``(region, year, value,
  indicator_id)``. Analogous to FRED ``state_indicator_values``.

Each configured indicator names a Eurostat ``dataset`` plus the ``filters`` that
pin every non-geo/non-time dimension to a single category;
:func:`eurostat_client.fetch_dataset` returns the whole region×year panel in one
keyless call and :func:`eurostat_client.parse_jsonstat` flattens it to long rows.
All HTTP access goes through the async ``httpx`` client in
:mod:`src.utils.eurostat_client`.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import polars as pl

from src.core.base_downloaders import BaseEurostatDownloader
from src.settings import load_settings
from src.utils import eurostat_client
from src.utils.downloads import _download_config, _get_sql_config, _test_sql, log_progress
from src.utils.incremental import group_max, read_rows
from src.utils.schema import (
    bootstrap_schema_group,
    get_table_definition,
    write_polars_to_table,
)

logger = logging.getLogger(__name__)


class EurostatDownloader(BaseEurostatDownloader):
    """Concrete Eurostat downloader that writes NUTS-2 indicators to Postgres via Polars."""

    SCHEMA_GROUP = "eurostat"

    def __init__(
        self,
        env_path: str | Path,
        geojson_path: str | Path,
        download_config_path: str | Path | None = None,
        nuts_level: int = 2,
        database_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture configuration; the Postgres URI is set in ``_initialize_connections``.

        Args:
            env_path: Path to the ``.env`` with Postgres credentials.
            geojson_path: Path to the bundled GISCO NUTS-2 GeoJSON used to build
                the regions catalogue.
            download_config_path: JSON file listing the indicators to fetch.
            nuts_level: NUTS level to ingest (2).
            database_schema: Parsed schema dict so this downloader can look up the
                canonical column list/dtypes for its tables.
        """
        self.env_path = Path(env_path)
        self.geojson_path = Path(geojson_path)
        self.download_config = _download_config(download_config_path)
        self.nuts_level = nuts_level
        self.sql_uri: Optional[str] = None

        self.regions_table_name = "eurostat_regions"
        self.indicators_table_name = "eurostat_indicators"
        self.values_table_name = "eurostat_indicator_values"

        self.database_schema = database_schema or {}
        self._region_ids: set[str] = set()

        self.download_max_retries = 5
        self.download_retry_delay_seconds = 5
        # Eurostat tolerates a handful of parallel requests fine.
        self.max_parallel_indicators = 4

    def _table_def(self, table_name: str) -> Dict[str, Any]:
        return get_table_definition(self.database_schema, self.SCHEMA_GROUP, table_name)

    def _require_sql_uri(self) -> str:
        """Return the initialised Postgres URI or raise if not yet connected."""
        if self.sql_uri is None:
            raise RuntimeError("sql_uri not initialised; call _initialize_connections() first")
        return self.sql_uri

    def _test_eurostat_api(self) -> bool:
        """Probe the Eurostat API via the async client's healthcheck in a throwaway loop."""

        async def _probe() -> bool:
            async with eurostat_client.build_async_client() as client:
                return await eurostat_client.healthcheck(client)

        try:
            ok = asyncio.run(_probe())
            if ok:
                logger.info("Successfully tested connection to Eurostat API")
            return ok
        except Exception:
            logger.exception("An error occured while testing connection to Eurostat API")
            return False

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
        _api_test = self._test_eurostat_api()
        return _sql_test and _api_test

    def download_regions(self) -> None:
        """Populate the ``eurostat_regions`` catalogue from the bundled GeoJSON."""
        logger.info("Starting download of Eurostat NUTS-%d regions catalogue", self.nuts_level)
        records = eurostat_client.regions_from_geojson(self.geojson_path, level=self.nuts_level)
        self._region_ids = {rec["id"] for rec in records}
        df_regions = pl.DataFrame(records)
        write_polars_to_table(
            df_regions,
            self._require_sql_uri(),
            self.regions_table_name,
            self._table_def(self.regions_table_name),
        )
        logger.info("Finished Eurostat regions catalogue (%d regions)", df_regions.height)

    async def download_indicator(
        self,
        client: httpx.AsyncClient,
        slug: str,
        dataset: str,
        filters: Dict[str, str],
        name: str,
        category: str,
    ) -> None:
        """Download one indicator's description row and its annual region panel."""
        logger.info("Starting download of Eurostat indicator (id=%s, dataset=%s)", slug, dataset)
        payload = await eurostat_client.call_with_retries(
            operation_name=f"eurostat.data({dataset})",
            request_coro_factory=lambda: eurostat_client.fetch_dataset(
                client, dataset, geo_level=f"nuts{self.nuts_level}", filters=filters
            ),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if not payload:
            logger.warning("Skipping Eurostat indicator with no data (id=%s)", slug)
            return

        rows, meta = eurostat_client.parse_jsonstat(payload, level=self.nuts_level)
        # Keep only regions present in the NUTS-2021 catalogue (drops codes from
        # older NUTS vintages that some datasets still return).
        rows = [r for r in rows if r["region"] in self._region_ids]
        if not rows:
            logger.warning("No panel rows for Eurostat indicator (id=%s) after region filter", slug)
            return

        df_values = (
            pl.DataFrame(rows)
            .with_columns(pl.lit(slug).alias("indicator_id"))
            .drop_nulls(subset=["region", "year"])
            .unique(subset=["region", "year", "indicator_id"], keep="last", maintain_order=True)
        )
        # rows carry plain int years (from parse_jsonstat), so min/max avoid the
        # polars column's wider (date-inclusive) return type.
        years = [int(r["year"]) for r in rows]
        min_year, max_year = min(years), max(years)

        description_row = {
            "indicator_id": slug,
            "name": name,
            "category": category,
            "dataset": dataset,
            "filters": json.dumps(filters, sort_keys=True),
            "units": self._config_units(slug, category) or meta.get("units"),
            "frequency": meta.get("frequency"),
            "nuts_level": self.nuts_level,
            "min_year": min_year,
            "max_year": max_year,
            "source_label": meta.get("source_label"),
            "notes": eurostat_client.synthesize_notes(meta, dataset, filters, min_year, max_year),
        }
        write_polars_to_table(
            pl.DataFrame([description_row]),
            self._require_sql_uri(),
            self.indicators_table_name,
            self._table_def(self.indicators_table_name),
        )
        write_polars_to_table(
            df_values,
            self._require_sql_uri(),
            self.values_table_name,
            self._table_def(self.values_table_name),
        )
        logger.info(
            "Finished Eurostat indicator (id=%s): %d region-year rows across %d regions",
            slug,
            df_values.height,
            df_values.get_column("region").n_unique(),
        )

    def _config_units(self, slug: str, category: str) -> Optional[str]:
        """Return the display ``units`` string configured for this indicator, if any."""
        for item in self.download_config.get(category, []):
            if item.get("id") == slug:
                return item.get("units")
        return None

    async def _download_indicator_safe(
        self,
        client: httpx.AsyncClient,
        slug: str,
        dataset: str,
        filters: Dict[str, str],
        name: str,
        category: str,
    ) -> None:
        """Run one indicator download, catching/logging so one failure never aborts the batch."""
        try:
            await self.download_indicator(client, slug, dataset, filters, name, category)
        except Exception:
            logger.exception(
                "Eurostat indicator download failed (id=%s, dataset=%s)", slug, dataset
            )

    async def _run_async(self) -> None:
        bootstrap_schema_group(self._require_sql_uri(), self.database_schema, self.SCHEMA_GROUP)

        # Regions must exist before the values table (FK), so write them first.
        await asyncio.to_thread(self.download_regions)

        # Flatten the config into (slug, dataset, filters, name, category) work units.
        work_units: list[tuple[str, str, Dict[str, str], str, str]] = []
        for category, items in self.download_config.items():
            for item in items:
                if "id" in item and "dataset" in item:
                    work_units.append(
                        (
                            item["id"],
                            item["dataset"],
                            item.get("filters", {}),
                            item.get("name", item["id"]),
                            category,
                        )
                    )

        semaphore = asyncio.Semaphore(self.max_parallel_indicators)

        async with eurostat_client.build_async_client() as client:

            async def _bounded(
                slug: str, dataset: str, filters: Dict[str, str], name: str, category: str
            ) -> None:
                async with semaphore:
                    await self._download_indicator_safe(
                        client, slug, dataset, filters, name, category
                    )

            tasks = [
                asyncio.create_task(_bounded(slug, dataset, filters, name, category))
                for slug, dataset, filters, name, category in work_units
            ]
            for coro in log_progress(
                asyncio.as_completed(tasks),
                label="Downloading Eurostat region indicators",
                total=len(tasks),
            ):
                await coro
        logger.info("Finished download of all Eurostat region indicators")

    async def _update_indicator(
        self,
        client: httpx.AsyncClient,
        slug: str,
        dataset: str,
        filters: Dict[str, str],
        last_year: int,
    ) -> None:
        """Append region-year rows for years strictly after ``last_year`` (values only).

        The description row already exists (PK ``indicator_id``), so an incremental
        refresh only writes to ``eurostat_indicator_values``.
        """
        payload = await eurostat_client.call_with_retries(
            operation_name=f"eurostat.data({dataset})",
            request_coro_factory=lambda: eurostat_client.fetch_dataset(
                client, dataset, geo_level=f"nuts{self.nuts_level}", filters=filters
            ),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if not payload:
            return
        rows, _ = eurostat_client.parse_jsonstat(payload, level=self.nuts_level)
        rows = [r for r in rows if r["region"] in self._region_ids]
        if not rows:
            return

        df_values = (
            pl.DataFrame(rows)
            .with_columns(
                pl.lit(slug).alias("indicator_id"),
                pl.col("year").cast(pl.Int64, strict=False).alias("year"),
            )
            .drop_nulls(subset=["region", "year"])
            .unique(subset=["region", "year", "indicator_id"], keep="last", maintain_order=True)
            .filter(pl.col("year") > last_year)
        )
        if df_values.is_empty():
            return
        await asyncio.to_thread(
            write_polars_to_table,
            df_values,
            self._require_sql_uri(),
            self.values_table_name,
            self._table_def(self.values_table_name),
        )
        logger.info(
            "Eurostat incremental: appended %d region-year rows for id=%s (year>%s)",
            df_values.height,
            slug,
            last_year,
        )

    async def _update_async(
        self, maxima: dict[tuple, int], meta_by_id: dict[str, tuple[str, Dict[str, str]]]
    ) -> None:
        semaphore = asyncio.Semaphore(self.max_parallel_indicators)
        async with eurostat_client.build_async_client() as client:

            async def _bounded(
                slug: str, dataset: str, filters: Dict[str, str], last_year: int
            ) -> None:
                async with semaphore:
                    try:
                        await self._update_indicator(client, slug, dataset, filters, last_year)
                    except Exception:
                        logger.exception("Eurostat incremental update failed (id=%s)", slug)

            tasks = []
            for (slug,), last_year in maxima.items():
                dataset, filters = meta_by_id.get(slug, ("", {}))
                if not dataset:
                    logger.warning("Skipping Eurostat update for %s: no dataset in catalogue", slug)
                    continue
                tasks.append(asyncio.create_task(_bounded(slug, dataset, filters, last_year)))
            for coro in log_progress(
                asyncio.as_completed(tasks),
                label="Updating Eurostat region indicators",
                total=len(tasks),
            ):
                await coro

    def update(self) -> None:
        """Incrementally refresh every already-stored indicator (append new years).

        Reads each indicator's latest year from ``eurostat_indicator_values`` and
        its ``dataset`` + ``filters`` from ``eurostat_indicators``, then appends only
        later years. Falls back to a full :meth:`run` when the value table doesn't
        exist yet or is empty.
        """
        if self.sql_uri is None:
            logger.warning("Eurostat update skipped: SQL connection not initialised")
            return
        maxima = group_max(
            self._require_sql_uri(), self.values_table_name, ["indicator_id"], "year"
        )
        if not maxima:
            logger.info("Eurostat: no existing indicator values; running full ingest")
            self.run()
            return

        catalog = read_rows(
            self._require_sql_uri(),
            self.indicators_table_name,
            ["indicator_id", "dataset", "filters"],
        )
        meta_by_id: dict[str, tuple[str, Dict[str, str]]] = {}
        for row in catalog or []:
            try:
                filters = json.loads(row.get("filters") or "{}")
            except (json.JSONDecodeError, TypeError):
                filters = {}
            meta_by_id[row["indicator_id"]] = (row.get("dataset") or "", filters)

        # The region catalogue already exists in Postgres; reload the id set from
        # the bundled GeoJSON (cheap) so the value filter matches the map polygons.
        records = eurostat_client.regions_from_geojson(self.geojson_path, level=self.nuts_level)
        self._region_ids = {rec["id"] for rec in records}

        asyncio.run(self._update_async(maxima, meta_by_id))

    def run(self) -> None:
        asyncio.run(self._run_async())
