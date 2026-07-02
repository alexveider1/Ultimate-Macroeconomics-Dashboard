"""FRED state-indicator ingestion pipeline.

Writes three tables (schema group ``fred``):

- ``states``                 — the 50 states + DC catalogue (abbrev, name, FIPS,
  Census region/division), analogous to World Bank ``countries``.
- ``state_indicators``       — one description row per indicator concept
  (name, units, frequency, series group…), analogous to WB ``metadata``.
- ``state_indicator_values`` — the long annual panel ``(state, year, value,
  indicator_id)``, analogous to WB ``indicators``.

Each configured indicator names a representative single-state series (e.g.
``CAUR``); :func:`fred_client.fetch_series_group` resolves it to a GeoFRED series
group, and a single :func:`fred_client.fetch_regional_panel` call returns the
whole annual cross-state panel. Observations are mapped to states by FIPS code
(the reliable key — see :mod:`src.utils.fred_client`). All HTTP access goes
through the async ``httpx`` client in :mod:`src.utils.fred_client`.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import polars as pl

from src.core.base_downloaders import BaseFredDownloader
from src.settings import load_settings
from src.utils import fred_client
from src.utils.downloads import _download_config, _get_sql_config, _test_sql, log_progress
from src.utils.schema import (
    bootstrap_schema_group,
    get_table_definition,
    write_polars_to_table,
)

logger = logging.getLogger(__name__)

# CAPOP (Resident Population) covers all 50 states + DC across the longest span,
# so it is the reference series used to populate the states catalogue.
_STATES_REFERENCE_SERIES = "CAPOP"


class FredDownloader(BaseFredDownloader):
    """Concrete FRED downloader that writes state indicators to Postgres via Polars."""

    SCHEMA_GROUP = "fred"

    def __init__(
        self,
        env_path: str | Path,
        download_config_path: str | Path | None = None,
        database_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture configuration; the Postgres URI + API key are set in ``_initialize_connections``.

        Args:
            env_path: Path to the ``.env`` with Postgres credentials + FRED key.
            download_config_path: JSON file listing the indicators to fetch.
            database_schema: Parsed schema dict so this downloader can look up the
                canonical column list/dtypes for its tables.
        """
        self.env_path = Path(env_path)
        self.download_config = _download_config(download_config_path)
        self.sql_uri: Optional[str] = None
        self.fred_api_key: str = ""

        self.states_table_name = "states"
        self.indicators_table_name = "state_indicators"
        self.values_table_name = "state_indicator_values"

        self.database_schema = database_schema or {}

        self.download_max_retries = 3
        self.download_retry_delay_seconds = 5
        # Polite ceiling on parallel FRED API calls; a handful is fine, more starts
        # tripping rate limits.
        self.max_parallel_indicators = 4

    def _table_def(self, table_name: str) -> Dict[str, Any]:
        return get_table_definition(self.database_schema, self.SCHEMA_GROUP, table_name)

    def _require_sql_uri(self) -> str:
        """Return the initialised Postgres URI or raise if not yet connected."""
        if self.sql_uri is None:
            raise RuntimeError("sql_uri not initialised; call _initialize_connections() first")
        return self.sql_uri

    def _test_fred_api(self) -> bool:
        """Probe the FRED API via the async client's healthcheck in a throwaway loop."""
        if not self.fred_api_key:
            logger.warning("No FRED_API_KEY configured; skipping FRED download")
            return False

        async def _probe() -> bool:
            async with fred_client.build_async_client(self.fred_api_key) as client:
                return await fred_client.healthcheck(client)

        try:
            ok = asyncio.run(_probe())
            if ok:
                logger.info("Successfully tested connection to FRED API")
            return ok
        except Exception:
            logger.exception("An error occured while testing connection to FRED API")
            return False

    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        secrets = load_settings(self.env_path)
        self.fred_api_key = secrets.fred_api_key
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
        _fred_test = self._test_fred_api()
        return _sql_test and _fred_test

    async def download_states(self, client: httpx.AsyncClient) -> Dict[str, str]:
        """Populate the ``states`` table and return the FRED ``{fips: name}`` mapping.

        The state names come from a live regional panel (the population series
        group), enriched with the static Census region/division reference so the
        table always holds all 50 states + DC even if a panel omits one.
        """
        logger.info("Starting download of FRED states catalogue")
        fred_names: Dict[str, str] = {}
        group = await fred_client.call_with_retries(
            operation_name=f"series.group({_STATES_REFERENCE_SERIES})",
            request_coro_factory=lambda: fred_client.fetch_series_group(
                client, _STATES_REFERENCE_SERIES
            ),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if group:
            panel = await fred_client.call_with_retries(
                operation_name="regional.data(states)",
                request_coro_factory=lambda: fred_client.fetch_regional_panel(
                    client,
                    series_group=group["series_group"],
                    region_type=group["region_type"],
                    start_date=group["min_date"],
                    end_date=group["max_date"],
                    units=group["units"],
                    season=group.get("season", "NSA"),
                ),
                max_retries=self.download_max_retries,
                retry_delay_seconds=self.download_retry_delay_seconds,
            )
            if panel:
                _, fred_names = fred_client.parse_regional_panel(panel)
        else:
            logger.warning("Could not resolve states reference group; using static state names")

        records = fred_client.state_records_from_names(fred_names)
        df_states = pl.DataFrame(records)
        await asyncio.to_thread(
            write_polars_to_table,
            df_states,
            self._require_sql_uri(),
            self.states_table_name,
            self._table_def(self.states_table_name),
        )
        logger.info("Finished FRED states catalogue (%d states)", df_states.height)
        return fred_names

    async def download_indicator(
        self, client: httpx.AsyncClient, slug: str, series_id: str, name: str, category: str
    ) -> None:
        """Download one indicator's description row and its annual state panel."""
        logger.info("Starting download of FRED indicator (id=%s, series=%s)", slug, series_id)
        group = await fred_client.call_with_retries(
            operation_name=f"series.group({series_id})",
            request_coro_factory=lambda: fred_client.fetch_series_group(client, series_id),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if group is None:
            logger.warning(
                "Skipping FRED indicator with no resolvable series group (id=%s, series=%s)",
                slug,
                series_id,
            )
            return

        panel = await fred_client.call_with_retries(
            operation_name=f"regional.data({slug})",
            request_coro_factory=lambda: fred_client.fetch_regional_panel(
                client,
                series_group=group["series_group"],
                region_type=group["region_type"],
                start_date=group["min_date"],
                end_date=group["max_date"],
                units=group["units"],
                season=group.get("season", "NSA"),
            ),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        if panel is None:
            logger.warning("Skipping FRED indicator data after all retries (id=%s)", slug)
            return

        rows, _ = fred_client.parse_regional_panel(panel)

        description_row = {
            "indicator_id": slug,
            "name": name,
            "category": category,
            "series_group": str(group.get("series_group") or ""),
            "example_series_id": series_id,
            "units": group.get("units"),
            "frequency": group.get("frequency"),
            "seasonal_adjustment": group.get("season"),
            "region_type": group.get("region_type"),
            "min_date": group.get("min_date"),
            "max_date": group.get("max_date"),
            "notes": fred_client.synthesize_notes(group),
        }
        await asyncio.to_thread(
            write_polars_to_table,
            pl.DataFrame([description_row]),
            self._require_sql_uri(),
            self.indicators_table_name,
            self._table_def(self.indicators_table_name),
        )

        if not rows:
            logger.warning("No panel rows for FRED indicator (id=%s)", slug)
            return

        df_values = pl.DataFrame(rows).with_columns(pl.lit(slug).alias("indicator_id"))
        df_values = df_values.drop_nulls(subset=["state", "year"]).unique(
            subset=["state", "year", "indicator_id"], keep="last", maintain_order=True
        )
        await asyncio.to_thread(
            write_polars_to_table,
            df_values,
            self._require_sql_uri(),
            self.values_table_name,
            self._table_def(self.values_table_name),
        )
        logger.info(
            "Finished FRED indicator (id=%s): %d state-year rows across %d states",
            slug,
            df_values.height,
            df_values.get_column("state").n_unique(),
        )

    async def _download_indicator_safe(
        self, client: httpx.AsyncClient, slug: str, series_id: str, name: str, category: str
    ) -> None:
        """Run one indicator download, catching/logging so one failure never aborts the batch."""
        try:
            await self.download_indicator(client, slug, series_id, name, category)
        except Exception:
            logger.exception("FRED indicator download failed (id=%s, series=%s)", slug, series_id)

    async def _run_async(self) -> None:
        bootstrap_schema_group(self._require_sql_uri(), self.database_schema, self.SCHEMA_GROUP)

        # Flatten the config into (slug, series_id, name, category) work units.
        work_units: list[tuple[str, str, str, str]] = []
        for category, items in self.download_config.items():
            for item in items:
                if "id" in item and "series_id" in item:
                    work_units.append(
                        (item["id"], item["series_id"], item.get("name", item["id"]), category)
                    )

        semaphore = asyncio.Semaphore(self.max_parallel_indicators)

        async with fred_client.build_async_client(self.fred_api_key) as client:
            # States must exist before the values table (FK), so download it first.
            await self.download_states(client)

            async def _bounded(slug: str, series_id: str, name: str, category: str) -> None:
                async with semaphore:
                    await self._download_indicator_safe(client, slug, series_id, name, category)

            tasks = [
                asyncio.create_task(_bounded(slug, series_id, name, category))
                for slug, series_id, name, category in work_units
            ]
            for coro in log_progress(
                asyncio.as_completed(tasks),
                label="Downloading FRED state indicators",
                total=len(tasks),
            ):
                await coro
        logger.info("Finished download of all FRED state indicators")

    def run(self) -> None:
        asyncio.run(self._run_async())
