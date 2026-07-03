"""Shared utilities for the World Bank / Yahoo / news downloaders.

Holds retry wrappers, connectivity probes, schema-flattening helpers, the
git ``CloneProgress`` adapter that logs git-clone telemetry, and
:func:`log_progress` — a logging-based replacement for ``tqdm`` progress bars.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import stat
from time import monotonic, sleep
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

from git import RemoteProgress
import polars as pl
from sqlalchemy import create_engine, text

from src.utils import wb_client

logger = logging.getLogger(__name__)

_PROGRESS_LOG_INTERVAL_SECONDS = 5.0


def log_progress(
    iterable: Iterable[Any],
    *,
    label: str,
    total: Optional[int] = None,
    interval_seconds: float = _PROGRESS_LOG_INTERVAL_SECONDS,
) -> Iterator[Any]:
    """Yield from ``iterable`` while emitting throttled progress log records.

    A logging-based stand-in for ``tqdm``: terminal progress bars don't render
    usefully in container logs, so instead of a live bar this logs a progress
    line at most once per ``interval_seconds`` (plus a final line on the last
    item). Work is expected to happen in the loop body, so the count reflects
    items whose body has finished. ``total`` enables percentage reporting; when
    it is unknown only the running count is logged.

    Args:
        iterable: Items to iterate over.
        label: Human-readable prefix identifying the operation in the log.
        total: Total item count when known, for percentage reporting.
        interval_seconds: Minimum seconds between progress log records.

    Yields:
        Each item from ``iterable`` unchanged.
    """
    logger.info("%s: starting (%s items)", label, total if total is not None else "?")
    count = 0
    last_log = monotonic()
    for item in iterable:
        yield item
        count += 1
        is_last = total is not None and count >= total
        now = monotonic()
        if is_last or now - last_log >= interval_seconds:
            if total:
                logger.info("%s: %d/%d (%.0f%%)", label, count, total, count / total * 100)
            else:
                logger.info("%s: %d processed", label, count)
            last_log = now


def _remove_readonly(func, path, exc_info):
    """``shutil.rmtree`` error handler: clear read-only bit and retry.

    The git working tree on Windows contains files with the read-only bit
    set (under ``.git/objects/`` notably); the standard ``rmtree`` cannot
    delete them until that bit is cleared.

    Args:
        func: The failed function (typically ``os.unlink``).
        path: Filesystem path the failure was on.
        exc_info: Original exception info (unused).
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


class CloneProgress(RemoteProgress):
    """Log GitPython clone progress periodically (replaces the old tqdm bar)."""

    def __init__(self):
        """Initialise the underlying ``RemoteProgress`` and log-throttle state."""
        super().__init__()
        self._last_log = 0.0

    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str = "",
    ) -> None:
        """Log one clone-progress sample, throttled to one line per interval.

        Args:
            op_code: GitPython operation code (unused — we just report totals).
            cur_count: Current operation count.
            max_count: Total operation count, or ``None`` when unknown.
            message: Optional human-readable status (unused).
        """
        complete = max_count is not None and float(cur_count) >= float(max_count)
        now = monotonic()
        if not complete and now - self._last_log < _PROGRESS_LOG_INTERVAL_SECONDS:
            return
        self._last_log = now
        if max_count:
            pct = float(cur_count) / float(max_count) * 100
            logger.info("Cloning repository: %d%% (%s/%s)", int(pct), cur_count, max_count)
        else:
            logger.info("Cloning repository: %s objects", cur_count)


def _call_with_retries(
    operation_name: str,
    request_callable: Callable[[], Optional[object]],
    retry_delay_seconds: float,
    max_retries: int,
    max_delay: float = 60.0,
):
    """Call ``request_callable`` with bounded retry-on-exception.

    The synchronous twin of :func:`wb_client.call_with_retries`: retries use the
    same exponential backoff with jitter (:func:`wb_client.compute_backoff_delay`)
    so flaky Yahoo / git / embedding calls back off from an overloaded upstream
    instead of retrying on a fixed cadence.

    Args:
        operation_name: Label used in log messages so failures can be traced.
        request_callable: Zero-arg callable that performs the request.
        retry_delay_seconds: Base delay for the first retry; doubles each attempt.
        max_retries: Number of retries *after* the first attempt — total
            attempts will be ``max_retries + 1``.
        max_delay: Ceiling on the per-attempt backoff delay.

    Returns:
        Whatever ``request_callable`` returns on success, or ``None`` if every
        attempt raised.
    """
    attempt = 0

    while attempt <= max_retries:
        try:
            return request_callable()
        except Exception as exc:
            if attempt == max_retries:
                logger.exception(
                    "Operation '%s' failed after %d attempt(s), giving up",
                    operation_name,
                    attempt + 1,
                )
                return None
            delay = wb_client.compute_backoff_delay(
                retry_delay_seconds, attempt, max_delay=max_delay
            )
            logger.warning(
                "Retry %d/%d for operation '%s' failed: %s; retrying in %.1fs",
                attempt + 1,
                max_retries,
                operation_name,
                exc,
                delay,
                exc_info=True,
            )
            sleep(delay)
            attempt += 1


def _flatten_record(record: dict) -> dict:
    """Flatten one-level nested dicts to ``parent.child`` keys.

    Args:
        record: One record from a WB API response.

    Returns:
        Dict with nested fields renamed (e.g. ``region.value``) and leaf
        scalars preserved as-is.
    """
    flattened = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{key}.{nested_key}"] = nested_value
        else:
            flattened[key] = value
    return flattened


def _polars_from_world_bank_records(records: Optional[object]) -> pl.DataFrame:
    """Convert a World Bank record iterable (or pre-built frame) into Polars.

    Tolerates already-built DataFrames, ``None`` (empty frame), and arbitrary
    iterables of dicts or non-dict scalars; flattens one level of nested
    dictionaries via :func:`_flatten_record`.

    Args:
        records: WB response — DataFrame, iterable, or None.

    Returns:
        Polars DataFrame; empty when ``records`` is None or yields no rows.
    """
    if isinstance(records, pl.DataFrame):
        return records

    if records is None:
        return pl.DataFrame()

    iterable_records: Iterable[Any] = records  # type: ignore[assignment]
    rows = []
    for record in iterable_records:
        if isinstance(record, dict):
            rows.append(_flatten_record(record))
        else:
            rows.append(record)

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts(rows, infer_schema_length=len(rows))


async def _download_source_indicators(
    client,
    db_id: int,
    sql_uri: str,
    table_name: str,
    table_def: Dict[str, Any],
    api_max_retries: int,
    api_retry_delay_seconds: float,
) -> bool:
    """Pull the indicator catalogue for one WB database into Postgres.

    Args:
        client: Shared ``httpx.AsyncClient`` for WB API calls.
        db_id: World Bank database id.
        sql_uri: Postgres URI to write to.
        table_name: Destination table.
        table_def: Schema definition (column types + PKs) for ``table_name``.
        api_max_retries: Retry budget for the underlying WB call.
        api_retry_delay_seconds: Sleep between WB-call retries.

    Returns:
        ``True`` on success (including the "no rows" case); ``False`` when
        all retries failed.
    """
    from src.utils.schema import write_polars_to_table

    indicator_records = await wb_client.call_with_retries(
        operation_name=f"series.list(db={db_id})",
        request_coro_factory=lambda: wb_client.fetch_series(client, db_id),
        max_retries=api_max_retries,
        retry_delay_seconds=api_retry_delay_seconds,
    )

    if indicator_records is None:
        logger.warning("Skipping source indicators for db_id=%s after all retries failed", db_id)
        return False

    df_indicators = _polars_from_world_bank_records(indicator_records)

    if df_indicators.is_empty():
        logger.info("No indicators returned for db_id=%s; skipping write", db_id)
        return True

    df_indicators = df_indicators.with_columns(pl.lit(db_id).alias("database_id"))
    df_indicators = df_indicators.rename({"value": "description"})
    await asyncio.to_thread(
        write_polars_to_table,
        df_indicators,
        sql_uri,
        table_name,
        table_def,
    )
    return True


def _download_config(path: str | Path) -> dict:
    """Read and parse a JSON download-config file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a Python dict.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _get_sql_config(username: str, password: str, host: str, port: int, db: str) -> str:
    """Assemble a ``postgresql://...`` URI from individual parts.

    Args:
        username: Postgres username.
        password: Postgres password.
        host: Postgres host (container name in Compose).
        port: Postgres port.
        db: Database name; pass an empty string for the cluster-level URI.

    Returns:
        Connection URI string suitable for SQLAlchemy / psycopg.
    """
    if db:
        uri = f"postgresql://{username}:{password}@{host}:{port}/{db}"
    else:
        uri = f"postgresql://{username}:{password}@{host}:{port}"
    return uri


def _test_sql(uri: str) -> bool:
    """Probe a Postgres connection with ``SELECT 1``.

    Args:
        uri: SQLAlchemy URI.

    Returns:
        ``True`` if the probe returned ``1``; ``False`` on any error.
    """
    try:
        with create_engine(uri).connect() as connection:
            _test = connection.execute(text("SELECT 1 AS number")).scalar_one()
        logger.info("Successfully tested connection to `PostgreSQL`")
        return bool(_test)
    except Exception:
        logger.exception("An error occured while testing connection to `PostgreSQL`")
        return False


def _test_world_bank_api() -> bool:
    """Probe the World Bank API via the async client's ``/source`` healthcheck.

    Runs the async probe in a throwaway event loop so the sync
    ``_initialize_connections`` caller stays unchanged.

    Returns:
        ``True`` if the call returned any rows; ``False`` on any error.
    """

    async def _probe() -> bool:
        async with wb_client.build_async_client() as client:
            return await wb_client.healthcheck(client)

    try:
        ok = asyncio.run(_probe())
        if ok:
            logger.info("Successfully tested connection to World Bank API")
        return ok
    except Exception:
        logger.exception("An error occured while testing connection to World Bank API")
        return False
