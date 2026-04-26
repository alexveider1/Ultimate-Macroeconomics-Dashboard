import os
import stat

from tqdm import tqdm
from git import RemoteProgress
from sqlalchemy import create_engine, text

import json
import logging
import wbgapi as wb
import polars as pl

from typing import Callable, Optional

from time import sleep

logger = logging.getLogger(__name__)


def _remove_readonly(func, path, exc_info):
    """Clear the read-only bit and retry the removal (for git files on Windows)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


class CloneProgress(RemoteProgress):
    """
    Subclasses GitPython's RemoteProgress to route git cloning
    telemetry directly into a visual tqdm progress bar.
    """

    def __init__(self):
        super().__init__()
        self.pbar = tqdm(desc="Cloning Repository", unit="operations")

    def update(
        self,
        op_code: int,
        cur_count: int,
        max_count: Optional[int] = None,
        message: str = "",
    ) -> None:
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


def _call_with_retries(
    operation_name: str,
    request_callable: Callable[[], Optional[object]],
    retry_delay_seconds: float,
    max_retries: int,
):
    """Call a request function with retries and logging."""
    attempt = 0
    delay = retry_delay_seconds

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
            logger.warning(
                "Retry %d/%d for operation '%s' failed: %s",
                attempt + 1,
                max_retries,
                operation_name,
                exc,
                exc_info=True,
            )
            sleep(delay)
            attempt += 1


def _flatten_record(record: dict) -> dict:
    """Flatten one-level nested dictionaries from World Bank API responses."""
    flattened = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{key}.{nested_key}"] = nested_value
        else:
            flattened[key] = value
    return flattened


def _polars_from_world_bank_records(records: Optional[object]) -> pl.DataFrame:
    """Convert World Bank API iterables/custom objects to a Polars DataFrame."""
    if isinstance(records, pl.DataFrame):
        return records

    rows = []
    for record in records:
        if isinstance(record, dict):
            rows.append(_flatten_record(record))
        else:
            rows.append(record)

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts(rows, infer_schema_length=len(rows))


def _download_source_indicators(
    db_id: int,
    sql_uri: str,
    table_name: str,
    table_exists_mode: str,
    api_max_retries: int,
    api_retry_delay_seconds: float,
) -> bool:
    """Download indicators for a given World Bank database and save to `PostgreSQL`"""
    indicator_records = _call_with_retries(
        operation_name=f"series.list(db={db_id})",
        request_callable=lambda: list(wb.series.list(db=db_id)),
        max_retries=api_max_retries,
        retry_delay_seconds=api_retry_delay_seconds,
    )

    if indicator_records is None:
        logger.warning(
            "Skipping source indicators for db_id=%s after all retries failed", db_id
        )
        return False

    df_indicators = _polars_from_world_bank_records(indicator_records)

    df_indicators = df_indicators.with_columns(pl.lit(db_id).alias("database_id"))
    df_indicators = df_indicators.rename({"value": "description"})
    df_indicators.write_database(
        table_name,
        connection=sql_uri,
        if_table_exists=table_exists_mode,
        engine="sqlalchemy",
    )
    return True


def _download_config(path: str) -> dict:
    """Download config for downloads"""
    download_config = json.load(open(path))
    return download_config


def _get_sql_config(username: str, password: str, host: str, port: int, db: str) -> str:
    """Download config from .env for setting up an connection with `PostgreSQL`"""
    if db:
        uri = f"postgresql://{username}:{password}@{host}:{port}/{db}"
    else:
        uri = f"postgresql://{username}:{password}@{host}:{port}"
    return uri


def _test_sql(uri: str) -> bool:
    """Test if provided connection string works correct"""
    try:
        with create_engine(uri).connect() as connection:
            _test = connection.execute(text("SELECT 1 AS number")).scalar_one()
        logger.info("Successfully tested connection to `PostgreSQL`")
        return bool(_test)
    except Exception:
        logger.exception("An error occured while testing connection to `PostgreSQL`")
        return False


def _test_world_bank_api() -> bool:
    """Test if connection to `world-bank` can be established"""
    try:
        _records = _call_with_retries(
            operation_name="source.list",
            request_callable=lambda: wb.source.list(),
            max_retries=2,
            retry_delay_seconds=1.0,
        )
        if _records is None:
            return False
        _test = _polars_from_world_bank_records(_records)
        logger.info("Successfully tested connection to World Bank API")
        return _test.height > 0
    except Exception:
        logger.exception("An error occured while testing connection to World Bank API")
        return False
