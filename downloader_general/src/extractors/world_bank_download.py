import os
import logging
import polars as pl
import wbgapi as wb

from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.utils.downloads import (
    _get_sql_config,
    _test_sql,
    _test_world_bank_api,
    _download_config,
    _call_with_retries,
    _polars_from_world_bank_records,
    _download_source_indicators,
)

from src.core.base_downloaders import BaseWorldBankDownloader

logger = logging.getLogger(__name__)


class WorldBankDownloader(BaseWorldBankDownloader):
    """Downloader for World Bank data"""

    def __init__(self, env_path: str, download_config_path: str = None):
        self.env_path = env_path
        self.download_config = _download_config(download_config_path)
        self.sql_uri = None
        self.sql_database_name = None

        self.database_table_name = "databases"
        self.database_indicators_table_name = "database_indicators"
        self.metadata_table_name = "metadata"
        self.indicators_table_name = "indicators"

        self.successfull_connections = False
        self._basic_tables_initialized = False
        self._metadata_table_initialized = False
        self._indicators_table_initialized = False
        self.download_max_retries = 3
        self.download_retry_delay_seconds = 5

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
            self.sql_database_name = db
        else:
            self.sql_uri = None
            self.sql_database_name = None
            logger.warning("Connection test to SQL database failed")
        _world_bank_test = _test_world_bank_api()
        result = _sql_test and _world_bank_test
        self.successfull_connections = result
        return result

    @staticmethod
    def _quote_sql_identifier(value: str) -> str:
        return '"' + str(value).replace('"', '""') + '"'

    @staticmethod
    def _quote_sql_literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    def create_llm_readonly_user(self):
        load_dotenv(self.env_path)
        llm_username = str(os.getenv("POSTGRES_LLM_USERNAME") or "").strip()
        llm_password = str(os.getenv("POSTGRES_LLM_PASSWORD") or "")

        if not self.sql_uri or not self.sql_database_name:
            raise RuntimeError(
                "SQL connection must be initialized before creating the LLM user."
            )

        if not llm_username or not llm_password:
            raise RuntimeError(
                "POSTGRES_LLM_USERNAME and POSTGRES_LLM_PASSWORD must be set."
            )

        role_identifier = self._quote_sql_identifier(llm_username)
        database_identifier = self._quote_sql_identifier(self.sql_database_name)
        password_literal = self._quote_sql_literal(llm_password)
        engine = create_engine(self.sql_uri, pool_pre_ping=True)

        try:
            with engine.begin() as connection:
                role_exists = (
                    connection.execute(
                        text(
                            "SELECT 1 FROM pg_catalog.pg_roles "
                            "WHERE rolname = :role_name"
                        ),
                        {"role_name": llm_username},
                    ).scalar()
                    is not None
                )

                role_statement = "ALTER ROLE" if role_exists else "CREATE ROLE"
                connection.execute(
                    text(
                        f"{role_statement} {role_identifier} "
                        f"WITH LOGIN PASSWORD {password_literal} "
                        "NOSUPERUSER NOCREATEDB NOCREATEROLE "
                        "NOINHERIT NOREPLICATION NOBYPASSRLS"
                    )
                )
                connection.execute(
                    text(
                        f"REVOKE ALL PRIVILEGES ON DATABASE {database_identifier} "
                        f"FROM {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        f"GRANT CONNECT ON DATABASE {database_identifier} "
                        f"TO {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        f"REVOKE ALL PRIVILEGES ON SCHEMA public FROM {role_identifier}"
                    )
                )
                connection.execute(
                    text(f"GRANT USAGE ON SCHEMA public TO {role_identifier}")
                )
                connection.execute(
                    text(
                        "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public "
                        f"FROM {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        "REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public "
                        f"FROM {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        f"GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                        f"REVOKE ALL ON TABLES FROM {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                        f"GRANT SELECT ON TABLES TO {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                        f"REVOKE ALL ON SEQUENCES FROM {role_identifier}"
                    )
                )
                connection.execute(
                    text(
                        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
                        f"GRANT SELECT ON SEQUENCES TO {role_identifier}"
                    )
                )
        finally:
            engine.dispose()

        logger.info(
            "Configured read-only LLM SQL user for database "
            f"(username={llm_username}, database={self.sql_database_name})"
        )

    def download_basic_tables(self):
        logger.info("Starting download of World Bank basic tables")
        source_records = _call_with_retries(
            operation_name="source.list",
            request_callable=lambda: wb.source.list(),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        df = _polars_from_world_bank_records(source_records)
        df = df.with_columns(pl.col("id").alias("database_id"))
        df.write_database(
            self.database_table_name,
            connection=self.sql_uri,
            if_table_exists="replace",
            engine="sqlalchemy",
        )
        logger.info("Starting download of World Bank countries table")
        country_records = _call_with_retries(
            operation_name="economy.list",
            request_callable=lambda: wb.economy.list(skipAggs=True, db=2, labels=True),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )
        df_countries = _polars_from_world_bank_records(country_records)
        df_countries.write_database(
            "countries",
            connection=self.sql_uri,
            if_table_exists="replace",
            engine="sqlalchemy",
        )
        logger.info("Finished downloading World Bank countries table")

        logger.info("Starting download of World Bank source indicators")
        source_ids = df.get_column("database_id").to_list()
        for source_id in tqdm(source_ids, desc="Downloading source indicators"):
            _download_source_indicators(
                db_id=source_id,
                sql_uri=self.sql_uri,
                table_name=self.database_indicators_table_name,
                table_exists_mode="append",
                api_max_retries=self.download_max_retries,
                api_retry_delay_seconds=self.download_retry_delay_seconds,
            )
        logger.info("Finished downloading World Bank source indicators")
        logger.info("Finished download of World Bank basic tables")
        self._basic_tables_initialized = True

    def download_db(self, indicator_id, db):
        logger.info(
            f"Starting download of World Bank indicator data (indicator_id={indicator_id}, db={db})"
        )
        data_records = _call_with_retries(
            operation_name=f"data.fetch(indicator_id={indicator_id}, db={db})",
            request_callable=lambda: wb.data.fetch(
                indicator_id,
                db=db,
                skipAggs=True,
                economy="all",
                time="all",
                skipBlanks=False,
                numericTimeKeys=True,
            ),
            max_retries=self.download_max_retries,
            retry_delay_seconds=self.download_retry_delay_seconds,
        )

        df = _polars_from_world_bank_records(data_records)

        if df.is_empty():
            logger.warning(
                f"No data found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return

        economy_column = "economy"
        year_column = "time"

        df = df.select(
            [
                pl.col(economy_column).alias("economy"),
                pl.col(year_column).alias("year"),
                pl.col("value"),
            ]
        ).with_columns(
            [
                pl.lit(indicator_id).alias("indicator_id"),
                pl.lit(db).alias("db_id"),
            ]
        )
        table_exists_mode = (
            "replace" if not self._indicators_table_initialized else "append"
        )
        df.write_database(
            self.indicators_table_name,
            connection=self.sql_uri,
            if_table_exists=table_exists_mode,
            engine="sqlalchemy",
        )
        logger.info(
            f"Finished download of World Bank indicator data (indicator_id={indicator_id}, db={db})"
        )
        sleep(30)
        self._indicators_table_initialized = True

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

        if metadata_response is None:
            logger.warning(
                f"No metadata found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return

        metadata = metadata_response.metadata
        indicator_name = metadata.get("IndicatorName")
        unit_of_measure = metadata.get("Unitofmeasure")
        source = metadata.get("Source")
        dev_relevance = metadata.get("Developmentrelevance")
        limitations_and_exceptions = metadata.get("Limitationsandexceptions")
        statistical_concept_and_methodology = metadata.get(
            "Statisticalconceptandmethodology"
        )
        dataframe_dict = {
            "indicator_id": indicator_id,
            "db_id": db,
            "indicator_name": indicator_name,
            "units": unit_of_measure,
            "source": source,
            "development_relevance": dev_relevance,
            "limitations_and_exceptions": limitations_and_exceptions,
            "statistical_concept_and_methodology": statistical_concept_and_methodology,
        }
        df = pl.DataFrame([dataframe_dict])
        if df.is_empty():
            logger.warning(
                f"No metadata found for World Bank indicator (indicator_id={indicator_id}, db={db})"
            )
            return
        table_exists_mode = (
            "replace" if not self._metadata_table_initialized else "append"
        )
        df.write_database(
            self.metadata_table_name,
            connection=self.sql_uri,
            if_table_exists=table_exists_mode,
            engine="sqlalchemy",
        )
        logger.info(
            f"Finished download of World Bank indicator metadata (indicator_id={indicator_id}, db={db})"
        )
        sleep(30)
        self._metadata_table_initialized = True

    def run(self):
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
            ):
                self.download_metadata(indicator_id, db_id)
                self.download_db(indicator_id, db_id)
            logger.info(f"Finished downloads for World Bank database (db_id={db_id})")

        self.create_llm_readonly_user()
