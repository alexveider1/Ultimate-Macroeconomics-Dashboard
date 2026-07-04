"""Entry point for the ingestion container.

On every container start: load env + config, then **always** bootstrap the
read-only LLM Postgres role (idempotent CREATE/ALTER + SELECT grants — cheap
to re-run, and required for password rotation, upgrades, and grants on tables
that didn't exist at the last bootstrap). After that, run the three
downloaders (World Bank → news → Yahoo) **once only**, gated by a marker
file (``.download_completed``) written after a successful run; subsequent
boots see the marker and skip downloads but still re-apply the bootstrap.
The downloaders run in order: World Bank → news → Yahoo → Binance → FRED →
Eurostat → Actually Relevant → World Bank articles.
"""

import logging
import os
from pathlib import Path
import sys

from src.config import load_config
from src.extractors import (
    ActuallyRelevantDownloader,
    BinanceDownloader,
    EurostatDownloader,
    FredDownloader,
    NewsDownloader,
    WorldBankArticlesDownloader,
    WorldBankDownloader,
    YahooDownloader,
)
from src.settings import load_settings
from src.utils.db_bootstrap import ensure_llm_role
from src.utils.downloads import _get_sql_config
from src.utils.schema import load_database_schema

CONFIG_PATH = Path("config.yaml")
DEFAULT_DOWNLOAD_MARKER = Path("_container_data/.download_completed")


def main() -> None:
    """Run the bootstrap (LLM role + grants) and, if not yet done, the downloads.

    The bootstrap runs on every container start — it's idempotent and cheap,
    and it's the path that grants ``SELECT`` on tables added since the marker
    was first written (e.g. tables created on a fresh deploy after a schema
    upgrade). Downloads themselves stay one-shot, gated by
    ``DEFAULT_DOWNLOAD_MARKER`` (overrideable via ``DOWNLOADER_ONCE_MARKER``).
    Each downloader's ``_initialize_connections`` is checked before ``run()``
    so a failed health check skips that source rather than aborting the whole
    job. A failure in the LLM-role bootstrap is logged and ignored — the rest
    of the ingestion can still proceed.
    """
    container_data_dir = Path("_container_data")
    news_output_dir = container_data_dir / "news"
    marker_path = Path(os.getenv("DOWNLOADER_ONCE_MARKER", str(DEFAULT_DOWNLOAD_MARKER)))

    container_data_dir.mkdir(parents=True, exist_ok=True)
    news_output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                container_data_dir / "app.log",
                mode="w",
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    config = load_config(CONFIG_PATH)
    shared = config.shared

    # config.yaml holds the fallback DB name; POSTGRES_DB in .env wins because
    # that's the value the postgres image uses on first volume init.
    env_file = shared.env_file
    secrets = load_settings(env_file)
    postgres_host = config.postgres.host
    postgres_port = config.postgres.port
    postgres_db = secrets.postgres_db or config.postgres.database
    qdrant_host = config.qdrant.host
    qdrant_port = config.qdrant.port
    database_schema = load_database_schema(shared.database_schema)
    world_bank_download_config = shared.world_bank_download_config
    news_download_config = shared.news_download_config
    yahoo_download_config = shared.yahoo_download_config
    binance_download_config = shared.binance_download_config
    fred_download_config = shared.fred_download_config
    eurostat_download_config = shared.eurostat_download_config
    actually_relevant_download_config = shared.actually_relevant_download_config
    world_bank_articles_download_config = shared.world_bank_articles_download_config
    nuts_geojson = shared.nuts_geojson
    eurostat_nuts_level = shared.eurostat_nuts_level
    repo_url = config.downloader_general.repo_url
    openai_base_url = shared.openai_base_url
    openai_embedding_model = shared.openai_embedding_model
    openai_embedding_model_max_tokens = shared.openai_embedding_model_max_tokens
    openai_model_dimensions = shared.openai_embedding_model_dimensions

    superuser_uri = _get_sql_config(
        username=secrets.postgres_user,
        password=secrets.postgres_password,
        host=str(postgres_host),
        port=int(postgres_port),
        db=str(postgres_db),
    )
    try:
        ensure_llm_role(
            sql_uri=superuser_uri,
            llm_username=secrets.postgres_llm_user,
            llm_password=secrets.postgres_llm_password,
        )
    except Exception:
        logging.exception("LLM role bootstrap failed; continuing with downloads")

    if marker_path.exists():
        logging.info(
            "Download marker present (%s); bootstrap re-applied, skipping downloads.",
            marker_path,
        )
        return

    world_bank_downloader = WorldBankDownloader(
        env_path=env_file,
        download_config_path=world_bank_download_config,
        database_schema=database_schema,
    )
    if world_bank_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        world_bank_downloader.run()

    news_downloader = NewsDownloader(
        env_file=env_file,
        repo_url=repo_url,
        qdrant_host=qdrant_host,
        qdrant_port=str(qdrant_port),
        config_path=news_download_config,
        save_path=news_output_dir,
        openai_base_url=openai_base_url,
        openai_embedding_model=openai_embedding_model,
        openai_token_limit=openai_embedding_model_max_tokens,
        openai_model_dimensions=openai_model_dimensions,
    )
    if news_downloader._initialize_connections():
        news_downloader.run()

    yahoo_downloader = YahooDownloader(
        env_path=env_file,
        download_config_path=yahoo_download_config,
        database_schema=database_schema,
    )
    if yahoo_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        yahoo_downloader.run()

    binance_downloader = BinanceDownloader(
        env_path=env_file,
        download_config_path=binance_download_config,
        database_schema=database_schema,
    )
    if binance_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        binance_downloader.run()

    fred_downloader = FredDownloader(
        env_path=env_file,
        download_config_path=fred_download_config,
        database_schema=database_schema,
    )
    if fred_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        fred_downloader.run()

    eurostat_downloader = EurostatDownloader(
        env_path=env_file,
        geojson_path=nuts_geojson,
        download_config_path=eurostat_download_config,
        nuts_level=eurostat_nuts_level,
        database_schema=database_schema,
    )
    if eurostat_downloader._initialize_connections(
        host=postgres_host,
        port=postgres_port,
        db=postgres_db,
    ):
        eurostat_downloader.run()

    actually_relevant_downloader = ActuallyRelevantDownloader(
        env_file=env_file,
        download_config_path=actually_relevant_download_config,
        qdrant_host=qdrant_host,
        qdrant_port=str(qdrant_port),
        openai_base_url=openai_base_url,
        openai_embedding_model=openai_embedding_model,
        openai_token_limit=openai_embedding_model_max_tokens,
        openai_model_dimensions=openai_model_dimensions,
    )
    if actually_relevant_downloader._initialize_connections():
        actually_relevant_downloader.run()

    world_bank_articles_downloader = WorldBankArticlesDownloader(
        env_file=env_file,
        download_config_path=world_bank_articles_download_config,
        qdrant_host=qdrant_host,
        qdrant_port=str(qdrant_port),
        openai_base_url=openai_base_url,
        openai_embedding_model=openai_embedding_model,
        openai_token_limit=openai_embedding_model_max_tokens,
        openai_model_dimensions=openai_model_dimensions,
    )
    if world_bank_articles_downloader._initialize_connections():
        world_bank_articles_downloader.run()

    # Mark the one-shot download as completed so future container starts
    # only re-apply the bootstrap and skip the multi-hour ingestion.
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
        logging.info("Download marker written: %s", marker_path)
    except OSError:
        logging.exception("Could not write download marker at %s", marker_path)


if __name__ == "__main__":
    main()
