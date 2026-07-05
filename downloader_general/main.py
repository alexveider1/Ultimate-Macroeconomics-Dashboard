"""Entry point for the ingestion container.

On every container start: load env + config, then **always** bootstrap the
read-only LLM Postgres role (idempotent CREATE/ALTER + SELECT grants — cheap
to re-run, and required for password rotation, upgrades, and grants on tables
that didn't exist at the last bootstrap). After that, run the eight downloaders
**once only**, gated by a marker file (``.download_completed``) written after a
successful run; subsequent boots see the marker and skip the initial ingest but
still re-apply the bootstrap. The downloaders run in order: World Bank → news →
Yahoo → Binance → FRED → Eurostat → Actually Relevant → World Bank articles.

Once the initial ingest is done, the process does **not** exit: if
``scheduler.enabled`` it hands off to :func:`src.scheduler.run_scheduler`, which
keeps the container alive and refreshes each source on its own interval by
calling that source's incremental ``update()`` (append-only). The container is
therefore long-running (``restart: unless-stopped`` in Compose) and reports
healthy via the ``.download_completed`` marker so dependents that used to wait on
``service_completed_successfully`` now wait on ``service_healthy``.
"""

import atexit
from collections.abc import Callable
import logging
import os
from pathlib import Path
import sys
from typing import Any

from src.config import DownloaderGeneralConfig, load_config
from src.core import tracing
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
from src.scheduler import SourceJob, run_scheduler
from src.settings import Settings, load_settings
from src.utils.db_bootstrap import ensure_llm_role
from src.utils.downloads import _get_sql_config
from src.utils.schema import load_database_schema

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")
DEFAULT_DOWNLOAD_MARKER = Path("_container_data/.download_completed")

# Order the initial full ingest runs in (FK targets first within each source).
INGEST_ORDER = [
    "world_bank",
    "news",
    "yahoo",
    "binance",
    "fred",
    "eurostat",
    "actually_relevant",
    "world_bank_articles",
]


def build_downloaders(
    config: DownloaderGeneralConfig,
    secrets: Settings,
    news_output_dir: Path,
) -> dict[str, tuple[Any, Callable[[], bool]]]:
    """Construct every downloader plus a zero-arg connection-init callable.

    Returns an ordered ``{name: (downloader, init)}`` map shared by the initial
    ingest (which calls ``downloader.run()``) and the scheduler (which calls
    ``downloader.update()``). ``init()`` returns whether that source's
    connections are healthy — the caller skips ``run``/``update`` when it's
    ``False``. The map's key order matches :data:`INGEST_ORDER`.

    Args:
        config: Parsed ``config.yaml`` slice for this service.
        secrets: Typed secrets (Postgres roles, OpenAI/FRED/Qdrant keys).
        news_output_dir: Directory the news pipeline clones + unzips into.
    """
    shared = config.shared
    postgres_host = config.postgres.host
    postgres_port = config.postgres.port
    postgres_db = secrets.postgres_db or config.postgres.database
    qdrant_host = config.qdrant.host
    qdrant_port = str(config.qdrant.port)
    database_schema = load_database_schema(shared.database_schema)

    def pg_init(dl: Any) -> Callable[[], bool]:
        return lambda: dl._initialize_connections(
            host=postgres_host, port=postgres_port, db=postgres_db
        )

    def rag_init(dl: Any) -> Callable[[], bool]:
        return lambda: dl._initialize_connections()

    world_bank = WorldBankDownloader(
        env_path=shared.env_file,
        download_config_path=shared.world_bank_download_config,
        database_schema=database_schema,
    )
    news = NewsDownloader(
        env_file=shared.env_file,
        repo_url=config.downloader_general.repo_url,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        config_path=shared.news_download_config,
        save_path=news_output_dir,
        openai_base_url=shared.openai_base_url,
        openai_embedding_model=shared.openai_embedding_model,
        openai_token_limit=shared.openai_embedding_model_max_tokens,
        openai_model_dimensions=shared.openai_embedding_model_dimensions,
    )
    yahoo = YahooDownloader(
        env_path=shared.env_file,
        download_config_path=shared.yahoo_download_config,
        database_schema=database_schema,
    )
    binance = BinanceDownloader(
        env_path=shared.env_file,
        download_config_path=shared.binance_download_config,
        database_schema=database_schema,
    )
    fred = FredDownloader(
        env_path=shared.env_file,
        download_config_path=shared.fred_download_config,
        database_schema=database_schema,
    )
    eurostat = EurostatDownloader(
        env_path=shared.env_file,
        geojson_path=shared.nuts_geojson,
        download_config_path=shared.eurostat_download_config,
        nuts_level=shared.eurostat_nuts_level,
        database_schema=database_schema,
    )
    actually_relevant = ActuallyRelevantDownloader(
        env_file=shared.env_file,
        download_config_path=shared.actually_relevant_download_config,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        openai_base_url=shared.openai_base_url,
        openai_embedding_model=shared.openai_embedding_model,
        openai_token_limit=shared.openai_embedding_model_max_tokens,
        openai_model_dimensions=shared.openai_embedding_model_dimensions,
    )
    world_bank_articles = WorldBankArticlesDownloader(
        env_file=shared.env_file,
        download_config_path=shared.world_bank_articles_download_config,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        openai_base_url=shared.openai_base_url,
        openai_embedding_model=shared.openai_embedding_model,
        openai_token_limit=shared.openai_embedding_model_max_tokens,
        openai_model_dimensions=shared.openai_embedding_model_dimensions,
    )

    downloaders: dict[str, tuple[Any, Callable[[], bool]]] = {
        "world_bank": (world_bank, pg_init(world_bank)),
        "news": (news, rag_init(news)),
        "yahoo": (yahoo, pg_init(yahoo)),
        "binance": (binance, pg_init(binance)),
        "fred": (fred, pg_init(fred)),
        "eurostat": (eurostat, pg_init(eurostat)),
        "actually_relevant": (actually_relevant, rag_init(actually_relevant)),
        "world_bank_articles": (world_bank_articles, rag_init(world_bank_articles)),
    }
    return downloaders


def run_initial_ingest(downloaders: dict[str, tuple[Any, Callable[[], bool]]]) -> None:
    """Run each source's full ``run()`` once, skipping sources whose init fails."""
    for name in INGEST_ORDER:
        downloader, init = downloaders[name]
        if init():
            downloader.run()
        else:
            logger.warning("Skipping initial ingest for %s: connection init failed", name)


def build_jobs(
    downloaders: dict[str, tuple[Any, Callable[[], bool]]],
    config: DownloaderGeneralConfig,
) -> list[SourceJob]:
    """Build the enabled :class:`SourceJob` list from the ``scheduler`` config.

    Each job's ``run_tick`` re-establishes the source's connections and, if
    healthy, calls its incremental ``update()``; a failed init skips that tick
    (retried next interval) instead of raising.
    """
    jobs: list[SourceJob] = []
    for name, source_cfg in config.scheduler.sources.items():
        if not source_cfg.enabled:
            continue
        entry = downloaders.get(name)
        if entry is None:
            logger.warning("Scheduler: unknown source %r in config; skipping", name)
            continue
        downloader, init = entry
        interval_seconds = max(source_cfg.interval_minutes * 60.0, 60.0)

        def make_tick(name: str, downloader: Any, init: Callable[[], bool]) -> Callable[[], None]:
            def _tick() -> None:
                if init():
                    downloader.update()
                else:
                    logger.warning(
                        "Scheduler: connection init failed for %s; skipping this tick", name
                    )

            return _tick

        jobs.append(
            SourceJob(
                name=name,
                interval_seconds=interval_seconds,
                run_tick=make_tick(name, downloader, init),
            )
        )
    return jobs


def _setup_logging(container_data_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(container_data_dir / "app.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    """Bootstrap the LLM role, run the one-shot ingest if needed, then schedule updates.

    The bootstrap runs on every container start (idempotent + cheap, and the path
    that grants ``SELECT`` on tables added since the marker was written). The
    initial full ingest is gated by ``DEFAULT_DOWNLOAD_MARKER`` (overrideable via
    ``DOWNLOADER_ONCE_MARKER``). Regardless of whether the ingest ran, if
    ``scheduler.enabled`` the process then hands off to the incremental scheduler
    and stays alive.
    """
    container_data_dir = Path("_container_data")
    news_output_dir = container_data_dir / "news"
    marker_path = Path(os.getenv("DOWNLOADER_ONCE_MARKER", str(DEFAULT_DOWNLOAD_MARKER)))

    container_data_dir.mkdir(parents=True, exist_ok=True)
    news_output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(container_data_dir)

    config = load_config(CONFIG_PATH)
    secrets = load_settings(config.shared.env_file)

    # Initialise Langfuse tracing (no-op unless enabled + keys set) before the
    # downloaders build their embedding clients, and flush on exit so the last
    # traces aren't lost when the scheduler is signalled to stop.
    tracing.init_tracing(
        config.langfuse,
        public_key=secrets.langfuse_public_key,
        secret_key=secrets.langfuse_secret_key,
        release="downloader_general",
    )
    atexit.register(tracing.flush)

    # config.yaml holds the fallback DB name; POSTGRES_DB in .env wins because
    # that's the value the postgres image uses on first volume init.
    postgres_db = secrets.postgres_db or config.postgres.database
    superuser_uri = _get_sql_config(
        username=secrets.postgres_user,
        password=secrets.postgres_password,
        host=str(config.postgres.host),
        port=int(config.postgres.port),
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

    downloaders = build_downloaders(config, secrets, news_output_dir)

    if marker_path.exists():
        logger.info(
            "Download marker present (%s); bootstrap re-applied, skipping initial ingest.",
            marker_path,
        )
    else:
        run_initial_ingest(downloaders)
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()
            logger.info("Download marker written: %s", marker_path)
        except OSError:
            logging.exception("Could not write download marker at %s", marker_path)

    if config.scheduler.enabled:
        jobs = build_jobs(downloaders, config)
        run_scheduler(jobs, run_on_start=config.scheduler.run_on_start)
    else:
        logger.info("Scheduler disabled (scheduler.enabled=false); exiting after ingest.")


if __name__ == "__main__":
    main()
