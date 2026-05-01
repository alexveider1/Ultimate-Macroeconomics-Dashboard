import sys
import yaml
import logging
import os
import tqdm

from src.extractors import WorldBankDownloader, NewsDownloader, YahooDownloader
from src.utils.schema import load_database_schema


class _TqdmHandler(logging.StreamHandler):
    """Routes log records through tqdm.write() to prevent progress bar overlap."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


CONFIG_PATH = "config.yaml"


def _require(mapping: dict, *path: str) -> object:
    """Look up a nested config key and raise a clear error if any segment is missing."""
    current: object = mapping
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(
                f"Missing required config key '{'.'.join(path)}' in {CONFIG_PATH}"
            )
        current = current[segment]
    return current


def main() -> None:
    """Main function to run the downloaders."""
    container_data_dir = os.path.join("_container_data")
    news_output_dir = os.path.join(container_data_dir, "news")

    os.makedirs(container_data_dir, exist_ok=True)
    os.makedirs(news_output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(
                os.path.join(container_data_dir, "app.log"),
                mode="w",
                encoding="utf-8",
            ),
            _TqdmHandler(),
        ],
    )

    with open(CONFIG_PATH) as f:
        args = yaml.safe_load(f)

    env_file = _require(args, "shared", "env_file")
    postgres_host = _require(args, "postgres", "host")
    postgres_port = _require(args, "postgres", "port")
    postgres_db = _require(args, "postgres", "database")
    qdrant_host = _require(args, "qdrant", "host")
    qdrant_port = _require(args, "qdrant", "port")
    database_schema_path = _require(args, "shared", "database_schema")
    database_schema = load_database_schema(database_schema_path)
    world_bank_download_config = _require(args, "shared", "world_bank_download_config")
    news_download_config = _require(args, "shared", "news_download_config")
    yahoo_download_config = _require(args, "shared", "yahoo_download_config")
    repo_url = _require(args, "downloader_general", "repo_url")
    openai_base_url = _require(args, "shared", "openai_base_url")
    openai_embedding_model = _require(args, "shared", "openai_embedding_model")
    openai_embedding_model_max_tokens = _require(
        args, "shared", "openai_embedding_model_max_tokens"
    )
    openai_model_dimensions = _require(
        args, "shared", "openai_embedding_model_dimensions"
    )

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
        qdrant_port=qdrant_port,
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


if __name__ == "__main__":
    main()
