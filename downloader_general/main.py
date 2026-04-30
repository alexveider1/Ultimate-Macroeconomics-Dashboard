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

    args = yaml.safe_load(open("config.yaml"))
    env_file = args["shared"]["env_file"]
    postgres_host = args["postgres"]["host"]
    postgres_port = args["postgres"]["port"]
    postgres_db = args["postgres"]["database"]
    qdrant_host = args["qdrant"]["host"]
    qdrant_port = args["qdrant"]["port"]
    database_schema_path = args["shared"]["database_schema"]
    database_schema = load_database_schema(database_schema_path)
    world_bank_download_config = args["shared"]["world_bank_download_config"]
    news_download_config = args["shared"]["news_download_config"]
    yahoo_download_config = args["shared"]["yahoo_download_config"]
    repo_url = args["downloader_general"]["repo_url"]
    openai_base_url = args["shared"]["openai_base_url"]
    openai_embedding_model = args["shared"]["openai_embedding_model"]
    openai_embedding_model_max_tokens = args["shared"][
        "openai_embedding_model_max_tokens"
    ]
    openai_model_dimensions = args["shared"]["openai_embedding_model_dimensions"]

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
