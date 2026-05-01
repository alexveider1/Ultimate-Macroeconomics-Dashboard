import sys

from src.core.base_downloaders import BaseNewsDownloader

from src.utils.downloads import (
    CloneProgress,
    _call_with_retries,
    _download_config,
    _remove_readonly,
)

from git import Repo
from zipfile import ZipFile
from tqdm import tqdm
from time import sleep
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import OpenAI
from tiktoken import encoding_for_model

import os
import json
import logging
import warnings
import shutil
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

SUPPORTED_ARTICLE_LANGUAGES = {"english", "en"}


class NewsDownloader(BaseNewsDownloader):
    """Downloader and Embedder for news articles from a GitHub repository."""

    def __init__(
        self,
        env_file: str,
        repo_url: str,
        save_path: str,
        qdrant_host: str,
        qdrant_port: str,
        config_path: str,
        openai_base_url: str | None = None,
        openai_embedding_model: str = "openai/text-embedding-3-small",
        openai_token_limit: int = 8192,
        openai_model_dimensions: int = 1536,
    ) -> None:
        self.env_path = env_file
        self.github_api_url = "https://api.github.com"

        self.repo_url = repo_url
        self.save_path = save_path

        self.is_downloaded = False
        self.is_parsed = False

        self.parsed_metadata = {}

        self.download_config = _download_config(config_path)
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.download_retry_delay_seconds = 5
        self.batch_size = 100

        self.openai_base_url = openai_base_url
        self.openai_embedding_model = openai_embedding_model
        self.embedding_token_limit = openai_token_limit
        self.openai_model_dimensions = openai_model_dimensions
        self.embedding_encoding = self._build_embedding_encoding()

    def _initialize_connections(self) -> bool:
        load_dotenv(self.env_path)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error(
                "OPENAI_API_KEY is not set; news embeddings cannot be generated"
            )
            return False
        self.openai_client = OpenAI(
            base_url=self.openai_base_url, api_key=openai_api_key
        )
        try:
            qdrant_api_key = os.getenv("QDRANT_API_KEY") or os.getenv(
                "QDRANT__SERVICE__API_KEY"
            )
            response = requests.get(self.github_api_url, timeout=5)
            qdrant_host = str(self.qdrant_host).strip()
            parsed_host = urlparse(qdrant_host)
            if parsed_host.scheme:
                qdrant_url = qdrant_host
            else:
                qdrant_url = f"http://{qdrant_host}:{int(self.qdrant_port)}"

            if not qdrant_api_key:
                raise ValueError(
                    "Missing Qdrant API key. Set QDRANT__SERVICE__API_KEY in env file."
                )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    prefer_grpc=False,
                )

            collections_response = self.qdrant_client.get_collections()
        except Exception as exc:
            logger.exception(
                "Failed to initialize downloader connections",
                extra={"operation": "Initializing connections", "error": str(exc)},
            )
            return False
        if response.status_code == 200 and bool(collections_response):
            parent_dir = os.path.dirname(os.path.abspath(self.save_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            return True
        else:
            return False

    def _build_embedding_encoding(self):
        model_name = str(self.openai_embedding_model).split("/")[-1]
        try:
            return encoding_for_model(model_name)
        except Exception:
            return encoding_for_model(model_name)

    def _truncate_for_embedding(self, text: str, article_path: str) -> str:
        if self.embedding_encoding is not None:
            token_ids = self.embedding_encoding.encode(text)
            token_count = len(token_ids)
            if token_count <= self.embedding_token_limit:
                return text

            truncated_text = self.embedding_encoding.decode(
                token_ids[: self.embedding_token_limit]
            )
            logger.warning(
                "Article text truncated for embeddings token limit",
                extra={
                    "operation": "Embedding and Uploading",
                    "article_file_path": article_path,
                    "original_token_count": token_count,
                    "truncated_token_count": self.embedding_token_limit,
                    "embedding_model": self.openai_embedding_model,
                },
            )
            return truncated_text

        max_chars = self.embedding_token_limit * 4
        if len(text) <= max_chars:
            return text

        logger.warning(
            "Article text truncated with character fallback",
            extra={
                "operation": "Embedding and Uploading",
                "article_file_path": article_path,
                "original_char_count": len(text),
                "truncated_char_count": max_chars,
                "embedding_model": self.openai_embedding_model,
            },
        )
        return text[:max_chars]

    def download_repository(self) -> bool:
        repo_url = self.repo_url
        os.makedirs(self.save_path, exist_ok=True)

        for item_name in os.listdir(self.save_path):
            item_path = os.path.join(self.save_path, item_name)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, onexc=_remove_readonly)
            else:
                try:
                    os.remove(item_path)
                except PermissionError:
                    os.chmod(item_path, 0o700)
                    os.remove(item_path)

        clone_result = _call_with_retries(
            "Cloning from github",
            lambda: Repo.clone_from(repo_url, self.save_path, progress=CloneProgress()),
            retry_delay_seconds=30,
            max_retries=3,
        )
        self.is_downloaded = clone_result is not None
        return self.is_downloaded

    def parse_repository(self) -> None:
        metadata = {}
        source_datasets_dir = os.path.join(self.save_path, "News_Datasets")
        allowed_topics = self.download_config

        iter_files = [
            f
            for f in os.listdir(source_datasets_dir)
            if f.split("_")[0] in allowed_topics
        ]
        for filename in tqdm(
            iter_files,
            desc="Unzipping files",
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            if not filename.endswith(".zip"):
                continue

            base_name = filename[:-4]
            parts = base_name.rsplit("_", 2)
            if len(parts) != 3:
                logger.warning(
                    "Skipping news archive with unexpected name format: %s",
                    filename,
                )
                continue
            topic, sentiment, date_str = parts
            if topic in allowed_topics:
                topic_normalized = topic.strip().lower()
                parsed_date = (
                    datetime.strptime(date_str, "%Y%m%d%H%M%S").date().isoformat()
                )

                zip_path = os.path.join(source_datasets_dir, filename)
                extract_dir = os.path.join(self.save_path, base_name)

                collection_name = (
                    f"{topic_normalized}_{sentiment}".replace(" ", "_")
                    .replace(",", " ")
                    .lower()
                )

                os.makedirs(extract_dir, exist_ok=True)

                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

                nested_extract_dir = os.path.join(extract_dir, base_name)
                if os.path.isdir(nested_extract_dir):
                    for nested_item in os.listdir(nested_extract_dir):
                        shutil.move(
                            os.path.join(nested_extract_dir, nested_item),
                            os.path.join(extract_dir, nested_item),
                        )
                    shutil.rmtree(nested_extract_dir, onexc=_remove_readonly)

                article_file_paths = []
                for root_dir, _, files in os.walk(extract_dir):
                    for article_filename in files:
                        if article_filename.endswith(".json"):
                            article_file_paths.append(
                                os.path.abspath(
                                    os.path.join(root_dir, article_filename)
                                )
                            )

                article_file_paths.sort()

                for article_file_path in article_file_paths:
                    try:
                        with open(
                            article_file_path, "r", encoding="utf-8"
                        ) as article_file:
                            article_payload = json.load(article_file)
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.warning(
                            f"Skipping invalid article file: {article_file_path}",
                            extra={
                                "operation": "Parsing article JSON",
                                "error": str(exc),
                            },
                        )
                        continue

                    article_language = (
                        str(article_payload.get("language", "")).strip().lower()
                    )
                    if article_language not in SUPPORTED_ARTICLE_LANGUAGES:
                        logger.info(
                            "Skipping article with unsupported language metadata",
                            extra={
                                "operation": "Parsing article JSON",
                                "article_file_path": article_file_path,
                                "language": article_payload.get("language"),
                            },
                        )
                        continue

                    metadata.setdefault(collection_name, []).append(
                        {
                            "topic": topic,
                            "sentiment": sentiment,
                            "date": parsed_date,
                            "archive_file": os.path.abspath(zip_path),
                            "extracted_dir": os.path.abspath(extract_dir),
                            "article_file_path": os.path.abspath(article_file_path),
                            "archive_name": base_name,
                            "article": article_payload,
                        }
                    )

        self.parsed_metadata = metadata
        self.is_parsed = True

    def clean_repository(self) -> None:
        preserved_dirs = {
            os.path.abspath(metadata_entry["extracted_dir"])
            for metadata_entries in self.parsed_metadata.values()
            for metadata_entry in metadata_entries
        }

        for item in os.listdir(self.save_path):
            item_path = os.path.join(self.save_path, item)
            item_abs_path = os.path.abspath(item_path)
            if item_abs_path in preserved_dirs:
                continue

            if os.path.isdir(item_path):
                shutil.rmtree(item_path, onexc=_remove_readonly)
            else:
                try:
                    os.remove(item_path)
                except PermissionError:
                    os.chmod(item_path, 0o700)
                    os.remove(item_path)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.openai_client.embeddings.create(
                input=texts, model=self.openai_embedding_model
            )

            sorted_data = sorted(response.data, key=lambda x: x.index)

            result = [item.embedding for item in sorted_data]
            return result

        except Exception as exc:
            logger.exception("Getting embeddings failed", exc_info=exc)
            return []

    def upload_to_qdrant(self) -> None:
        for collection_name, metadata_entries in self.parsed_metadata.items():
            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.openai_model_dimensions,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )

            for i in tqdm(
                range(0, len(metadata_entries), self.batch_size),
                desc=f"Embedding and Uploading: {collection_name}",
                dynamic_ncols=True,
                file=sys.stdout,
            ):
                batch_metadata = metadata_entries[i : i + self.batch_size]

                texts_to_embed = []
                for meta in batch_metadata:
                    article_data = meta.get("article", {})
                    text = article_data.get("text")
                    texts_to_embed.append(
                        self._truncate_for_embedding(
                            str(text), meta.get("article_file_path", "unknown")
                        )
                    )

                embeddings = _call_with_retries(
                    "get_embeddings",
                    lambda: self.get_embeddings(texts_to_embed),
                    retry_delay_seconds=3,
                    max_retries=3,
                )

                if not embeddings:
                    logger.warning(
                        "Skipping batch because embeddings are empty",
                        extra={
                            "operation": "Embedding and Uploading",
                            "collection": collection_name,
                            "batch_start": i,
                            "batch_size": len(batch_metadata),
                        },
                    )
                    continue

                if len(embeddings) != len(batch_metadata):
                    logger.warning(
                        "Skipping batch due to embedding count mismatch",
                        extra={
                            "operation": "Embedding and Uploading",
                            "collection": collection_name,
                            "batch_start": i,
                            "metadata_count": len(batch_metadata),
                            "embedding_count": len(embeddings),
                        },
                    )
                    continue

                points = []
                for meta, embedding_vector in zip(batch_metadata, embeddings):
                    point_id = str(uuid4())
                    points.append(
                        models.PointStruct(
                            id=point_id, payload=meta, vector=embedding_vector
                        )
                    )

                if points:
                    self.qdrant_client.upsert(
                        collection_name=collection_name, points=points
                    )

            sleep(self.download_retry_delay_seconds)

    def run(self) -> None:
        if not self.download_repository():
            return
        self.parse_repository()
        self.clean_repository()
        self.upload_to_qdrant()
