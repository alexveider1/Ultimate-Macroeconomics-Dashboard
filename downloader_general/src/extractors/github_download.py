"""News pipeline: clone the Webhose news repo, unzip, embed, push to Qdrant.

End-to-end this runs once per clean boot via :meth:`NewsDownloader.run`. Per-zip
extraction and embedding batches are parallelised with ``ThreadPoolExecutor``;
collection uploads to Qdrant remain serial because we ``recreate_collection``
at the start of each.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
from time import sleep
from urllib.parse import urlparse
from uuid import uuid4
import warnings
from zipfile import ZipFile

from git import Repo
import httpx
from openai import OpenAI
from qdrant_client import QdrantClient, models
from tiktoken import encoding_for_model

from src.core.base_downloaders import BaseNewsDownloader
from src.core.qdrant_uploader import ensure_collection, existing_payload_values
from src.settings import load_settings
from src.utils.downloads import (
    CloneProgress,
    _call_with_retries,
    _download_config,
    _remove_readonly,
    log_progress,
)

logger = logging.getLogger(__name__)

SUPPORTED_ARTICLE_LANGUAGES = {"english", "en"}


class NewsDownloader(BaseNewsDownloader):
    """Clone a news-dataset repo, extract articles, embed, upload to Qdrant."""

    def __init__(
        self,
        env_file: str | Path,
        repo_url: str,
        save_path: str | Path,
        qdrant_host: str,
        qdrant_port: str,
        config_path: str | Path,
        openai_base_url: str | None = None,
        openai_embedding_model: str = "openai/text-embedding-3-small",
        openai_token_limit: int = 8192,
        openai_model_dimensions: int = 1536,
    ) -> None:
        """Capture configuration; OpenAI/Qdrant clients are built lazily.

        Args:
            env_file: Path to the ``.env`` with ``OPENAI_API_KEY`` and
                ``QDRANT__SERVICE__API_KEY``.
            repo_url: GitHub URL of the news dataset to clone.
            save_path: Local directory where the repo is cloned + unzipped.
            qdrant_host: Hostname (or full URL) of the Qdrant service.
            qdrant_port: Port (ignored when ``qdrant_host`` includes a scheme).
            config_path: Path to the JSON file listing allowed topics.
            openai_base_url: OpenAI-compatible base URL, or ``None`` for default.
            openai_embedding_model: Embedding model identifier.
            openai_token_limit: Max tokens per embedding call (input truncated).
            openai_model_dimensions: Embedding dimensionality (Qdrant param).
        """
        self.env_path = Path(env_file)
        self.github_api_url = "https://api.github.com"

        self.repo_url = repo_url
        self.save_path = Path(save_path)

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
        # Concurrent embedding batches per collection. The OpenAI SDK is
        # thread-safe; keep this conservative so we don't blow through the
        # tokens-per-minute quota in a burst.
        self.max_parallel_embed_batches = 4

    def _initialize_connections(self) -> bool:
        secrets = load_settings(self.env_path)
        openai_api_key = secrets.openai_api_key
        if not openai_api_key:
            logger.error("OPENAI_API_KEY is not set; news embeddings cannot be generated")
            return False
        self.openai_client = OpenAI(base_url=self.openai_base_url, api_key=openai_api_key)
        try:
            qdrant_api_key = secrets.qdrant_api_key or None
            response = _call_with_retries(
                operation_name="github_api_probe",
                request_callable=lambda: httpx.get(self.github_api_url, timeout=30.0),
                retry_delay_seconds=5.0,
                max_retries=5,
            )
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
        if response is None:
            logger.error("GitHub API probe failed after all retries; skipping news download")
            return False
        if response.status_code == 200 and bool(collections_response):
            parent_dir = self.save_path.resolve().parent
            parent_dir.mkdir(parents=True, exist_ok=True)
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

            truncated_text = self.embedding_encoding.decode(token_ids[: self.embedding_token_limit])
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
        self.save_path.mkdir(parents=True, exist_ok=True)

        for item_path in self.save_path.iterdir():
            if item_path.is_dir():
                shutil.rmtree(item_path, onexc=_remove_readonly)
            else:
                try:
                    item_path.unlink()
                except PermissionError:
                    item_path.chmod(0o700)
                    item_path.unlink()

        clone_result = _call_with_retries(
            "Cloning from github",
            lambda: Repo.clone_from(repo_url, str(self.save_path), progress=CloneProgress()),
            retry_delay_seconds=30,
            max_retries=3,
        )
        self.is_downloaded = clone_result is not None
        return self.is_downloaded

    @staticmethod
    def _collection_name_for(topic: str, sentiment: str) -> str:
        """Derive the Qdrant collection name from a topic + sentiment (news convention)."""
        topic_normalized = topic.strip().lower()
        return f"{topic_normalized}_{sentiment}".replace(" ", "_").replace(",", " ").lower()

    def _parse_archive_name(
        self, archive_path: Path, allowed_topics: list[str]
    ) -> tuple[str, str] | None:
        """Return ``(collection_name, archive_base_name)`` for an allowed zip, else ``None``.

        Filename-only (no unzip), so an incremental update can decide whether an
        archive is already ingested before doing any extraction work.
        """
        if archive_path.suffix != ".zip":
            return None
        base_name = archive_path.stem
        parts = base_name.rsplit("_", 2)
        if len(parts) != 3:
            return None
        topic, sentiment, _date_str = parts
        if topic not in allowed_topics:
            return None
        return self._collection_name_for(topic, sentiment), base_name

    def _filter_new_archives(self, archives: list[Path], allowed_topics: list[str]) -> list[Path]:
        """Drop archives whose ``archive_name`` is already present in its Qdrant collection."""
        parsed: dict[Path, tuple[str, str]] = {}
        for archive_path in archives:
            info = self._parse_archive_name(archive_path, allowed_topics)
            if info is not None:
                parsed[archive_path] = info

        existing_by_collection: dict[str, set[str]] = {}
        for collection, _base in parsed.values():
            if collection not in existing_by_collection:
                existing_by_collection[collection] = existing_payload_values(
                    self.qdrant_client, collection, "archive_name"
                )

        kept = [
            archive_path
            for archive_path, (collection, base_name) in parsed.items()
            if base_name not in existing_by_collection.get(collection, set())
        ]
        skipped = len(parsed) - len(kept)
        if skipped:
            logger.info("News incremental: skipping %d archive(s) already in Qdrant", skipped)
        return kept

    def _process_archive(
        self, archive_path: Path, allowed_topics: list[str]
    ) -> tuple[str, list[dict]] | None:
        """Extract one zip and collect its JSON article payloads.

        Returns ``(collection_name, entries)`` or ``None`` when the archive is
        skipped. Pure per-archive work — no shared state — so it is safe to
        call from a ``ThreadPoolExecutor`` worker.
        """
        if archive_path.suffix != ".zip":
            return None

        base_name = archive_path.stem
        parts = base_name.rsplit("_", 2)
        if len(parts) != 3:
            logger.warning(
                "Skipping news archive with unexpected name format: %s",
                archive_path.name,
            )
            return None
        topic, sentiment, date_str = parts
        if topic not in allowed_topics:
            return None

        parsed_date = datetime.strptime(date_str, "%Y%m%d%H%M%S").date().isoformat()

        zip_path = archive_path
        extract_dir = self.save_path / base_name
        collection_name = self._collection_name_for(topic, sentiment)

        extract_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        nested_extract_dir = extract_dir / base_name
        if nested_extract_dir.is_dir():
            for nested_item in nested_extract_dir.iterdir():
                shutil.move(
                    str(nested_item),
                    str(extract_dir / nested_item.name),
                )
            shutil.rmtree(nested_extract_dir, onexc=_remove_readonly)

        entries: list[dict] = []
        for article_file_path in sorted(extract_dir.rglob("*.json")):
            try:
                article_payload = json.loads(article_file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    f"Skipping invalid article file: {article_file_path}",
                    extra={
                        "operation": "Parsing article JSON",
                        "error": str(exc),
                    },
                )
                continue

            article_language = str(article_payload.get("language", "")).strip().lower()
            if article_language not in SUPPORTED_ARTICLE_LANGUAGES:
                logger.info(
                    "Skipping article with unsupported language metadata",
                    extra={
                        "operation": "Parsing article JSON",
                        "article_file_path": str(article_file_path),
                        "language": article_payload.get("language"),
                    },
                )
                continue

            entries.append(
                {
                    "topic": topic,
                    "sentiment": sentiment,
                    "date": parsed_date,
                    "archive_file": str(zip_path.resolve()),
                    "extracted_dir": str(extract_dir.resolve()),
                    "article_file_path": str(article_file_path.resolve()),
                    "archive_name": base_name,
                    "article": article_payload,
                }
            )

        return collection_name, entries

    def parse_repository(self, incremental: bool = False) -> None:
        metadata: dict[str, list[dict]] = {}
        source_datasets_dir = self.save_path / "News_Datasets"
        allowed_topics = self.download_config

        iter_files = [
            entry
            for entry in source_datasets_dir.iterdir()
            if entry.name.split("_")[0] in allowed_topics
        ]
        if incremental:
            # Skip archives already embedded so an update only unzips + embeds new
            # stories (dedup by archive_name against the existing Qdrant points).
            iter_files = self._filter_new_archives(iter_files, allowed_topics)

        # Per-archive work is I/O-heavy (zip extraction + lots of small JSON
        # reads); ThreadPoolExecutor gives a big win without needing processes.
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_archive, archive_path, allowed_topics): archive_path
                for archive_path in iter_files
            }
            for future in log_progress(
                as_completed(futures),
                label="Unzipping files",
                total=len(futures),
            ):
                archive_path = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Failed to process archive: %s", archive_path)
                    continue
                if result is None:
                    continue
                collection_name, entries = result
                metadata.setdefault(collection_name, []).extend(entries)

        self.parsed_metadata = metadata
        self.is_parsed = True

    def clean_repository(self) -> None:
        preserved_dirs = {
            str(Path(metadata_entry["extracted_dir"]).resolve())
            for metadata_entries in self.parsed_metadata.values()
            for metadata_entry in metadata_entries
        }

        for item_path in self.save_path.iterdir():
            if str(item_path.resolve()) in preserved_dirs:
                continue

            if item_path.is_dir():
                shutil.rmtree(item_path, onexc=_remove_readonly)
            else:
                try:
                    item_path.unlink()
                except PermissionError:
                    item_path.chmod(0o700)
                    item_path.unlink()

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

    def _embed_batch(
        self, collection_name: str, batch_start: int, batch_metadata: list[dict]
    ) -> list[models.PointStruct] | None:
        """Embed a single batch and build PointStructs. Returns None on failure."""
        texts_to_embed = [
            self._truncate_for_embedding(
                str(meta.get("article", {}).get("text")),
                meta.get("article_file_path", "unknown"),
            )
            for meta in batch_metadata
        ]

        embeddings = _call_with_retries(
            "get_embeddings",
            lambda: self.get_embeddings(texts_to_embed),
            retry_delay_seconds=3,
            max_retries=5,
        )

        if not embeddings:
            logger.warning(
                "Skipping batch because embeddings are empty",
                extra={
                    "operation": "Embedding and Uploading",
                    "collection": collection_name,
                    "batch_start": batch_start,
                    "batch_size": len(batch_metadata),
                },
            )
            return None

        if len(embeddings) != len(batch_metadata):
            logger.warning(
                "Skipping batch due to embedding count mismatch",
                extra={
                    "operation": "Embedding and Uploading",
                    "collection": collection_name,
                    "batch_start": batch_start,
                    "metadata_count": len(batch_metadata),
                    "embedding_count": len(embeddings),
                },
            )
            return None

        return [
            models.PointStruct(id=str(uuid4()), payload=meta, vector=vector)
            for meta, vector in zip(batch_metadata, embeddings)
        ]

    def _embed_and_upsert_collection(
        self, collection_name: str, metadata_entries: list[dict]
    ) -> None:
        """Embed ``metadata_entries`` in concurrent batches and upsert to one collection.

        Shared by :meth:`upload_to_qdrant` (recreate-first) and
        :meth:`upsert_to_qdrant` (ensure-first) so the batching path is defined once.
        """
        batch_starts = list(range(0, len(metadata_entries), self.batch_size))
        with ThreadPoolExecutor(max_workers=self.max_parallel_embed_batches) as executor:
            futures = {
                executor.submit(
                    self._embed_batch,
                    collection_name,
                    batch_start,
                    metadata_entries[batch_start : batch_start + self.batch_size],
                ): batch_start
                for batch_start in batch_starts
            }
            for future in log_progress(
                as_completed(futures),
                label=f"Embedding and Uploading: {collection_name}",
                total=len(futures),
            ):
                batch_start = futures[future]
                try:
                    points = future.result()
                except Exception:
                    logger.exception(
                        "Embedding batch failed (collection=%s, batch_start=%s)",
                        collection_name,
                        batch_start,
                    )
                    continue
                if points:
                    # qdrant_client is thread-safe; upserts can overlap freely.
                    self.qdrant_client.upsert(collection_name=collection_name, points=points)

        sleep(self.download_retry_delay_seconds)

    def upload_to_qdrant(self) -> None:
        """Recreate each collection then embed + upload its parsed entries (full reload)."""
        for collection_name, metadata_entries in self.parsed_metadata.items():
            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.openai_model_dimensions,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
            self._embed_and_upsert_collection(collection_name, metadata_entries)

    def upsert_to_qdrant(self) -> None:
        """Append parsed entries to each collection without dropping it (incremental)."""
        for collection_name, metadata_entries in self.parsed_metadata.items():
            if not metadata_entries:
                continue
            ensure_collection(self.qdrant_client, collection_name, self.openai_model_dimensions)
            self._embed_and_upsert_collection(collection_name, metadata_entries)

    def cleanup_downloaded_files(self) -> None:
        """Delete everything under ``save_path`` (clone + extracted dirs) after upload.

        Called at the end of :meth:`run` and :meth:`update` so the multi-GB news
        clone/unzip never lingers on the host volume once its articles live in
        Qdrant. Dedup for future incremental runs uses Qdrant as the source of
        truth, so no on-disk state needs to survive.
        """
        if not self.save_path.exists():
            return
        for item_path in self.save_path.iterdir():
            if item_path.is_dir():
                shutil.rmtree(item_path, onexc=_remove_readonly)
            else:
                try:
                    item_path.unlink()
                except PermissionError:
                    item_path.chmod(0o700)
                    item_path.unlink()
        logger.info("News: removed downloaded files under %s after Qdrant upload", self.save_path)

    def run(self) -> None:
        if not self.download_repository():
            return
        self.parse_repository()
        self.clean_repository()
        self.upload_to_qdrant()
        self.cleanup_downloaded_files()

    def update(self) -> None:
        """Incrementally ingest only news archives not already embedded, then clean up.

        Re-clones the news repo, parses only the archives whose ``archive_name`` is
        not yet in its Qdrant collection, upserts those (no ``recreate``), and
        finally removes the downloaded files (Feature 2).
        """
        if not self.download_repository():
            return
        self.parse_repository(incremental=True)
        if self.parsed_metadata:
            self.upsert_to_qdrant()
        else:
            logger.info("News incremental: no new archives to ingest")
        self.cleanup_downloaded_files()
