"""Shared OpenAI-embedding + Qdrant-upload machinery for RAG downloaders.

Extracted from :class:`~src.extractors.github_download.NewsDownloader` so the
newer Actually-Relevant and World-Bank-articles downloaders reuse the exact same
embedding + upsert path (same model, dimensions, batching, retry policy and
``recreate_collection`` semantics) rather than duplicating it. A downloader mixes
this in and only has to build a ``parsed_metadata`` dict mapping each Qdrant
collection name to a list of *entry* payloads whose embed text lives at
``entry["article"]["text"]`` — then calls :meth:`upload_collections`.

The payload shape is deliberately identical to the news pipeline's (a nested
``article`` dict plus top-level ``topic``/``sentiment``/``date``) so the app news
page and the agent RAG worker read the new collections with no changes.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import sleep
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
import warnings

from openai import OpenAI
from qdrant_client import QdrantClient, models
from tiktoken import Encoding, encoding_for_model

from src.core import tracing
from src.settings import Settings
from src.utils.downloads import _call_with_retries, log_progress

logger = logging.getLogger(__name__)


def _walk_payload(payload: dict | None, field_path: str) -> Any:
    """Descend a (possibly nested) payload by a dotted ``field_path``.

    ``"archive_name"`` reads a top-level key; ``"article.id"`` descends into the
    nested ``article`` dict. Returns ``None`` if any segment is missing.
    """
    node: Any = payload or {}
    for segment in field_path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
    return node


def existing_payload_values(client: QdrantClient, collection: str, field_path: str) -> set[str]:
    """Return the set of distinct payload values at ``field_path`` in a collection.

    Scrolls the whole collection (payload-only, no vectors) and collects the value
    at ``field_path`` from every point — the dedup key an incremental update uses
    to skip documents already ingested. Returns an empty set when the collection
    does not exist yet.

    Args:
        client: Connected Qdrant client.
        collection: Collection name to scan.
        field_path: Dotted payload path of the dedup key (e.g. ``"article.id"``).
    """
    if not client.collection_exists(collection):
        return set()

    top_key = field_path.split(".")[0]
    values: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            with_payload=[top_key],
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            value = _walk_payload(point.payload, field_path)
            if value not in (None, ""):
                values.add(str(value))
        if offset is None:
            break
    return values


def ensure_collection(client: QdrantClient, collection: str, dimensions: int) -> None:
    """Create ``collection`` with the cosine ``dimensions`` space only if absent.

    Unlike ``recreate_collection`` this never drops an existing collection, so an
    incremental update appends to it rather than wiping the prior points.
    """
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
            on_disk_payload=True,
        )


class QdrantEmbeddingUploaderMixin:
    """Reusable embed-and-upsert behaviour for Qdrant-backed RAG downloaders.

    Subclasses must set the following instance attributes (typically in
    ``__init__``) before calling any method here:

    - ``openai_base_url``, ``openai_embedding_model``, ``embedding_token_limit``,
      ``openai_model_dimensions`` — embedding config (mirrors ``config.yaml``).
    - ``qdrant_host``, ``qdrant_port`` — Qdrant connection (host may be a full URL).
    - ``embedding_encoding`` — result of :meth:`_build_embedding_encoding`.

    ``batch_size``, ``max_parallel_embed_batches`` and
    ``download_retry_delay_seconds`` have sensible class-level defaults.
    """

    openai_base_url: str | None
    openai_embedding_model: str
    embedding_token_limit: int
    openai_model_dimensions: int
    qdrant_host: str
    qdrant_port: str
    embedding_encoding: Encoding
    openai_client: OpenAI
    qdrant_client: QdrantClient

    batch_size: int = 100
    # Concurrent embedding batches per collection. The OpenAI SDK is thread-safe;
    # keep this conservative so we don't blow through the tokens-per-minute quota.
    max_parallel_embed_batches: int = 4
    download_retry_delay_seconds: int = 5

    def _build_embedding_encoding(self) -> Encoding:
        """Build the ``tiktoken`` encoding for the configured embedding model."""
        model_name = str(self.openai_embedding_model).split("/")[-1]
        return encoding_for_model(model_name)

    def _truncate_for_embedding(self, text: str, article_path: str) -> str:
        """Truncate ``text`` to the embedding token budget (char fallback)."""
        if self.embedding_encoding is not None:
            token_ids = self.embedding_encoding.encode(text)
            if len(token_ids) <= self.embedding_token_limit:
                return text
            truncated_text = self.embedding_encoding.decode(token_ids[: self.embedding_token_limit])
            logger.warning(
                "Text truncated for embeddings token limit",
                extra={
                    "operation": "Embedding and Uploading",
                    "article_file_path": article_path,
                    "original_token_count": len(token_ids),
                    "truncated_token_count": self.embedding_token_limit,
                    "embedding_model": self.openai_embedding_model,
                },
            )
            return truncated_text

        max_chars = self.embedding_token_limit * 4
        return text if len(text) <= max_chars else text[:max_chars]

    def _connect_embedding_and_qdrant(self, secrets: Settings) -> bool:
        """Build the OpenAI + Qdrant clients and probe Qdrant.

        Returns ``True`` when both clients are ready (OpenAI key present, Qdrant
        reachable). The subclass ``_initialize_connections`` calls this and then
        probes its own upstream API separately.
        """
        openai_api_key = secrets.openai_api_key
        if not openai_api_key:
            logger.error("OPENAI_API_KEY is not set; embeddings cannot be generated")
            return False
        client_cls = tracing.openai_client_class()
        self.openai_client = client_cls(base_url=self.openai_base_url, api_key=openai_api_key)

        qdrant_api_key = secrets.qdrant_api_key or None
        if not qdrant_api_key:
            logger.error("Missing Qdrant API key. Set QDRANT__SERVICE__API_KEY in env file.")
            return False

        qdrant_host = str(self.qdrant_host).strip()
        parsed_host = urlparse(qdrant_host)
        if parsed_host.scheme:
            qdrant_url = qdrant_host
        else:
            qdrant_url = f"http://{qdrant_host}:{int(self.qdrant_port)}"

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    prefer_grpc=False,
                )
            self.qdrant_client.get_collections()
        except Exception as exc:
            logger.exception(
                "Failed to initialize Qdrant connection",
                extra={"operation": "Initializing connections", "error": str(exc)},
            )
            return False
        return True

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with the configured model, preserving input order."""
        try:
            response = self.openai_client.embeddings.create(
                input=texts, model=self.openai_embedding_model
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as exc:
            logger.exception("Getting embeddings failed", exc_info=exc)
            return []

    def _embed_batch(
        self, collection_name: str, batch_start: int, batch_metadata: list[dict]
    ) -> list[models.PointStruct] | None:
        """Embed one batch's ``article.text`` values and build PointStructs.

        Returns ``None`` when the batch should be skipped (empty embeddings or a
        count mismatch), mirroring the news pipeline's defensive behaviour.
        """
        texts_to_embed = [
            self._truncate_for_embedding(
                str(meta.get("article", {}).get("text")),
                meta.get("article_file_path", collection_name),
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
                extra={"collection": collection_name, "batch_start": batch_start},
            )
            return None

        if len(embeddings) != len(batch_metadata):
            logger.warning(
                "Skipping batch due to embedding count mismatch",
                extra={
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

    def _existing_payload_values(self, collection: str, field_path: str) -> set[str]:
        """Distinct dedup-key values already in ``collection`` (see module helper)."""
        return existing_payload_values(self.qdrant_client, collection, field_path)

    def ensure_collection(self, collection: str) -> None:
        """Create ``collection`` (cosine, ``openai_model_dimensions``) if it's absent."""
        ensure_collection(self.qdrant_client, collection, self.openai_model_dimensions)

    def _embed_and_upsert(self, collection_name: str, metadata_entries: list[dict]) -> None:
        """Embed ``metadata_entries`` in concurrent batches and upsert them.

        Shared by :meth:`upload_collections` (which recreates first) and
        :meth:`upsert_collections` (which only ensures the collection exists), so
        the batching / retry / progress-logging path is defined once.
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
                    self.qdrant_client.upsert(collection_name=collection_name, points=points)

        sleep(self.download_retry_delay_seconds)

    def upload_collections(self, parsed_metadata: dict[str, list[dict]]) -> None:
        """Recreate each collection then embed + upsert its entries in batches.

        ``parsed_metadata`` maps a Qdrant collection name to a list of entry
        payloads (each with its embed text at ``entry["article"]["text"]``). Each
        collection is dropped + recreated (idempotent full reload) with a
        cosine, ``openai_model_dimensions``-sized vector space; batches are
        embedded concurrently and upserted (the Qdrant client is thread-safe).
        """
        for collection_name, metadata_entries in parsed_metadata.items():
            if not metadata_entries:
                logger.info("No entries for collection %s; skipping upload", collection_name)
                continue

            self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.openai_model_dimensions,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
            self._embed_and_upsert(collection_name, metadata_entries)

    def upsert_collections(self, parsed_metadata: dict[str, list[dict]]) -> None:
        """Append new entries to each collection **without** dropping it.

        The incremental counterpart of :meth:`upload_collections`: each collection
        is created only if absent (:meth:`ensure_collection`) and the entries —
        which the caller has already filtered down to documents not yet present —
        are embedded and upserted. Collections with no new entries are skipped.
        """
        for collection_name, metadata_entries in parsed_metadata.items():
            if not metadata_entries:
                logger.info("No new entries for collection %s; skipping", collection_name)
                continue
            self.ensure_collection(collection_name)
            self._embed_and_upsert(collection_name, metadata_entries)
