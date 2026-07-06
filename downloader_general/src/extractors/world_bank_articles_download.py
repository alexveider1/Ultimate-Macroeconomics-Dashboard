"""World Bank documents pipeline: search WDS, fetch text, chunk, embed, push.

Runs once per clean boot via :meth:`WorldBankArticlesDownloader.run`. For each
configured query-topic it fetches the top-N matching documents from the WDS API,
pulls each document's plain text from its ``txturl`` (no docling — see
:mod:`src.utils.wds_client`), splits the text into overlapping token windows, and
uploads one Qdrant point per chunk into a per-topic collection
(``world_bank_<slug>``).

Embedding + upload reuse :class:`~src.core.qdrant_uploader.QdrantEmbeddingUploaderMixin`;
the payload shape matches the news pipeline so the app + agent read it unchanged.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import random
import re
import time
from typing import Any, Protocol

import httpx

from src.core.base_downloaders import BaseWorldBankArticlesDownloader
from src.core.qdrant_uploader import QdrantEmbeddingUploaderMixin
from src.settings import load_settings
from src.utils.downloads import _call_with_retries, _download_config, log_progress
from src.utils.wds_client import DEFAULT_BASE_URL, build_client, fetch_text, search, warm_up

logger = logging.getLogger(__name__)

# documents.worldbank.org is Cloudflare-fronted and 403s bursty bot-like traffic
# (see src.utils.wds_client); keep the parallelism modest and warm the bot cookie
# once per query so the burst reuses it.
_MAX_PARALLEL_TEXT_FETCHES = 4


class TokenEncoding(Protocol):
    """Minimal token-codec surface used by :func:`chunk_text` (tiktoken-shaped)."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


def clean_text(text: str) -> str:
    """Normalise WB plain text: drop form-feeds and collapse blank-line runs."""
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, encoding: TokenEncoding, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping token windows of ``chunk_size`` tokens.

    Windows advance by ``chunk_size - overlap`` tokens. A text within one window
    is returned whole. Each returned chunk decodes back to text.
    """
    token_ids = encoding.encode(text)
    if not token_ids:
        return []
    if len(token_ids) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_size]
        if not window:
            break
        chunks.append(encoding.decode(window))
        if start + chunk_size >= len(token_ids):
            break
    return chunks


def build_chunk_entry(
    doc: dict[str, Any],
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    query: str,
    collection: str,
) -> dict[str, Any]:
    """Build one Qdrant entry payload for a document chunk (news payload shape)."""
    display_title = str(doc.get("display_title") or doc.get("id") or "Untitled")
    docdt = str(doc.get("docdt") or "")
    docty = doc.get("docty") or ""
    country = doc.get("count") or ""
    title = (
        f"{display_title} (part {chunk_index + 1}/{total_chunks})"
        if total_chunks > 1
        else display_title
    )

    return {
        "topic": query,
        "sentiment": "neutral",
        "date": docdt[:10],
        "source": "world_bank",
        "archive_name": collection,
        "article": {
            "title": title,
            "text": chunk,
            "url": doc.get("url") or "",
            "pdfurl": doc.get("pdfurl") or "",
            "published": docdt,
            "author": "",
            "language": doc.get("lang") or "",
            "categories": [docty, country],
            "thread": {"site": "World Bank", "site_section": docty, "country": country},
            "doc_id": doc.get("id") or "",
            "doc_title": display_title,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "docty": docty,
        },
    }


def dedup_with_text(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep records that have a ``txturl``, deduplicated by document id."""
    seen: set[str] = set()
    docs: list[dict[str, Any]] = []
    for record in records:
        doc_id = record.get("id")
        if not doc_id or not record.get("txturl") or doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(record)
    return docs


class WorldBankArticlesDownloader(QdrantEmbeddingUploaderMixin, BaseWorldBankArticlesDownloader):
    """Per query fetch top-N WB docs, chunk their text, embed, upload to Qdrant."""

    def __init__(
        self,
        env_file: str | Path,
        download_config_path: str | Path,
        qdrant_host: str,
        qdrant_port: str,
        openai_base_url: str | None = None,
        openai_embedding_model: str = "openai/text-embedding-3-small",
        openai_token_limit: int = 8192,
        openai_model_dimensions: int = 1536,
    ) -> None:
        """Capture config; OpenAI/Qdrant/HTTP clients are built in ``_initialize_connections``."""
        self.env_path = Path(env_file)

        config = _download_config(download_config_path)
        self.base_url = str(config.get("base_url", DEFAULT_BASE_URL))
        self.rows_per_query = int(config.get("rows_per_query", 100))
        self.lang: str | None = config.get("lang") or None
        self.from_year: int | None = config.get("from_year")
        self.doc_types: list[str] = list(config.get("doc_types") or [])
        self.chunk_size_tokens = int(config.get("chunk_size_tokens", 800))
        self.chunk_overlap_tokens = int(config.get("chunk_overlap_tokens", 100))
        self.queries: list[dict[str, str]] = [
            {"query": str(item["query"]), "collection": str(item["collection"])}
            for item in config.get("queries", [])
            if item.get("query") and item.get("collection")
        ]

        self.qdrant_host = qdrant_host
        self.qdrant_port = str(qdrant_port)
        self.openai_base_url = openai_base_url
        self.openai_embedding_model = openai_embedding_model
        self.embedding_token_limit = openai_token_limit
        self.openai_model_dimensions = openai_model_dimensions
        self.embedding_encoding = self._build_embedding_encoding()

        self._client: httpx.Client | None = None

    def _initialize_connections(self) -> bool:
        secrets = load_settings(self.env_path)
        if not self._connect_embedding_and_qdrant(secrets):
            return False

        self._client = build_client()
        probe = _call_with_retries(
            "wds.probe",
            lambda: search(
                self._require_client(),
                qterm="inflation",
                rows=1,
                base_url=self.base_url,
                lang=self.lang,
            ),
            retry_delay_seconds=5,
            max_retries=5,
        )
        if probe is None:
            logger.error("World Bank WDS API probe failed; skipping source")
            return False
        return True

    def _require_client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("HTTP client not initialised; call _initialize_connections first")
        return self._client

    def _fetch_doc_text(self, doc: dict[str, Any]) -> str:
        """Fetch + retry one document's ``txturl`` plain text (empty on failure).

        A Cloudflare ``403`` raises inside ``fetch_text`` and is retried with
        backoff here; blocks are transient so an extra attempt usually clears it.
        A small random pre-fetch delay desynchronises the parallel workers and
        keeps the aggregate request rate under Cloudflare's bot threshold (the
        same jitter rationale as the WB indicator client).
        """
        time.sleep(random.uniform(0.1, 0.6))
        return (
            _call_with_retries(
                f"wds.text({doc.get('id')})",
                lambda: fetch_text(self._require_client(), str(doc["txturl"])),
                retry_delay_seconds=3,
                max_retries=4,
            )
            or ""
        )

    def _build_entries_for_docs(
        self, docs: list[dict[str, Any]], query: str, collection: str
    ) -> list[dict[str, Any]]:
        """Fetch each doc's text concurrently, chunk it, and build entries."""
        entries: list[dict[str, Any]] = []
        # Prime Cloudflare's __cf_bm cookie before the parallel burst so the
        # concurrent fetches reuse it instead of each racing a cookie-less 403.
        if docs:
            warm_up(self._require_client(), str(docs[0]["txturl"]))
        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_TEXT_FETCHES) as executor:
            futures = {executor.submit(self._fetch_doc_text, doc): doc for doc in docs}
            for future in log_progress(
                as_completed(futures),
                label=f"Fetching WB texts: {collection}",
                total=len(futures),
            ):
                doc = futures[future]
                try:
                    text = future.result()
                except Exception:
                    logger.exception("Failed to fetch txturl for doc %s", doc.get("id"))
                    continue
                if not text:
                    continue
                chunks = chunk_text(
                    clean_text(text),
                    self.embedding_encoding,
                    self.chunk_size_tokens,
                    self.chunk_overlap_tokens,
                )
                total = len(chunks)
                for index, chunk in enumerate(chunks):
                    entries.append(build_chunk_entry(doc, chunk, index, total, query, collection))
        return entries

    def run(self) -> None:
        parsed: dict[str, list[dict[str, Any]]] = {}
        for item in log_progress(
            self.queries, label="World Bank query-topics", total=len(self.queries)
        ):
            query = item["query"]
            collection = item["collection"]
            records = _call_with_retries(
                f"wds.search({query})",
                lambda q=query: search(
                    self._require_client(),
                    qterm=q,
                    rows=self.rows_per_query,
                    base_url=self.base_url,
                    doc_types=self.doc_types or None,
                    lang=self.lang,
                    from_year=self.from_year,
                ),
                retry_delay_seconds=5,
                max_retries=5,
            )
            if not records:
                logger.warning("No WDS documents for query %r", query)
                parsed[collection] = []
                continue

            docs = dedup_with_text(records)
            entries = self._build_entries_for_docs(docs, query, collection)
            parsed[collection] = entries
            logger.info("World Bank %s: %d docs -> %d chunks", collection, len(docs), len(entries))

        self.upload_collections(parsed)
        if self._client is not None:
            self._client.close()

    def update(self) -> None:
        """Incrementally ingest only documents not already embedded (dedup by doc id).

        Per query re-runs the WDS search, keeps only documents whose ``id`` is not
        already in the collection (``article.doc_id``), fetches + chunks only those,
        and upserts the new chunks (no ``recreate``).
        """
        parsed: dict[str, list[dict[str, Any]]] = {}
        for item in log_progress(
            self.queries, label="World Bank query-topics (update)", total=len(self.queries)
        ):
            query = item["query"]
            collection = item["collection"]
            records = _call_with_retries(
                f"wds.search({query})",
                lambda q=query: search(
                    self._require_client(),
                    qterm=q,
                    rows=self.rows_per_query,
                    base_url=self.base_url,
                    doc_types=self.doc_types or None,
                    lang=self.lang,
                    from_year=self.from_year,
                ),
                retry_delay_seconds=5,
                max_retries=5,
            )
            if not records:
                logger.warning("No WDS documents for query %r", query)
                parsed[collection] = []
                continue

            docs = dedup_with_text(records)
            existing_doc_ids = self._existing_payload_values(collection, "article.doc_id")
            new_docs = [doc for doc in docs if str(doc.get("id")) not in existing_doc_ids]
            entries = self._build_entries_for_docs(new_docs, query, collection)
            parsed[collection] = entries
            logger.info(
                "World Bank %s incremental: %d new docs -> %d chunks",
                collection,
                len(new_docs),
                len(entries),
            )

        self.upsert_collections(parsed)
        if self._client is not None:
            self._client.close()
