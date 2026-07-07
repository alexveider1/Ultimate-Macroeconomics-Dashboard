"""World Bank documents pipeline: search WDS, fetch text, chunk, embed, push.

Runs once per clean boot via :meth:`WorldBankArticlesDownloader.run`. For each
configured query-topic it fetches the top-N matching documents from the WDS API,
pulls each document's plain text from its ``txturl`` (no docling — see
:mod:`src.utils.wds_client`), and builds one entry per document. The overlapping
token-window chunking + embedding + upload all live in the shared
:class:`~src.core.qdrant_uploader.QdrantEmbeddingUploaderMixin` (same path every
news source uses), so a long document is split into multiple Qdrant points
rather than truncated. The payload shape matches the news pipeline so the app +
agent read it unchanged.

``documents.worldbank.org`` sits behind Cloudflare Bot Management, so this
downloader takes extra anti-bot care: modest concurrency, a randomised
pre-fetch delay per request, a per-query cookie warm-up, a re-warm on every
blocked (403/429) response before the retry backs off, generous retry budgets,
and a pause between query-topics. All of those are tunable from the download
config (see :meth:`__init__`).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import random
import time
from typing import Any

import httpx

from src.core.base_downloaders import BaseWorldBankArticlesDownloader
from src.core.qdrant_uploader import QdrantEmbeddingUploaderMixin
from src.settings import load_settings
from src.utils.downloads import _call_with_retries, _download_config, log_progress
from src.utils.wds_client import DEFAULT_BASE_URL, build_client, fetch_text, search, warm_up

logger = logging.getLogger(__name__)

# Anti-bot defaults (overridable from the download config). documents.worldbank.org
# is Cloudflare-fronted and 403s bursty bot-like traffic, so the parallelism is
# modest, retries are generous, and requests are jittered + paced.
_DEFAULT_MAX_PARALLEL_TEXT_FETCHES = 3
_DEFAULT_DOWNLOAD_MAX_RETRIES = 6
_DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS = 5.0
_DEFAULT_INTER_QUERY_PAUSE_SECONDS = 5.0
_DEFAULT_FETCH_JITTER_MIN_SECONDS = 0.5
_DEFAULT_FETCH_JITTER_MAX_SECONDS = 2.0

# HTTP statuses that mean "Cloudflare blocked this as a bot" — worth re-warming
# the __cf_bm cookie before the retry wrapper backs off and tries again.
_ANTIBOT_STATUS_CODES = frozenset({403, 429})


def build_doc_entry(doc: dict[str, Any], text: str, query: str, collection: str) -> dict[str, Any]:
    """Build one Qdrant entry payload for a document (news payload shape).

    The full document text is stored on the entry; the shared uploader splits it
    into overlapping token-window chunks (one Qdrant point each) and adds the
    ``chunk_index`` / ``total_chunks`` / ``"(part i/N)"`` title metadata.
    """
    display_title = str(doc.get("display_title") or doc.get("id") or "Untitled")
    docdt = str(doc.get("docdt") or "")
    docty = doc.get("docty") or ""
    country = doc.get("count") or ""

    return {
        "topic": query,
        "sentiment": "neutral",
        "date": docdt[:10],
        "source": "world_bank",
        "archive_name": collection,
        "article": {
            "title": display_title,
            "text": text,
            "url": doc.get("url") or "",
            "pdfurl": doc.get("pdfurl") or "",
            "published": docdt,
            "author": "",
            "language": doc.get("lang") or "",
            "categories": [docty, country],
            "thread": {"site": "World Bank", "site_section": docty, "country": country},
            "doc_id": doc.get("id") or "",
            "doc_title": display_title,
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
        chunk_size_tokens: int = 800,
        chunk_overlap_tokens: int = 100,
    ) -> None:
        """Capture config; OpenAI/Qdrant/HTTP clients are built in ``_initialize_connections``."""
        self.env_path = Path(env_file)

        config = _download_config(download_config_path)
        self.base_url = str(config.get("base_url", DEFAULT_BASE_URL))
        self.rows_per_query = int(config.get("rows_per_query", 100))
        self.lang: str | None = config.get("lang") or None
        self.from_year: int | None = config.get("from_year")
        self.doc_types: list[str] = list(config.get("doc_types") or [])
        self.queries: list[dict[str, str]] = [
            {"query": str(item["query"]), "collection": str(item["collection"])}
            for item in config.get("queries", [])
            if item.get("query") and item.get("collection")
        ]

        # Anti-bot / resilience knobs (bigger pauses + more retries than the other
        # sources because of the Cloudflare bot manager on the document host).
        self.max_parallel_text_fetches = int(
            config.get("max_parallel_text_fetches", _DEFAULT_MAX_PARALLEL_TEXT_FETCHES)
        )
        self.download_max_retries = int(
            config.get("download_max_retries", _DEFAULT_DOWNLOAD_MAX_RETRIES)
        )
        self.download_retry_delay_seconds = float(
            config.get("download_retry_delay_seconds", _DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS)
        )
        self.inter_query_pause_seconds = float(
            config.get("inter_query_pause_seconds", _DEFAULT_INTER_QUERY_PAUSE_SECONDS)
        )
        self.fetch_jitter_min_seconds = float(
            config.get("fetch_jitter_min_seconds", _DEFAULT_FETCH_JITTER_MIN_SECONDS)
        )
        self.fetch_jitter_max_seconds = float(
            config.get("fetch_jitter_max_seconds", _DEFAULT_FETCH_JITTER_MAX_SECONDS)
        )

        self.qdrant_host = qdrant_host
        self.qdrant_port = str(qdrant_port)
        self.openai_base_url = openai_base_url
        self.openai_embedding_model = openai_embedding_model
        self.embedding_token_limit = openai_token_limit
        self.openai_model_dimensions = openai_model_dimensions
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
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
            retry_delay_seconds=self.download_retry_delay_seconds,
            max_retries=self.download_max_retries,
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

        A Cloudflare block (``403``/``429``) raises inside ``fetch_text``; before
        re-raising so the retry wrapper backs off, the ``__cf_bm`` cookie is
        re-warmed so the next attempt looks like a returning browser. A small
        random pre-fetch delay desynchronises the parallel workers and keeps the
        aggregate request rate under Cloudflare's bot threshold (the same jitter
        rationale as the WB indicator client).
        """
        time.sleep(random.uniform(self.fetch_jitter_min_seconds, self.fetch_jitter_max_seconds))
        txturl = str(doc["txturl"])

        def _attempt() -> str:
            try:
                return fetch_text(self._require_client(), txturl)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in _ANTIBOT_STATUS_CODES:
                    logger.warning(
                        "WDS text fetch blocked (%s) for %s; re-warming bot cookie",
                        exc.response.status_code,
                        doc.get("id"),
                    )
                    warm_up(self._require_client(), txturl)
                raise

        return (
            _call_with_retries(
                f"wds.text({doc.get('id')})",
                _attempt,
                retry_delay_seconds=self.download_retry_delay_seconds,
                max_retries=self.download_max_retries,
            )
            or ""
        )

    def _build_entries_for_docs(
        self, docs: list[dict[str, Any]], query: str, collection: str
    ) -> list[dict[str, Any]]:
        """Fetch each doc's text concurrently and build one entry per document."""
        entries: list[dict[str, Any]] = []
        # Prime Cloudflare's __cf_bm cookie before the parallel burst so the
        # concurrent fetches reuse it instead of each racing a cookie-less 403.
        if docs:
            warm_up(self._require_client(), str(docs[0]["txturl"]))
        with ThreadPoolExecutor(max_workers=self.max_parallel_text_fetches) as executor:
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
                entries.append(build_doc_entry(doc, text, query, collection))
        return entries

    def _search_query(self, query: str) -> list[dict[str, Any]] | None:
        """Run one WDS search with the shared retry budget."""
        return _call_with_retries(
            f"wds.search({query})",
            lambda: search(
                self._require_client(),
                qterm=query,
                rows=self.rows_per_query,
                base_url=self.base_url,
                doc_types=self.doc_types or None,
                lang=self.lang,
                from_year=self.from_year,
            ),
            retry_delay_seconds=self.download_retry_delay_seconds,
            max_retries=self.download_max_retries,
        )

    def _pace_between_queries(self, index: int) -> None:
        """Pause before every query after the first to stay under the bot threshold."""
        if index > 0 and self.inter_query_pause_seconds > 0:
            time.sleep(self.inter_query_pause_seconds)

    def run(self) -> None:
        parsed: dict[str, list[dict[str, Any]]] = {}
        for index, item in enumerate(
            log_progress(self.queries, label="World Bank query-topics", total=len(self.queries))
        ):
            self._pace_between_queries(index)
            query = item["query"]
            collection = item["collection"]
            records = self._search_query(query)
            if not records:
                logger.warning("No WDS documents for query %r", query)
                parsed[collection] = []
                continue

            docs = dedup_with_text(records)
            entries = self._build_entries_for_docs(docs, query, collection)
            parsed[collection] = entries
            logger.info("World Bank %s: %d docs -> %d entries", collection, len(docs), len(entries))

        self.upload_collections(parsed)
        if self._client is not None:
            self._client.close()

    def update(self) -> None:
        """Incrementally ingest only documents not already embedded (dedup by doc id).

        Per query re-runs the WDS search, keeps only documents whose ``id`` is not
        already in the collection (``article.doc_id``), fetches only those, and
        upserts them (no ``recreate``). The shared uploader chunks each new
        document into its Qdrant points.
        """
        parsed: dict[str, list[dict[str, Any]]] = {}
        for index, item in enumerate(
            log_progress(
                self.queries, label="World Bank query-topics (update)", total=len(self.queries)
            )
        ):
            self._pace_between_queries(index)
            query = item["query"]
            collection = item["collection"]
            records = self._search_query(query)
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
                "World Bank %s incremental: %d new docs -> %d entries",
                collection,
                len(new_docs),
                len(entries),
            )

        self.upsert_collections(parsed)
        if self._client is not None:
            self._client.close()
