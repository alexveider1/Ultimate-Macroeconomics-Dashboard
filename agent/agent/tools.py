"""Tool implementations invoked by the LangGraph workers.

Each LangGraph worker in :mod:`agent.graph` calls one or more of the
async helpers in this module: SQL execution, Qdrant news search, sandbox
code execution, DuckDuckGo fallback search, and the on-demand WB indicator
ingestion endpoint. Shared client instances (Postgres engine, Qdrant,
OpenAI) are lazily built and cached in the module-level ``_runtime`` dict.
"""

import asyncio
import base64
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ddgs import DDGS
import httpx
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
import yaml

logger = logging.getLogger(__name__)


_runtime: Dict[str, Any] = {}

MAX_SQL_ROWS = 500

# Connection-pooled HTTP client shared by every backend call (sandbox +
# downloader_extra). Cuts TLS / TCP handshake from each request and lets httpx
# reuse keep-alive sockets across the agent's lifetime.
_HTTPX_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)


def configure_runtime(
    *,
    database_schema_path: Path,
    news_topics_path: Path,
    qdrant_url: str,
    qdrant_api_key: str,
    postgres_database_uri: str,
    python_sandbox_base_url: str,
    downloader_extra_base_url: str,
    openai_api_key: str,
    openai_base_url: str,
    openai_embedding_model: str,
) -> None:
    """Populate the module-level runtime dict with every external dependency.

    Called once at FastAPI startup; subsequent calls overwrite values. Keeps
    the worker code free of global state imports — every worker pulls its
    dependencies from this dict via the lazy getters below.

    Args:
        database_schema_path: Path to ``database_schema.yaml``.
        news_topics_path: Path to the JSON list of allowed news topics.
        qdrant_url: Full ``http(s)://host:port`` URL for Qdrant.
        qdrant_api_key: Bearer token for Qdrant.
        postgres_database_uri: SQLAlchemy URI for the read-only LLM role.
        python_sandbox_base_url: Base URL of the sandbox FastAPI service.
        downloader_extra_base_url: Base URL of the on-demand WB ingester.
        openai_api_key: OpenAI-compatible API key.
        openai_base_url: OpenAI-compatible base URL.
        openai_embedding_model: Embedding model identifier.
    """
    _runtime["qdrant_url"] = qdrant_url
    _runtime["qdrant_api_key"] = qdrant_api_key
    _runtime["postgres_database_uri"] = postgres_database_uri
    _runtime["python_sandbox_base_url"] = python_sandbox_base_url
    _runtime["downloader_extra_base_url"] = downloader_extra_base_url
    _runtime["openai_api_key"] = openai_api_key
    _runtime["openai_base_url"] = openai_base_url
    _runtime["openai_embedding_model"] = openai_embedding_model

    _runtime["database_schema"] = yaml.safe_load(
        Path(database_schema_path).read_text(encoding="utf-8")
    )
    _runtime["news_topics"] = json.loads(Path(news_topics_path).read_text(encoding="utf-8"))

    _runtime["_engine"] = None
    _runtime["_qdrant_client"] = None
    _runtime["_openai_async_client"] = None
    _runtime["_httpx_client"] = None
    # Invalidate the cached schema-text rendering so a YAML reload at runtime
    # takes effect on the next worker call.
    get_database_schema_text.cache_clear()


def _get_engine():
    """Return the cached SQLAlchemy engine, building it on first use."""
    if _runtime.get("_engine") is None:
        _runtime["_engine"] = create_engine(
            _runtime["postgres_database_uri"],
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _runtime["_engine"]


def _get_qdrant_client() -> QdrantClient:
    """Return the cached Qdrant HTTP client, building it on first use."""
    if _runtime.get("_qdrant_client") is None:
        _runtime["_qdrant_client"] = QdrantClient(
            url=_runtime["qdrant_url"],
            api_key=_runtime.get("qdrant_api_key"),
            prefer_grpc=False,
        )
    return _runtime["_qdrant_client"]


def _get_openai_async_client() -> AsyncOpenAI:
    """Return the cached async OpenAI client, building it on first use.

    Kept as the plain client on purpose: the Langfuse ``openai`` integration
    patches the ``openai`` module *globally*, which would double-trace every
    LangGraph ``ChatOpenAI`` call (already captured by the graph-level Langfuse
    callback). The RAG query embedding is negligible cost and is covered by the
    ``rag_agent`` node span in the trace.
    """
    if _runtime.get("_openai_async_client") is None:
        _runtime["_openai_async_client"] = AsyncOpenAI(
            api_key=_runtime["openai_api_key"],
            base_url=_runtime["openai_base_url"],
        )
    return _runtime["_openai_async_client"]


def _get_httpx_client() -> httpx.AsyncClient:
    """Return the cached connection-pooled async HTTP client.

    A single ``httpx.AsyncClient`` is reused for every call to the
    ``python_sandbox`` and ``downloader_extra`` services. This avoids the
    per-request TCP/TLS handshake that ``async with httpx.AsyncClient()``
    incurred previously and lets ``httpx`` recycle keep-alive sockets.
    """
    if _runtime.get("_httpx_client") is None:
        _runtime["_httpx_client"] = httpx.AsyncClient(limits=_HTTPX_LIMITS)
    return _runtime["_httpx_client"]


async def aclose_runtime_clients() -> None:
    """Close the cached httpx client. Called from the FastAPI shutdown hook."""
    client = _runtime.get("_httpx_client")
    if client is not None:
        await client.aclose()
        _runtime["_httpx_client"] = None


@lru_cache(maxsize=1)
def get_database_schema_text() -> str:
    """Render the YAML schema as a human-readable text block.

    Used by the ``sql_agent`` system prompt so the LLM can ground table /
    column references without us shipping the raw YAML.

    Returns:
        Multi-line text describing every database / table / column / FK.
    """
    schema = _runtime.get("database_schema", {})
    lines: list[str] = []
    for db_name, tables in schema.get("databases", {}).items():
        lines.append(f"--- Database: {db_name} ---")
        for table_name, table_def in tables.items():
            lines.append(f"\nTable: {table_name}")

            pk_cols = table_def.get("primary_key") or []
            if pk_cols:
                lines.append(f"  Primary key: ({', '.join(pk_cols)})")

            fks = table_def.get("foreign_keys") or []
            for fk in fks:
                cols = ", ".join(fk.get("columns", []))
                ref_table = fk.get("references_table", "")
                ref_cols = ", ".join(fk.get("references_columns", []))
                enforced = "enforced" if fk.get("enforce", True) else "documentation only"
                lines.append(f"  Foreign key: ({cols}) -> {ref_table}({ref_cols}) [{enforced}]")

            lines.append("  Columns:")
            for col_name, col_info in (table_def.get("columns") or {}).items():
                col_type = col_info.get("type", "UNKNOWN")
                col_desc = col_info.get("description", "")
                lines.append(f"    - {col_name} ({col_type}): {col_desc}")
    return "\n".join(lines)


def get_news_topics() -> List[str]:
    """Return the list of allowed RAG topics loaded from JSON config."""
    return _runtime.get("news_topics", [])


def _sync_run_sql_query(sql_query: str) -> Dict[str, Any]:
    """Run one read-only SQL query and return rows + columns + truncation flag.

    Refuses anything that isn't a ``SELECT`` so the LLM can't accidentally
    issue DDL through the read-only role. Result rows are capped at
    ``MAX_SQL_ROWS`` with a ``truncated`` flag so the agent can decide
    whether to re-query with tighter filters.

    Args:
        sql_query: The SQL produced by the ``sql_agent`` worker.

    Returns:
        Dict with ``columns``, ``rows``, ``row_count``, ``truncated`` and
        ``query`` keys, or an ``error`` key on failure.
    """
    normalized = sql_query.strip().upper()
    if not normalized.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed.", "rows": [], "row_count": 0}

    engine = _get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(MAX_SQL_ROWS + 1)]
            truncated = len(rows) > MAX_SQL_ROWS
            if truncated:
                rows = rows[:MAX_SQL_ROWS]

            for row in rows:
                for key, value in row.items():
                    if isinstance(value, (bytes, memoryview)):
                        row[key] = str(value)
                    elif hasattr(value, "isoformat"):
                        row[key] = value.isoformat()

            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": truncated,
                "query": sql_query,
            }
    except Exception as exc:
        return {"error": str(exc), "rows": [], "row_count": 0, "query": sql_query}


async def run_sql_query(sql_query: str) -> Dict[str, Any]:
    """Async wrapper around :func:`_sync_run_sql_query` (off-loop)."""
    return await asyncio.to_thread(_sync_run_sql_query, sql_query)


async def execute_code_in_sandbox(
    code: str,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """POST ``code`` to the python_sandbox service and return its result.

    Args:
        code: Python source to run.
        timeout_seconds: Sandbox-side wall-clock budget.

    Returns:
        Sandbox response dict (``success`` / ``stdout`` / ``stderr`` / ``returncode``).
        On HTTP failure, returns a synthetic record with ``success=False`` and
        ``returncode=-1`` so the LLM can act on the error.
    """
    url = f"{_runtime['python_sandbox_base_url']}/execute"
    payload = {"code": code, "timeout_seconds": timeout_seconds}

    client = _get_httpx_client()
    response = await client.post(url, json=payload, timeout=max(timeout_seconds + 10, 70))
    if response.status_code != 200:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Sandbox HTTP {response.status_code}: {response.text}",
            "returncode": -1,
        }
    return response.json()


def encode_data_for_sandbox(rows: list[dict]) -> str:
    """Base64-encode rows so they can be safely interpolated into sandbox code.

    Args:
        rows: List of row dicts.

    Returns:
        Base64-encoded UTF-8 JSON string. ``default=str`` is used so datetime
        and decimal values survive the round-trip.
    """
    raw = json.dumps(rows, default=str)
    return base64.b64encode(raw.encode()).decode()


# Source suffix appended to every Webhose news collection name by the writer
# (``downloader_general/src/extractors/github_download.py:NEWS_SOURCE_SUFFIX``).
# Must match byte-for-byte so topic/sentiment filters still resolve to the right
# collection.
NEWS_COLLECTION_SUFFIX = "_webhose"


def _make_collection_name(topic: str, sentiment: str) -> str:
    """Replicate the Qdrant collection naming used by ``downloader_general``.

    Args:
        topic: Topic label (e.g. ``Economy Business and Finance``).
        sentiment: ``positive`` or ``negative``.

    Returns:
        A Qdrant-safe collection name (lowercased, underscores, ``_webhose`` suffix).
    """
    topic_normalized = topic.strip().lower()
    name = f"{topic_normalized}_{sentiment}"
    name = name.replace(" ", "_").replace(",", " ").lower()
    return f"{name}{NEWS_COLLECTION_SUFFIX}"


async def _get_embedding(text_input: str) -> List[float]:
    """Embed one string via the configured OpenAI-compatible model."""
    client = _get_openai_async_client()
    model = _runtime["openai_embedding_model"]
    response = await client.embeddings.create(input=[text_input], model=model)
    return response.data[0].embedding


# Curated sources ingested by ``downloader_general`` under their own per-topic
# collection names (they don't follow the ``{topic}_{sentiment}`` convention), so
# a topic/sentiment filter would otherwise skip them. They are always searched.
ALWAYS_SEARCH_COLLECTION_PREFIXES = ("actually_relevant_", "world_bank_")


def _sync_qdrant_search(
    query_embedding: List[float],
    topic_filter: str | None,
    sentiment_filter: str | None,
    top_k: int,
) -> Dict[str, Any]:
    """Search every relevant Qdrant collection and merge the top ``top_k`` hits.

    Args:
        query_embedding: Pre-computed embedding for the user query.
        topic_filter: When set, restrict to ``topic_*`` collections.
        sentiment_filter: When set, restrict to ``*_sentiment`` collections.
        top_k: Total number of articles to return after merging collections.

    Returns:
        Dict with an ``articles`` key listing top results, or a ``message``
        explaining why no collections matched.
    """
    qdrant = _get_qdrant_client()

    collections_response = qdrant.get_collections()
    all_collections = [c.name for c in collections_response.collections]

    target_collections: list[str] = []
    if topic_filter and sentiment_filter:
        name = _make_collection_name(topic_filter, sentiment_filter)
        if name in all_collections:
            target_collections = [name]
    elif topic_filter:
        for sent in ("positive", "negative"):
            name = _make_collection_name(topic_filter, sent)
            if name in all_collections:
                target_collections.append(name)
    elif sentiment_filter:
        suffix = f"_{sentiment_filter}{NEWS_COLLECTION_SUFFIX}"
        target_collections = [c for c in all_collections if c.endswith(suffix)]
    else:
        target_collections = all_collections

    # Always fold in the curated per-topic sources (Actually Relevant, World Bank
    # documents); irrelevant collections just return low scores and drop out of
    # the merged top_k ranking below.
    always_on = [c for c in all_collections if c.startswith(ALWAYS_SEARCH_COLLECTION_PREFIXES)]
    target_collections = list(dict.fromkeys([*target_collections, *always_on]))

    if not target_collections:
        return {"articles": [], "message": "No matching collections found."}

    per_coll = max(1, top_k // len(target_collections) + 1)
    all_results: list[dict] = []

    for coll_name in target_collections:
        try:
            if hasattr(qdrant, "query_points"):
                resp = qdrant.query_points(
                    collection_name=coll_name,
                    query=query_embedding,
                    limit=per_coll,
                    with_payload=True,
                    with_vectors=False,
                )
                hits = resp.points
            else:
                hits = qdrant.search(
                    collection_name=coll_name,
                    query_vector=query_embedding,
                    limit=per_coll,
                    with_payload=True,
                )

            for hit in hits:
                payload = hit.payload or {}
                article = payload.get("article", {})
                all_results.append(
                    {
                        "score": getattr(hit, "score", 0),
                        "title": article.get("title", ""),
                        "text": (article.get("text", "") or "")[:2000],
                        "url": article.get("url", ""),
                        "published": article.get("published", ""),
                        "sentiment": payload.get("sentiment", ""),
                        "topic": payload.get("topic", ""),
                        "source": article.get("thread", {}).get("site", ""),
                        "collection": coll_name,
                    }
                )
        except Exception as exc:
            logger.warning("Qdrant search failed for %s: %s", coll_name, exc)

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"articles": all_results[:top_k]}


async def search_qdrant_news(
    query: str,
    topic_filter: str | None = None,
    sentiment_filter: str | None = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Embed ``query`` and search Qdrant news collections.

    Args:
        query: User search string.
        topic_filter: Optional topic restriction.
        sentiment_filter: Optional sentiment restriction.
        top_k: Number of articles to return.

    Returns:
        Dict with ``articles`` or an ``error`` key.
    """
    try:
        query_embedding = await _get_embedding(query)
        return await asyncio.to_thread(
            _sync_qdrant_search, query_embedding, topic_filter, sentiment_filter, top_k
        )
    except Exception as exc:
        return {"articles": [], "error": str(exc)}


def _sync_web_search(
    queries: List[str],
    max_results_per_query: int = 5,
) -> Dict[str, Any]:
    """Run one or more DuckDuckGo text searches and collect the hits.

    Args:
        queries: List of search strings.
        max_results_per_query: Hits to keep per query.

    Returns:
        Dict with ``results`` (and an ``error`` key on failure).
    """
    results: list[dict] = []
    try:
        ddgs = DDGS()
        for q in queries:
            for hit in ddgs.text(q, max_results=max_results_per_query):
                results.append(
                    {
                        "query": q,
                        "title": hit.get("title", ""),
                        "body": hit.get("body", ""),
                        "href": hit.get("href", ""),
                    }
                )
    except Exception as exc:
        return {"results": results, "error": str(exc)}
    return {"results": results}


async def web_search(
    queries: List[str],
    max_results_per_query: int = 5,
) -> Dict[str, Any]:
    """Async wrapper around :func:`_sync_web_search` (off-loop)."""
    return await asyncio.to_thread(_sync_web_search, queries, max_results_per_query)


async def ingest_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ask ``downloader_extra`` to ingest one unit of data into Postgres.

    The unified ``/ingest`` endpoint dispatches on ``payload["source"]``
    (``worldbank`` / ``yahoo`` / ``binance``); the caller builds the
    source-specific body (indicator_id+db_id, ticker, or symbol).

    Args:
        payload: The ``/ingest`` request body, including a ``source`` key.

    Returns:
        ``downloader_extra``'s ingest response, augmented with ``success=True``,
        or ``{"success": False, "error": ...}`` on HTTP failure.
    """
    url = f"{_runtime['downloader_extra_base_url']}/ingest"

    client = _get_httpx_client()
    # Yahoo/Binance pull full history (paged klines + yfinance), so allow a
    # generous budget — larger than the old WB-only 120s.
    response = await client.post(url, json=payload, timeout=300)
    if response.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text}",
        }
    data = response.json()
    data["success"] = True
    return data
