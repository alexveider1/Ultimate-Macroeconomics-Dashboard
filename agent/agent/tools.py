import asyncio
import base64
import json
import logging
from typing import Any, Dict, List

import httpx
import yaml
from ddgs import DDGS
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


_runtime: Dict[str, Any] = {}

MAX_SQL_ROWS = 500


def configure_runtime(
    *,
    database_schema_path: str,
    world_bank_catalog_path: str,
    news_topics_path: str,
    qdrant_url: str,
    qdrant_api_key: str,
    postgres_database_uri: str,
    python_sandbox_base_url: str,
    downloader_extra_base_url: str,
    openai_api_key: str,
    openai_base_url: str,
    openai_embedding_model: str,
) -> None:
    _runtime["qdrant_url"] = qdrant_url
    _runtime["qdrant_api_key"] = qdrant_api_key
    _runtime["postgres_database_uri"] = postgres_database_uri
    _runtime["python_sandbox_base_url"] = python_sandbox_base_url
    _runtime["downloader_extra_base_url"] = downloader_extra_base_url
    _runtime["openai_api_key"] = openai_api_key
    _runtime["openai_base_url"] = openai_base_url
    _runtime["openai_embedding_model"] = openai_embedding_model

    with open(database_schema_path) as fh:
        _runtime["database_schema"] = yaml.safe_load(fh)

    with open(world_bank_catalog_path) as fh:
        _runtime["world_bank_catalog"] = json.load(fh)

    with open(news_topics_path) as fh:
        _runtime["news_topics"] = json.load(fh)

    _runtime["_engine"] = None
    _runtime["_qdrant_client"] = None
    _runtime["_openai_async_client"] = None


def _get_engine():
    if _runtime.get("_engine") is None:
        _runtime["_engine"] = create_engine(
            _runtime["postgres_database_uri"],
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _runtime["_engine"]


def _get_qdrant_client() -> QdrantClient:
    if _runtime.get("_qdrant_client") is None:
        _runtime["_qdrant_client"] = QdrantClient(
            url=_runtime["qdrant_url"],
            api_key=_runtime.get("qdrant_api_key"),
            prefer_grpc=False,
        )
    return _runtime["_qdrant_client"]


def _get_openai_async_client() -> AsyncOpenAI:
    if _runtime.get("_openai_async_client") is None:
        _runtime["_openai_async_client"] = AsyncOpenAI(
            api_key=_runtime["openai_api_key"],
            base_url=_runtime["openai_base_url"],
        )
    return _runtime["_openai_async_client"]


def get_database_schema_text() -> str:
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
                lines.append(
                    f"  Foreign key: ({cols}) -> {ref_table}({ref_cols}) [{enforced}]"
                )

            lines.append("  Columns:")
            for col_name, col_info in (table_def.get("columns") or {}).items():
                col_type = col_info.get("type", "UNKNOWN")
                col_desc = col_info.get("description", "")
                lines.append(f"    - {col_name} ({col_type}): {col_desc}")
    return "\n".join(lines)


def get_world_bank_catalog_text() -> str:
    catalog = _runtime.get("world_bank_catalog", {})
    lines: list[str] = []
    for category, indicators in catalog.items():
        lines.append(f"\n--- {category} ---")
        for ind in indicators:
            lines.append(f"  {ind['name']} (id={ind['id']}, db={ind['db']})")
    return "\n".join(lines)


def get_news_topics() -> List[str]:
    return _runtime.get("news_topics", [])


def _sync_run_sql_query(sql_query: str) -> Dict[str, Any]:
    normalized = sql_query.strip().upper()
    if not normalized.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed.", "rows": [], "row_count": 0}

    engine = _get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows = [
                dict(zip(columns, row)) for row in result.fetchmany(MAX_SQL_ROWS + 1)
            ]
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
    return await asyncio.to_thread(_sync_run_sql_query, sql_query)


async def execute_code_in_sandbox(
    code: str,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    url = f"{_runtime['python_sandbox_base_url']}/execute"
    payload = {"code": code, "timeout_seconds": timeout_seconds}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url, json=payload, timeout=max(timeout_seconds + 10, 70)
        )
        if response.status_code != 200:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Sandbox HTTP {response.status_code}: {response.text}",
                "returncode": -1,
            }
        return response.json()


def encode_data_for_sandbox(rows: list[dict]) -> str:
    """Base64-encode a list of dicts so it can be safely embedded in sandbox code."""
    raw = json.dumps(rows, default=str)
    return base64.b64encode(raw.encode()).decode()


def _make_collection_name(topic: str, sentiment: str) -> str:
    """Replicate the collection-naming logic used by the downloader."""
    topic_normalized = topic.strip().lower()
    name = f"{topic_normalized}_{sentiment}"
    name = name.replace(" ", "_").replace(",", " ").lower()
    return name


async def _get_embedding(text_input: str) -> List[float]:
    client = _get_openai_async_client()
    model = _runtime["openai_embedding_model"]
    response = await client.embeddings.create(input=[text_input], model=model)
    return response.data[0].embedding


def _sync_qdrant_search(
    query_embedding: List[float],
    topic_filter: str | None,
    sentiment_filter: str | None,
    top_k: int,
) -> Dict[str, Any]:
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
        target_collections = [
            c for c in all_collections if c.endswith(f"_{sentiment_filter}")
        ]
    else:
        target_collections = all_collections

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
    return await asyncio.to_thread(_sync_web_search, queries, max_results_per_query)




async def download_indicator(indicator_id: str, db_id: int) -> Dict[str, Any]:
    url = f"{_runtime['downloader_extra_base_url']}/ingest"
    payload = {"indicator_id": indicator_id, "db_id": db_id}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=120)
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
        data = response.json()
        data["success"] = True
        return data
