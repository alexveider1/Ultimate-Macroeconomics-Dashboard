"""Qdrant + embedding helpers for the BFF's news / RAG endpoints.

Async, mirroring the agent's RAG worker (``agent/agent/tools.py``): a query is
embedded with the shared OpenAI-compatible embedding model, then every relevant
collection is searched and the hits are merged and re-ranked by score. The
curated per-topic sources (``actually_relevant_*`` / ``world_bank_*``) don't
follow the ``{topic}_{sentiment}`` naming, so they are always folded in.

Every operation degrades to an empty result on failure, so a frontend renders an
informative empty state instead of a 500 when Qdrant is unavailable.
"""

import logging
from typing import Any

from models import NewsArticle, NewsSearchHit
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)

# Curated sources ingested under their own per-topic collection names (they don't
# follow the ``{topic}_{sentiment}`` convention) — always searched regardless of
# the topic/sentiment filter, exactly like the agent's RAG worker.
ALWAYS_SEARCH_COLLECTION_PREFIXES = ("actually_relevant_", "world_bank_")


def build_qdrant_client(host: str, port: int, api_key: str) -> AsyncQdrantClient:
    """Build the async Qdrant client from host/port/key."""
    return AsyncQdrantClient(
        url=f"http://{host}:{port}", api_key=api_key or None, prefer_grpc=False
    )


def build_openai_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Build the async OpenAI-compatible client used for query embeddings."""
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _make_collection_name(topic: str, sentiment: str) -> str:
    """Return the ``{topic}_{sentiment}`` Qdrant collection name."""
    return f"{topic}_{sentiment}"


def _article_from_payload(payload: dict, point_id: Any, collection: str) -> dict:
    """Flatten a stored point payload into the common news-article shape."""
    article = payload.get("article", {}) or {}
    thread = article.get("thread", {}) or {}
    return {
        "id": str(point_id),
        "title": article.get("title", "") or "",
        "text": (article.get("text", "") or "")[:2000],
        "url": article.get("url", "") or "",
        "published": str(article.get("published", "") or ""),
        "source": thread.get("site", "") or "",
        "topic": payload.get("topic", "") or "",
        "sentiment": payload.get("sentiment", "") or "",
        "collection": collection,
    }


async def list_collections(qdrant: AsyncQdrantClient) -> list[str]:
    """Return every collection name, or an empty list on failure."""
    try:
        response = await qdrant.get_collections()
        return sorted(c.name for c in response.collections)
    except Exception as exc:
        logger.warning("Qdrant list_collections failed: %s", exc)
        return []


async def browse_collection(
    qdrant: AsyncQdrantClient, collection: str, limit: int
) -> list[NewsArticle]:
    """Scroll up to ``limit`` stored documents from one collection."""
    try:
        records, _ = await qdrant.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning("Qdrant browse failed for '%s': %s", collection, exc)
        return []

    return [
        NewsArticle(**_article_from_payload(record.payload or {}, record.id, collection))
        for record in records
    ]


async def embed_query(openai: AsyncOpenAI, model: str, text: str) -> list[float]:
    """Embed one query string via the configured embedding model."""
    response = await openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


async def _resolve_target_collections(
    all_collections: list[str], topic: str | None, sentiment: str | None
) -> list[str]:
    """Pick which collections to search from a topic/sentiment filter."""
    if topic and sentiment:
        name = _make_collection_name(topic, sentiment)
        targets = [name] if name in all_collections else []
    elif topic:
        targets = [
            _make_collection_name(topic, sent)
            for sent in ("positive", "negative")
            if _make_collection_name(topic, sent) in all_collections
        ]
    elif sentiment:
        targets = [c for c in all_collections if c.endswith(f"_{sentiment}")]
    else:
        targets = list(all_collections)

    always_on = [c for c in all_collections if c.startswith(ALWAYS_SEARCH_COLLECTION_PREFIXES)]
    # Preserve order, drop dupes.
    return list(dict.fromkeys([*targets, *always_on]))


async def search_news(
    qdrant: AsyncQdrantClient,
    query_embedding: list[float],
    topic: str | None,
    sentiment: str | None,
    top_k: int,
) -> tuple[list[NewsSearchHit], str | None]:
    """Search matching collections and merge the top ``top_k`` hits by score.

    Returns:
        ``(hits, message)`` — ``message`` is non-null only when no collection
        matched the filter (so the caller can surface an explanation).
    """
    try:
        collections_response = await qdrant.get_collections()
        all_collections = [c.name for c in collections_response.collections]
    except Exception as exc:
        logger.warning("Qdrant get_collections failed during search: %s", exc)
        return [], "Vector store is currently unavailable."

    target_collections = await _resolve_target_collections(all_collections, topic, sentiment)
    if not target_collections:
        return [], "No matching collections found."

    per_coll = max(1, top_k // len(target_collections) + 1)
    hits: list[NewsSearchHit] = []

    for collection in target_collections:
        try:
            response = await qdrant.query_points(
                collection_name=collection,
                query=query_embedding,
                limit=per_coll,
                with_payload=True,
                with_vectors=False,
            )
            for point in response.points:
                article = _article_from_payload(point.payload or {}, point.id, collection)
                hits.append(NewsSearchHit(score=getattr(point, "score", 0.0) or 0.0, **article))
        except Exception as exc:
            logger.warning("Qdrant search failed for '%s': %s", collection, exc)

    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k], None
