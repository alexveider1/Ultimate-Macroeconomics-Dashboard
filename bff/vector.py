"""Qdrant + embedding helpers for the BFF's news / RAG endpoints.

Async, mirroring the agent's RAG worker (``agent/agent/tools.py``): a query is
embedded with the shared OpenAI-compatible embedding model, then every relevant
collection is searched and the hits are merged and re-ranked by score. The
curated sources (each a single source-named collection, ``actually_relevant`` /
``world_bank``) don't follow the ``{topic}_{sentiment}`` naming, so they are
always folded in.

Every operation degrades to an empty result on failure, so a frontend renders an
informative empty state instead of a 500 when Qdrant is unavailable.
"""

import logging
import math
from typing import Any

from models import NewsArticle, NewsSearchHit
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
import tracing

logger = logging.getLogger(__name__)

# Curated sources ingested each into a single source-named collection
# (``actually_relevant`` / ``world_bank``); they don't follow the
# ``{topic}_{sentiment}`` convention — always searched regardless of the
# topic/sentiment filter (matched via ``str.startswith``), like the agent's worker.
ALWAYS_SEARCH_COLLECTION_PREFIXES = ("actually_relevant", "world_bank")

# Source suffix the Webhose news writer appends to every collection name
# (``downloader_general/src/extractors/github_download.py:NEWS_SOURCE_SUFFIX``);
# must match byte-for-byte so topic/sentiment filters resolve correctly.
NEWS_COLLECTION_SUFFIX = "_webhose"


def build_qdrant_client(host: str, port: int, api_key: str) -> AsyncQdrantClient:
    """Build the async Qdrant client from host/port/key."""
    return AsyncQdrantClient(
        url=f"http://{host}:{port}", api_key=api_key or None, prefer_grpc=False
    )


def build_openai_client(api_key: str, base_url: str) -> AsyncOpenAI:
    """Build the async OpenAI-compatible client used for query embeddings.

    Uses the Langfuse-instrumented ``AsyncOpenAI`` when tracing is enabled so the
    news-search embedding calls are traced; the plain client otherwise.
    """
    client_cls = tracing.async_openai_client_class()
    return client_cls(api_key=api_key, base_url=base_url)


def _make_collection_name(topic: str, sentiment: str) -> str:
    """Return the ``{topic}_{sentiment}_webhose`` Qdrant collection name.

    Must match the writer in ``downloader_general``
    (``extractors/github_download.py:_collection_name_for``) and the agent's reader
    (``agent/agent/tools.py:_make_collection_name``) byte-for-byte, since Qdrant
    collection names are case-sensitive: the corpus is stored lowercased with spaces
    ``->`` ``_``, commas ``->`` ``" "`` and a ``_webhose`` source suffix. A bare
    ``f"{topic}_{sentiment}"`` would look up e.g. ``"politics_positive"`` and miss
    the real ``"politics_positive_webhose"``.
    """
    topic_normalized = topic.strip().lower()
    base = f"{topic_normalized}_{sentiment}".replace(" ", "_").replace(",", " ").lower()
    return f"{base}{NEWS_COLLECTION_SUFFIX}"


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
        suffix = f"_{sentiment}{NEWS_COLLECTION_SUFFIX}"
        targets = [c for c in all_collections if c.endswith(suffix)]
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


def _extract_vector(raw: Any) -> list[float] | None:
    """Coerce a stored point vector (plain list or named-vector dict) to a float list."""
    if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]
    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                return [float(x) for x in value]
    return None


async def load_collection_vectors(
    qdrant: AsyncQdrantClient, collection: str, max_points: int
) -> tuple[list[str], list[str], list[list[float]]]:
    """Scroll up to ``max_points`` points **with vectors**, returning ids/titles/vectors."""
    try:
        records, _ = await qdrant.scroll(
            collection_name=collection,
            limit=max_points,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as exc:
        logger.warning("Qdrant vector scroll failed for '%s': %s", collection, exc)
        return [], [], []

    ids: list[str] = []
    titles: list[str] = []
    vectors: list[list[float]] = []
    for record in records:
        vec = _extract_vector(getattr(record, "vector", None))
        if not vec:
            continue
        article = (record.payload or {}).get("article", {}) or {}
        ids.append(str(record.id))
        titles.append(str(article.get("title") or record.id))
        vectors.append(vec)
    return ids, titles, vectors


async def get_point_vector(
    qdrant: AsyncQdrantClient, collection: str, point_id: str
) -> list[float] | None:
    """Retrieve one point's vector (for a query article outside the sampled page)."""
    try:
        records = await qdrant.retrieve(
            collection_name=collection, ids=[point_id], with_vectors=True
        )
    except Exception as exc:
        logger.warning("Qdrant retrieve failed for '%s/%s': %s", collection, point_id, exc)
        return None
    if not records:
        return None
    return _extract_vector(getattr(records[0], "vector", None))


def cosine_distances(query: list[float], others: list[list[float]]) -> list[float]:
    """Cosine distance (``1 − cos``, clamped to ``[0, 2]``) from ``query`` to each other vector."""
    q_norm = math.sqrt(sum(x * x for x in query)) or 1e-12
    normalized_query = [x / q_norm for x in query]
    distances: list[float] = []
    for vec in others:
        v_norm = math.sqrt(sum(x * x for x in vec)) or 1e-12
        sim = sum(a * (b / v_norm) for a, b in zip(normalized_query, vec))
        distances.append(max(0.0, min(2.0, 1.0 - sim)))
    return distances
