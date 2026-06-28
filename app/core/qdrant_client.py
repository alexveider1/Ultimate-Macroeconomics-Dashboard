"""Read-side Qdrant helpers used by the news / RAG pages.

A module-level :class:`QdrantClient` is built from ``config.yaml`` + ``.env``;
every function accepts an optional override for testability. All operations
swallow failures and degrade to empty results — the news page renders an
informative empty state instead of crashing when Qdrant is down.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.app_logging import log_vector_query
from core.config import load_config
from core.settings import get_settings

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)
SETTINGS = get_settings()

logger = logging.getLogger(__name__)

_QDRANT_HOST = CONFIG.qdrant.host
_QDRANT_PORT = CONFIG.qdrant.port
_QDRANT_URL = f"http://{_QDRANT_HOST}:{_QDRANT_PORT}"
_QDRANT_API_KEY = SETTINGS.qdrant_api_key or None

_DEFAULT_CLIENT = QdrantClient(
    url=_QDRANT_URL,
    api_key=_QDRANT_API_KEY,
    prefer_grpc=False,
)


def is_qdrant_available(client: Optional[QdrantClient] = None) -> bool:
    """Return ``True`` iff ``client.get_collections()`` succeeds.

    Args:
        client: Optional override (default: module-level client).

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    client = client or _DEFAULT_CLIENT
    log_vector_query(operation="health_check")
    try:
        client.get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return False


def list_collections(client: Optional[QdrantClient] = None) -> List[str]:
    """Return the names of every collection visible to the client.

    Args:
        client: Optional override (default: module-level client).

    Returns:
        List of collection names; empty list when the API call fails.
    """
    client = client or _DEFAULT_CLIENT
    log_vector_query(operation="list_collections")
    try:
        response = client.get_collections()
        return [item.name for item in response.collections]
    except Exception as exc:
        logger.warning("Qdrant list_collections failed: %s", exc)
        return []


def scroll_collection(
    collection_name: str,
    client: Optional[QdrantClient] = None,
    page_limit: int = 256,
    with_vectors: bool = False,
) -> List[models.Record]:
    """Page through every point in a collection and return them in one list.

    Args:
        collection_name: Target Qdrant collection.
        client: Optional override (default: module-level client).
        page_limit: Number of records fetched per scroll page.
        with_vectors: When ``True``, embeddings are returned alongside the
            payloads (used by the embedding-visualisation panel; expensive
            for large collections).

    Returns:
        List of payload-carrying records (vectors included when requested);
        empty on error.
    """
    client = client or _DEFAULT_CLIENT
    log_vector_query(
        operation="scroll_collection",
        collection_name=collection_name,
        summary=f"page_limit={page_limit} with_vectors={with_vectors}",
    )
    all_records: List[models.Record] = []
    offset = None

    try:
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            if not records:
                break

            all_records.extend(records)
            if offset is None:
                break

        return all_records
    except Exception as exc:
        logger.warning("Qdrant scroll_collection failed for '%s': %s", collection_name, exc)
        return []


def get_point(
    collection_name: str,
    point_id: str,
    client: Optional[QdrantClient] = None,
    with_vector: bool = False,
) -> Optional[models.Record]:
    """Retrieve a single point from a collection.

    Args:
        collection_name: Target Qdrant collection.
        point_id: Identifier of the point to fetch.
        client: Optional override (default: module-level client).
        with_vector: When ``True`` the embedding is returned alongside the payload.

    Returns:
        The matching record, or ``None`` when no point with the given id exists
        (or the request fails).
    """
    client = client or _DEFAULT_CLIENT
    log_vector_query(
        operation="get_point",
        collection_name=collection_name,
        summary=f"point_id={point_id} with_vector={with_vector}",
    )
    try:
        points = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=with_vector,
        )
        if points:
            return points[0]
        return None
    except Exception as exc:
        logger.warning(
            "Qdrant get_point failed for '%s' id=%s: %s",
            collection_name,
            point_id,
            exc,
        )
        return None


def find_nearest_embeddings(
    collection_name: str,
    query_vector: List[float],
    client: Optional[QdrantClient] = None,
    limit: int = 5,
    exact_match_filter: Optional[Dict[str, Any]] = None,
    return_payload_fields: Optional[List[str]] = None,
    exclude_point_id: str | None = None,
) -> List[models.ScoredPoint]:
    """Run a nearest-neighbour search over a Qdrant collection.

    Prefers ``client.query_points`` when available (newer API) and falls back
    to ``client.search`` otherwise.

    Args:
        collection_name: Target Qdrant collection.
        query_vector: Query embedding.
        client: Optional override (default: module-level client).
        limit: Maximum number of hits to return.
        exact_match_filter: Optional ``payload_key -> value`` map applied as
            an AND of ``MatchValue`` conditions.
        return_payload_fields: When supplied, only those payload fields are
            returned (the rest are dropped server-side).
        exclude_point_id: Optional id to filter out of the result set
            (useful for "related items, excluding the current one").

    Returns:
        Up to ``limit`` scored points sorted by descending similarity;
        empty list on error.
    """
    client = client or _DEFAULT_CLIENT
    log_vector_query(
        operation="find_nearest_embeddings",
        collection_name=collection_name,
        summary=(
            f"limit={limit} vector_size={len(query_vector)} "
            f"exclude_point_id={exclude_point_id or '-'}"
        ),
    )
    query_filter = None
    must_conditions = []
    must_not_conditions = []

    if exact_match_filter:
        for key, value in exact_match_filter.items():
            must_conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )

    if exclude_point_id is not None:
        must_not_conditions.append(models.HasIdCondition(has_id=[exclude_point_id]))

    if must_conditions or must_not_conditions:
        query_filter = models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

    with_payload: Any = True
    if return_payload_fields:
        with_payload = models.PayloadSelectorInclude(include=return_payload_fields)

    try:
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=False,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
            )

        return hits[:limit]
    except Exception as exc:
        logger.warning(
            "Qdrant find_nearest_embeddings failed for '%s': %s",
            collection_name,
            exc,
        )
        return []
