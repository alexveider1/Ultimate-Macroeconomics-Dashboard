import os
from typing import List, Dict, Any, Optional

import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.app_logging import log_vector_query

CONFIG = yaml.safe_load(open("config.yaml"))
load_dotenv(".env")

_QDRANT_HOST = CONFIG.get("qdrant", {}).get("host", "vector_db")
_QDRANT_PORT = CONFIG.get("qdrant", {}).get("port", 6333)
_QDRANT_URL = f"http://{_QDRANT_HOST}:{_QDRANT_PORT}"
_QDRANT_API_KEY = (
    os.getenv("QDRANT_API_KEY")
    or os.getenv("QDRANT__API_KEY")
    or os.getenv("QDRANT__SERVICE__API_KEY")
    or None
)

_DEFAULT_CLIENT = QdrantClient(
    url=_QDRANT_URL,
    api_key=_QDRANT_API_KEY,
    prefer_grpc=False,
)


def is_qdrant_available(client: Optional[QdrantClient] = None) -> bool:
    client = client or _DEFAULT_CLIENT
    log_vector_query(operation="health_check")
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def list_collections(client: Optional[QdrantClient] = None) -> List[str]:
    client = client or _DEFAULT_CLIENT
    log_vector_query(operation="list_collections")
    try:
        response = client.get_collections()
        return [item.name for item in response.collections]
    except Exception:
        return []


def scroll_collection(
    collection_name: str,
    client: Optional[QdrantClient] = None,
    page_limit: int = 256,
) -> List[models.Record]:
    client = client or _DEFAULT_CLIENT
    log_vector_query(
        operation="scroll_collection",
        collection_name=collection_name,
        summary=f"page_limit={page_limit}",
    )
    all_records: List[models.Record] = []
    offset = None

    try:
        all_records = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break

            all_records.extend(records)
            if offset is None:
                break

        return all_records
    except Exception:
        return []


def get_point(
    collection_name: str,
    point_id: str,
    client: Optional[QdrantClient] = None,
    with_vector: bool = False,
) -> Optional[models.Record]:
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
    except Exception:
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
    except Exception:
        return []
