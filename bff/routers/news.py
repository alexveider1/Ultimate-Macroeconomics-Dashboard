"""News / RAG endpoints: browse Qdrant collections, semantic search, embedding map."""

import logging

import clients
from fastapi import APIRouter, HTTPException, Query, Request
from models import (
    EmbeddingProjectionPoint,
    EmbeddingProjectionRequest,
    EmbeddingProjectionResponse,
    NewsArticle,
    NewsCollectionsOut,
    NewsSearchRequest,
    NewsSearchResponse,
)
import vector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/collections", response_model=NewsCollectionsOut)
async def list_news_collections(request: Request) -> NewsCollectionsOut:
    """Return every Qdrant collection available to browse / search."""
    collections = await vector.list_collections(request.app.state.qdrant)
    return NewsCollectionsOut(collections=collections)


@router.get("/collections/{collection}/articles", response_model=list[NewsArticle])
async def browse_news(
    collection: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[NewsArticle]:
    """Return up to ``limit`` stored documents from one collection."""
    return await vector.browse_collection(request.app.state.qdrant, collection, limit)


@router.post("/search", response_model=NewsSearchResponse)
async def search_news(payload: NewsSearchRequest, request: Request) -> NewsSearchResponse:
    """Embed the query and return score-ranked hits across the RAG corpus."""
    if not request.app.state.news_search_enabled:
        raise HTTPException(
            status_code=503,
            detail="Semantic search is disabled: OPENAI_API_KEY is not configured.",
        )

    try:
        embedding = await vector.embed_query(
            request.app.state.openai,
            request.app.state.embedding_model,
            payload.query,
        )
    except Exception as exc:
        logger.warning("News query embedding failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Query embedding failed: {exc}") from exc

    hits, message = await vector.search_news(
        request.app.state.qdrant,
        embedding,
        payload.topic,
        payload.sentiment,
        payload.top_k,
    )
    return NewsSearchResponse(articles=hits, message=message)


@router.post(
    "/collections/{collection}/projection",
    response_model=EmbeddingProjectionResponse,
)
async def project_collection(
    collection: str,
    payload: EmbeddingProjectionRequest,
    request: Request,
) -> EmbeddingProjectionResponse:
    """Project one collection's embeddings to 2D/3D via the clustering service.

    The heavy vectors stay server-side: the BFF scrolls them, forwards them to
    the clustering ``/cluster`` endpoint (clustering + dim-reduction), and returns
    only the coordinates + cluster labels. When ``query_id`` is set it also returns
    the cosine-distance distribution from that article to the rest of the sample.
    """
    ids, titles, vectors = await vector.load_collection_vectors(
        request.app.state.qdrant, collection, payload.max_points
    )
    if len(ids) < 4:
        return EmbeddingProjectionResponse(
            points=[],
            output_dim=payload.output_dim,
            mode="none",
            message="Need at least 4 articles with embeddings to project.",
        )

    dim = len(vectors[0])
    feature_cols = [f"f{i}" for i in range(dim)]
    rows = [
        {"__article_id": pid, "__title": title, **{fc: v for fc, v in zip(feature_cols, vec)}}
        for pid, title, vec in zip(ids, titles, vectors)
    ]
    body = {
        "method": payload.method,
        "dataframe": rows,
        "feature_columns": feature_cols,
        "k": payload.k,
        "reduction_method": payload.reduction_method,
        "output_dim": payload.output_dim,
    }
    url = f"{request.app.state.clustering_url}/cluster"
    result = await clients.post_json(
        request.app.state.http_client, "clustering", url, body, timeout=120.0
    )

    viz_cols = result.get("visualization_columns", []) or []
    points: list[EmbeddingProjectionPoint] = []
    for row in result.get("dataframe", []) or []:
        x = float(row.get(viz_cols[0], 0.0)) if len(viz_cols) > 0 else 0.0
        y = float(row.get(viz_cols[1], 0.0)) if len(viz_cols) > 1 else 0.0
        z = float(row.get(viz_cols[2], 0.0)) if len(viz_cols) > 2 else None
        points.append(
            EmbeddingProjectionPoint(
                id=str(row.get("__article_id", "")),
                title=str(row.get("__title", "")),
                cluster=str(row.get("cluster", "?")),
                x=x,
                y=y,
                z=z,
            )
        )

    distances: list[float] | None = None
    query_title: str | None = None
    if payload.query_id:
        if payload.query_id in ids:
            index = ids.index(payload.query_id)
            query_vec: list[float] | None = vectors[index]
            query_title = titles[index]
            others = [v for j, v in enumerate(vectors) if j != index]
        else:
            query_vec = await vector.get_point_vector(
                request.app.state.qdrant, collection, payload.query_id
            )
            others = vectors
        if query_vec and others:
            distances = vector.cosine_distances(query_vec, others)

    return EmbeddingProjectionResponse(
        points=points,
        output_dim=payload.output_dim,
        mode=str(result.get("visualization_mode", "none")),
        distances=distances,
        query_id=payload.query_id,
        query_title=query_title,
    )
