"""News / RAG endpoints: browse Qdrant collections and semantic search."""

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from models import (
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
