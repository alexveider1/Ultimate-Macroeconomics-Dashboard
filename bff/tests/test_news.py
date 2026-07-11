"""Tests for the news/RAG router using fake async Qdrant + OpenAI clients."""

from collections.abc import Iterator
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest
from routers import news


class _FakeQdrant:
    """Minimal async stand-in for ``AsyncQdrantClient``."""

    def __init__(self, points_by_collection: dict[str, list[SimpleNamespace]]) -> None:
        self._points = points_by_collection

    async def get_collections(self) -> SimpleNamespace:
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self._points])

    async def scroll(self, collection_name, limit, with_payload, with_vectors):
        return self._points.get(collection_name, [])[:limit], None

    async def query_points(self, collection_name, query, limit, with_payload, with_vectors):
        points = [
            SimpleNamespace(id=p.id, payload=p.payload, score=0.9)
            for p in self._points.get(collection_name, [])[:limit]
        ]
        return SimpleNamespace(points=points)


class _FakeEmbeddings:
    async def create(self, input, model):  # noqa: A002 - mirrors OpenAI's kwarg name.
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3, 0.4])])


class _FakeOpenAI:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


def _point(point_id: str, title: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=point_id,
        payload={
            "topic": "economy",
            "sentiment": "positive",
            "article": {
                "title": title,
                "text": "body",
                "url": "http://x",
                "published": "2026-01-01",
                "thread": {"site": "site.com"},
            },
        },
    )


def _build_client(*, news_search_enabled: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(news.router)
    app.state.qdrant = _FakeQdrant(
        {
            "economy_positive": [_point("1", "Boom")],
            "world_bank": [_point("2", "WB report")],
        }
    )
    app.state.openai = _FakeOpenAI()
    app.state.embedding_model = "text-embedding-3-small"
    app.state.news_search_enabled = news_search_enabled
    return TestClient(app)


@pytest.fixture()
def news_client() -> Iterator[TestClient]:
    with _build_client() as client:
        yield client


def test_list_collections(news_client: TestClient) -> None:
    body = news_client.get("/news/collections").json()
    assert body["collections"] == ["economy_positive", "world_bank"]


def test_browse_collection(news_client: TestClient) -> None:
    body = news_client.get("/news/collections/economy_positive/articles").json()
    assert len(body) == 1
    assert body[0]["title"] == "Boom"
    assert body[0]["source"] == "site.com"


def test_search_merges_and_ranks(news_client: TestClient) -> None:
    body = news_client.post("/news/search", json={"query": "growth", "top_k": 5}).json()
    titles = {hit["title"] for hit in body["articles"]}
    # No filter → all collections + always-on curated world_bank folded in.
    assert titles == {"Boom", "WB report"}
    assert body["message"] is None


def test_search_disabled_returns_503() -> None:
    with _build_client(news_search_enabled=False) as client:
        response = client.post("/news/search", json={"query": "x"})
    assert response.status_code == 503


def test_search_rejects_empty_query(news_client: TestClient) -> None:
    assert news_client.post("/news/search", json={"query": ""}).status_code == 422


# --------------------------------------------------------------------------- #
# Embedding projection (server-side dim-reduction via the clustering service)
# --------------------------------------------------------------------------- #


def _vec_point(point_id: str, title: str, vector: list[float]) -> SimpleNamespace:
    return SimpleNamespace(id=point_id, vector=vector, payload={"article": {"title": title}})


class _VecQdrant:
    """Fake Qdrant that returns points *with* vectors for the projection path."""

    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._records = records

    async def scroll(self, collection_name, limit, with_payload, with_vectors):
        return self._records[:limit], None

    async def retrieve(self, collection_name, ids, with_vectors):
        wanted = {str(i) for i in ids}
        return [r for r in self._records if str(r.id) in wanted]


def _cluster_handler(request: httpx.Request) -> httpx.Response:
    """Stand in for the clustering service: echo rows with cluster + 2D coords."""
    body = json.loads(request.content)
    rows = body["dataframe"]
    out = [
        {
            "__article_id": row["__article_id"],
            "__title": row["__title"],
            "cluster": index % 2,
            "__viz_x": float(index),
            "__viz_y": float(-index),
        }
        for index, row in enumerate(rows)
    ]
    return httpx.Response(
        200,
        json={
            "method_used": body["method"],
            "dataframe": out,
            "visualization_mode": "tsne",
            "visualization_columns": ["__viz_x", "__viz_y"],
            "visualization_labels": ["x", "y"],
        },
    )


def _build_projection_client() -> TestClient:
    app = FastAPI()
    app.include_router(news.router)
    records = [
        _vec_point(str(i), f"Article {i}", [float(i), float(i) + 1.0, 0.5]) for i in range(5)
    ]
    app.state.qdrant = _VecQdrant(records)
    app.state.clustering_url = "http://clustering:8002"
    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(_cluster_handler))
    return TestClient(app)


def test_projection_returns_points() -> None:
    with _build_projection_client() as client:
        body = client.post("/news/collections/economy/projection", json={}).json()
    assert len(body["points"]) == 5
    assert {p["cluster"] for p in body["points"]} == {"0", "1"}
    assert body["points"][0]["x"] == 0.0
    assert body["distances"] is None


def test_projection_with_query_returns_distances() -> None:
    with _build_projection_client() as client:
        body = client.post("/news/collections/economy/projection", json={"query_id": "0"}).json()
    # Distances to every *other* article (5 total → 4 others).
    assert body["distances"] is not None
    assert len(body["distances"]) == 4
    assert body["query_title"] == "Article 0"


def test_projection_too_few_points() -> None:
    app = FastAPI()
    app.include_router(news.router)
    app.state.qdrant = _VecQdrant([_vec_point("0", "solo", [1.0, 0.0])])
    app.state.clustering_url = "http://clustering:8002"
    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(_cluster_handler))
    with TestClient(app) as client:
        body = client.post("/news/collections/economy/projection", json={}).json()
    assert body["points"] == []
    assert "at least 4" in body["message"]
