"""Tests for the news/RAG router using fake async Qdrant + OpenAI clients."""

from collections.abc import Iterator
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
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
            "world_bank_growth": [_point("2", "WB report")],
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
    assert body["collections"] == ["economy_positive", "world_bank_growth"]


def test_browse_collection(news_client: TestClient) -> None:
    body = news_client.get("/news/collections/economy_positive/articles").json()
    assert len(body) == 1
    assert body[0]["title"] == "Boom"
    assert body[0]["source"] == "site.com"


def test_search_merges_and_ranks(news_client: TestClient) -> None:
    body = news_client.post("/news/search", json={"query": "growth", "top_k": 5}).json()
    titles = {hit["title"] for hit in body["articles"]}
    # No filter → all collections + always-on curated world_bank_* folded in.
    assert titles == {"Boom", "WB report"}
    assert body["message"] is None


def test_search_disabled_returns_503() -> None:
    with _build_client(news_search_enabled=False) as client:
        response = client.post("/news/search", json={"query": "x"})
    assert response.status_code == 503


def test_search_rejects_empty_query(news_client: TestClient) -> None:
    assert news_client.post("/news/search", json={"query": ""}).status_code == 422
