"""Tests for the incremental Qdrant helpers (dedup + append-not-recreate).

A small fake Qdrant client stands in for the real one; crucially it has **no**
``recreate_collection`` method, so any accidental full-reload path would surface
as an ``AttributeError`` rather than silently wiping a collection.
"""

from typing import Any

from src.core.qdrant_uploader import (
    QdrantEmbeddingUploaderMixin,
    ensure_collection,
    existing_payload_values,
)


class _FakePoint:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload


class _FakeQdrant:
    """Minimal Qdrant stand-in: collection_exists / scroll / create_collection / upsert."""

    def __init__(self, points_by_collection: dict[str, list[dict[str, Any]]]) -> None:
        self._points = points_by_collection
        self.created: list[str] = []
        self.upserts: list[tuple[str, list]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._points

    def scroll(
        self,
        collection_name: str,
        limit: int,
        with_payload: Any,
        with_vectors: bool,
        offset: Any,
    ) -> tuple[list[_FakePoint], None]:
        points = [_FakePoint(p) for p in self._points.get(collection_name, [])]
        return points, None  # single page

    def create_collection(
        self, collection_name: str, vectors_config: Any, on_disk_payload: bool
    ) -> None:
        self.created.append(collection_name)
        self._points.setdefault(collection_name, [])

    def upsert(self, collection_name: str, points: list) -> None:
        self.upserts.append((collection_name, points))


def test_existing_payload_values_top_level_key() -> None:
    client = _FakeQdrant({"c": [{"archive_name": "a1"}, {"archive_name": "a2"}]})
    assert existing_payload_values(client, "c", "archive_name") == {"a1", "a2"}


def test_existing_payload_values_nested_key() -> None:
    client = _FakeQdrant({"c": [{"article": {"id": "x"}}, {"article": {"id": "y"}}]})
    assert existing_payload_values(client, "c", "article.id") == {"x", "y"}


def test_existing_payload_values_missing_collection_is_empty() -> None:
    client = _FakeQdrant({})
    assert existing_payload_values(client, "nope", "archive_name") == set()


def test_ensure_collection_creates_once() -> None:
    client = _FakeQdrant({})
    ensure_collection(client, "new", 1536)
    ensure_collection(client, "new", 1536)  # idempotent
    assert client.created == ["new"]


def test_upsert_collections_ensures_and_skips_empty() -> None:
    mixin = QdrantEmbeddingUploaderMixin.__new__(QdrantEmbeddingUploaderMixin)
    mixin.qdrant_client = _FakeQdrant({})
    mixin.openai_model_dimensions = 1536

    embedded: list[tuple[str, int]] = []
    # Stub the embed+upsert step so no OpenAI call is needed.
    mixin._embed_and_upsert = lambda name, entries: embedded.append((name, len(entries)))  # type: ignore[method-assign]

    mixin.upsert_collections({"c1": [{"article": {"text": "t"}}], "c2": []})

    # c1 gets ensured + embedded; c2 (empty) is skipped entirely.
    assert embedded == [("c1", 1)]
    assert mixin.qdrant_client.created == ["c1"]
