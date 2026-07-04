"""Unit tests for the pure Qdrant/news helper functions (no live client)."""

import asyncio

import vector


def test_make_collection_name() -> None:
    assert vector._make_collection_name("economy", "positive") == "economy_positive"


def test_article_from_payload_flattens_nested_shape() -> None:
    payload = {
        "topic": "trade",
        "sentiment": "negative",
        "article": {
            "title": "Tariffs rise",
            "text": "x" * 5000,
            "url": "http://example.com",
            "published": "2026-01-01",
            "thread": {"site": "example.com"},
        },
    }
    article = vector._article_from_payload(payload, point_id=42, collection="trade_negative")

    assert article["id"] == "42"
    assert article["title"] == "Tariffs rise"
    assert len(article["text"]) == 2000  # truncated to 2000 chars.
    assert article["source"] == "example.com"
    assert article["topic"] == "trade"
    assert article["sentiment"] == "negative"
    assert article["collection"] == "trade_negative"


def test_article_from_payload_tolerates_missing_fields() -> None:
    article = vector._article_from_payload({}, point_id="abc", collection="c")
    assert article["id"] == "abc"
    assert article["title"] == ""
    assert article["source"] == ""


def _resolve(all_collections, topic, sentiment):
    return asyncio.run(vector._resolve_target_collections(all_collections, topic, sentiment))


def test_resolve_target_collections_topic_and_sentiment() -> None:
    cols = ["economy_positive", "economy_negative", "trade_positive", "actually_relevant_economy"]
    result = _resolve(cols, "economy", "positive")
    # Exact match + always-on curated source folded in.
    assert result == ["economy_positive", "actually_relevant_economy"]


def test_resolve_target_collections_topic_only() -> None:
    cols = ["economy_positive", "economy_negative", "world_bank_growth"]
    result = _resolve(cols, "economy", None)
    assert "economy_positive" in result
    assert "economy_negative" in result
    assert "world_bank_growth" in result  # always-on curated prefix.


def test_resolve_target_collections_no_filter_includes_all() -> None:
    cols = ["economy_positive", "world_bank_growth"]
    result = _resolve(cols, None, None)
    assert set(result) == set(cols)


def test_resolve_target_collections_dedupes() -> None:
    cols = ["world_bank_growth"]
    # world_bank_growth would match "no filter" AND the always-on prefix.
    result = _resolve(cols, None, None)
    assert result == ["world_bank_growth"]
