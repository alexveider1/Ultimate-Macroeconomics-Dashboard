"""Unit tests for the pure Qdrant/news helper functions (no live client)."""

import asyncio

import vector


def test_make_collection_name() -> None:
    assert vector._make_collection_name("economy", "positive") == "economy_positive_webhose"


def test_make_collection_name_matches_writer_normalization() -> None:
    # Qdrant names are case-sensitive and the corpus is stored lowercased, so a raw
    # capitalized topic label must fold to the real (lowercase) collection name.
    assert vector._make_collection_name("Politics", "positive") == "politics_positive_webhose"
    # Comma/space handling + the _webhose source suffix must match downloader_general's
    # writer byte-for-byte (github_download.py:_collection_name_for): spaces -> "_",
    # commas -> " ", then a trailing "_webhose".
    assert (
        vector._make_collection_name("Economy, Business and Finance", "positive")
        == "economy _business_and_finance_positive_webhose"
    )


def test_resolve_target_collections_normalizes_raw_topic_label() -> None:
    # A caller passing the human topic label ("Politics") must still resolve to the
    # stored lowercase collection — this is the regression the bare f-string missed.
    cols = ["politics_positive_webhose", "politics_negative_webhose", "world_bank"]
    result = _resolve(cols, "Politics", "positive")
    assert result == ["politics_positive_webhose", "world_bank"]


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
    cols = [
        "economy_positive_webhose",
        "economy_negative_webhose",
        "trade_positive_webhose",
        "actually_relevant",
    ]
    result = _resolve(cols, "economy", "positive")
    # Exact match + always-on curated source folded in.
    assert result == ["economy_positive_webhose", "actually_relevant"]


def test_resolve_target_collections_sentiment_only() -> None:
    cols = ["economy_positive_webhose", "economy_negative_webhose", "world_bank"]
    result = _resolve(cols, None, "positive")
    # Only the matching-sentiment webhose collection + the always-on curated source.
    assert result == ["economy_positive_webhose", "world_bank"]


def test_resolve_target_collections_topic_only() -> None:
    cols = ["economy_positive_webhose", "economy_negative_webhose", "world_bank"]
    result = _resolve(cols, "economy", None)
    assert "economy_positive_webhose" in result
    assert "economy_negative_webhose" in result
    assert "world_bank" in result  # always-on curated prefix.


def test_resolve_target_collections_no_filter_includes_all() -> None:
    cols = ["economy_positive", "world_bank"]
    result = _resolve(cols, None, None)
    assert set(result) == set(cols)


def test_resolve_target_collections_dedupes() -> None:
    cols = ["world_bank"]
    # world_bank would match "no filter" AND the always-on prefix.
    result = _resolve(cols, None, None)
    assert result == ["world_bank"]
