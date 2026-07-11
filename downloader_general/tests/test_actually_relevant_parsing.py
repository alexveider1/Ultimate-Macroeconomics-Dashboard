"""Tests for the Actually Relevant downloader: taxonomy, text, payload, bucketing."""

from __future__ import annotations

from src.extractors.actually_relevant_download import (
    ActuallyRelevantDownloader,
    build_entry,
    compose_story_text,
)
from src.utils.actually_relevant_client import build_macro_map

SAMPLE_ISSUES = [
    {
        "slug": "existential-threats",
        "name": "Existential Threats",
        "children": [
            {"slug": "nuclear-war", "name": "(Nuclear) War"},
            {"slug": "pandemics", "name": "Pandemics"},
        ],
    },
    {"slug": "planet-climate", "name": "Planet & Climate", "children": []},
]


def test_build_macro_map_maps_children_to_parent() -> None:
    mapping = build_macro_map(SAMPLE_ISSUES)
    assert mapping["nuclear-war"] == ("existential-threats", "Existential Threats")
    assert mapping["pandemics"] == ("existential-threats", "Existential Threats")
    # A top-level slug maps to itself.
    assert mapping["existential-threats"] == ("existential-threats", "Existential Threats")
    assert mapping["planet-climate"] == ("planet-climate", "Planet & Climate")


def test_compose_story_text_orders_fields_and_skips_empty() -> None:
    story = {
        "title": "TITLE",
        "summary": "SUMMARY",
        "relevanceSummary": "RELSUM",
        "relevanceReasons": "REASONS",
        "antifactors": "CAVEAT",
        "quote": "QUOTE",
        "quoteAttribution": "PERSON",
        "marketingBlurb": "BLURB",
    }
    text = compose_story_text(story)
    order = [
        text.index("TITLE"),
        text.index("SUMMARY"),
        text.index("RELSUM"),
        text.index("REASONS"),
        text.index("CAVEAT"),
        text.index("QUOTE"),
        text.index("BLURB"),
    ]
    assert order == sorted(order)
    assert "Why it matters: RELSUM" in text
    assert "Caveats:" in text
    assert '"QUOTE" — PERSON' in text

    # Empty fields are dropped entirely.
    assert compose_story_text({"title": "Only"}) == "Only"


def test_build_entry_has_news_payload_shape() -> None:
    story = {
        "title": "Copper story",
        "summary": "sum",
        "slug": "copper-story",
        "sourceUrl": "https://news.example.com/a",
        "datePublished": "2026-07-04T02:15:00.495Z",
        "quoteAttribution": "Analyst",
        "issue": {"name": "Planet & Climate", "slug": "planet-climate"},
        "feed": {"title": "Mongabay", "displayTitle": "Mongabay"},
        "id": "abc-123",
        "emotionTag": "frustrating",
    }
    entry = build_entry(story, "actually_relevant", "Planet & Climate")

    # The macro topic is kept in the payload `topic`; the collection is the
    # single source-named `actually_relevant` (its `archive_name`).
    assert entry["topic"] == "Planet & Climate"
    assert entry["archive_name"] == "actually_relevant"
    assert entry["sentiment"] == "neutral"
    assert entry["date"] == "2026-07-04"
    assert entry["source"] == "actually_relevant"

    article = entry["article"]
    assert article["title"] == "Copper story"
    assert article["url"] == "https://news.example.com/a"
    assert article["story_url"].endswith("/stories/copper-story")
    assert article["thread"]["site"] == "Mongabay"
    assert article["issue_slug"] == "planet-climate"
    assert "sum" in article["text"]


def test_build_metadata_consolidates_into_single_collection() -> None:
    downloader = ActuallyRelevantDownloader.__new__(ActuallyRelevantDownloader)
    # Every macro topic points at the single consolidated collection.
    downloader.collection_by_macro = {
        "existential-threats": "actually_relevant",
        "planet-climate": "actually_relevant",
    }
    macro_map = build_macro_map(SAMPLE_ISSUES)
    stories = [
        {"title": "war", "issue": {"slug": "nuclear-war"}, "datePublished": "2026-01-01T00:00:00Z"},
        {
            "title": "climate",
            "issue": {"slug": "planet-climate"},
            "datePublished": "2026-01-02T00:00:00Z",
        },
        {"title": "unknown", "issue": {"slug": "not-a-real-slug"}},
    ]

    parsed = downloader._build_metadata(stories, macro_map)

    # All macro topics collapse into the single actually_relevant collection...
    assert set(parsed.keys()) == {"actually_relevant"}
    # ...both mapped stories land there (the unmapped one is skipped, not errored).
    assert len(parsed["actually_relevant"]) == 2
    # The macro topic is preserved per-point in the payload `topic` field.
    topics = {entry["topic"] for entry in parsed["actually_relevant"]}
    assert topics == {"Existential Threats", "Planet & Climate"}
