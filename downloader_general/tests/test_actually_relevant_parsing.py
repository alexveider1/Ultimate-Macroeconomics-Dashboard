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
    entry = build_entry(story, "actually_relevant_planet_climate", "Planet & Climate")

    assert entry["topic"] == "Planet & Climate"
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


def test_build_metadata_buckets_stories_by_macro_parent() -> None:
    downloader = ActuallyRelevantDownloader.__new__(ActuallyRelevantDownloader)
    downloader.collection_by_macro = {
        "existential-threats": "actually_relevant_existential_threats",
        "planet-climate": "actually_relevant_planet_climate",
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

    # Child slug bucketed into the macro-parent collection.
    assert len(parsed["actually_relevant_existential_threats"]) == 1
    assert len(parsed["actually_relevant_planet_climate"]) == 1
    # Every configured collection is present (unmapped story skipped, not errored).
    assert set(parsed.keys()) == set(downloader.collection_by_macro.values())
