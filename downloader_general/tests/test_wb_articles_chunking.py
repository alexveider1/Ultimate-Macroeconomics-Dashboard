"""Tests for the shared news chunker + the World Bank documents doc-entry builder.

``clean_text`` / ``chunk_text`` / ``chunk_entries`` now live in the shared
``src.core.qdrant_uploader`` (every news source funnels through them); the WB
documents extractor only builds one entry per document.
"""

from __future__ import annotations

from src.core.qdrant_uploader import chunk_entries, chunk_text, clean_text
from src.extractors.world_bank_articles_download import build_doc_entry, dedup_with_text


class FakeEncoding:
    """Char-as-token codec (exact inverse) so chunk boundaries are deterministic."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def test_chunk_text_splits_with_overlap() -> None:
    enc = FakeEncoding()
    text = "abcdefghijklmnopqrstuvwxy"  # 25 "tokens"
    chunks = chunk_text(text, enc, chunk_size=10, overlap=3)

    assert len(chunks) == 4
    # No chunk exceeds the window size.
    assert all(len(c) <= 10 for c in chunks)
    # Consecutive windows advance by (size - overlap) = 7 and share `overlap` tokens.
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "hijklmnopq"
    assert chunks[0][-3:] == chunks[1][:3]


def test_chunk_text_edge_cases() -> None:
    enc = FakeEncoding()
    assert chunk_text("", enc, chunk_size=10, overlap=3) == []
    # Text within one window is returned whole.
    assert chunk_text("short", enc, chunk_size=10, overlap=3) == ["short"]


def test_chunk_entries_expands_and_labels_parts() -> None:
    enc = FakeEncoding()
    entries = [
        {
            "topic": "trade",
            "archive_name": "trade_positive_webhose",
            "article": {"title": "Long Story", "text": "abcdefghijklmnopqrstuvwxy"},
        }
    ]
    chunked = chunk_entries(entries, enc, chunk_size=10, overlap=3)

    # 25 "tokens" -> 4 chunks, each its own entry sharing the top-level payload.
    assert len(chunked) == 4
    assert all(e["archive_name"] == "trade_positive_webhose" for e in chunked)
    assert [e["article"]["chunk_index"] for e in chunked] == [0, 1, 2, 3]
    assert all(e["article"]["total_chunks"] == 4 for e in chunked)
    # Multi-chunk articles get a "(part i/N)" title suffix.
    assert chunked[0]["article"]["title"] == "Long Story (part 1/4)"
    assert chunked[3]["article"]["title"] == "Long Story (part 4/4)"
    # The original entry is not mutated.
    assert entries[0]["article"]["text"] == "abcdefghijklmnopqrstuvwxy"


def test_chunk_entries_single_chunk_keeps_title() -> None:
    enc = FakeEncoding()
    entries = [{"article": {"title": "Short", "text": "hello"}}]
    chunked = chunk_entries(entries, enc, chunk_size=10, overlap=3)

    assert len(chunked) == 1
    assert chunked[0]["article"]["title"] == "Short"  # no "(part i/N)" suffix
    assert chunked[0]["article"]["chunk_index"] == 0
    assert chunked[0]["article"]["total_chunks"] == 1


def test_dedup_with_text_skips_missing_txturl_and_duplicates() -> None:
    records = [
        {"id": "D1", "txturl": "u1"},
        {"id": "D1", "txturl": "u1b"},  # duplicate id
        {"id": "D2"},  # no txturl
        {"id": "D3", "txturl": "u3"},
        {"txturl": "u4"},  # no id
    ]
    docs = dedup_with_text(records)
    assert [d["id"] for d in docs] == ["D1", "D3"]


def test_build_doc_entry_payload_shape() -> None:
    doc = {
        "id": "D9",
        "display_title": "Inflation Report",
        "docty": "Policy Research Working Paper",
        "count": "World",
        "docdt": "2020-01-15T00:00:00Z",
        "url": "https://documents.worldbank.org/9",
        "pdfurl": "https://documents.worldbank.org/9.pdf",
        "lang": "English",
    }
    entry = build_doc_entry(doc, "full document body", "inflation", "world_bank_inflation")

    assert entry["topic"] == "inflation"
    assert entry["sentiment"] == "neutral"
    assert entry["date"] == "2020-01-15"
    assert entry["source"] == "world_bank"

    article = entry["article"]
    # One entry per document holds the full text + the plain display title; the
    # shared chunker adds the "(part i/N)" suffix + chunk metadata at embed time.
    assert article["title"] == "Inflation Report"
    assert article["text"] == "full document body"
    assert article["thread"] == {
        "site": "World Bank",
        "site_section": "Policy Research Working Paper",
        "country": "World",
    }
    assert article["doc_id"] == "D9"
    assert article["doc_title"] == "Inflation Report"


def test_clean_text_drops_form_feeds_and_collapses_blank_lines() -> None:
    cleaned = clean_text("line1\x0cline2\n\n\n\nline3   \n")
    assert "\x0c" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("line1")
    assert cleaned.endswith("line3")
