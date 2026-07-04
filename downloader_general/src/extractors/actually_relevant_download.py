"""Actually Relevant pipeline: page the curated-news API, embed, push to Qdrant.

Runs once per clean boot via :meth:`ActuallyRelevantDownloader.run`. The API
serves *curated analysis* (not full source bodies), so each Qdrant point embeds a
composed document built from the story's analysis fields (see
:func:`compose_story_text`). Stories are bucketed into one collection per macro
topic (``actually_relevant_<macro>``) by mapping each story's granular
``issue.slug`` up to its macro parent via the ``/api/issues`` taxonomy.

Embedding + upload reuse :class:`~src.core.qdrant_uploader.QdrantEmbeddingUploaderMixin`;
the payload shape matches the news pipeline so the app + agent read it unchanged.
"""

import logging
from pathlib import Path
from typing import Any

import httpx

from src.core.base_downloaders import BaseActuallyRelevantDownloader
from src.core.qdrant_uploader import QdrantEmbeddingUploaderMixin
from src.settings import load_settings
from src.utils.actually_relevant_client import (
    DEFAULT_BASE_URL,
    build_client,
    build_macro_map,
    fetch_issues,
    fetch_stories_page,
)
from src.utils.downloads import _call_with_retries, _download_config, log_progress

logger = logging.getLogger(__name__)

SITE_BASE_URL = "https://actuallyrelevant.news"


def compose_story_text(story: dict[str, Any]) -> str:
    """Compose the embed/display text from a story's curated analysis fields.

    Order: title, summary, relevance summary, the multi-paragraph relevance
    reasoning, caveats (antifactors), the pull quote, and the marketing blurb.
    Empty fields are dropped so the text stays clean.
    """
    parts: list[str] = []

    def add(value: Any, prefix: str = "") -> None:
        text = str(value or "").strip()
        if text:
            parts.append(f"{prefix}{text}" if prefix else text)

    add(story.get("title"))
    add(story.get("summary"))
    add(story.get("relevanceSummary"), prefix="Why it matters: ")
    add(story.get("relevanceReasons"))
    add(story.get("antifactors"), prefix="Caveats:\n")

    quote = str(story.get("quote") or "").strip()
    if quote:
        attribution = str(story.get("quoteAttribution") or "").strip()
        parts.append(f'"{quote}" — {attribution}' if attribution else f'"{quote}"')

    add(story.get("marketingBlurb"))
    return "\n\n".join(parts)


def build_entry(story: dict[str, Any], collection: str, macro_name: str) -> dict[str, Any]:
    """Build one Qdrant entry payload for a story (news-pipeline payload shape)."""
    issue = story.get("issue") or {}
    feed = story.get("feed") or {}
    site = feed.get("displayTitle") or feed.get("title") or story.get("sourceTitle") or ""
    published = str(story.get("datePublished") or "")
    slug = str(story.get("slug") or "")

    return {
        "topic": macro_name,
        "sentiment": "neutral",
        "date": published[:10],
        "source": "actually_relevant",
        "archive_name": collection,
        "article": {
            "title": story.get("title") or "",
            "text": compose_story_text(story),
            "url": story.get("sourceUrl") or "",
            "story_url": f"{SITE_BASE_URL}/stories/{slug}" if slug else "",
            "published": published,
            "author": story.get("quoteAttribution") or "",
            "language": "en",
            "categories": [issue.get("name", ""), site],
            "thread": {
                "site": site,
                "site_section": issue.get("name", ""),
                "country": "",
            },
            "summary": story.get("summary") or "",
            "relevanceSummary": story.get("relevanceSummary") or "",
            "emotionTag": story.get("emotionTag") or "",
            "relevance": story.get("relevance"),
            "issue_slug": issue.get("slug", ""),
            "id": story.get("id", ""),
        },
    }


class ActuallyRelevantDownloader(QdrantEmbeddingUploaderMixin, BaseActuallyRelevantDownloader):
    """Fetch every Actually Relevant story, bucket by macro topic, embed, upload."""

    def __init__(
        self,
        env_file: str | Path,
        download_config_path: str | Path,
        qdrant_host: str,
        qdrant_port: str,
        openai_base_url: str | None = None,
        openai_embedding_model: str = "openai/text-embedding-3-small",
        openai_token_limit: int = 8192,
        openai_model_dimensions: int = 1536,
    ) -> None:
        """Capture config; OpenAI/Qdrant/HTTP clients are built in ``_initialize_connections``."""
        self.env_path = Path(env_file)

        config = _download_config(download_config_path)
        self.base_url = str(config.get("base_url", DEFAULT_BASE_URL))
        self.page_size = int(config.get("page_size", 100))
        self.collection_by_macro: dict[str, str] = {
            str(topic["slug"]): str(topic["collection"])
            for topic in config.get("topics", [])
            if topic.get("slug") and topic.get("collection")
        }

        self.qdrant_host = qdrant_host
        self.qdrant_port = str(qdrant_port)
        self.openai_base_url = openai_base_url
        self.openai_embedding_model = openai_embedding_model
        self.embedding_token_limit = openai_token_limit
        self.openai_model_dimensions = openai_model_dimensions
        self.embedding_encoding = self._build_embedding_encoding()

        self._client: httpx.Client | None = None
        self._issues: list[dict[str, Any]] = []

    def _initialize_connections(self) -> bool:
        secrets = load_settings(self.env_path)
        if not self._connect_embedding_and_qdrant(secrets):
            return False

        self._client = build_client()
        issues = _call_with_retries(
            "actually_relevant.issues",
            lambda: fetch_issues(self._require_client(), self.base_url),
            retry_delay_seconds=5,
            max_retries=5,
        )
        if not issues:
            logger.error("Actually Relevant API probe failed; skipping source")
            return False
        self._issues = issues
        return True

    def _require_client(self) -> httpx.Client:
        if self._client is None:
            raise RuntimeError("HTTP client not initialised; call _initialize_connections first")
        return self._client

    def _fetch_all_stories(self) -> list[dict[str, Any]]:
        """Page through ``/api/stories`` and return every story record."""
        first = _call_with_retries(
            "actually_relevant.stories(page=1)",
            lambda: fetch_stories_page(self._require_client(), 1, self.page_size, self.base_url),
            retry_delay_seconds=5,
            max_retries=5,
        )
        if not first:
            logger.error("Could not fetch the first page of Actually Relevant stories")
            return []

        total_pages = int(first.get("totalPages", 1) or 1)
        stories: list[dict[str, Any]] = list(first.get("data", []) or [])
        logger.info(
            "Actually Relevant: %s stories across %d page(s)",
            first.get("total", "?"),
            total_pages,
        )

        for page in log_progress(
            range(2, total_pages + 1),
            label="Fetching Actually Relevant stories",
            total=max(0, total_pages - 1),
        ):
            payload = _call_with_retries(
                f"actually_relevant.stories(page={page})",
                lambda p=page: fetch_stories_page(
                    self._require_client(), p, self.page_size, self.base_url
                ),
                retry_delay_seconds=5,
                max_retries=5,
            )
            if payload:
                stories.extend(payload.get("data", []) or [])
        return stories

    def _build_metadata(
        self, stories: list[dict[str, Any]], macro_map: dict[str, tuple[str, str]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Bucket stories into per-macro-topic collections (empties allowed)."""
        parsed: dict[str, list[dict[str, Any]]] = {
            collection: [] for collection in self.collection_by_macro.values()
        }
        skipped = 0
        for story in stories:
            issue = story.get("issue") or {}
            slug = issue.get("slug")
            mapped = macro_map.get(slug) if slug else None
            if mapped is None:
                skipped += 1
                continue
            macro_slug, macro_name = mapped
            collection = self.collection_by_macro.get(macro_slug)
            if not collection:
                skipped += 1
                continue
            parsed[collection].append(build_entry(story, collection, macro_name))

        if skipped:
            logger.info("Skipped %d Actually Relevant stories with an unmapped issue slug", skipped)
        for collection, entries in parsed.items():
            logger.info("Actually Relevant %s: %d stories", collection, len(entries))
        return parsed

    def run(self) -> None:
        macro_map = build_macro_map(self._issues)
        stories = self._fetch_all_stories()
        parsed = self._build_metadata(stories, macro_map)
        self.upload_collections(parsed)
        if self._client is not None:
            self._client.close()
