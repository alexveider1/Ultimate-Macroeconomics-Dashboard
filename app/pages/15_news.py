import json
import re
from threading import RLock
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

from core.app_logging import log_page_render
from core.qdrant_client import (
    find_nearest_embeddings,
    get_point,
    is_qdrant_available,
    list_collections,
    scroll_collection,
)


PAGE_TITLE = "News Explorer"
DEFAULT_NEAREST_COUNT = 5
WORD_CLOUD_WIDTH = 1600
WORD_CLOUD_HEIGHT = 800
WORD_CLOUD_MAX_WORDS = 200
MARKDOWN_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
SETEXT_HEADER_RE = re.compile(r"^\s*[=-]{3,}\s*$")
HTML_HEADING_TAG_RE = re.compile(r"</?h[1-6][^>]*>", flags=re.IGNORECASE)
MATPLOTLIB_LOCK = RLock()

METADATA_FIELDS: list = [
    "published",
    "title",
    "thread/country",
    "thread/site_section",
    "url",
    "author",
    "categories",
    "language",
]


def _as_non_empty_string(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _extract_article(payload: dict[str, Any]) -> dict[str, Any]:
    article = payload.get("article")
    if isinstance(article, dict):
        return article
    return {}


def _sanitize_article_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = HTML_HEADING_TAG_RE.sub("", raw_line).rstrip()
        if not line.strip():
            cleaned_lines.append("")
            continue
        if SETEXT_HEADER_RE.match(line):
            continue
        line = MARKDOWN_HEADER_RE.sub("", line)
        cleaned_lines.append(line)

    sanitized = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", sanitized).strip()


def _is_non_empty_metadata_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _get_nested_metadata_value(source: dict[str, Any], field_name: str) -> Any:
    if field_name in source:
        return source.get(field_name)

    current: Any = source
    for key in field_name.split("/"):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current.get(key)

    return current


def _find_metadata_value(
    article: dict[str, Any], payload: dict[str, Any], field_name: str
) -> Any:
    article_value = _get_nested_metadata_value(article, field_name)
    if _is_non_empty_metadata_value(article_value):
        return article_value

    payload_value = _get_nested_metadata_value(payload, field_name)
    if _is_non_empty_metadata_value(payload_value):
        return payload_value

    return None


def _extract_title(payload: dict[str, Any], point_id: str) -> str:
    article = _extract_article(payload)
    candidates = [
        article.get("title"),
        article.get("headline"),
        article.get("name"),
        payload.get("title"),
        payload.get("headline"),
        payload.get("archive_name"),
    ]
    for candidate in candidates:
        text = _as_non_empty_string(candidate)
        if text:
            return text
    return f"Untitled news ({point_id})"


def _extract_thread_text(payload: dict[str, Any]) -> str:
    thread = payload.get("thread")
    if not isinstance(thread, dict):
        return ""

    text = _as_non_empty_string(thread.get("text"))
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def _extract_text(payload: dict[str, Any]) -> str:
    article = _extract_article(payload)
    candidates = [
        article.get("text"),
        article.get("content"),
        article.get("body"),
        payload.get("text"),
        payload.get("content"),
    ]
    for candidate in candidates:
        text = _as_non_empty_string(candidate)
        if text:
            sanitized = _sanitize_article_text(text)
            if sanitized:
                return sanitized
    return "No article text is available for this record."


def _extract_source_url(payload: dict[str, Any]) -> str:
    article = _extract_article(payload)
    candidates = [
        article.get("url"),
        article.get("link"),
        article.get("source_url"),
        payload.get("url"),
        payload.get("source_url"),
    ]
    for candidate in candidates:
        text = _as_non_empty_string(candidate)
        if text:
            return text
    return ""


def _build_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    article = _extract_article(payload)
    filtered_metadata: dict[str, Any] = {}
    for field_name in METADATA_FIELDS:
        value = _find_metadata_value(
            article=article, payload=payload, field_name=field_name
        )
        if _is_non_empty_metadata_value(value):
            filtered_metadata[field_name] = value
    return filtered_metadata


def _extract_query_vector(vector: Any) -> list[float] | None:
    if isinstance(vector, list):
        return vector
    if isinstance(vector, tuple):
        return list(vector)
    if isinstance(vector, dict):
        for candidate in vector.values():
            if isinstance(candidate, list):
                return candidate
            if isinstance(candidate, tuple):
                return list(candidate)
    return None


@st.cache_data(show_spinner=False, ttl=300)
def _load_topic_news_bundle(collection_name: str) -> dict[str, Any]:
    records = scroll_collection(collection_name=collection_name)
    items: list[dict[str, Any]] = []
    thread_text_parts: list[str] = []

    for record in records:
        payload = record.payload if isinstance(record.payload, dict) else {}

        point_id = str(record.id)
        title = _extract_title(payload, point_id)
        article_text = _extract_text(payload)
        date_text = _as_non_empty_string(payload.get("date"))
        if date_text:
            label = f"{title} | {date_text}"
        else:
            label = title

        if (
            article_text
            and article_text != "No article text is available for this record."
        ):
            thread_text_parts.append(re.sub(r"\s+", " ", article_text).strip())

        items.append(
            {
                "id": point_id,
                "title": title,
                "label": label,
                "text": article_text,
                "metadata": _build_metadata(payload),
                "source_url": _extract_source_url(payload),
            }
        )

    return {
        "items": items,
        "wordcloud_text": " ".join(thread_text_parts),
        "wordcloud_source_count": len(thread_text_parts),
    }


@st.cache_data(show_spinner=False, ttl=300)
def _build_wordcloud_image(topic_corpus: str) -> Any:
    normalized_corpus = re.sub(r"\s+", " ", topic_corpus).strip()
    if not normalized_corpus:
        return None

    try:
        return (
            WordCloud(
                width=WORD_CLOUD_WIDTH,
                height=WORD_CLOUD_HEIGHT,
                background_color="white",
                collocations=False,
                max_words=WORD_CLOUD_MAX_WORDS,
            )
            .generate(normalized_corpus)
            .to_array()
        )
    except ValueError:
        return None


@st.fragment
def _render_topic_wordcloud(topic_corpus: str, source_count: int) -> None:
    with st.container(border=True):
        st.markdown("### Topic Word Cloud")
        st.caption(f"Included records with text: {source_count}.")

        image_array = _build_wordcloud_image(topic_corpus)
        if image_array is None:
            st.info("No article texts were found for this topic.")
            return

        with MATPLOTLIB_LOCK:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
            ax.imshow(image_array, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)


def _show_selected_news(selected_item: dict[str, Any], show_metadata: bool) -> None:
    st.subheader(selected_item["title"])

    source_url = selected_item.get("source_url")
    if source_url:
        st.link_button("Open original source", source_url)

    st.markdown(selected_item["text"])

    if show_metadata:
        st.markdown("### Metadata")
        st.code(
            json.dumps(selected_item.get("metadata", {}), ensure_ascii=True, indent=2),
            language="json",
        )


def _render_nearest_news(
    collection_name: str,
    selected_item: dict[str, Any],
    id_to_item: dict[str, dict[str, Any]],
    select_state_key: str,
) -> None:
    if not st.button(
        f"Find {DEFAULT_NEAREST_COUNT} nearest news",
        use_container_width=True,
        type="primary",
    ):
        return

    point = get_point(
        collection_name=collection_name,
        point_id=selected_item["id"],
        with_vector=True,
    )
    if not point or not point.vector:
        st.warning("Unable to fetch the selected vector. Try another news item.")
        return

    query_vector = _extract_query_vector(point.vector)
    if not query_vector:
        st.warning("Selected news has an unsupported vector format.")
        return

    nearest_hits = find_nearest_embeddings(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=DEFAULT_NEAREST_COUNT,
        exclude_point_id=selected_item["id"],
    )
    if not nearest_hits:
        st.info("No nearest news were found for this item.")
        return

    st.markdown("### Nearest News")
    for idx, hit in enumerate(nearest_hits, start=1):
        hit_id = str(hit.id)
        if hit_id in id_to_item:
            title = id_to_item[hit_id]["title"]
        else:
            hit_payload = hit.payload if isinstance(hit.payload, dict) else {}
            title = _extract_title(hit_payload, hit_id)

        score_text = f"{hit.score:.4f}" if hit.score is not None else "N/A"
        label = f"{idx}. {title} (score: {score_text})"
        if st.button(
            label, key=f"nearest_{collection_name}_{hit_id}", use_container_width=True
        ):
            st.session_state[select_state_key] = hit_id
            st.rerun()


def render_news_page() -> None:
    log_page_render(PAGE_TITLE)
    st.title(PAGE_TITLE)
    st.caption(
        "Pick a topic (Qdrant collection), find a news item via searchable selector, inspect text/metadata, and explore nearest neighbors."
    )

    if not is_qdrant_available():
        st.error(
            "Qdrant client is not available. Check qdrant.url/host/port in config.yaml and QDRANT__API_KEY/QDRANT_API_KEY/QDRANT__SERVICE__API_KEY in .env."
        )
        return

    topic_options = list_collections()
    if not topic_options:
        st.info("No Qdrant collections were found.")
        return

    selected_topic = st.selectbox(
        "News topic",
        options=topic_options,
        help="Each Qdrant collection is treated as a topic.",
    )

    topic_news_bundle = _load_topic_news_bundle(selected_topic)
    _render_topic_wordcloud(
        topic_corpus=str(topic_news_bundle.get("wordcloud_text", "")),
        source_count=int(topic_news_bundle.get("wordcloud_source_count", 0)),
    )

    news_items = topic_news_bundle.get("items", [])
    if not news_items:
        st.warning("No news found in the selected topic.")
        return

    id_to_item = {item["id"]: item for item in news_items}
    ordered_ids = [item["id"] for item in news_items]

    select_state_key = f"news_selected_{selected_topic}"
    if (
        select_state_key not in st.session_state
        or st.session_state[select_state_key] not in id_to_item
    ):
        st.session_state[select_state_key] = ordered_ids[0]

    st.markdown("### News Finder")
    st.caption(
        "Starts with Qdrant ordering. Type to filter suggestions in the selector."
    )
    selected_news_id = st.selectbox(
        "Find and select a piece of news",
        options=ordered_ids,
        key=select_state_key,
        format_func=lambda point_id: id_to_item[point_id]["label"],
    )

    show_metadata = st.toggle("Show metadata for selected news", value=False)
    selected_item = id_to_item[selected_news_id]
    _show_selected_news(selected_item=selected_item, show_metadata=show_metadata)

    st.divider()
    _render_nearest_news(
        collection_name=selected_topic,
        selected_item=selected_item,
        id_to_item=id_to_item,
        select_state_key=select_state_key,
    )


render_news_page()
