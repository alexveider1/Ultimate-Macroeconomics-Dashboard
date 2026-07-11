"""News explorer — browse the Qdrant news corpus by topic.

Each Qdrant collection is one topic; the page lists collections, lets
the user pick an article, renders the cleaned title/body/metadata,
shows a word-cloud built from every article in the topic, and finds
nearest-neighbour articles via the article's embedding.
"""

import json
import re
from threading import RLock
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from core.api_client import cluster_dataframe
from core.app_logging import log_page_render
from core.plotting import apply_plotly_theme
from core.qdrant_client import (
    find_nearest_embeddings,
    get_point,
    is_qdrant_available,
    list_collections,
    scroll_collection,
)
from core.theming import get_color, get_colorway

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
    """Coerce ``value`` to a stripped string (empty string when ``None``)."""
    text = str(value or "").strip()
    return text


def _extract_article(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the nested ``article`` dict from a Qdrant point payload."""
    article = payload.get("article")
    if isinstance(article, dict):
        return article
    return {}


def _sanitize_article_text(text: str) -> str:
    """Strip markdown headers / HTML heading tags and collapse blank lines."""
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
    """Return ``True`` when ``value`` is a non-empty scalar or non-empty container."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _get_nested_metadata_value(source: dict[str, Any], field_name: str) -> Any:
    """Resolve ``"thread/country"``-style paths against a nested dict."""
    if field_name in source:
        return source.get(field_name)

    current: Any = source
    for key in field_name.split("/"):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current.get(key)

    return current


def _find_metadata_value(article: dict[str, Any], payload: dict[str, Any], field_name: str) -> Any:
    """Look up ``field_name`` in the article first, then fall back to the payload."""
    article_value = _get_nested_metadata_value(article, field_name)
    if _is_non_empty_metadata_value(article_value):
        return article_value

    payload_value = _get_nested_metadata_value(payload, field_name)
    if _is_non_empty_metadata_value(payload_value):
        return payload_value

    return None


def _extract_title(payload: dict[str, Any], point_id: str) -> str:
    """Return the first non-empty title candidate or ``"Untitled news (id)"`` as a fallback."""
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
    """Return the article-thread body collapsed onto a single line."""
    thread = payload.get("thread")
    if not isinstance(thread, dict):
        return ""

    text = _as_non_empty_string(thread.get("text"))
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def _extract_text(payload: dict[str, Any]) -> str:
    """Return the sanitised article body (text/content/body), or a placeholder when missing."""
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
    """Return the first non-empty article URL candidate (or ``""``)."""
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
    """Return the subset of :data:`METADATA_FIELDS` that are populated for this article."""
    article = _extract_article(payload)
    filtered_metadata: dict[str, Any] = {}
    for field_name in METADATA_FIELDS:
        value = _find_metadata_value(article=article, payload=payload, field_name=field_name)
        if _is_non_empty_metadata_value(value):
            filtered_metadata[field_name] = value
    return filtered_metadata


def _extract_query_vector(vector: Any) -> list[float] | None:
    """Normalise Qdrant's various ``vector`` shapes into a single ``list[float]``."""
    if isinstance(vector, list):
        return [float(v) for v in vector]
    if isinstance(vector, tuple):
        return [float(v) for v in vector]
    if isinstance(vector, dict):
        for candidate in vector.values():
            if isinstance(candidate, list):
                return [float(v) for v in candidate]
            if isinstance(candidate, tuple):
                return [float(v) for v in candidate]
    return None


@st.cache_data(show_spinner=False, ttl=300)
def _load_topic_news_bundle(collection_name: str) -> dict[str, Any]:
    """Scroll every point in a topic and return display items + word-cloud corpus."""
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

        if article_text and article_text != "No article text is available for this record.":
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
    """Render the topic-corpus word cloud as a numpy array; return ``None`` when empty."""
    normalized_corpus = re.sub(r"\s+", " ", topic_corpus).strip()
    if not normalized_corpus:
        return None

    try:
        return (
            WordCloud(
                width=WORD_CLOUD_WIDTH,
                height=WORD_CLOUD_HEIGHT,
                background_color=get_color("wordcloud_background"),
                colormap=get_color("wordcloud_colormap"),
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
    """Render the topic word-cloud panel; rerenders independently as a fragment."""
    with st.container(border=True):
        st.markdown("### Topic Word Cloud")
        st.caption(f"Included records with text: {source_count}.")

        image_array = _build_wordcloud_image(topic_corpus)
        if image_array is None:
            st.info("No article texts were found for this topic.")
            return

        background = get_color("wordcloud_background")
        with MATPLOTLIB_LOCK:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
            fig.patch.set_facecolor(background)
            ax.set_facecolor(background)
            ax.imshow(image_array, interpolation="bilinear")
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig, clear_figure=True, width="stretch")
            plt.close(fig)


def _show_selected_news(selected_item: dict[str, Any], show_metadata: bool) -> None:
    """Render the selected article's title, source-link button, body, and optional metadata JSON."""
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


@st.cache_data(show_spinner=False, ttl=300)
def _load_topic_vectors(collection_name: str, max_points: int) -> dict[str, Any]:
    """Scroll the topic with vectors and return aligned id/title/vector lists.

    Sampled down to ``max_points`` with a fixed seed so the embedding map
    stays stable across reruns. Points without a recoverable vector are
    dropped silently — see :func:`_extract_query_vector`.
    """
    records = scroll_collection(collection_name=collection_name, with_vectors=True)
    ids: list[str] = []
    titles: list[str] = []
    vectors: list[list[float]] = []
    for record in records:
        vec = _extract_query_vector(getattr(record, "vector", None))
        if not vec:
            continue
        payload = record.payload if isinstance(record.payload, dict) else {}
        ids.append(str(record.id))
        titles.append(_extract_title(payload, str(record.id)))
        vectors.append(vec)

    if max_points > 0 and len(ids) > max_points:
        rng = np.random.default_rng(seed=42)
        sample_idx = np.sort(rng.choice(len(ids), size=max_points, replace=False))
        ids = [ids[i] for i in sample_idx]
        titles = [titles[i] for i in sample_idx]
        vectors = [vectors[i] for i in sample_idx]

    return {"ids": ids, "titles": titles, "vectors": vectors}


def _build_embedding_scatter(
    plot_rows: list[dict[str, Any]],
    viz_cols: list[str],
    selected_id: str,
    output_dim: int,
) -> go.Figure:
    """Render a 2D or 3D scatter coloured by cluster, with the selected article highlighted."""
    palette = get_colorway()
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for row in plot_rows:
        cluster_label = str(row.get("cluster", "unassigned"))
        by_cluster.setdefault(cluster_label, []).append(row)

    fig = go.Figure()
    is_3d = output_dim == 3
    x_col, y_col = viz_cols[0], viz_cols[1]
    z_col = viz_cols[2] if is_3d and len(viz_cols) >= 3 else None

    for idx, (cluster_label, cluster_rows) in enumerate(sorted(by_cluster.items())):
        colour = palette[idx % len(palette)]
        xs = [row[x_col] for row in cluster_rows]
        ys = [row[y_col] for row in cluster_rows]
        zs = [row[z_col] for row in cluster_rows] if z_col else None
        titles = [str(row.get("__title", row.get("__article_id", ""))) for row in cluster_rows]
        hovertemplate = "<b>%{text}</b><br>cluster=" + cluster_label + "<extra></extra>"
        trace_kwargs: dict[str, Any] = {
            "mode": "markers",
            "name": f"cluster {cluster_label}",
            "marker": dict(size=6, color=colour, opacity=0.85),
            "text": titles,
            "hovertemplate": hovertemplate,
        }
        if z_col:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, **trace_kwargs))
        else:
            fig.add_trace(go.Scatter(x=xs, y=ys, **trace_kwargs))

    selected_row = next(
        (row for row in plot_rows if str(row.get("__article_id")) == selected_id),
        None,
    )
    if selected_row is not None:
        marker_kwargs: dict[str, Any] = dict(
            size=14,
            color=get_color("selected_marker"),
            symbol="diamond" if z_col else "star",
            line=dict(width=1, color="black"),
        )
        if z_col:
            fig.add_trace(
                go.Scatter3d(
                    x=[selected_row[x_col]],
                    y=[selected_row[y_col]],
                    z=[selected_row[z_col]],
                    mode="markers",
                    name="selected",
                    marker=marker_kwargs,
                    text=[str(selected_row.get("__title", ""))],
                    hovertemplate="<b>Selected: %{text}</b><extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[selected_row[x_col]],
                    y=[selected_row[y_col]],
                    mode="markers",
                    name="selected",
                    marker=marker_kwargs,
                    text=[str(selected_row.get("__title", ""))],
                    hovertemplate="<b>Selected: %{text}</b><extra></extra>",
                )
            )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return apply_plotly_theme(fig)


@st.fragment
def _render_embedding_map(collection_name: str, selected_id: str) -> None:
    """Render the topic-wide embedding scatter, projected via the clustering service.

    Controls live inside an ``st.form`` so the projection only fires when the
    "Run" submit button is pressed; the result is cached in session state and
    only the selected-article highlight refreshes when the user picks a
    different article elsewhere on the page.
    """
    with st.container(border=True):
        st.markdown("### Embedding Map")
        st.caption(
            "2D/3D projection of every article in the topic, clustered and reduced "
            "via the clustering service. The selected article is highlighted."
        )

        cache_key = f"emb_map_result_{collection_name}"

        with st.form(key=f"emb_map_form_{collection_name}", border=False):
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                method = st.selectbox(
                    "Cluster method",
                    ["kmeans", "dbscan", "hdbscan", "hierarchical"],
                    key=f"emb_map_method_{collection_name}",
                )
            with col_b:
                reduction = st.selectbox(
                    "Reducer",
                    ["tsne", "umap", "pca"],
                    key=f"emb_map_reducer_{collection_name}",
                )
            with col_c:
                output_dim = st.selectbox(
                    "Dim",
                    [2, 3],
                    key=f"emb_map_dim_{collection_name}",
                )
            with col_d:
                max_points = st.slider(
                    "Max points",
                    min_value=50,
                    max_value=1000,
                    value=300,
                    step=50,
                    key=f"emb_map_max_{collection_name}",
                )

            k = st.slider(
                "k (kmeans, ignored otherwise)",
                min_value=2,
                max_value=12,
                value=4,
                key=f"emb_map_k_{collection_name}",
            )

            run_clicked = st.form_submit_button(
                "Run embedding map",
                type="primary",
                width="stretch",
            )

        if run_clicked:
            bundle = _load_topic_vectors(collection_name, int(max_points))
            ids = bundle["ids"]
            if not ids:
                st.info("No embeddings available for this topic.")
                return
            if len(ids) < 4:
                st.info("Need at least 4 articles to project — try a richer topic.")
                return

            vectors = bundle["vectors"]
            titles = bundle["titles"]
            embedding_dim = len(vectors[0])
            feature_cols = [f"f{i}" for i in range(embedding_dim)]
            rows: list[dict[str, Any]] = []
            for pid, title, vec in zip(ids, titles, vectors):
                row: dict[str, Any] = {"__article_id": pid, "__title": title}
                row.update({fc: float(v) for fc, v in zip(feature_cols, vec)})
                rows.append(row)

            with st.spinner("Projecting embeddings via the clustering service..."):
                try:
                    response = cluster_dataframe(
                        dataframe=rows,
                        method=method,
                        feature_columns=feature_cols,
                        k=int(k),
                        reduction_method=reduction,
                        output_dim=int(output_dim),
                    )
                except Exception as exc:
                    st.warning(f"Clustering service is unavailable: {exc}")
                    return

            plot_rows = response.get("dataframe", [])
            viz_cols = response.get("visualization_columns", [])
            if not plot_rows or not viz_cols:
                st.info("Clustering returned no projection.")
                return

            st.session_state[cache_key] = {
                "plot_rows": plot_rows,
                "viz_cols": viz_cols,
                "output_dim": int(output_dim),
            }

        cached = st.session_state.get(cache_key)
        if not cached:
            st.info("Set the parameters above and press **Run embedding map** to compute.")
            return

        fig = _build_embedding_scatter(
            plot_rows=cached["plot_rows"],
            viz_cols=cached["viz_cols"],
            selected_id=selected_id,
            output_dim=int(cached["output_dim"]),
        )
        st.plotly_chart(
            fig,
            width="stretch",
            key=f"emb_map_chart_{collection_name}",
        )


@st.fragment
def _render_distance_histogram(collection_name: str, selected_id: str, selected_title: str) -> None:
    """Plot the cosine-distance distribution from the selected article to every other in the topic.

    Like :func:`_render_embedding_map`, the histogram only computes on Run. The
    result is keyed by collection in session state and includes a label for the
    article it was computed against so the user can see when their current
    selection differs from the cached query.
    """
    with st.container(border=True):
        st.markdown("### Distance from Selected Article")
        st.caption(
            "Cosine distance from the selected article to every other article in the topic. "
            "Lower = more similar; the mass on the left indicates how many close neighbours exist."
        )

        cache_key = f"emb_hist_result_{collection_name}"

        with st.form(key=f"emb_hist_form_{collection_name}", border=False):
            max_points = st.slider(
                "Sample size",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
                key=f"emb_hist_max_{collection_name}",
            )
            run_clicked = st.form_submit_button(
                "Run distance histogram",
                type="primary",
                width="stretch",
            )

        if run_clicked:
            bundle = _load_topic_vectors(collection_name, int(max_points))
            ids: list[str] = bundle["ids"]
            vectors: list[list[float]] = bundle["vectors"]
            if not ids:
                st.info("No embeddings available for this topic.")
                return

            if selected_id in ids:
                i = ids.index(selected_id)
                query_vec: list[float] = vectors[i]
                other_vectors: list[list[float]] = [v for j, v in enumerate(vectors) if j != i]
            else:
                point = get_point(
                    collection_name=collection_name, point_id=selected_id, with_vector=True
                )
                if not point or not getattr(point, "vector", None):
                    st.info("Selected article has no recoverable vector.")
                    return
                query_candidate = _extract_query_vector(point.vector)
                if not query_candidate:
                    st.info("Selected article vector is in an unsupported format.")
                    return
                query_vec = query_candidate
                other_vectors = vectors

            if not other_vectors:
                st.info("Need at least 2 articles to compute a distance distribution.")
                return

            q = np.asarray(query_vec, dtype=float)
            q_norm = q / (np.linalg.norm(q) + 1e-12)
            v_matrix = np.asarray(other_vectors, dtype=float)
            row_norms = np.linalg.norm(v_matrix, axis=1, keepdims=True) + 1e-12
            v_normed = v_matrix / row_norms
            sims = v_normed @ q_norm
            distances = np.clip(1.0 - sims, 0.0, 2.0)

            st.session_state[cache_key] = {
                "distances": distances.tolist(),
                "queried_id": selected_id,
                "queried_title": selected_title,
            }

        cached = st.session_state.get(cache_key)
        if not cached:
            st.info("Press **Run distance histogram** to compute.")
            return

        distances_arr = np.asarray(cached["distances"], dtype=float)
        queried_id = str(cached.get("queried_id") or "")
        queried_title = str(cached.get("queried_title") or queried_id)
        if queried_id != selected_id:
            st.caption(
                f"Showing distances for **{queried_title}** — press Run again to refresh "
                "for the article you have now selected."
            )

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=distances_arr,
                nbinsx=30,
                marker=dict(color=get_colorway()[0]),
                hovertemplate="distance: %{x:.3f}<br>count: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Cosine distance",
            yaxis_title="Article count",
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(
            apply_plotly_theme(fig),
            width="stretch",
            key=f"emb_hist_chart_{collection_name}",
        )

        col_min, col_med, col_max = st.columns(3)
        col_min.metric("Min distance", f"{float(distances_arr.min()):.3f}")
        col_med.metric("Median", f"{float(np.median(distances_arr)):.3f}")
        col_max.metric("Max distance", f"{float(distances_arr.max()):.3f}")


def _render_nearest_news(
    collection_name: str,
    selected_item: dict[str, Any],
    id_to_item: dict[str, dict[str, Any]],
    select_state_key: str,
) -> None:
    """Render the "Find similar articles" button and, when clicked, the result list."""
    if not st.button(
        f"Find {DEFAULT_NEAREST_COUNT} nearest news",
        width="stretch",
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
        if st.button(label, key=f"nearest_{collection_name}_{hit_id}", width="stretch"):
            st.session_state[select_state_key] = hit_id
            st.rerun()


def render_news_page() -> None:
    """Page entry-point: topic picker, article picker, body, word-cloud, nearest articles."""
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
    st.caption("Starts with Qdrant ordering. Type to filter suggestions in the selector.")
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

    st.divider()
    _render_embedding_map(
        collection_name=selected_topic,
        selected_id=selected_news_id,
    )
    _render_distance_histogram(
        collection_name=selected_topic,
        selected_id=selected_news_id,
        selected_title=str(selected_item.get("title", selected_news_id)),
    )


render_news_page()
