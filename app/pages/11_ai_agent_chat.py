"""AI Analyst chat page — streams responses from the LangGraph agent.

Streamlit re-renders this script on every interaction, so chat state
lives in ``st.session_state[CHAT_STATE_KEY]``. The page wraps the SSE
stream from :func:`core.api_client.agent_chat_stream` and pretty-prints
each event kind (``step`` / ``token`` / ``final`` / ``error``), then
renders any returned artifacts (plot JSON, tabular data) inline.

Supports multimodal input: file attachments (documents, images, text, audio)
via the chat box plus a voice recorder. Non-text inputs are normalized to text /
images by :mod:`core.multimodal` (docling for documents, Whisper for audio,
base64 data URIs for images) before the turn is streamed to the agent.
"""

import hashlib
import re
import uuid

from core.api_client import agent_chat_stream
from core.app_logging import log_page_render
from core.multimodal import augment_message, process_uploads, supported_upload_extensions
from core.plotting import apply_plotly_theme
import plotly.io as pio
import polars as pl
import streamlit as st

CHAT_STATE_KEY = "agent_chat_messages"
SESSION_ID_KEY = "agent_chat_session_id"
VOICE_INPUT_KEY = "agent_chat_voice_input"
VOICE_CONSUMED_KEY = "agent_chat_voice_consumed_hash"
TABLE_PREVIEW_LIMIT = 100

STEP_DISPLAY_NAMES = {
    "guardrail": "guardrail",
    "supervisor": "router",
    "sql_agent": "sql_agent",
    "plotly_agent": "plotly_agent",
    "table_agent": "table_agent",
    "rag_agent": "rag_agent",
    "web_search": "web_search",
    "downloader_agent": "downloader_agent",
    "chat_agent": "chat_agent",
    "FINISH": "FINISH",
}


_LATEX_BLOCK_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def _normalize_math_delimiters(text: str) -> str:
    """Convert LaTeX-style `\\(...\\)` and `\\[...\\]` into Streamlit-friendly
    `$...$` / `$$...$$` so math renders in `st.markdown`."""
    if not text or "\\" not in text:
        return text
    text = _LATEX_BLOCK_RE.sub(lambda m: f"$$\n{m.group(1).strip()}\n$$", text)
    text = _LATEX_INLINE_RE.sub(lambda m: f"${m.group(1).strip()}$", text)
    return text


def _ensure_chat_state() -> None:
    """Create the empty chat history list + a session id on first render."""
    if CHAT_STATE_KEY not in st.session_state:
        st.session_state[CHAT_STATE_KEY] = []
    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = uuid.uuid4().hex


def _as_artifacts(value: object) -> dict:
    """Coerce an unknown value to ``dict``; return ``{}`` when not a dict."""
    return value if isinstance(value, dict) else {}


def _build_data_table_view(artifacts: dict) -> dict | None:
    """Materialise the data table that should appear under the answer."""
    data = artifacts.get("latest_table") or artifacts.get("latest_data")
    if not isinstance(data, dict):
        return None

    rows = data.get("rows") or data.get("records") or []
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return None

    full_df = pl.DataFrame(rows)
    columns = [str(c) for c in (data.get("columns") or full_df.columns)]
    for column in columns:
        if column not in full_df.columns:
            full_df = full_df.with_columns(pl.lit(None).alias(column))
    full_df = full_df.select(columns)

    preview_df = full_df.head(TABLE_PREVIEW_LIMIT)

    file_name = "agent_query_result.csv"
    query_text = str(data.get("query", "") or "").strip().lower()
    if " from " in query_text:
        relation_name = query_text.split(" from ", 1)[1].split()[0].strip('"')
        if relation_name:
            safe_name = "".join(
                ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in relation_name
            ).strip("_")
            if safe_name:
                file_name = f"{safe_name}.csv"

    return {
        "preview_df": preview_df,
        "row_count": int(data.get("row_count") or full_df.height),
        "preview_row_count": preview_df.height,
        "csv_text": full_df.write_csv(),
        "file_name": file_name,
        "truncated": bool(data.get("truncated", False)),
    }


def _render_data_expander(artifacts: dict, message_key: str) -> None:
    """Render the "Show data table" expander under an assistant message."""
    view = _build_data_table_view(artifacts)
    if view is None:
        return

    with st.expander("Show data table", expanded=False):
        preview_df = view["preview_df"]
        dataframe_height = max(180, min(640, 35 * (preview_df.height + 1)))
        st.dataframe(
            preview_df,
            width="stretch",
            hide_index=True,
            height=dataframe_height,
        )

        row_count = view["row_count"]
        preview_row_count = view["preview_row_count"]
        if row_count > preview_row_count > 0:
            st.caption(
                f"Showing the first {preview_row_count} of {row_count} row(s). "
                "Download includes the full table."
            )
        elif view["truncated"] and row_count > 0:
            st.caption(
                f"The result was truncated to {row_count} row(s). "
                "The download contains the same returned rows."
            )
        elif row_count > 0:
            st.caption(f"Table contains {row_count} row(s).")

        st.download_button(
            "Download full table as CSV",
            data=view["csv_text"].encode("utf-8"),
            file_name=view["file_name"],
            mime="text/csv",
            key=f"{message_key}_table_download",
            width="content",
        )


def _render_plot_artifact(plot_artifact: object, message_key: str) -> None:
    """Render a Plotly figure JSON artifact produced by the plotly_agent worker."""
    if not isinstance(plot_artifact, dict):
        return

    figure_json = str(plot_artifact.get("figure_json", "") or "").strip()
    if not figure_json:
        return

    try:
        figure = pio.from_json(figure_json)
    except Exception as exc:
        st.warning(f"Plot artifact could not be rendered: {exc}")
        return

    figure = apply_plotly_theme(figure)

    title = str(plot_artifact.get("title", "") or "").strip()
    if title:
        st.caption(f"Rendered plot: {title}")
    st.plotly_chart(figure, width="stretch", key=f"{message_key}_plot")


def _render_execution_log(steps: list[str], placeholder, finished: bool) -> None:
    """Render the worker-step breadcrumb (e.g. ``router → sql_agent → chat_agent``)."""
    if not steps:
        placeholder.empty()
        return
    arrow_chain = " → ".join(STEP_DISPLAY_NAMES.get(s, s) for s in steps)
    suffix = "" if finished else " …"
    placeholder.markdown(f"`{arrow_chain}{suffix}`")


def _render_assistant_artifacts(artifacts: dict, message_key: str) -> None:
    """Render the plot + data-table block emitted by the agent's ``final`` event."""
    _render_plot_artifact(artifacts.get("latest_plotly"), message_key=message_key)
    _render_data_expander(artifacts, message_key=message_key)


def _render_messages() -> None:
    """Replay the chat history from ``st.session_state`` on every rerun."""
    for index, message in enumerate(st.session_state[CHAT_STATE_KEY]):
        role = message.get("role", "assistant")
        content = str(message.get("content", ""))
        with st.chat_message(role):
            steps = message.get("steps") if role == "assistant" else None
            if steps:
                arrow_chain = " → ".join(STEP_DISPLAY_NAMES.get(s, s) for s in steps)
                st.markdown(f"`{arrow_chain}`")
            if role == "assistant":
                st.markdown(_normalize_math_delimiters(content))
            else:
                st.markdown(content)
                attachments = message.get("attachments")
                if attachments:
                    st.caption("📎 " + ", ".join(str(name) for name in attachments))
            artifacts = _as_artifacts(message.get("artifacts"))
            if role == "assistant":
                _render_assistant_artifacts(
                    artifacts,
                    message_key=f"chat_message_{index}",
                )


def _trim_history_for_api() -> list[dict[str, str]]:
    """Return the last 24 chat turns reduced to ``{"role", "content"}`` dicts."""
    raw_messages = st.session_state[CHAT_STATE_KEY]
    history: list[dict[str, str]] = []
    for message in raw_messages[-24:]:
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))
        if role in {"user", "assistant"} and content.strip():
            history.append({"role": role, "content": content})
    return history


def _dedupe_step(steps: list[str], node: str) -> None:
    """Append node unless it is identical to the last step."""
    if not node:
        return
    if steps and steps[-1] == node:
        return
    steps.append(node)


def _collect_voice_attachment() -> tuple[str, str | None, bytes] | None:
    """Return the recorded-voice attachment when it's new (not already consumed).

    The ``st.audio_input`` widget keeps its last recording in session state, so a
    content hash is tracked to avoid re-transcribing the same clip on a later
    text-only turn.
    """
    audio = st.session_state.get(VOICE_INPUT_KEY)
    if audio is None:
        return None
    try:
        audio_bytes = audio.getvalue()
    except Exception:  # noqa: BLE001 - a malformed widget value shouldn't crash the page
        return None
    if not audio_bytes:
        return None
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash == st.session_state.get(VOICE_CONSUMED_KEY):
        return None
    st.session_state[VOICE_CONSUMED_KEY] = audio_hash
    name = getattr(audio, "name", None) or "voice_message.wav"
    content_type = getattr(audio, "type", None) or "audio/wav"
    return (name, content_type, audio_bytes)


def _handle_chat() -> None:
    """Read one user turn (text + attachments + voice) and stream the response.

    File attachments and a recorded voice clip are normalized to text / images by
    :mod:`core.multimodal` before streaming. Handles the SSE event types
    (``step`` / ``token`` / ``final`` / ``error``) and renders any artifacts on
    the ``final`` event. Token/cost accounting lives in Langfuse, so the ``usage``
    block on the ``final`` event is ignored here.
    """
    submission = st.chat_input(
        "Ask the AI analyst...",
        accept_file="multiple",
        file_type=supported_upload_extensions(),
    )
    if not submission:
        return

    # With accept_file set, chat_input yields an object carrying .text + .files;
    # fall back to a bare string if a Streamlit build returns one.
    prompt_text = str(getattr(submission, "text", submission) or "").strip()
    uploaded_files = list(getattr(submission, "files", []) or [])

    attachments: list[tuple[str, str | None, bytes]] = [
        (uploaded.name, uploaded.type, uploaded.getvalue()) for uploaded in uploaded_files
    ]
    voice = _collect_voice_attachment()
    if voice is not None:
        attachments.append(voice)

    if not prompt_text and not attachments:
        return

    attachment_names = [name for name, _, _ in attachments]

    images: list[str] = []
    if attachments:
        with st.spinner("Processing attachments…"):
            processed = process_uploads(attachments)
        images = processed.images
        agent_message = augment_message(prompt_text, processed.text_blocks)
    else:
        agent_message = prompt_text

    display_text = prompt_text or "_(sent attachments)_"
    st.session_state[CHAT_STATE_KEY].append(
        {"role": "user", "content": display_text, "attachments": attachment_names}
    )
    with st.chat_message("user"):
        st.markdown(display_text)
        if attachment_names:
            st.caption("📎 " + ", ".join(attachment_names))

    with st.chat_message("assistant"):
        message_key = (
            f"pending_{len(st.session_state[CHAT_STATE_KEY])}_"
            f"{hashlib.sha256(agent_message.encode('utf-8')).hexdigest()[:12]}"
        )

        log_placeholder = st.empty()
        answer_placeholder = st.empty()

        steps: list[str] = []
        answer_buffer: list[str] = []
        artifacts: dict = {}
        final_answer = ""
        error_text = ""

        try:
            for event in agent_chat_stream(
                user_message=agent_message,
                chat_history=_trim_history_for_api(),
                session_id=st.session_state.get(SESSION_ID_KEY),
                images=images or None,
            ):
                event_type = event.get("type", "")
                if event_type == "step":
                    _dedupe_step(steps, str(event.get("node", "")))
                    _render_execution_log(steps, log_placeholder, finished=False)
                elif event_type == "token":
                    delta = str(event.get("delta", ""))
                    if delta:
                        answer_buffer.append(delta)
                        answer_placeholder.markdown(
                            _normalize_math_delimiters("".join(answer_buffer))
                        )
                elif event_type == "final":
                    final_answer = str(event.get("answer", "")) or "".join(answer_buffer)
                    artifacts = _as_artifacts(event.get("artifacts"))
                    break
                elif event_type == "error":
                    error_text = str(event.get("answer", "Agent error."))
                    break
        except Exception as exc:
            error_text = f"Agent request failed: {exc}"

        _render_execution_log(steps, log_placeholder, finished=True)

        if error_text and not final_answer:
            answer_placeholder.error(error_text)
            st.session_state[CHAT_STATE_KEY].append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "artifacts": {},
                    "steps": steps,
                }
            )
            return

        if not final_answer:
            final_answer = "".join(answer_buffer) or "No answer returned."

        answer_placeholder.markdown(_normalize_math_delimiters(final_answer))
        _render_assistant_artifacts(artifacts, message_key=message_key)

        st.session_state[CHAT_STATE_KEY].append(
            {
                "role": "assistant",
                "content": final_answer,
                "artifacts": artifacts,
                "steps": steps,
            }
        )


def render_page() -> None:
    """Page entry-point: title, prior messages, then the chat input handler."""
    log_page_render("AI Analyst")
    st.title("AI Analyst")
    st.caption(
        "Chat with the AI analyst. Attach documents, images, or text files via "
        "the ＋ in the chat box, or record a voice message below."
    )

    _ensure_chat_state()

    left_col, right_col = st.columns([0.7, 0.3])
    with left_col:
        st.audio_input("🎤 Voice message (optional)", key=VOICE_INPUT_KEY)
    with right_col:
        if st.button("Clear chat", width="stretch"):
            st.session_state[CHAT_STATE_KEY] = []
            # New conversation → new Langfuse session id.
            st.session_state[SESSION_ID_KEY] = uuid.uuid4().hex
            st.session_state.pop(VOICE_CONSUMED_KEY, None)
            st.rerun()

    _render_messages()
    _handle_chat()


render_page()
