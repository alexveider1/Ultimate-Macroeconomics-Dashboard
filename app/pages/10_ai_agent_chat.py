import hashlib

import polars as pl
import plotly.io as pio
import streamlit as st

from core.app_logging import log_page_render
from core.api_client import agent_chat, agent_chat_stream
from core.plotting import apply_plotly_theme


CHAT_STATE_KEY = "agent_chat_messages"
TABLE_PREVIEW_LIMIT = 100


def _ensure_chat_state() -> None:
    if CHAT_STATE_KEY not in st.session_state:
        st.session_state[CHAT_STATE_KEY] = []


def _as_artifacts(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _coerce_latest_data_to_table_artifact(data_artifact: object) -> dict | None:
    if not isinstance(data_artifact, dict):
        return None

    rows = data_artifact.get("rows", [])
    columns = [str(column) for column in data_artifact.get("columns", []) or []]
    if not isinstance(rows, list) or not rows:
        return None

    full_records = [row for row in rows if isinstance(row, dict)]
    if not full_records:
        return None

    full_df = pl.DataFrame(full_records)
    preview_df = full_df.head(TABLE_PREVIEW_LIMIT)
    if columns:
        for column in columns:
            if column not in full_df.columns:
                full_df = full_df.with_columns(pl.lit(None).alias(column))
            if column not in preview_df.columns:
                preview_df = preview_df.with_columns(pl.lit(None).alias(column))
        full_df = full_df.select(columns)
        preview_df = preview_df.select(columns)

    file_name = "agent_query_result.csv"
    query_text = str(data_artifact.get("query", "") or "").strip().lower()
    if " from " in query_text:
        relation_name = query_text.split(" from ", 1)[1].split()[0].strip('"')
        if relation_name:
            safe_name = "".join(
                character if character.isalnum() or character in {"_", "-"} else "_"
                for character in relation_name
            ).strip("_")
            if safe_name:
                file_name = f"{safe_name}.csv"

    return {
        "row_count": int(data_artifact.get("row_count") or full_df.height),
        "preview_row_count": int(preview_df.height),
        "columns": columns or [str(column) for column in full_df.columns],
        "records": preview_df.to_dicts(),
        "table_text": preview_df.to_pandas().to_string(index=False),
        "csv_text": full_df.write_csv(),
        "file_name": file_name,
        "truncated": bool(data_artifact.get("truncated", False)),
        "source": "latest_data",
    }


def _render_table_artifact(table_artifact: object, message_key: str) -> None:
    if not isinstance(table_artifact, dict):
        return

    preview_records = table_artifact.get("records", [])
    columns = [str(column) for column in table_artifact.get("columns", []) or []]
    row_count = int(table_artifact.get("row_count") or 0)
    preview_row_count = int(table_artifact.get("preview_row_count") or 0)
    truncated = bool(table_artifact.get("truncated", False))
    csv_text = str(table_artifact.get("csv_text", "") or "")
    file_name = str(
        table_artifact.get("file_name", "agent_table_result.csv")
        or "agent_table_result.csv"
    )

    if preview_records:
        preview_df = pl.DataFrame(preview_records)
        if columns:
            for column in columns:
                if column not in preview_df.columns:
                    preview_df = preview_df.with_columns(pl.lit(None).alias(column))
            preview_df = preview_df.select(columns)

        dataframe_height = max(
            180,
            min(640, 35 * (min(preview_df.height, TABLE_PREVIEW_LIMIT) + 1)),
        )
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            height=dataframe_height,
        )
    elif row_count > 0:
        st.info(
            f"Table generated with {row_count} row(s), but no preview rows were returned."
        )
    else:
        return

    if row_count > preview_row_count > 0:
        st.caption(
            f"Showing the first {preview_row_count} row(s) of {row_count}. Download includes the full table."
        )
    elif truncated and row_count > 0:
        st.caption(
            f"The SQL result was truncated to {row_count} row(s) before rendering. The download contains the same returned rows."
        )
    elif row_count > 0:
        st.caption(f"Table contains {row_count} row(s).")

    if csv_text:
        st.download_button(
            "Download full table as CSV",
            data=csv_text.encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            key=f"{message_key}_table_download",
            use_container_width=False,
        )


def _render_plot_artifact(plot_artifact: object, message_key: str) -> None:
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
    st.plotly_chart(figure, use_container_width=True, key=f"{message_key}_plot")


def _render_progress_updates(progress_updates: object) -> None:
    if not isinstance(progress_updates, list) or not progress_updates:
        return

    with st.expander("Execution updates", expanded=False):
        st.markdown(
            "\n".join(
                f"- {str(item)}" for item in progress_updates if str(item).strip()
            )
        )


def _render_assistant_artifacts(artifacts: dict, message_key: str) -> None:
    _render_progress_updates(artifacts.get("progress_updates"))
    _render_plot_artifact(artifacts.get("latest_plotly"), message_key=message_key)
    display_preferences = artifacts.get("display_preferences")
    if (
        isinstance(display_preferences, dict)
        and display_preferences.get("show_table") is False
    ):
        return
    table_artifact = artifacts.get("latest_table")
    if not isinstance(table_artifact, dict):
        table_artifact = _coerce_latest_data_to_table_artifact(
            artifacts.get("latest_data")
        )
    _render_table_artifact(table_artifact, message_key=message_key)


def _render_messages() -> None:
    for index, message in enumerate(st.session_state[CHAT_STATE_KEY]):
        role = message.get("role", "assistant")
        content = str(message.get("content", ""))
        with st.chat_message(role):
            st.markdown(content)
            artifacts = _as_artifacts(message.get("artifacts"))
            if role == "assistant":
                _render_assistant_artifacts(
                    artifacts,
                    message_key=f"chat_message_{index}",
                )


def _trim_history_for_api() -> list[dict[str, str]]:
    raw_messages = st.session_state[CHAT_STATE_KEY]
    history: list[dict[str, str]] = []
    for message in raw_messages[-24:]:
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))
        if role in {"user", "assistant"} and content.strip():
            history.append({"role": role, "content": content})
    return history


def _handle_chat() -> None:
    prompt = st.chat_input("Ask the AI analyst...")
    if not prompt:
        return

    st.session_state[CHAT_STATE_KEY].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pending_key = (
            f"pending_{len(st.session_state[CHAT_STATE_KEY])}_"
            f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"
        )

        progress_container = st.empty()
        progress_items: list[str] = []
        result = None

        try:
            for event in agent_chat_stream(
                user_message=prompt,
                chat_history=_trim_history_for_api(),
            ):
                event_type = event.get("type", "progress")
                if event_type == "progress":
                    updates = event.get("updates", [])
                    progress_items.extend(updates)
                    if progress_items:
                        progress_container.markdown(
                            "**Agent is working...**\n"
                            + "\n".join(f"- {u}" for u in progress_items)
                        )
                elif event_type in ("final", "error"):
                    result = event
                    break
        except Exception:
            result = None

        if result is None:
            progress_container.empty()
            with st.spinner("Agent is working..."):
                try:
                    result = agent_chat(
                        user_message=prompt,
                        chat_history=_trim_history_for_api(),
                    )
                except Exception as exc:
                    error_text = f"Agent request failed: {exc}"
                    st.error(error_text)
                    st.session_state[CHAT_STATE_KEY].append(
                        {"role": "assistant", "content": error_text, "artifacts": {}}
                    )
                    return

        progress_container.empty()

        answer = str(result.get("answer", "No answer returned."))
        artifacts = _as_artifacts(result.get("artifacts"))
        progress_updates = result.get("progress_updates") or result.get("trace") or []
        if progress_items and not progress_updates:
            progress_updates = progress_items
        artifacts["progress_updates"] = progress_updates

        st.markdown(answer)
        _render_assistant_artifacts(artifacts, message_key=pending_key)

        st.session_state[CHAT_STATE_KEY].append(
            {
                "role": "assistant",
                "content": answer,
                "artifacts": artifacts,
                "trace": result.get("trace", []),
            }
        )


def render_page() -> None:
    log_page_render("AI Analyst")
    st.title("AI Analyst")
    st.caption("Chat interface backed by the agent server in task-mode by default.")

    _ensure_chat_state()

    _, right_col = st.columns([0.7, 0.3])
    with right_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state[CHAT_STATE_KEY] = []
            st.rerun()

    _render_messages()
    _handle_chat()


render_page()
