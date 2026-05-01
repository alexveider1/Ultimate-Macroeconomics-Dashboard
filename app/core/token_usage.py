"""Per-Streamlit-session token usage accounting.

The agent service returns prompt/completion/total token counts in the `final`
event of `/chat/stream` (and inline in `/plots/interpret` responses). The app
accumulates these into `st.session_state` under a single key, broken down by
model. Counts are in-memory only and disappear when the session ends.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


SESSION_STATE_KEY = "token_usage_by_model"


def _bucket() -> dict[str, dict[str, int]]:
    bucket = st.session_state.get(SESSION_STATE_KEY)
    if not isinstance(bucket, dict):
        bucket = {}
        st.session_state[SESSION_STATE_KEY] = bucket
    return bucket


def record_usage(usage: dict[str, Any] | None) -> None:
    """Add a single LLM-call usage record to the session totals."""
    if not isinstance(usage, dict):
        return
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    if prompt == 0 and completion == 0 and total == 0:
        return

    model = str(usage.get("model") or "unknown").strip() or "unknown"
    bucket = _bucket()
    entry = bucket.setdefault(
        model,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    entry["prompt_tokens"] += prompt
    entry["completion_tokens"] += completion
    entry["total_tokens"] += total


def get_session_token_usage() -> dict[str, dict[str, int]]:
    return dict(_bucket())


def total_session_tokens() -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for entry in _bucket().values():
        totals["prompt_tokens"] += int(entry.get("prompt_tokens", 0))
        totals["completion_tokens"] += int(entry.get("completion_tokens", 0))
        totals["total_tokens"] += int(entry.get("total_tokens", 0))
    return totals


def reset_session_token_usage() -> None:
    st.session_state[SESSION_STATE_KEY] = {}
