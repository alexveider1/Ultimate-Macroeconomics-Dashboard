"""Per-Streamlit-session LLM credential storage and gating.

In the hosting build the operator never holds the chat / agentic LLM key;
the end user fills it in a small form on first visit. Credentials live in
`st.session_state` for the lifetime of the browser tab and are forwarded
to the agent service as request headers (X-LLM-Base-URL, X-LLM-API-Key,
X-LLM-Model). They are never written to disk.
"""
from __future__ import annotations

import streamlit as st


SESSION_STATE_KEY = "llm_creds"


def get_llm_creds() -> dict[str, str] | None:
    creds = st.session_state.get(SESSION_STATE_KEY)
    if not isinstance(creds, dict):
        return None
    if not creds.get("base_url") or not creds.get("api_key"):
        return None
    return creds


def llm_auth_headers() -> dict[str, str]:
    creds = get_llm_creds()
    if not creds:
        return {}
    headers = {
        "X-LLM-Base-URL": creds["base_url"],
        "X-LLM-API-Key": creds["api_key"],
    }
    model = creds.get("model", "").strip()
    if model:
        headers["X-LLM-Model"] = model
    return headers


def clear_llm_creds() -> None:
    st.session_state[SESSION_STATE_KEY] = {}


def require_llm_creds() -> None:
    """Render a credential form and halt the page until creds are provided."""
    if get_llm_creds():
        return

    st.title("Configure LLM access")
    st.markdown(
        "This dashboard uses your own LLM provider for the AI Analyst, "
        "chart interpretation, and other generative features. Enter the "
        "OpenAI-compatible base URL and API key for your provider — they "
        "are kept only in your browser session and forwarded to the agent "
        "service via request headers. Nothing is written to disk."
    )

    with st.form("llm_creds_form"):
        base_url = st.text_input(
            "LLM base URL",
            placeholder="https://api.openai.com/v1",
            help="Any OpenAI-compatible chat-completions endpoint.",
        ).strip()
        api_key = st.text_input(
            "LLM API key",
            type="password",
            help="Sent as the bearer token to the provider above.",
        ).strip()
        model = st.text_input(
            "Model (optional)",
            placeholder="Leave blank to use the server default.",
            help="Overrides the model configured in config.yaml for this session.",
        ).strip()
        submitted = st.form_submit_button("Continue", type="primary")

    if submitted:
        if not base_url or not api_key:
            st.error("Base URL and API key are both required.")
            st.stop()
        st.session_state[SESSION_STATE_KEY] = {
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }
        st.rerun()

    st.stop()
