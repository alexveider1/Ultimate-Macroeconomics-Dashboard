import streamlit as st
import yaml

from core.app_logging import log_page_render
from core.api_client import list_agent_models, resolve_agent_base_url
from core.session import clear_llm_creds, get_llm_creds
from core.token_usage import (
    get_session_token_usage,
    reset_session_token_usage,
    total_session_tokens,
)


CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


def _resolve_agent_base_url() -> str:
    return resolve_agent_base_url()


def _read_shared_config() -> tuple[dict, str | None]:
    try:
        loaded = CONFIG
        if isinstance(loaded, dict):
            return loaded, CONFIG_PATH
    except Exception:
        pass
    return {}, None


def _write_shared_config(config_data: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as file:
            yaml.safe_dump(config_data, file, sort_keys=False)
        return CONFIG_PATH
    except Exception:
        raise PermissionError("Could not write config.yaml from this runtime context.")


def _mask(value: str) -> str:
    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]}"


log_page_render("System Settings")
st.title("System Settings")
st.caption(
    "Review the LLM credentials in use this session, configure the agent "
    "model, and monitor token usage."
)

st.subheader("LLM credentials (this session)")

session_creds = get_llm_creds()
if session_creds:
    st.markdown(
        f"- **Base URL:** `{session_creds.get('base_url', '')}`\n"
        f"- **API key:** `{_mask(session_creds.get('api_key', ''))}`\n"
        f"- **Model override:** "
        f"`{session_creds.get('model') or '(server default)'}`"
    )
    st.caption(
        "Credentials live only in your browser session. Sign out to clear "
        "them and force the credential form on the next request."
    )
    if st.button("Sign out / clear credentials", width="content"):
        clear_llm_creds()
        st.rerun()
else:
    st.info("No LLM credentials in this session yet.")

st.divider()
st.subheader("AI Model")

config_data, loaded_from = _read_shared_config()
shared_cfg = config_data.get("shared", {}) if isinstance(config_data, dict) else {}
configured_model = str(shared_cfg.get("openai_llm_model", "")).strip()

agent_base_url = _resolve_agent_base_url()
available_models: list[str] = []
models_error: str | None = None
try:
    available_models = list_agent_models(agent_base_url)
except Exception as exc:
    models_error = str(exc)

if available_models:
    model_options = sorted(set(available_models))
    if configured_model and configured_model not in model_options:
        model_options = [configured_model] + model_options

    default_model = configured_model or model_options[0]
    selected_index = model_options.index(default_model)
    selected_model = st.selectbox(
        "Agent model",
        options=model_options,
        index=selected_index,
        help="Fetched from the agent API /models endpoint.",
    )
else:
    st.warning(
        "Could not fetch models from agent API. You can still set model manually."
    )
    if models_error:
        st.caption(f"Agent models error: {models_error}")
    selected_model = st.text_input(
        "Agent model",
        value=configured_model,
        placeholder="Example: gpt-5.4-nano",
    ).strip()

save_clicked = st.button("Save AI model", type="primary", width="content")
if save_clicked:
    if not selected_model:
        st.error("Please choose or enter a model name.")
    else:
        config_data.setdefault("shared", {})["openai_llm_model"] = selected_model
        try:
            written_to = _write_shared_config(config_data)
            st.success(f"Saved model '{selected_model}' to {written_to}.")
            st.info("Restart the agent container/service to apply the new model.")
        except Exception as exc:
            st.error(f"Could not write config.yaml: {exc}")

if loaded_from:
    st.caption(f"Config loaded from: {loaded_from}")

st.divider()
st.subheader("Session token usage")
st.caption(
    "Tokens consumed by the AI agent during this Streamlit session. "
    "Counts reset when you reload the browser tab or click the button below."
)

usage_by_model = get_session_token_usage()
if not usage_by_model:
    st.info("No tokens consumed yet in this session.")
else:
    table_rows = [
        {
            "Model": model,
            "Prompt": usage.get("prompt_tokens", 0),
            "Completion": usage.get("completion_tokens", 0),
            "Total": usage.get("total_tokens", 0),
        }
        for model, usage in sorted(usage_by_model.items())
    ]
    totals = total_session_tokens()
    table_rows.append(
        {
            "Model": "ALL",
            "Prompt": totals["prompt_tokens"],
            "Completion": totals["completion_tokens"],
            "Total": totals["total_tokens"],
        }
    )
    st.dataframe(table_rows, hide_index=True, width="stretch")

if st.button("Reset token counter", width="content"):
    reset_session_token_usage()
    st.rerun()
