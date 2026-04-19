import streamlit as st
import yaml

from core.app_logging import log_page_render
from core.api_client import list_agent_models, resolve_agent_base_url


PLOTLY_THEMES = [
    "plotly",
    "plotly_white",
    "plotly_dark",
    "ggplot2",
    "seaborn",
    "simple_white",
    "presentation",
    "xgridoff",
    "ygridoff",
]

CONFIG_PATH = "config.yaml"
CONFIG = yaml.safe_load(open(CONFIG_PATH))


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


log_page_render("System Settings")
st.title("System Settings")
st.caption("Configure dashboard-wide visualization preferences.")

if "user_settings" not in st.session_state:
    st.session_state.user_settings = {
        "theme": "dark",
        "plotly_theme": "plotly",
    }

current_theme = st.session_state.user_settings.get("plotly_theme", "plotly")
default_index = (
    PLOTLY_THEMES.index(current_theme) if current_theme in PLOTLY_THEMES else 0
)

selected_theme = st.selectbox(
    "Plotly chart theme",
    options=PLOTLY_THEMES,
    index=default_index,
    help="Applies to all dashboard Plotly charts.",
)

st.session_state.user_settings["plotly_theme"] = selected_theme
st.success(f"Active Plotly theme: {selected_theme}")

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

save_clicked = st.button("Save AI model", type="primary", use_container_width=False)
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
