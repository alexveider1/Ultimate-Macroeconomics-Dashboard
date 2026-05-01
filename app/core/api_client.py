import os
import requests
import json
from typing import Any

from core.app_logging import log_http_request
from core.session import llm_auth_headers


def _resolve_base_url(
    explicit_base_url: str | None,
    env_var_name: str,
    default_base_url: str,
) -> str:
    candidates = [
        explicit_base_url,
        os.getenv(env_var_name),
        default_base_url,
    ]

    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip().rstrip("/")

    return default_base_url.rstrip("/")


def resolve_forecaster_base_url(base_url: str | None = None) -> str:
    return _resolve_base_url(base_url, "FORECASTER_BASE_URL", "http://forecaster:8001")


def resolve_agent_base_url(base_url: str | None = None) -> str:
    return _resolve_base_url(base_url, "AGENT_BASE_URL", "http://agent:8000")


def resolve_clustering_base_url(base_url: str | None = None) -> str:
    return _resolve_base_url(base_url, "CLUSTERING_BASE_URL", "http://clustering:8002")


def forecast_timeseries(
    base_url: str,
    dates: list[str],
    values: list[float],
    n_prev: int,
    n_predict: int,
    alpha: float = 0.05,
    model_type: str = "prophet",
) -> dict[str, Any]:
    resolved_base_url = resolve_forecaster_base_url(base_url)
    payload = {
        "model_type": model_type,
        "dates": dates,
        "values": values,
        "n_prev": n_prev,
        "n_predict": n_predict,
        "alpha": alpha,
    }
    try:
        log_http_request(
            resolved_base_url,
            "/predict",
            "POST",
            summary=(
                f"model_type={model_type} history_points={len(values)} "
                f"n_prev={n_prev} n_predict={n_predict}"
            ),
        )
        response = requests.post(
            f"{resolved_base_url}/predict", json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "No available forecaster base URL candidates for /predict"
        ) from exc


def agent_chat_stream(
    user_message: str,
    chat_history: list[dict[str, str]] | None = None,
    base_url: str | None = None,
):
    resolved_base_url = resolve_agent_base_url(base_url)
    payload = {
        "user_message": user_message,
        "chat_history": chat_history or [],
    }
    try:
        log_http_request(
            resolved_base_url,
            "/chat/stream",
            "POST",
            summary=(
                f"message_length={len(user_message)} "
                f"history_items={len(chat_history or [])} stream=true"
            ),
        )
        with requests.post(
            f"{resolved_base_url}/chat/stream",
            json=payload,
            timeout=(10, 300),
            stream=True,
            headers=llm_auth_headers(),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=False):
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw_line or "").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line.removeprefix("data:").strip()
                if not line or line.startswith(":"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid streaming payload from agent service: {exc}"
                    ) from exc
    except requests.HTTPError as exc:
        raise RuntimeError(
            "No available agent base URL candidates for /chat/stream"
        ) from exc


def interpret_plot_image(
    image_base64: str,
    mode: str,
    chart_context: str = "",
    base_url: str | None = None,
) -> dict[str, Any]:
    resolved_base_url = resolve_agent_base_url(base_url)
    payload = {
        "image_base64": image_base64,
        "mode": mode,
        "chart_context": chart_context,
    }
    try:
        log_http_request(
            resolved_base_url,
            "/plots/interpret",
            "POST",
            summary=(
                f"mode={mode} chart_context_length={len(chart_context)} "
                f"image_base64_length={len(image_base64)}"
            ),
        )
        response = requests.post(
            f"{resolved_base_url}/plots/interpret",
            json=payload,
            timeout=90,
            headers=llm_auth_headers(),
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict):
            return result
        return {"description": str(result), "mode": mode}
    except requests.HTTPError as exc:
        raise RuntimeError(
            "No available agent base URL candidates for /plots/interpret"
        ) from exc


def cluster_dataframe(
    dataframe: list[dict[str, Any]],
    method: str,
    feature_columns: list[str],
    k: int = 3,
    n_init: int = 10,
    random_state: int = 42,
    eps: float = 0.5,
    min_samples: int = 5,
    base_url: str | None = None,
) -> dict[str, Any]:
    resolved_base_url = resolve_clustering_base_url(base_url)
    payload = {
        "method": method,
        "dataframe": dataframe,
        "feature_columns": feature_columns,
        "k": k,
        "n_init": n_init,
        "random_state": random_state,
        "eps": eps,
        "min_samples": min_samples,
    }
    try:
        log_http_request(
            resolved_base_url,
            "/cluster",
            "POST",
            summary=(
                f"method={method} rows={len(dataframe)} "
                f"feature_columns={len(feature_columns)}"
            ),
        )
        response = requests.post(
            f"{resolved_base_url}/cluster", json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "No available clustering base URL candidates for /cluster"
        ) from exc


def list_agent_models(base_url: str | None = None) -> list[str]:
    resolved_base_url = resolve_agent_base_url(base_url)
    try:
        log_http_request(resolved_base_url, "/models", "GET")
        response = requests.get(
            f"{resolved_base_url}/models",
            timeout=30,
            headers=llm_auth_headers(),
        )
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            models = payload.get("models", [])
            if isinstance(models, list):
                return [str(model) for model in models if str(model).strip()]

        if isinstance(payload, list):
            return [str(model) for model in payload if str(model).strip()]
    except requests.RequestException as exc:
        raise RuntimeError(
            "No available agent base URL candidates for /models"
        ) from exc
