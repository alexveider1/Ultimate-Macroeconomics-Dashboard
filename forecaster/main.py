"""FastAPI adapter exposing the forecasting models — compute lives in Triton.

This service keeps its original HTTP contract (``/predict``, ``/models``,
``/health``) but no longer runs any ML itself. It validates + cleans the request
(date parsing, dedup, ``n_prev`` truncation), enforces the ``config.yaml`` model
toggles, then forwards the prepared series to the matching python-backend model
in the ``triton`` container over gRPC and reshapes the reply into the same
``ForecastResponse`` callers already expect.
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from config import load_config
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import polars as pl
from schemas import ForecastPoint, ForecastRequest, ForecastResponse
from triton_client import TritonError, create_client, infer_json, resolve_triton_url

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("FORECASTER_CONFIG_PATH", "config.yaml"))

CONFIG = load_config(CONFIG_PATH)
FORECASTER_CONFIG = CONFIG.forecaster
TRITON_CONFIG = CONFIG.triton

ARIMA_AVAILABLE = FORECASTER_CONFIG.ARIMA_AVAILABLE
PROPHET_AVAILABLE = FORECASTER_CONFIG.PROPHET_AVAILABLE
CHRONOS_AVAILABLE = FORECASTER_CONFIG.CHRONOS_AVAILABLE
CHRONOS_MODEL_NAME = FORECASTER_CONFIG.CHRONOS_MODEL
CHRONOS_DEFAULT_MODEL_NAME = "amazon/chronos-t5-small"

# `auto_arima`, `arima`, `sarima` share the ARIMA dep family toggle. Moving-average
# and XGBoost are always available (XGBoost's GPU work happens inside Triton).
ARIMA_FAMILY_MODELS = {"auto_arima", "arima", "sarima"}


def _ensure_model_enabled(model_type: str) -> None:
    """Reject a disabled model family with a 400 before hitting Triton."""
    if model_type in ARIMA_FAMILY_MODELS and not ARIMA_AVAILABLE:
        raise HTTPException(status_code=400, detail=f"Model '{model_type}' is disabled.")
    if model_type == "prophet" and not PROPHET_AVAILABLE:
        raise HTTPException(status_code=400, detail="Model 'prophet' is disabled.")
    if model_type == "chronos" and not CHRONOS_AVAILABLE:
        raise HTTPException(status_code=400, detail="Model 'chronos' is disabled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the shared Triton gRPC client (lazy connect) on startup."""
    url = resolve_triton_url(TRITON_CONFIG.host, TRITON_CONFIG.grpc_port)
    logger.info("Forecaster adapter targeting Triton at %s", url)
    app.state.triton = create_client(url)
    try:
        yield
    finally:
        try:
            app.state.triton.close()
        except Exception:  # noqa: BLE001 - best-effort shutdown
            logger.debug("Triton client close failed", exc_info=True)


app = FastAPI(
    title="Time Series Forecasting API",
    description="Adapter for ARIMA / SARIMA / Prophet / Chronos / MA / XGBoost served by Triton.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Time Series Forecasting API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, list[str]]:
    """Return the labels of every enabled model (unchanged contract)."""
    available_models: list[str] = []
    if ARIMA_AVAILABLE:
        available_models.extend(["auto_arima", "arima", "sarima"])
    if PROPHET_AVAILABLE:
        available_models.append("prophet")
    if CHRONOS_AVAILABLE:
        chronos_label = CHRONOS_MODEL_NAME or CHRONOS_DEFAULT_MODEL_NAME
        available_models.append(f"chronos ({chronos_label})")
    available_models.extend(["moving_average", "xgboost"])

    return {"available_models": available_models}


def _prepare_series(request: ForecastRequest) -> tuple[list[str], list[float]]:
    """Validate + clean the history and return ISO dates aligned with values.

    Mirrors the previous in-process preprocessing: parse timestamps (reject
    unparseable ones), collapse duplicate timestamps keeping the last value,
    sort ascending, and truncate to the trailing ``n_prev`` points.
    """
    bad_dates = HTTPException(
        status_code=400,
        detail="Invalid date format found in 'dates'. Use ISO datetime-compatible strings.",
    )
    try:
        df = pl.DataFrame({"ds": request.dates, "y": request.values}).with_columns(
            pl.col("ds").str.to_datetime(strict=False)
        )
    except Exception:
        # polars raises (not nulls) when it can't even infer a format for the
        # column — treat that the same as an unparseable-date 400.
        raise bad_dates
    if df["ds"].null_count() > 0:
        raise bad_dates

    df = df.group_by("ds", maintain_order=True).agg(pl.col("y").last()).sort("ds")
    if request.n_prev is not None and request.n_prev < len(df):
        df = df.tail(request.n_prev)

    df = df.with_columns(pl.col("ds").dt.strftime("%Y-%m-%d %H:%M:%S"))
    return df["ds"].to_list(), df["y"].to_list()


@app.post("/predict", response_model=ForecastResponse)
async def generate_prediction(request: ForecastRequest) -> ForecastResponse:
    """Forward the cleaned history to the matching Triton model and reshape it.

    Raises:
        HTTPException: 400 for unparseable dates, disabled models, or invalid
            model inputs; 500 for model failures; 502 when Triton is unreachable.
    """
    _ensure_model_enabled(request.model_type)
    dates, values = _prepare_series(request)

    payload = {
        "dates": dates,
        "values": values,
        "n_predict": request.n_predict,
        "alpha": request.alpha,
        "model_params": request.model_params,
    }

    try:
        result = await run_in_threadpool(
            infer_json, app.state.triton, f"forecast_{request.model_type}", payload
        )
    except TritonError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    points = [
        ForecastPoint(ds=ds, yhat=yhat, yhat_lower=lower, yhat_upper=upper)
        for ds, yhat, lower, upper in zip(
            result["ds"], result["yhat"], result["yhat_lower"], result["yhat_upper"]
        )
    ]
    return ForecastResponse(model_used=request.model_type, forecast=points)
