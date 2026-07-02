"""FastAPI service exposing the forecasting models.

Heavy ML imports happen lazily inside :func:`_get_forecaster` so the container
boots fast even when only a subset of models is enabled. Each model is
instantiated at most once per process; subsequent requests reuse the cached
instance behind an ``asyncio.Lock`` that protects the first-call race.
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from config import load_config
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import polars as pl
from schemas import ForecastPoint, ForecastRequest, ForecastResponse

from forecasters.core.base import BaseForecaster

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("FORECASTER_CONFIG_PATH", "config.yaml"))

CONFIG = load_config(CONFIG_PATH)
FORECASTER_CONFIG = CONFIG.forecaster

ARIMA_AVAILABLE = FORECASTER_CONFIG.ARIMA_AVAILABLE
PROPHET_AVAILABLE = FORECASTER_CONFIG.PROPHET_AVAILABLE
CHRONOS_AVAILABLE = FORECASTER_CONFIG.CHRONOS_AVAILABLE
CHRONOS_MODEL_NAME = FORECASTER_CONFIG.CHRONOS_MODEL
CHRONOS_DEFAULT_MODEL_NAME = "amazon/chronos-t5-small"

# `auto_arima`, `arima`, `sarima` share the ARIMA dep stack (pmdarima / statsmodels)
# so they ride on the same toggle. Moving-average and XGBoost have lightweight
# deps and stay always-available.
ARIMA_FAMILY_MODELS = {"auto_arima", "arima", "sarima"}


async def _get_forecaster(app: FastAPI, model_type: str) -> BaseForecaster:
    """Return a cached forecaster, lazily importing + constructing under a lock.

    Without the lock, two concurrent first-time requests for the same model
    would each import the heavy ML library and race on the dict assignment.
    The lock makes initialization atomic across the asyncio event loop;
    heavy fit/predict still runs off-loop via ``run_in_threadpool``.

    Args:
        app: FastAPI instance whose ``state.model_cache`` holds the singletons.
        model_type: Model id from :data:`schemas.ModelType`.

    Returns:
        A ready-to-use :class:`BaseForecaster` subclass.

    Raises:
        HTTPException: 400 when the requested model is disabled or unknown;
            500 when the underlying library fails to import or instantiate.
    """
    cache: dict[str, BaseForecaster] = app.state.model_cache
    lock: asyncio.Lock = app.state.model_cache_lock

    if model_type == "prophet":
        if not PROPHET_AVAILABLE:
            raise HTTPException(status_code=400, detail="Model 'prophet' is disabled.")
        if "prophet" in cache:
            return cache["prophet"]
        async with lock:
            if "prophet" in cache:
                return cache["prophet"]
            try:
                from forecasters.prophet_model import ProphetForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize Prophet forecaster: {str(e)}",
                )
            cache["prophet"] = ProphetForecaster()
            return cache["prophet"]

    if model_type == "chronos":
        if not CHRONOS_AVAILABLE:
            raise HTTPException(status_code=400, detail="Model 'chronos' is disabled.")
        if "chronos" in cache:
            return cache["chronos"]
        async with lock:
            if "chronos" in cache:
                return cache["chronos"]
            try:
                from forecasters.chronos_model import ChronosForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize Chronos forecaster: {str(e)}",
                )
            cache["chronos"] = (
                ChronosForecaster(CHRONOS_MODEL_NAME) if CHRONOS_MODEL_NAME else ChronosForecaster()
            )
            return cache["chronos"]

    if model_type in ARIMA_FAMILY_MODELS:
        if not ARIMA_AVAILABLE:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' is disabled.")
        if model_type in cache:
            return cache[model_type]
        async with lock:
            if model_type in cache:
                return cache[model_type]
            try:
                if model_type == "auto_arima":
                    from forecasters.auto_arima_model import AutoArimaForecaster

                    cache[model_type] = AutoArimaForecaster()
                elif model_type == "arima":
                    from forecasters.arima_model import ArimaForecaster

                    cache[model_type] = ArimaForecaster()
                else:  # sarima
                    from forecasters.sarima_model import SarimaForecaster

                    cache[model_type] = SarimaForecaster()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize {model_type} forecaster: {str(e)}",
                )
            return cache[model_type]

    if model_type == "moving_average":
        if "moving_average" in cache:
            return cache["moving_average"]
        async with lock:
            if "moving_average" in cache:
                return cache["moving_average"]
            try:
                from forecasters.moving_average_model import MovingAverageForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize moving-average forecaster: {str(e)}",
                )
            cache["moving_average"] = MovingAverageForecaster()
            return cache["moving_average"]

    if model_type == "xgboost":
        if "xgboost" in cache:
            return cache["xgboost"]
        async with lock:
            if "xgboost" in cache:
                return cache["xgboost"]
            try:
                from forecasters.xgboost_model import XgboostForecaster
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize XGBoost forecaster: {str(e)}",
                )
            cache["xgboost"] = XgboostForecaster()
            return cache["xgboost"]

    raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the per-process model cache and lock on startup.

    Chronos is eagerly loaded onto RAM/GPU at boot so the first inference
    request doesn't pay the multi-second checkpoint-download + weight-load
    cost. Everything else stays lazy since they are cheap to construct.
    """
    app.state.model_cache = {}
    app.state.model_cache_lock = asyncio.Lock()

    if CHRONOS_AVAILABLE:
        try:
            from forecasters.chronos_model import ChronosForecaster

            chronos_label = CHRONOS_MODEL_NAME or CHRONOS_DEFAULT_MODEL_NAME
            logger.info("Preloading Chronos pipeline: %s", chronos_label)
            app.state.model_cache["chronos"] = (
                await run_in_threadpool(ChronosForecaster, CHRONOS_MODEL_NAME)
                if CHRONOS_MODEL_NAME
                else await run_in_threadpool(ChronosForecaster)
            )
            logger.info("Chronos pipeline ready (%s)", chronos_label)
        except Exception as exc:
            logger.warning("Failed to preload Chronos on startup: %s", exc, exc_info=True)

    yield


app = FastAPI(
    title="Time Series Forecasting API",
    description="A unified API for ARIMA / SARIMA / Prophet / Chronos / MA / XGBoost forecasting.",
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
    """Return the labels of every enabled model.

    Chronos additionally embeds the underlying checkpoint name in
    parentheses so the dashboard can show which weights are loaded.
    Moving-average and XGBoost are always available because their
    deps are lightweight.
    """
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


@app.post("/predict", response_model=ForecastResponse)
async def generate_prediction(request: ForecastRequest) -> ForecastResponse:
    """Build a forecast for the supplied history using the requested model.

    Args:
        request: Validated :class:`ForecastRequest` body.

    Returns:
        ForecastResponse with the model label and predicted points.

    Raises:
        HTTPException: 400 for unparseable dates or invalid inputs;
            500 if the underlying model raises during fit/predict.
    """
    df = pl.DataFrame({"ds": request.dates, "y": request.values}).with_columns(
        pl.col("ds").str.to_datetime(strict=False)
    )

    if df["ds"].null_count() > 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format found in 'dates'. Use ISO datetime-compatible strings.",
        )

    df = df.group_by("ds", maintain_order=True).agg(pl.col("y").last()).sort("ds")

    if request.n_prev is not None and request.n_prev < len(df):
        df_context = df.tail(request.n_prev)
    else:
        df_context = df

    forecaster = await _get_forecaster(app, request.model_type)

    try:
        forecast_df = await run_in_threadpool(
            forecaster.predict,
            df=df_context,
            n_predict=request.n_predict,
            alpha=request.alpha,
            **request.model_params,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid forecasting input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {str(e)}")

    forecast_df = forecast_df.with_columns(pl.col("ds").dt.strftime("%Y-%m-%d %H:%M:%S"))

    points = [ForecastPoint(**row) for row in forecast_df.to_dicts()]

    return ForecastResponse(model_used=request.model_type, forecast=points)
