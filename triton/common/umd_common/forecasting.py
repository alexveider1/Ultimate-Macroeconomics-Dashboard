"""Forecasting inference ported from the old ``forecaster/forecasters/*`` wrappers.

Each function takes a decoded request payload::

    {
        "dates": [iso, ...],
        "values": [float, ...],
        "n_predict": int,
        "alpha": float,
        "model_params": {...},
    }

and returns a JSON-clean dict::

    {"ds": [iso, ...], "yhat": [...], "yhat_lower": [...], "yhat_upper": [...]}

The maths is intentionally kept identical to the pre-Triton wrappers so the
numbers the dashboard renders don't move. Heavy libraries (statsmodels,
pmdarima, prophet, xgboost, torch/chronos) are imported lazily inside the
functions that need them, so importing this module is cheap and the CPU-only
paths stay usable in tests without a GPU.
"""

import math
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .timeutil import resolve_forecast_frequency


class InputError(ValueError):
    """Raised for bad user input; the adapter maps it to HTTP 400."""


def _future_index(dates: list[str], n_predict: int) -> pd.DatetimeIndex:
    """Return the ``n_predict`` future timestamps following ``dates``."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    last_date = idx.max()
    freq = resolve_forecast_frequency(idx)
    return pd.date_range(start=last_date, periods=n_predict + 1, freq=freq)[1:]


def _finalize(
    future: pd.DatetimeIndex,
    yhat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    """Assemble the response dict with ISO timestamps and plain float lists."""
    return {
        "ds": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future],
        "yhat": [float(v) for v in np.asarray(yhat, dtype=float)],
        "yhat_lower": [float(v) for v in np.asarray(lower, dtype=float)],
        "yhat_upper": [float(v) for v in np.asarray(upper, dtype=float)],
    }


def _auto_arima(dates: list[str], y: np.ndarray, n_predict: int, alpha: float, _: dict) -> dict:
    import pmdarima as pm

    model = pm.auto_arima(y, seasonal=False, suppress_warnings=True)
    forecasts, conf_int = model.predict(n_periods=n_predict, return_conf_int=True, alpha=alpha)
    future = _future_index(dates, n_predict)
    return _finalize(future, forecasts, conf_int[:, 0], conf_int[:, 1])


def _arima(dates: list[str], y: np.ndarray, n_predict: int, alpha: float, params: dict) -> dict:
    from statsmodels.tsa.arima.model import ARIMA

    order = (int(params.get("p", 1)), int(params.get("d", 1)), int(params.get("q", 1)))
    model = ARIMA(y, order=order).fit()
    forecast = model.get_forecast(steps=n_predict)
    yhat = np.asarray(forecast.predicted_mean, dtype=float)
    conf = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
    future = _future_index(dates, n_predict)
    return _finalize(future, yhat, conf[:, 0], conf[:, 1])


def _sarima(dates: list[str], y: np.ndarray, n_predict: int, alpha: float, params: dict) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order = (int(params.get("p", 1)), int(params.get("d", 1)), int(params.get("q", 1)))
    seasonal_order = (
        int(params.get("P", 0)),
        int(params.get("D", 0)),
        int(params.get("Q", 0)),
        int(params.get("s", 12)),
    )
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    forecast = model.get_forecast(steps=n_predict)
    yhat = np.asarray(forecast.predicted_mean, dtype=float)
    conf = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
    future = _future_index(dates, n_predict)
    return _finalize(future, yhat, conf[:, 0], conf[:, 1])


def _prophet(dates: list[str], y: np.ndarray, n_predict: int, alpha: float, _: dict) -> dict:
    from prophet import Prophet

    pdf = pd.DataFrame({"ds": pd.to_datetime(dates), "y": y})
    model = Prophet(interval_width=1.0 - alpha)
    model.fit(pdf)
    freq = resolve_forecast_frequency(pd.DatetimeIndex(pdf["ds"]))
    future = model.make_future_dataframe(periods=n_predict, freq=freq)
    forecast = model.predict(future).tail(n_predict)
    return _finalize(
        pd.DatetimeIndex(forecast["ds"]),
        forecast["yhat"].to_numpy(),
        forecast["yhat_lower"].to_numpy(),
        forecast["yhat_upper"].to_numpy(),
    )


def _moving_average(
    dates: list[str], y: np.ndarray, n_predict: int, alpha: float, params: dict
) -> dict:
    from scipy.stats import norm

    n = len(y)
    window = int(params.get("window", 5))
    if window <= 0 or window > n:
        window = max(1, min(n, n // 4 if n >= 4 else n))

    forecast_value = float(np.mean(y[-window:]))
    yhat = np.full(n_predict, forecast_value, dtype=float)

    if n > window:
        in_sample_means = np.array(
            [float(np.mean(y[i - window : i])) for i in range(window, n)],
            dtype=float,
        )
        residuals = y[window:] - in_sample_means
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    else:
        sigma = float(np.std(y, ddof=1)) if n > 1 else 0.0
    if not math.isfinite(sigma):
        sigma = 0.0

    z = float(norm.ppf(1.0 - alpha / 2.0))
    horizons = np.arange(1, n_predict + 1, dtype=float)
    margin = z * sigma * np.sqrt(horizons)
    future = _future_index(dates, n_predict)
    return _finalize(future, yhat, yhat - margin, yhat + margin)


def _build_training_matrix(values: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` aligned for supervised training over lag features."""
    n = len(values)
    rows: list[list[float]] = []
    targets: list[float] = []
    for t in range(lags, n):
        window = values[t - lags : t]
        feat: list[float] = list(window)
        feat.append(float(np.mean(window)))
        feat.append(float(np.std(window, ddof=0)))
        feat.append(float(t))
        rows.append(feat)
        targets.append(float(values[t]))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _features_for_step(history: np.ndarray, lags: int, t_index: int) -> np.ndarray:
    """Return the 1xF feature row used to predict the next step."""
    window = history[-lags:]
    feat: list[float] = list(window)
    feat.append(float(np.mean(window)))
    feat.append(float(np.std(window, ddof=0)))
    feat.append(float(t_index))
    return np.asarray(feat, dtype=float).reshape(1, -1)


def _xgboost(dates: list[str], y: np.ndarray, n_predict: int, alpha: float, params: dict) -> dict:
    from scipy.stats import norm
    from xgboost import XGBRegressor

    lags = max(1, int(params.get("lags", 5)))
    if len(y) <= lags + 1:
        raise InputError(
            f"Need at least {lags + 2} historical points to train XGBoost with lags={lags}."
        )

    X, target = _build_training_matrix(y, lags)

    # ``device="cuda"`` runs the tree build + inference on the GPU this
    # python-backend instance is pinned to (config.pbtxt KIND_GPU).
    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        objective="reg:squarederror",
        verbosity=0,
        random_state=42,
        tree_method="hist",
        device="cuda",
    )
    model.fit(X, target)

    in_sample_pred = np.asarray(model.predict(X), dtype=float)
    residuals = target - in_sample_pred
    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    if not math.isfinite(sigma):
        sigma = 0.0

    z = float(norm.ppf(1.0 - alpha / 2.0))
    history = y.copy()
    forecasts: list[float] = []
    for _ in range(n_predict):
        feat = _features_for_step(history, lags, len(history))
        next_pred = float(model.predict(feat)[0])
        forecasts.append(next_pred)
        history = np.append(history, next_pred)

    yhat = np.asarray(forecasts, dtype=float)
    horizons = np.arange(1, n_predict + 1, dtype=float)
    margin = z * sigma * np.sqrt(horizons)
    future = _future_index(dates, n_predict)
    return _finalize(future, yhat, yhat - margin, yhat + margin)


_DISPATCH = {
    "auto_arima": _auto_arima,
    "arima": _arima,
    "sarima": _sarima,
    "prophet": _prophet,
    "moving_average": _moving_average,
    "xgboost": _xgboost,
}


def _extract(payload: dict[str, Any]) -> tuple[list[str], np.ndarray, int, float, dict]:
    """Pull the common ``(dates, y, n_predict, alpha, params)`` tuple out of a payload."""
    dates = list(payload["dates"])
    y = np.asarray(payload["values"], dtype=float)
    n_predict = int(payload["n_predict"])
    alpha = float(payload.get("alpha", 0.05))
    params = dict(payload.get("model_params") or {})
    return dates, y, n_predict, alpha, params


def run(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a stateless forecaster by id and return the forecast dict.

    Args:
        model: One of ``auto_arima``, ``arima``, ``sarima``, ``prophet``,
            ``moving_average``, ``xgboost`` (``chronos`` is served by
            :class:`ChronosRunner` because it preloads weights).
        payload: Decoded request body.

    Raises:
        InputError: For unknown model ids or invalid inputs (mapped to 400).
    """
    fn = _DISPATCH.get(model)
    if fn is None:
        raise InputError(f"Unknown forecasting model: {model}")
    dates, y, n_predict, alpha, params = _extract(payload)
    return fn(dates, y, n_predict, alpha, params)


class ChronosRunner:
    """Holds an Amazon Chronos pipeline loaded once onto the GPU.

    Kept as a class (not a function in ``_DISPATCH``) because loading the
    checkpoint is expensive and is done once in the Triton model's
    ``initialize`` rather than per request.
    """

    def __init__(self, model_name: str | None = None):
        """Load the Chronos pipeline onto CUDA (bf16) — GPU is required here."""
        from chronos import ChronosPipeline
        import torch

        self._torch = torch
        name = model_name or "amazon/chronos-t5-small"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*`torch_dtype` is deprecated.*",
                category=FutureWarning,
            )
            self.pipeline = ChronosPipeline.from_pretrained(
                name, device_map=device, torch_dtype=dtype
            )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sample Chronos and return the median forecast + alpha CI dict."""
        torch = self._torch
        dates, y, n_predict, alpha, _ = _extract(payload)

        context = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32)
        with torch.inference_mode():
            forecast_tensor = self.pipeline.predict(context, prediction_length=n_predict)

        samples = forecast_tensor[0].detach().cpu().numpy()
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]

        yhat = np.median(samples, axis=0)
        yhat_lower = np.quantile(samples, alpha / 2.0, axis=0)
        yhat_upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
        future = _future_index(dates, n_predict)
        return _finalize(future, yhat, yhat_lower, yhat_upper)
