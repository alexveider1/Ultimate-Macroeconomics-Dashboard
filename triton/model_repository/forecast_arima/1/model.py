"""Triton python-backend model: manual-order ARIMA forecaster (CPU)."""

from umd_common.forecast_backend import ForecastModelBase


class TritonPythonModel(ForecastModelBase):
    """statsmodels ARIMA(p, d, q); orders come from ``model_params``."""

    MODEL = "arima"
