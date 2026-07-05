"""Triton python-backend model: auto-tuned ARIMA (pmdarima, CPU)."""

from umd_common.forecast_backend import ForecastModelBase


class TritonPythonModel(ForecastModelBase):
    """Non-seasonal ``pmdarima.auto_arima``; refits per request."""

    MODEL = "auto_arima"
