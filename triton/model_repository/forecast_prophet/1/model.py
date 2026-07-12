"""Triton python-backend model: Facebook Prophet forecaster (CPU)."""

from umd_common.forecast_backend import ForecastModelBase


class TritonPythonModel(ForecastModelBase):
    """Prophet; refits per request, ``interval_width = 1 - alpha``."""

    MODEL = "prophet"
