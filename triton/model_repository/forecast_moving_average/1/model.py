"""Triton python-backend model: moving-average baseline forecaster (CPU)."""

from umd_common.forecast_backend import ForecastModelBase


class TritonPythonModel(ForecastModelBase):
    """Flat trailing-window mean with random-walk-style CI growth."""

    MODEL = "moving_average"
