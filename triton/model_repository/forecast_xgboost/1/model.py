"""Triton python-backend model: recursive XGBoost forecaster (GPU)."""

from umd_common.forecast_backend import ForecastModelBase


class TritonPythonModel(ForecastModelBase):
    """Lag + rolling-feature XGBoost, trained on ``device="cuda"``."""

    MODEL = "xgboost"
