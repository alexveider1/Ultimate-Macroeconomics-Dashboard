"""Triton python-backend model: manual-order SARIMA forecaster (CPU)."""

from umd_common.forecast_backend import ForecastModelBase


class TritonPythonModel(ForecastModelBase):
    """statsmodels SARIMAX((p,d,q),(P,D,Q,s)); orders come from ``model_params``."""

    MODEL = "sarima"
