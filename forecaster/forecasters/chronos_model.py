"""Amazon Chronos foundation-model forecaster wrapper."""

import warnings

from chronos import ChronosPipeline
import numpy as np
import pandas as pd
import polars as pl
import torch

from .core.base import BaseForecaster, resolve_forecast_frequency


class ChronosForecaster(BaseForecaster):
    """Wrapper around a Chronos T5 pipeline; loads the model once per instance."""

    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        """Load the Chronos pipeline onto CUDA when available, else CPU.

        Args:
            model_name: Hugging Face id of the Chronos checkpoint to load.
                The deprecation warning for ``torch_dtype`` from the upstream
                ``from_pretrained`` is intentionally suppressed.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*`torch_dtype` is deprecated.*",
                category=FutureWarning,
            )
            self.pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=dtype,
            )

    def predict(
        self,
        df: pl.DataFrame,
        n_predict: int,
        alpha: float,
        **kwargs,
    ) -> pl.DataFrame:
        """Run Chronos sampling and produce a median forecast + alpha CI.

        Args:
            df: Two-column ``(ds, y)`` Polars frame sorted ascending by ``ds``.
            n_predict: Forecast horizon in points.
            alpha: Significance level; the confidence interval uses the
                ``alpha/2`` and ``1-alpha/2`` quantiles of the sample set.

        Returns:
            Polars frame with ``ds``, ``yhat`` (median), ``yhat_lower``,
            ``yhat_upper``.
        """
        context_array = np.ascontiguousarray(df["y"].to_numpy())
        context = torch.tensor(context_array, dtype=torch.float32)

        with torch.inference_mode():
            forecast_tensor = self.pipeline.predict(context, prediction_length=n_predict)

        samples = forecast_tensor[0].detach().cpu().numpy()
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]

        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0

        yhat = np.median(samples, axis=0)
        yhat_lower = np.quantile(samples, lower_q, axis=0)
        yhat_upper = np.quantile(samples, upper_q, axis=0)

        last_date = df["ds"].max()
        freq = resolve_forecast_frequency(pd.DatetimeIndex(df["ds"].to_list()))
        future_dates = pd.date_range(start=last_date, periods=n_predict + 1, freq=freq)[1:]

        return pl.DataFrame(
            {
                "ds": future_dates,
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
            }
        )
