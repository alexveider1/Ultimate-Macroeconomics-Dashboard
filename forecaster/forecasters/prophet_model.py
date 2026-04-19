import polars as pl
import pandas as pd
from prophet import Prophet
from .core.base import BaseForecaster, resolve_forecast_frequency


class ProphetForecaster(BaseForecaster):
    def __init__(self):
        super().__init__()

    def predict(self, df: pl.DataFrame, n_predict: int, alpha: float) -> pl.DataFrame:
        interval_width = 1.0 - alpha

        pdf = pd.DataFrame(
            {
                "ds": pd.to_datetime(df["ds"].to_list()),
                "y": df["y"].to_list(),
            }
        )

        model = Prophet(interval_width=interval_width)
        model.fit(pdf)

        freq = resolve_forecast_frequency(pd.DatetimeIndex(pdf["ds"]))
        future = model.make_future_dataframe(periods=n_predict, freq=freq)
        forecast = model.predict(future)

        future_forecast = forecast.tail(n_predict)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]

        return pl.from_pandas(future_forecast)
