import polars as pl
import pandas as pd
import pmdarima as pm

from .core.base import BaseForecaster, resolve_forecast_frequency


class ArimaForecaster(BaseForecaster):
    def __init__(self):
        super().__init__()

    def predict(self, df: pl.DataFrame, n_predict: int, alpha: float) -> pl.DataFrame:
        y = df["y"].to_numpy()

        model = pm.auto_arima(y, seasonal=False, suppress_warnings=True)

        forecasts, conf_int = model.predict(
            n_periods=n_predict, return_conf_int=True, alpha=alpha
        )

        last_date = df["ds"].max()
        freq = resolve_forecast_frequency(pd.DatetimeIndex(df["ds"].to_list()))
        future_dates = pd.date_range(start=last_date, periods=n_predict + 1, freq=freq)[
            1:
        ]

        return pl.DataFrame(
            {
                "ds": future_dates,
                "yhat": forecasts,
                "yhat_lower": conf_int[:, 0],
                "yhat_upper": conf_int[:, 1],
            }
        )
