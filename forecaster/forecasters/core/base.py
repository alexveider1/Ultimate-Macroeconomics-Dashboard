from abc import ABC, abstractmethod
import polars as pl
import pandas as pd


class BaseForecaster(ABC):
    """Class constructor for forecaster objects"""

    def __init__(self):
        pass

    @abstractmethod
    def predict(
        self,
        df: pl.DataFrame,
        n_predict: int,
        alpha: float,
    ) -> pl.DataFrame:
        pass


def resolve_forecast_frequency(
    datetimes: pd.DatetimeIndex | list[pd.Timestamp],
    default: str = "D",
) -> str:
    """Infer a stable frequency, with a robust fallback for short or irregular series."""
    idx = pd.DatetimeIndex(pd.to_datetime(datetimes)).sort_values()
    if len(idx) < 2:
        return default

    inferred = pd.infer_freq(idx)
    if inferred:
        return inferred

    deltas = idx.to_series().diff().dropna()
    positive_deltas = deltas[deltas > pd.Timedelta(0)]
    if positive_deltas.empty:
        return default

    most_common_delta = positive_deltas.mode().iloc[0]
    try:
        return pd.tseries.frequencies.to_offset(most_common_delta).freqstr
    except Exception:
        return default
