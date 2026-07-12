"""Frequency inference shared by every forecaster (ported verbatim).

Kept identical to the former ``forecaster/forecasters/core/base.py`` helper so
forecasts produced through Triton land on the same future timestamps as before.
"""

import pandas as pd


def resolve_forecast_frequency(
    datetimes: "pd.DatetimeIndex | list[pd.Timestamp]",
    default: str = "D",
) -> str:
    """Infer a stable pandas frequency string for ``datetimes``.

    First asks pandas to infer the frequency; if that fails (e.g. the series
    is irregular), falls back to the modal positive gap between successive
    timestamps. Returns ``default`` for series with fewer than two points or
    when no positive gap is available.

    Args:
        datetimes: Historical timestamps; need not be sorted.
        default: Frequency to return when nothing better can be inferred.

    Returns:
        A pandas frequency string suitable for ``pd.date_range(freq=...)``.
    """
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
