"""CPU-runnable smoke tests for the ported forecasting maths (``umd_common``).

These exercise the exact functions the Triton python-backend models call, so the
numbers match the pre-Triton wrappers. GPU-only models (chronos, GPU xgboost) are
not tested here — they need a device and are covered by the end-to-end run.
"""

from __future__ import annotations

from typing import Any

import pytest

from umd_common import forecasting


def _payload(model_params: dict[str, Any] | None = None, n_predict: int = 6) -> dict[str, Any]:
    dates = [f"2020-{m:02d}-01" for m in range(1, 13)] + [f"2021-{m:02d}-01" for m in range(1, 13)]
    values = [float(i) + (i % 3) * 0.5 for i in range(len(dates))]
    return {
        "dates": dates,
        "values": values,
        "n_predict": n_predict,
        "alpha": 0.05,
        "model_params": model_params or {},
    }


def _assert_shape(result: dict[str, Any], n: int) -> None:
    assert set(result.keys()) == {"ds", "yhat", "yhat_lower", "yhat_upper"}
    assert len(result["ds"]) == len(result["yhat"]) == n
    for lo, mid, hi in zip(result["yhat_lower"], result["yhat"], result["yhat_upper"]):
        assert lo <= mid <= hi


def test_moving_average_is_flat_and_shaped() -> None:
    result = forecasting.run("moving_average", _payload({"window": 4}))
    _assert_shape(result, 6)
    assert all(abs(v - result["yhat"][0]) < 1e-9 for v in result["yhat"])


def test_arima_shape() -> None:
    pytest.importorskip("statsmodels")
    result = forecasting.run("arima", _payload({"p": 1, "d": 1, "q": 1}))
    _assert_shape(result, 6)


def test_sarima_shape() -> None:
    pytest.importorskip("statsmodels")
    result = forecasting.run("sarima", _payload({"p": 1, "d": 0, "q": 1, "P": 1, "s": 12}))
    _assert_shape(result, 6)


def test_auto_arima_shape() -> None:
    pytest.importorskip("pmdarima")
    result = forecasting.run("auto_arima", _payload())
    _assert_shape(result, 6)


def test_unknown_model_raises_input_error() -> None:
    with pytest.raises(forecasting.InputError):
        forecasting.run("does_not_exist", _payload())


def test_future_dates_follow_history() -> None:
    result = forecasting.run("moving_average", _payload(n_predict=3))
    # Monthly history → first forecast timestamp is the next month start.
    assert result["ds"][0].startswith("2022-01")
