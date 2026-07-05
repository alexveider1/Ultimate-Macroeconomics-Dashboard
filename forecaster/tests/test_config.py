"""Validation tests for the forecaster's typed ``config.yaml`` view."""

from config import ForecasterConfig
from pydantic import ValidationError
import pytest


def test_valid_config_parses() -> None:
    cfg = ForecasterConfig.model_validate(
        {
            "forecaster": {
                "port": 8001,
                "ARIMA_AVAILABLE": True,
                "PROPHET_AVAILABLE": False,
                "CHRONOS_AVAILABLE": True,
                "CHRONOS_MODEL": "amazon/chronos-t5-tiny",
            },
            "app": {"port": 8501},  # foreign section is ignored
        }
    )
    assert cfg.forecaster.ARIMA_AVAILABLE is True
    assert cfg.forecaster.PROPHET_AVAILABLE is False
    assert cfg.forecaster.CHRONOS_MODEL == "amazon/chronos-t5-tiny"


def test_defaults_when_section_absent() -> None:
    cfg = ForecasterConfig.model_validate({})
    assert cfg.forecaster.ARIMA_AVAILABLE is False
    assert cfg.forecaster.CHRONOS_MODEL is None
    assert cfg.forecaster.port == 8001
    # Triton endpoint defaults to the compose service name + gRPC port.
    assert cfg.triton.host == "triton"
    assert cfg.triton.grpc_port == 8001


def test_triton_section_parses() -> None:
    cfg = ForecasterConfig.model_validate(
        {"triton": {"host": "triton", "grpc_port": 8001, "http_port": 8000}}
    )
    assert cfg.triton.grpc_port == 8001
    assert cfg.triton.http_port == 8000


def test_invalid_toggle_type_raises() -> None:
    with pytest.raises(ValidationError):
        ForecasterConfig.model_validate({"forecaster": {"ARIMA_AVAILABLE": "not-a-bool"}})
