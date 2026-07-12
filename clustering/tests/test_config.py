"""Validation tests for the clustering service's typed config view."""

from config import ClusteringConfig
from pydantic import ValidationError
import pytest


def test_valid_config_parses() -> None:
    cfg = ClusteringConfig.model_validate({"clustering": {"port": 8002}, "app": {"port": 8501}})
    assert cfg.clustering.port == 8002


def test_default_when_section_absent() -> None:
    cfg = ClusteringConfig.model_validate({})
    assert cfg.clustering.port == 8002
    # Triton endpoint defaults to the compose service name + gRPC port.
    assert cfg.triton.host == "triton"
    assert cfg.triton.grpc_port == 8001


def test_triton_section_parses() -> None:
    cfg = ClusteringConfig.model_validate(
        {"triton": {"host": "triton", "grpc_port": 8001, "http_port": 8000}}
    )
    assert cfg.triton.http_port == 8000


def test_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        ClusteringConfig.model_validate({"clustering": {"port": "not-an-int"}})
