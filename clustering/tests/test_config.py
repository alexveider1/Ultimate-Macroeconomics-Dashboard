"""Validation tests for the clustering service's typed config view."""

import pytest
from pydantic import ValidationError

from config import ClusteringConfig


def test_valid_config_parses() -> None:
    cfg = ClusteringConfig.model_validate({"clustering": {"port": 8002}, "app": {"port": 8501}})
    assert cfg.clustering.port == 8002


def test_default_when_section_absent() -> None:
    assert ClusteringConfig.model_validate({}).clustering.port == 8002


def test_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        ClusteringConfig.model_validate({"clustering": {"port": "not-an-int"}})
