"""Validation tests for the python_sandbox service's typed config view."""

from config import PythonSandboxConfig
from pydantic import ValidationError
import pytest


def test_valid_config_parses() -> None:
    cfg = PythonSandboxConfig.model_validate(
        {"python_sandbox": {"port": 8004}, "app": {"port": 8501}}
    )
    assert cfg.python_sandbox.port == 8004


def test_default_when_section_absent() -> None:
    assert PythonSandboxConfig.model_validate({}).python_sandbox.port == 8004


def test_invalid_port_raises() -> None:
    with pytest.raises(ValidationError):
        PythonSandboxConfig.model_validate({"python_sandbox": {"port": "nope"}})
