"""Test fixtures: ensure a stub config.yaml exists before ``main`` is imported.

``main.py`` reads ``config.yaml`` at module import time (for the model toggles
and the Triton endpoint), so the test process must have one in cwd. The toggles
here are deliberately mixed (ARIMA on, Prophet/Chronos off) so the adapter's
enable/disable gating is exercised without a running Triton server.
"""

from __future__ import annotations

from pathlib import Path

_STUB = """\
forecaster:
  port: 8001
  ARIMA_AVAILABLE: true
  PROPHET_AVAILABLE: false
  CHRONOS_AVAILABLE: false
  CHRONOS_MODEL: amazon/chronos-t5-tiny
triton:
  host: triton
  grpc_port: 8001
  http_port: 8000
"""


def _ensure_config_file() -> None:
    config_path = Path.cwd() / "config.yaml"
    if not config_path.is_file():
        config_path.write_text(_STUB, encoding="utf-8")


_ensure_config_file()
