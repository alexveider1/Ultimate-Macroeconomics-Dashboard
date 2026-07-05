"""Triton python-backend model: Amazon Chronos forecaster (GPU, preloaded).

Unlike the stateless forecasters, Chronos has real pretrained weights, so the
pipeline is loaded once in ``initialize`` (not per request). The checkpoint name
is read from the bind-mounted ``config.yaml`` (single source of truth), matching
the ``forecaster.CHRONOS_MODEL`` key the old service used.
"""

from pathlib import Path

from umd_common import forecasting
from umd_common.triton_io import make_json_response, parse_json_input
import yaml

_CONFIG_PATH = Path("/app/config.yaml")


def _chronos_model_name() -> str | None:
    """Read ``forecaster.CHRONOS_MODEL`` from the mounted config, if present."""
    try:
        cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return (cfg.get("forecaster") or {}).get("CHRONOS_MODEL")
    except Exception:
        return None


class TritonPythonModel:
    """Load Chronos onto the GPU at startup, then sample per request."""

    def initialize(self, args):
        """Preload the Chronos pipeline (weights download on first boot)."""
        self.runner = forecasting.ChronosRunner(_chronos_model_name())

    def execute(self, requests: list) -> list:
        """Sample Chronos for each request's history and return a JSON tensor."""
        responses = []
        for request in requests:
            try:
                payload = parse_json_input(request)
                result = self.runner.predict(payload)
                responses.append(make_json_response({"ok": True, "result": result}))
            except forecasting.InputError as exc:
                responses.append(
                    make_json_response({"ok": False, "error_code": 400, "detail": str(exc)})
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller as HTTP 500
                responses.append(
                    make_json_response(
                        {"ok": False, "error_code": 500, "detail": f"{type(exc).__name__}: {exc}"}
                    )
                )
        return responses
