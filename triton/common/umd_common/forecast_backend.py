"""Shared ``execute`` loop for the stateless forecasting python-backend models.

Each ``model.py`` subclasses :class:`ForecastModelBase` and sets ``MODEL`` to
its forecaster id. Chronos is *not* built on this base — it preloads weights in
``initialize`` and lives in its own ``model.py``.
"""

from . import forecasting
from .triton_io import make_json_response, parse_json_input


class ForecastModelBase:
    """Decode → dispatch to :func:`forecasting.run` → encode, per request."""

    MODEL: str = ""

    def execute(self, requests: list) -> list:
        """Handle a batch of Triton inference requests (one JSON blob each)."""
        return [self._handle(request) for request in requests]

    def _handle(self, request):
        try:
            payload = parse_json_input(request)
            result = forecasting.run(self.MODEL, payload)
            return make_json_response({"ok": True, "result": result})
        except forecasting.InputError as exc:
            return make_json_response({"ok": False, "error_code": 400, "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as HTTP 500
            return make_json_response(
                {"ok": False, "error_code": 500, "detail": f"{type(exc).__name__}: {exc}"}
            )
