"""Shared ``execute`` loop for the clustering python-backend model."""

from . import clustering
from .triton_io import make_json_response, parse_json_input


class ClusterModelBase:
    """Decode → :func:`clustering.run` → encode, per request."""

    def execute(self, requests: list) -> list:
        """Handle a batch of Triton inference requests (one JSON blob each)."""
        return [self._handle(request) for request in requests]

    def _handle(self, request):
        try:
            payload = parse_json_input(request)
            result = clustering.run(payload)
            return make_json_response({"ok": True, "result": result})
        except clustering.InputError as exc:
            return make_json_response({"ok": False, "error_code": 400, "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as HTTP 500
            return make_json_response(
                {"ok": False, "error_code": 500, "detail": f"{type(exc).__name__}: {exc}"}
            )
