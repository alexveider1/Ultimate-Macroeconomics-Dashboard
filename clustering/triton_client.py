"""Minimal Triton gRPC helper: send one JSON request tensor, get one back.

The clustering algorithms live in the ``triton`` container as a python-backend
model (cuML on GPU where available, scikit-learn on CPU otherwise) that speaks a
single ``TYPE_STRING`` in / out contract (see ``triton/common``). This adapter
keeps its former HTTP contract and forwards the compute to Triton. Kept as a
tiny per-service module (duplicated in ``forecaster``) per the repo's
no-shared-package convention.
"""

import json
import os
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient


class TritonError(Exception):
    """A model-reported error carrying the HTTP status the adapter should raise."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Store ``status_code`` (400/500) and human-readable ``detail``."""
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def resolve_triton_url(host: str, grpc_port: int) -> str:
    """Return the Triton gRPC endpoint (``TRITON_GRPC_URL`` env overrides)."""
    return os.environ.get("TRITON_GRPC_URL") or f"{host}:{grpc_port}"


def create_client(url: str) -> grpcclient.InferenceServerClient:
    """Create a gRPC client (lazy — no connection until the first infer)."""
    return grpcclient.InferenceServerClient(url=url, verbose=False)


def infer_json(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Run one inference and return the decoded ``result`` dict.

    Args:
        client: A connected Triton gRPC client.
        model_name: Triton model to target (e.g. ``cluster``).
        payload: Request body serialised as the model's ``INPUT`` JSON tensor.

    Returns:
        The ``result`` object from the model's response envelope.

    Raises:
        TritonError: When the model reports ``ok: false`` (mapped to its
            ``error_code``), or when the transport/response is malformed (502).
    """
    encoded = np.array([json.dumps(payload).encode("utf-8")], dtype=np.object_)
    infer_input = grpcclient.InferInput("INPUT", [1], "BYTES")
    infer_input.set_data_from_numpy(encoded)
    requested = grpcclient.InferRequestedOutput("OUTPUT")

    try:
        response = client.infer(model_name=model_name, inputs=[infer_input], outputs=[requested])
    except Exception as exc:  # noqa: BLE001 - transport failures → 502
        raise TritonError(502, f"Triton inference call failed: {exc}") from exc

    raw = response.as_numpy("OUTPUT")
    if raw is None:
        raise TritonError(502, "Triton returned no OUTPUT tensor.")
    value = raw.reshape(-1)[0]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")

    envelope = json.loads(value)
    if not envelope.get("ok", False):
        raise TritonError(
            int(envelope.get("error_code", 500)),
            str(envelope.get("detail", "Inference failed.")),
        )
    return envelope["result"]
