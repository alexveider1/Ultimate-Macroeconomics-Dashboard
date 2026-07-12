"""Thin glue between Triton's ``pb_utils`` tensors and plain JSON dicts.

Both inference directions carry a single ``TYPE_STRING`` tensor holding one
UTF-8 JSON document (``INPUT`` in, ``OUTPUT`` out). Modelling the heterogeneous,
keyword-heavy forecasting/clustering payloads as one JSON blob is far simpler
than mapping ~25 hyperparameters onto typed tensors, and it keeps the request
contract identical to the old FastAPI bodies.

This module imports ``triton_python_backend_utils`` and therefore only imports
cleanly inside a running Triton python-backend stub — never in unit tests.
"""

import json
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore[import-not-found]

_INPUT_NAME = "INPUT"
_OUTPUT_NAME = "OUTPUT"


def parse_json_input(request: Any, name: str = _INPUT_NAME) -> dict[str, Any]:
    """Decode the single JSON string tensor named ``name`` into a dict."""
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    raw = tensor.as_numpy().reshape(-1)[0]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def make_json_response(payload: dict[str, Any], name: str = _OUTPUT_NAME) -> Any:
    """Wrap ``payload`` as a one-element ``TYPE_STRING`` output tensor."""
    encoded = json.dumps(payload).encode("utf-8")
    tensor = pb_utils.Tensor(name, np.array([encoded], dtype=np.object_))
    return pb_utils.InferenceResponse(output_tensors=[tensor])
