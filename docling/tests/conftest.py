"""Test fixtures: ensure a stub config.yaml exists before ``main`` is imported.

``main.py`` reads ``config.yaml`` at module import time (for the docling port +
the Triton VLM endpoint), so the test process must have one in cwd. Triton is
never contacted — building the converter only constructs option objects, and the
convert tests inject a fake converter onto ``app.state``.
"""

from __future__ import annotations

from pathlib import Path

_STUB = """\
docling:
  port: 8006
  convert_timeout_seconds: 120
triton:
  host: triton
  openai_port: 9000
  vlm_model: granite_docling
"""


def _ensure_config_file() -> None:
    config_path = Path.cwd() / "config.yaml"
    if not config_path.is_file():
        config_path.write_text(_STUB, encoding="utf-8")


_ensure_config_file()
