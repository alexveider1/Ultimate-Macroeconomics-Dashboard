"""Test fixtures: ensure a stub config.yaml exists before `main` is imported.

`main.py` reads `config.yaml` at module import time, so the test process must
have one in cwd. The shared `_container_data/config.yaml` works fine if it is
available; otherwise we drop a minimal stub.
"""

from __future__ import annotations

from pathlib import Path


def _ensure_config_file() -> None:
    config_path = Path.cwd() / "config.yaml"
    if not config_path.is_file():
        config_path.write_text("clustering:\n  port: 8002\n", encoding="utf-8")


_ensure_config_file()
