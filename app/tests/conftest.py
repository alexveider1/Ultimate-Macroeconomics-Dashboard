"""Shared fixtures for the app smoke suite.

`theming.py` looks for `themes.yaml` in a small set of locations; in the test
environment we point it at the bundled `_container_data/themes.yaml` via cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _cwd_in_container_data(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    themes_path = repo_root / "_container_data" / "themes.yaml"
    if not themes_path.is_file():
        pytest.skip(f"themes.yaml not found at {themes_path}")
    monkeypatch.chdir(themes_path.parent)
    # Streamlit needs HOME to be writable for its caching backend.
    monkeypatch.setenv("HOME", os.environ.get("HOME", str(repo_root)))
