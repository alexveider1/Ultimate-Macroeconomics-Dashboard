"""Validation tests for the docling service's typed ``config.yaml`` view."""

from __future__ import annotations

from pathlib import Path

from config import DoclingConfig, load_config


def test_defaults_when_sections_absent() -> None:
    cfg = DoclingConfig.model_validate({})
    assert cfg.docling.port == 8006
    assert cfg.docling.convert_timeout_seconds == 120
    assert cfg.docling.vlm.base_url.endswith("/v1")
    assert cfg.docling.vlm.model == "ibm-granite/granite-docling-258M"


def test_load_config_reads_overrides(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(
        "docling:\n  port: 9006\n  convert_timeout_seconds: 45\n"
        "  vlm:\n    base_url: https://vlm-x/v1\n    model: mymodel\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.docling.port == 9006
    assert cfg.docling.convert_timeout_seconds == 45
    assert cfg.docling.vlm.base_url == "https://vlm-x/v1"
    assert cfg.docling.vlm.model == "mymodel"
