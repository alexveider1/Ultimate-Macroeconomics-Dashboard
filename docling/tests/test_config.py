"""Validation tests for the docling service's typed ``config.yaml`` view."""

from __future__ import annotations

from pathlib import Path

from config import DoclingConfig, load_config


def test_defaults_when_sections_absent() -> None:
    cfg = DoclingConfig.model_validate({})
    assert cfg.docling.port == 8006
    assert cfg.docling.convert_timeout_seconds == 120
    assert cfg.triton.host == "triton"
    assert cfg.triton.openai_port == 9000
    assert cfg.triton.vlm_model == "granite_docling"


def test_load_config_reads_overrides(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text(
        "docling:\n  port: 9006\n  convert_timeout_seconds: 45\n"
        "triton:\n  host: triton-x\n  openai_port: 9100\n  vlm_model: mymodel\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.docling.port == 9006
    assert cfg.docling.convert_timeout_seconds == 45
    assert cfg.triton.host == "triton-x"
    assert cfg.triton.openai_port == 9100
    assert cfg.triton.vlm_model == "mymodel"
