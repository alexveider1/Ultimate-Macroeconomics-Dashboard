"""Tests for the single-run orchestration and the loop's error swallowing."""

from pathlib import Path

import backup_runner
from config import BackupConfig, BackupServiceConfig, PostgresConfig, QdrantConfig
import pytest
from settings import Settings


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("POSTGRES_USER", "main")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pw")
    monkeypatch.setenv("POSTGRES_DB", "macro")
    monkeypatch.setenv("QDRANT__SERVICE__API_KEY", "k")
    return Settings()  # ty: ignore[missing-argument]  # populated from env


def _config(tmp_path: Path, **backup_overrides: object) -> BackupServiceConfig:
    rclone_conf = tmp_path / "rclone.conf"
    rclone_conf.write_text("[s3remote]\ntype = local\n", encoding="utf-8")
    defaults: dict[str, object] = {
        "enabled": True,
        "rclone_remote": "s3remote",
        "rclone_path": "macro-backups",
        "rclone_config_path": str(rclone_conf),
        "staging_dir": str(tmp_path / "staging"),
        "retention_days": 7,
    }
    defaults.update(backup_overrides)
    return BackupServiceConfig(
        postgres=PostgresConfig(host="db", port=5432, database="macro"),
        qdrant=QdrantConfig(host="vector_db", port=6333),
        backup=BackupConfig.model_validate(defaults),
    )


def _patch_steps(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr(
        backup_runner, "dump_postgres", lambda **kw: calls.append("dump") or kw["out_path"]
    )
    monkeypatch.setattr(
        backup_runner, "snapshot_qdrant", lambda **kw: calls.append("snapshot") or Path("snap")
    )
    monkeypatch.setattr(backup_runner.rclone_runner, "upload", lambda **kw: calls.append("upload"))
    monkeypatch.setattr(backup_runner.rclone_runner, "prune", lambda **kw: calls.append("prune"))


def test_run_backup_once_calls_steps_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_steps(monkeypatch, calls)
    backup_runner.run_backup_once(_config(tmp_path), _settings(monkeypatch))
    assert calls == ["dump", "snapshot", "upload", "prune"]


def test_run_backup_once_skips_when_no_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_steps(monkeypatch, calls)
    backup_runner.run_backup_once(_config(tmp_path, rclone_remote=""), _settings(monkeypatch))
    assert calls == []  # nothing dumped/uploaded without a configured remote


def test_run_backup_once_skips_when_rclone_conf_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    _patch_steps(monkeypatch, calls)
    backup_runner.run_backup_once(
        _config(tmp_path, rclone_config_path=str(tmp_path / "nope.conf")),
        _settings(monkeypatch),
    )
    assert calls == []


def test_run_backup_once_propagates_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_kw: object) -> Path:
        raise RuntimeError("pg_dump exploded")

    monkeypatch.setattr(backup_runner, "dump_postgres", boom)
    with pytest.raises(RuntimeError, match="pg_dump exploded"):
        backup_runner.run_backup_once(_config(tmp_path), _settings(monkeypatch))


def test_safe_run_swallows_exceptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def boom(_config: object, _settings: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_backup_once", boom)
    # Must not raise — the scheduler loop relies on this.
    main._safe_run(_config(tmp_path), _settings(monkeypatch))
