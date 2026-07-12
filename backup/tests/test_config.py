"""Tests for the typed ``config.yaml`` view."""

from pathlib import Path

from config import load_config


def test_load_config_reads_backup_block(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
postgres:
  host: db
  port: 5432
qdrant:
  host: vector_db
  port: 6333
backup:
  enabled: true
  interval_minutes: 30
  run_on_start: false
  rclone_remote: s3remote
  rclone_path: my-backups
  retention_days: 14
""",
        encoding="utf-8",
    )
    config = load_config(cfg_path)
    assert config.postgres.host == "db"
    assert config.qdrant.port == 6333
    assert config.backup.enabled is True
    assert config.backup.interval_minutes == 30
    assert config.backup.run_on_start is False
    assert config.backup.rclone_remote == "s3remote"
    assert config.backup.rclone_path == "my-backups"
    assert config.backup.retention_days == 14


def test_load_config_defaults_when_backup_block_absent(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("postgres:\n  host: db\n  port: 5432\n", encoding="utf-8")
    config = load_config(cfg_path)
    # A fresh deploy must not back up until explicitly enabled.
    assert config.backup.enabled is False
    assert config.backup.interval_minutes == 60.0
    assert config.backup.run_on_start is True
    assert config.backup.rclone_remote == ""
    assert config.backup.rclone_config_path == "_container_data/rclone.conf"
    assert config.backup.staging_dir == "_container_data/staging"
    assert config.backup.retention_days == 7
