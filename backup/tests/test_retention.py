"""Tests for the pure rclone argv builders."""

from rclone_runner import build_copy_args, build_prune_args


def test_build_copy_args() -> None:
    args = build_copy_args(
        config_path="/app/_container_data/rclone.conf",
        source="/app/_container_data/staging",
        remote="s3remote",
        remote_path="macro-backups",
    )
    assert args == [
        "rclone",
        "--config",
        "/app/_container_data/rclone.conf",
        "copy",
        "/app/_container_data/staging",
        "s3remote:macro-backups",
    ]


def test_build_prune_args_uses_min_age_days() -> None:
    args = build_prune_args(
        config_path="/app/_container_data/rclone.conf",
        remote="s3remote",
        remote_path="macro-backups",
        retention_days=7,
    )
    assert args[:5] == [
        "rclone",
        "--config",
        "/app/_container_data/rclone.conf",
        "delete",
        "s3remote:macro-backups",
    ]
    assert args[5:] == ["--min-age", "7d"]
