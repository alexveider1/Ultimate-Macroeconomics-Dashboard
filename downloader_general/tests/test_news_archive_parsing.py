"""Smoke test: per-archive worker helper extracts a zip and reads its JSON articles."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from src.extractors.github_download import NewsDownloader


def _make_archive(save_path: Path, base_name: str, payloads: list[dict]) -> Path:
    inner = save_path / f"{base_name}_inner"
    inner.mkdir(parents=True, exist_ok=True)
    for idx, payload in enumerate(payloads):
        (inner / f"article_{idx}.json").write_text(json.dumps(payload), encoding="utf-8")

    archive_path = save_path / f"{base_name}.zip"
    with ZipFile(archive_path, "w") as zf:
        for file_path in inner.iterdir():
            zf.write(file_path, arcname=f"{base_name}/{file_path.name}")
    return archive_path


def test_process_archive_extracts_english_articles(tmp_path: Path) -> None:
    base_name = "economy_positive_20230101000000"
    archive = _make_archive(
        tmp_path,
        base_name,
        [
            {"language": "english", "text": "Macro story"},
            {"language": "french", "text": "Filtered out"},
        ],
    )

    downloader = NewsDownloader.__new__(NewsDownloader)
    downloader.save_path = tmp_path

    result = downloader._process_archive(archive, allowed_topics=["economy"])
    assert result is not None
    collection_name, entries = result
    assert collection_name == "economy_positive_webhose"
    assert len(entries) == 1
    assert entries[0]["topic"] == "economy"
    assert entries[0]["sentiment"] == "positive"
    assert entries[0]["article"]["text"] == "Macro story"

    # The .zip is deleted right after extraction so peak disk stays low and the
    # raw archives never linger; the extracted JSONs remain for embedding.
    assert not archive.exists()
    assert (tmp_path / base_name).is_dir()


def test_process_archive_skips_non_zip(tmp_path: Path) -> None:
    file_path = tmp_path / "not_an_archive.txt"
    file_path.write_text("ignored")

    downloader = NewsDownloader.__new__(NewsDownloader)
    downloader.save_path = tmp_path

    assert downloader._process_archive(file_path, allowed_topics=["anything"]) is None
