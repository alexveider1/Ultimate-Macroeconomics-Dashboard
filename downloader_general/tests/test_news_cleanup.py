"""Test that the news pipeline deletes its downloaded files after Qdrant upload.

``cleanup_downloaded_files`` only touches ``self.save_path``, so the downloader is
built via ``__new__`` to avoid the network-touching ``__init__`` (tiktoken /
clients) — the test just needs the attribute set.
"""

from pathlib import Path

from src.extractors.github_download import NewsDownloader


def test_cleanup_removes_all_downloaded_files(tmp_path: Path) -> None:
    save_path = tmp_path / "news"
    save_path.mkdir()
    # A cloned-repo layout: a top-level file, a nested dataset dir with articles.
    (save_path / "some_archive.zip").write_text("zip-bytes")
    datasets = save_path / "News_Datasets"
    datasets.mkdir()
    (datasets / "article.json").write_text("{}")
    extracted = save_path / "business_positive_20240101000000"
    extracted.mkdir()
    (extracted / "a.json").write_text("{}")

    downloader = NewsDownloader.__new__(NewsDownloader)
    downloader.save_path = save_path

    downloader.cleanup_downloaded_files()

    assert save_path.exists()  # the dir itself stays; only its contents go
    assert list(save_path.iterdir()) == []


def test_cleanup_noop_when_save_path_absent(tmp_path: Path) -> None:
    downloader = NewsDownloader.__new__(NewsDownloader)
    downloader.save_path = tmp_path / "does_not_exist"
    # Must not raise when there's nothing to clean.
    downloader.cleanup_downloaded_files()
