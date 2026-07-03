"""Concrete downloader implementations for each external data source."""

from .binance_download import BinanceDownloader
from .eurostat_download import EurostatDownloader
from .fred_download import FredDownloader
from .github_download import NewsDownloader
from .world_bank_download import WorldBankDownloader
from .yahoo_download import YahooDownloader

__all__ = [
    "BinanceDownloader",
    "EurostatDownloader",
    "FredDownloader",
    "NewsDownloader",
    "WorldBankDownloader",
    "YahooDownloader",
]
