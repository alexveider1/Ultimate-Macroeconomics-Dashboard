"""Concrete downloader implementations for each external data source."""

from .actually_relevant_download import ActuallyRelevantDownloader
from .binance_download import BinanceDownloader
from .eurostat_download import EurostatDownloader
from .fred_download import FredDownloader
from .github_download import NewsDownloader
from .world_bank_articles_download import WorldBankArticlesDownloader
from .world_bank_download import WorldBankDownloader
from .yahoo_download import YahooDownloader

__all__ = [
    "ActuallyRelevantDownloader",
    "BinanceDownloader",
    "EurostatDownloader",
    "FredDownloader",
    "NewsDownloader",
    "WorldBankArticlesDownloader",
    "WorldBankDownloader",
    "YahooDownloader",
]
