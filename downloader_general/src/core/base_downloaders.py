"""Abstract base classes for the three one-shot downloaders.

Each concrete downloader in ``src/extractors/`` extends one of these to
guarantee a consistent ``_initialize_connections`` / per-step / ``run``
interface, so the entry-point in ``main.py`` can drive them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import httpx
from tiktoken import Encoding


class BaseWorldBankDownloader(ABC):
    """Abstract contract for any World Bank downloader implementation."""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        """Test whether sql and `world-bank` connections can be established"""
        pass

    @abstractmethod
    async def download_basic_tables(self, client: httpx.AsyncClient) -> None:
        """Download basic `world-bank` tables"""
        pass

    @abstractmethod
    async def download_metadata(
        self, client: httpx.AsyncClient, indicator_id: str, db: int
    ) -> None:
        """Download metadata for table from `world-bank`"""
        pass

    @abstractmethod
    async def download_db(self, client: httpx.AsyncClient, indicator_id: str, db: int) -> None:
        """Download table from `world-bank`"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for downloading all the needed tables from `world-bank`"""
        pass


class BaseNewsDownloader(ABC):
    """Abstract contract for any GitHub-sourced news downloader implementation."""

    @abstractmethod
    def _initialize_connections(self) -> bool:
        """Test whether `GitHub` can be established"""
        pass

    @abstractmethod
    def _build_embedding_encoding(self) -> Encoding:
        """Build encoding for OpenAI embedding model"""
        pass

    @abstractmethod
    def _truncate_for_embedding(self, text: str, article_path: str) -> str:
        """Truncate text to fit within OpenAI embedding token limit"""
        pass

    @abstractmethod
    def download_repository(self) -> bool:
        """Fetch repository with news from `github`"""
        pass

    @abstractmethod
    def parse_repository(self) -> None:
        """Parse news from repository"""
        pass

    @abstractmethod
    def clean_repository(self) -> None:
        """Clean up downloaded repository to free up space"""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for given texts"""
        pass

    @abstractmethod
    def upload_to_qdrant(self) -> None:
        """Upload news and embeddings to Qdrant vector database"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for fetching and parsing news from `github`"""
        pass


class BaseYahooDownloader(ABC):
    """Abstract contract for any Yahoo Finance downloader implementation."""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        """Test whether `yahoo-finance` can be established"""
        pass

    @abstractmethod
    def download_historical_data(self, ticker_id: str, category: str, period: str = "max") -> None:
        """Download historical data for given ticker from `yahoo-finance`"""
        pass

    @abstractmethod
    def download_metadata(self, ticker_id: str, asset_name: str, category: str) -> bool:
        """Download metadata for given ticker from `yahoo-finance`"""
        pass

    @abstractmethod
    def download_category(self, category: str, assets: List[Dict[str, str]]) -> None:
        """Download historical data and metadata for all tickers in given category"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for downloading all the needed data from `yahoo-finance`"""
        pass


class BaseFredDownloader(ABC):
    """Abstract contract for any FRED state-indicator downloader implementation."""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        """Test whether the SQL and FRED API connections can be established"""
        pass

    @abstractmethod
    async def download_states(self, client: httpx.AsyncClient) -> Dict[str, str]:
        """Write the states catalogue and return the FRED ``{fips: name}`` mapping"""
        pass

    @abstractmethod
    async def download_indicator(
        self, client: httpx.AsyncClient, slug: str, series_id: str, name: str, category: str
    ) -> None:
        """Download one indicator's description row + annual state panel from `fred`"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for downloading all the needed state indicators from `fred`"""
        pass


class BaseEurostatDownloader(ABC):
    """Abstract contract for any Eurostat NUTS-region downloader implementation."""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        """Test whether the SQL and Eurostat API connections can be established"""
        pass

    @abstractmethod
    def download_regions(self) -> None:
        """Write the NUTS-2 regions catalogue from the bundled GISCO GeoJSON"""
        pass

    @abstractmethod
    async def download_indicator(
        self,
        client: httpx.AsyncClient,
        slug: str,
        dataset: str,
        filters: Dict[str, str],
        name: str,
        category: str,
    ) -> None:
        """Download one indicator's description row + annual region panel from `eurostat`"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for downloading all the needed region indicators from `eurostat`"""
        pass


class BaseActuallyRelevantDownloader(ABC):
    """Abstract contract for the Actually Relevant curated-news downloader."""

    @abstractmethod
    def _initialize_connections(self) -> bool:
        """Test whether the Actually Relevant API, OpenAI and Qdrant are reachable"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Fetch every story, bucket by macro-topic, embed and upload to Qdrant"""
        pass


class BaseWorldBankArticlesDownloader(ABC):
    """Abstract contract for the World Bank documents (WDS) downloader."""

    @abstractmethod
    def _initialize_connections(self) -> bool:
        """Test whether the World Bank WDS API, OpenAI and Qdrant are reachable"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Per query fetch top-N docs, chunk their text, embed and upload to Qdrant"""
        pass


class BaseBinanceDownloader(ABC):
    """Abstract contract for any Binance crypto downloader implementation."""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        """Test whether the SQL connection can be established"""
        pass

    @abstractmethod
    async def select_top_symbols(self, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
        """Pick the most popular spot pairs from `binance` (ranked, with metadata)"""
        pass

    @abstractmethod
    def download_metadata(self, rows: List[Dict[str, Any]]) -> None:
        """Write the selected pairs' master data to Postgres"""
        pass

    @abstractmethod
    async def download_historical_data(
        self, client: httpx.AsyncClient, symbol: str, base_asset: str
    ) -> None:
        """Download the full candle history for one symbol from `binance`"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for downloading all the needed data from `binance`"""
        pass
