from abc import ABC, abstractmethod
from typing import List, Dict
from tiktoken import Encoding


class BaseWorldBankDownloader(ABC):
    """Class constructor for basic `world-bank` downloader"""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> None:
        """Test whether sql and `world-bank` connections can be established"""
        pass

    @abstractmethod
    def create_llm_readonly_user(self) -> None:
        """Create or update the read-only SQL user used by the LLM"""
        pass

    @abstractmethod
    def download_basic_tables(self) -> None:
        """Download basic `world-bank` tables"""
        pass

    @abstractmethod
    def download_metadata(self, indicator_id: str, db: int) -> None:
        """Download metadata for table from `world-bank`"""
        pass

    @abstractmethod
    def download_db(self, indicator_id: str, db: int) -> None:
        """Download table from `world-bank`"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Method for downloading all the needed tables from `world-bank`"""
        pass


class BaseNewsDownloader(ABC):
    """Class constructor for basic news downloader from `github`"""

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
    def download_repository(self) -> None:
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
    def get_embeddings(self, text: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for given text"""
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
    """Class constructor for basic `yahoo-finance` downloader"""

    @abstractmethod
    def _initialize_connections(self, host: str, port: int, db: str) -> bool:
        """Test whether `yahoo-finance` can be established"""
        pass

    @abstractmethod
    def download_historical_data(
        self, ticker_id: str, category: str, period: str = "max"
    ) -> None:
        """Download historical data for given ticker from `yahoo-finance`"""
        pass

    @abstractmethod
    def download_metadata(self, ticker_id: str, asset_name: str, category: str) -> None:
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
