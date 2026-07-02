"""Validation tests for the unified ``IngestRequest`` source/field matrix."""

from pydantic import ValidationError
import pytest
from schema import IngestRequest


def test_default_source_is_worldbank():
    req = IngestRequest(indicator_id="NY.GDP.MKTP.CD", db_id=2)
    assert req.source == "worldbank"
    assert req.indicator_id == "NY.GDP.MKTP.CD"
    assert req.db_id == 2


def test_worldbank_requires_indicator_and_db():
    with pytest.raises(ValidationError):
        IngestRequest(source="worldbank", indicator_id="NY.GDP.MKTP.CD")  # missing db_id
    with pytest.raises(ValidationError):
        IngestRequest(source="worldbank", db_id=2)  # missing indicator_id


def test_yahoo_requires_ticker():
    ok = IngestRequest(source="yahoo", ticker="AAPL")
    assert ok.ticker == "AAPL"
    with pytest.raises(ValidationError):
        IngestRequest(source="yahoo")


def test_binance_requires_symbol():
    ok = IngestRequest(source="binance", symbol="BTCUSDT")
    assert ok.symbol == "BTCUSDT"
    with pytest.raises(ValidationError):
        IngestRequest(source="binance")


def test_unknown_source_rejected():
    with pytest.raises(ValidationError):
        IngestRequest.model_validate({"source": "fred", "indicator_id": "x", "db_id": 1})
