"""Tests for the code-filter query-param parser."""

import pytest
from utils import parse_code_filter


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("ALL", []),
        ("all", []),
        ("USA", ["USA"]),
        ("USA,DEU", ["USA", "DEU"]),
        (" USA , DEU ", ["USA", "DEU"]),
        ("USA,ALL", []),  # ALL anywhere means "no filter".
        ("USA,,DEU", ["USA", "DEU"]),  # empty segments dropped.
    ],
)
def test_parse_code_filter(raw: str | None, expected: list[str]) -> None:
    assert parse_code_filter(raw) == expected
