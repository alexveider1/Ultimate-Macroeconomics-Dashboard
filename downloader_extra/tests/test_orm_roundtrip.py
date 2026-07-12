"""Integration test: round-trip MacroIndicator via the ORM through a real Postgres."""

from __future__ import annotations

from schema import MacroIndicator
from sqlalchemy import delete, select
from sqlalchemy.orm import Session


def test_insert_select_delete_roundtrip(session: Session) -> None:
    rows = [
        MacroIndicator(economy="USA", year=2020, value=10.0, indicator_id="GDP", db_id=2),
        MacroIndicator(economy="USA", year=2021, value=11.0, indicator_id="GDP", db_id=2),
        MacroIndicator(economy="FRA", year=2021, value=4.0, indicator_id="GDP", db_id=2),
    ]
    session.add_all(rows)
    session.commit()

    usa_rows = (
        session.execute(
            select(MacroIndicator)
            .where(MacroIndicator.economy == "USA")
            .order_by(MacroIndicator.year)
        )
        .scalars()
        .all()
    )
    assert [(r.year, r.value) for r in usa_rows] == [(2020, 10.0), (2021, 11.0)]

    session.execute(
        delete(MacroIndicator).where(
            MacroIndicator.indicator_id == "GDP",
            MacroIndicator.db_id == 2,
        )
    )
    session.commit()

    remaining = session.execute(select(MacroIndicator)).scalars().all()
    assert remaining == []


def test_distinct_indicator_ids(session: Session) -> None:
    session.add_all(
        [
            MacroIndicator(economy="USA", year=2020, value=1.0, indicator_id="A", db_id=2),
            MacroIndicator(economy="USA", year=2021, value=2.0, indicator_id="A", db_id=2),
            MacroIndicator(economy="USA", year=2020, value=3.0, indicator_id="B", db_id=2),
        ]
    )
    session.commit()

    distinct_ids = (
        session.execute(
            select(MacroIndicator.indicator_id).distinct().order_by(MacroIndicator.indicator_id)
        )
        .scalars()
        .all()
    )
    assert distinct_ids == ["A", "B"]
