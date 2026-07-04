"""End-to-end tests for the FRED + Eurostat regional read routers."""

from fastapi.testclient import TestClient
from schema import (
    EurostatIndicator,
    EurostatIndicatorValue,
    Region,
    State,
    StateIndicator,
    StateIndicatorValue,
)
from sqlalchemy.orm import Session


def _seed_fred(session: Session) -> None:
    # Parent rows (states) flushed before the child values so the FK is satisfied
    # — SQLAlchemy doesn't order FK-only tables within a single flush.
    session.add_all(
        [
            State(id="CA", name="California", fips="06", region="West", division="Pacific"),
            State(id="NY", name="New York", fips="36", region="Northeast"),
            StateIndicator(indicator_id="unemployment_rate", name="Unemployment Rate", units="%"),
        ]
    )
    session.flush()
    session.add_all(
        [
            StateIndicatorValue(state="CA", year=2024, value=4.9, indicator_id="unemployment_rate"),
            StateIndicatorValue(state="NY", year=2024, value=4.2, indicator_id="unemployment_rate"),
            StateIndicatorValue(
                state="CA", year=2023, value=None, indicator_id="unemployment_rate"
            ),
        ]
    )
    session.commit()


def _seed_eurostat(session: Session) -> None:
    session.add_all(
        [
            Region(id="DE21", name="Oberbayern", country_code="DE", country_name="Germany"),
            Region(id="FR10", name="Île-de-France", country_code="FR", country_name="France"),
            EurostatIndicator(
                indicator_id="gdp_per_capita", name="GDP per capita", dataset="nama_10r_2gdp"
            ),
        ]
    )
    session.flush()
    session.add_all(
        [
            EurostatIndicatorValue(
                region="DE21", year=2022, value=60000.0, indicator_id="gdp_per_capita"
            ),
            EurostatIndicatorValue(
                region="FR10", year=2022, value=58000.0, indicator_id="gdp_per_capita"
            ),
        ]
    )
    session.commit()


def test_fred_states(client: TestClient, session: Session) -> None:
    _seed_fred(session)
    body = client.get("/fred/states").json()
    assert [s["id"] for s in body] == ["CA", "NY"]


def test_fred_indicator_404(client: TestClient, session: Session) -> None:
    _seed_fred(session)
    assert client.get("/fred/indicators/nope").status_code == 404


def test_fred_values_skip_nulls_and_filter(client: TestClient, session: Session) -> None:
    _seed_fred(session)
    body = client.get("/fred/indicators/unemployment_rate/values").json()
    # The null-valued 2023 CA row is excluded.
    assert len(body) == 2
    assert all(row["value"] is not None for row in body)

    filtered = client.get(
        "/fred/indicators/unemployment_rate/values", params={"states": "CA"}
    ).json()
    assert {row["region"] for row in filtered} == {"CA"}


def test_eurostat_regions_and_values(client: TestClient, session: Session) -> None:
    _seed_eurostat(session)
    regions = client.get("/eurostat/regions").json()
    assert {r["id"] for r in regions} == {"DE21", "FR10"}

    values = client.get("/eurostat/indicators/gdp_per_capita/values").json()
    assert len(values) == 2
    assert {v["region"] for v in values} == {"DE21", "FR10"}


def test_eurostat_indicator_lookup(client: TestClient, session: Session) -> None:
    _seed_eurostat(session)
    body = client.get("/eurostat/indicators/gdp_per_capita").json()
    assert body["dataset"] == "nama_10r_2gdp"
