"""End-to-end tests for the World Bank read router (over a real Postgres)."""

from fastapi.testclient import TestClient
from schema import Country, DatabaseIndicator, MacroIndicator, MetadataRow
from sqlalchemy.orm import Session


def _seed(session: Session) -> None:
    session.add_all(
        [
            Country(
                id="USA",
                value="United States",
                aggregate=False,
                region_value="North America",
                income_level_value="High income",
                latitude=38.0,
                longitude=-77.0,
                capital_city="Washington D.C.",
            ),
            Country(id="WLD", value="World", aggregate=True),
            DatabaseIndicator(id="NY.GDP.MKTP.CD", description="GDP (current US$)", database_id=2),
            DatabaseIndicator(id="NY.GDP.MKTP.CD", description="GDP other db", database_id=11),
            MetadataRow(
                indicator_id="NY.GDP.MKTP.CD",
                db_id=2,
                indicator_name="GDP (current US$)",
                units="US$",
                source="World Bank",
            ),
            MacroIndicator(
                economy="USA", year=2020, value=21e12, indicator_id="NY.GDP.MKTP.CD", db_id=2
            ),
            MacroIndicator(
                economy="USA", year=2021, value=23e12, indicator_id="NY.GDP.MKTP.CD", db_id=2
            ),
            MacroIndicator(
                economy="DEU", year=2020, value=3.8e12, indicator_id="NY.GDP.MKTP.CD", db_id=2
            ),
        ]
    )
    session.commit()


def test_list_countries(client: TestClient, session: Session) -> None:
    _seed(session)
    body = client.get("/worldbank/countries").json()
    assert {c["id"] for c in body} == {"USA", "WLD"}
    usa = next(c for c in body if c["id"] == "USA")
    assert usa["name"] == "United States"
    assert usa["region"] == "North America"
    assert usa["income_level"] == "High income"


def test_list_countries_excludes_aggregates(client: TestClient, session: Session) -> None:
    _seed(session)
    body = client.get("/worldbank/countries", params={"include_aggregates": False}).json()
    assert [c["id"] for c in body] == ["USA"]


def test_list_country_codes(client: TestClient, session: Session) -> None:
    _seed(session)
    assert client.get("/worldbank/countries/codes").json() == ["DEU", "USA"]


def test_indicator_info_prefers_wdi(client: TestClient, session: Session) -> None:
    _seed(session)
    body = client.get("/worldbank/indicators/NY.GDP.MKTP.CD").json()
    assert body["name"] == "GDP (current US$)"  # db=2 wins over db=11.
    assert body["units"] == "US$"
    assert body["source"] == "World Bank"


def test_indicator_values_ordered(client: TestClient, session: Session) -> None:
    _seed(session)
    body = client.get("/worldbank/indicators/NY.GDP.MKTP.CD/values").json()
    assert body["name"] == "GDP (current US$)"
    points = body["points"]
    assert len(points) == 3
    # Ordered by (year, economy): 2020 DEU, 2020 USA, 2021 USA.
    assert (points[0]["year"], points[0]["economy"]) == (2020, "DEU")
    assert (points[1]["year"], points[1]["economy"]) == (2020, "USA")


def test_indicator_values_country_filter(client: TestClient, session: Session) -> None:
    _seed(session)
    body = client.get(
        "/worldbank/indicators/NY.GDP.MKTP.CD/values", params={"countries": "USA"}
    ).json()
    assert {p["economy"] for p in body["points"]} == {"USA"}
    assert len(body["points"]) == 2
