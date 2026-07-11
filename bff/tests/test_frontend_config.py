"""Tests for the frontend config endpoints (/config theme + dashboard, /geo).

These exercise the routers against the real bundled files under
``_container_data`` (resolved by ``utils.resolve_data_file``), so they also
double as a smoke test that ``ui_themes.yaml`` is well-formed.
"""

from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from routers import config as config_router, geo

_REQUIRED_TOKEN_GROUPS = {"chrome", "series", "semantic", "charts", "wordcloud"}


@pytest.fixture()
def config_client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(config_router.router)
    app.include_router(geo.router)
    with TestClient(app) as client:
        yield client


def test_active_theme_has_all_token_groups(config_client: TestClient) -> None:
    body = config_client.get("/config/theme").json()
    assert body["name"]
    theme = body["theme"]
    assert _REQUIRED_TOKEN_GROUPS.issubset(theme.keys())
    assert theme["chrome"]["background"].startswith("#")
    assert isinstance(theme["series"]["colorway"], list) and theme["series"]["colorway"]


def test_all_themes_include_bundled_palettes(config_client: TestClient) -> None:
    body = config_client.get("/config/themes").json()
    assert body["active"] in body["themes"]
    assert {"dark", "dark-blue", "light-green"}.issubset(body["themes"].keys())


def test_dashboard_config_is_section_map(config_client: TestClient) -> None:
    body = config_client.get("/config/dashboard").json()
    assert isinstance(body, dict) and body
    # At least one section is a list of indicator dicts carrying id + name.
    list_sections = [v for v in body.values() if isinstance(v, list)]
    assert list_sections
    sample = next(item for section in list_sections for item in section if isinstance(item, dict))
    assert "id" in sample and "name" in sample


def test_geo_nuts2_streams_geojson(config_client: TestClient) -> None:
    response = config_client.get("/geo/nuts2")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")


def test_geo_unknown_dataset_404(config_client: TestClient) -> None:
    assert config_client.get("/geo/nope").status_code == 404


def test_geo_us_states_streams_geojson(config_client: TestClient) -> None:
    # Shipped with the FRED regional page (M2): ECharts USA w/ AK/HI insets.
    response = config_client.get("/geo/us-states")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
