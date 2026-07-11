"""GeoJSON endpoints for the ECharts choropleth maps.

Serves the bundled boundary files the frontend registers via ``echarts.registerMap``:
NUTS-2 (Eurostat regional page) and US states (FRED regional page). Files are
streamed straight from disk (they are large) rather than parsed. A missing file
degrades to a clean 503 so the map renders an empty state instead of crashing.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from utils import resolve_data_file

router = APIRouter(prefix="/geo", tags=["geo"])

# Logical name → path under _container_data. The US-states file is added when the
# FRED regional page is built (M2); until then that name returns 503.
_GEO_FILES: dict[str, str] = {
    "world": "_configs/world_countries.geojson",
    "nuts2": "_configs/nuts_level2_2021.geojson",
    "us-states": "_configs/us_states.geojson",
}


@router.get("/{name}")
def get_geojson(name: str) -> FileResponse:
    """Stream one bundled GeoJSON boundary file by logical name."""
    relpath = _GEO_FILES.get(name)
    if relpath is None:
        raise HTTPException(status_code=404, detail=f"Unknown geo dataset: {name}")
    path = resolve_data_file(relpath)
    if path is None:
        raise HTTPException(status_code=503, detail=f"GeoJSON '{name}' is not available")
    return FileResponse(path, media_type="application/geo+json")
