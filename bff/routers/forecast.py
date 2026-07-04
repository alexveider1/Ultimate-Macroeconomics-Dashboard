"""Forecasting proxy — forwards to the forecaster service's ``/predict``."""

from typing import Any

import clients
from fastapi import APIRouter, Request
from models import ForecastRequest

router = APIRouter(tags=["forecast"])


@router.post("/forecast")
async def forecast(payload: ForecastRequest, request: Request) -> dict[str, Any]:
    """Proxy a forecast request to the forecaster service and return its JSON."""
    url = f"{request.app.state.forecaster_url}/predict"
    return await clients.post_json(
        request.app.state.http_client,
        "forecaster",
        url,
        payload.model_dump(),
        timeout=60.0,
    )
