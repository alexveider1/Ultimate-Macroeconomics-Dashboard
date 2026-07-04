"""Clustering proxy — forwards to the clustering service's ``/cluster``."""

from typing import Any

import clients
from fastapi import APIRouter, Request
from models import ClusterRequest

router = APIRouter(tags=["clustering"])


@router.post("/cluster")
async def cluster(payload: ClusterRequest, request: Request) -> dict[str, Any]:
    """Proxy a clustering request to the clustering service and return its JSON."""
    url = f"{request.app.state.clustering_url}/cluster"
    return await clients.post_json(
        request.app.state.http_client,
        "clustering",
        url,
        payload.model_dump(),
        timeout=120.0,
    )
