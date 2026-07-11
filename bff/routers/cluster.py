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


@router.get("/cluster/methods")
async def cluster_methods(request: Request) -> dict[str, Any]:
    """Proxy the clustering service's method list (drives the sandbox UI)."""
    url = f"{request.app.state.clustering_url}/methods"
    return await clients.get_json(request.app.state.http_client, "clustering", url, timeout=30.0)
