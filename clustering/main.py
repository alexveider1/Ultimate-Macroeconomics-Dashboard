"""FastAPI adapter exposing tabular clustering — compute lives in Triton.

This service keeps its original HTTP contract (``/cluster``, ``/methods``,
``/health``) but no longer runs any clustering itself. It validates the request
with the same Pydantic schema, forwards it to the ``cluster`` python-backend
model in the ``triton`` container (cuML on GPU for KMeans/DBSCAN/PCA/t-SNE/UMAP,
scikit-learn on CPU for the rest), and reshapes the reply into the same
``ClusterResponse`` callers already expect.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from config import load_config
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from schemas import ClusterRequest, ClusterResponse
from triton_client import TritonError, create_client, infer_json, resolve_triton_url

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)
TRITON_CONFIG = CONFIG.triton


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the shared Triton gRPC client (lazy connect) on startup."""
    url = resolve_triton_url(TRITON_CONFIG.host, TRITON_CONFIG.grpc_port)
    logger.info("Clustering adapter targeting Triton at %s", url)
    app.state.triton = create_client(url)
    try:
        yield
    finally:
        try:
            app.state.triton.close()
        except Exception:  # noqa: BLE001 - best-effort shutdown
            logger.debug("Triton client close failed", exc_info=True)


app = FastAPI(
    title="Clustering API",
    description=(
        "Adapter for tabular clustering (KMeans, DBSCAN, Mean-Shift, HDBSCAN, "
        "Spectral, Hierarchical) with 2D/3D projection (t-SNE, PCA, UMAP, Kernel "
        "PCA), served by Triton."
    ),
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a static welcome banner — used as a liveness signal."""
    return {"message": "Welcome to the Clustering API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for the Compose healthcheck."""
    return {"status": "ok"}


@app.get("/methods")
def list_methods() -> dict[str, list[str]]:
    """Expose the algorithms and dim-reduction methods supported by this service."""
    return {
        "available_methods": [
            "kmeans",
            "dbscan",
            "meanshift",
            "hdbscan",
            "spectral",
            "hierarchical",
        ],
        "available_reductions": ["tsne", "pca", "umap", "kpca"],
    }


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_dataframe(request: ClusterRequest) -> ClusterResponse:
    """Forward the validated request to Triton and reshape the reply.

    Args:
        request: Payload selecting the algorithm, hyperparameters, projection
            method, and target output dimensionality (see
            :class:`schemas.ClusterRequest`).

    Returns:
        ClusterResponse with the original rows augmented with ``cluster`` plus
        the visualisation coordinates and projection metadata.

    Raises:
        HTTPException: 400 for invalid inputs; 500 for model failures; 502 when
            Triton is unreachable.
    """
    try:
        result = await run_in_threadpool(
            infer_json, app.state.triton, "cluster", request.model_dump()
        )
    except TritonError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    return ClusterResponse(**result)
