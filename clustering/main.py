"""FastAPI service exposing unsupervised clustering on tabular input.

Wraps scikit-learn clustering algorithms (KMeans, DBSCAN, Mean-Shift, HDBSCAN,
SpectralClustering, AgglomerativeClustering) and projects the feature matrix
into 2D or 3D for visualisation. When ``n_features > output_dim`` the projection
is built with t-SNE, PCA, UMAP, or Kernel PCA (per ``reduction_method``); with
``n_features <= output_dim`` the original space is preserved as-is.
"""

from pathlib import Path

import numpy as np
import umap
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

from config import load_config
from schemas import ClusterRequest, ClusterResponse

CONFIG_PATH = Path("config.yaml")

CONFIG = load_config(CONFIG_PATH)

VIZ_X_COL = "__viz_x"
VIZ_Y_COL = "__viz_y"
VIZ_Z_COL = "__viz_z"
VIZ_COLS = (VIZ_X_COL, VIZ_Y_COL, VIZ_Z_COL)

app = FastAPI(
    title="Clustering API",
    description=(
        "Tabular clustering (KMeans, DBSCAN, Mean-Shift, HDBSCAN, Spectral, "
        "Hierarchical) with 2D/3D projection (t-SNE, PCA, UMAP, Kernel PCA)."
    ),
)


def _infer_numeric_columns(rows: list[dict[str, object]]) -> list[str]:
    """Return the keys whose values are numeric in *every* row of ``rows``.

    Args:
        rows: Non-empty list of row dicts (caller guarantees ``len(rows) >= 1``).

    Returns:
        Column names that are numeric across all rows. Order matches ``rows[0]``.
    """
    first_row = rows[0]
    numeric_columns: list[str] = []
    for col in first_row:
        values = [row.get(col) for row in rows]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            numeric_columns.append(col)
    return numeric_columns


def _build_feature_matrix(
    rows: list[dict[str, object]], feature_columns: list[str] | None
) -> tuple[np.ndarray, list[str]]:
    """Cast ``rows`` to a 2D float matrix using ``feature_columns``.

    Args:
        rows: Tabular input as list of row dicts.
        feature_columns: Explicit feature list, or ``None`` to auto-detect
            numeric columns via :func:`_infer_numeric_columns`.

    Returns:
        Tuple of ``(matrix, columns_used)``.

    Raises:
        HTTPException: When no numeric features are available, a feature
            column is missing from a row, or a value isn't a number.
    """
    if feature_columns is None:
        feature_columns = _infer_numeric_columns(rows)

    if len(feature_columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="No numeric features available. Provide numeric columns in 'feature_columns'.",
        )

    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        values: list[float] = []
        for col in feature_columns:
            if col not in row:
                raise HTTPException(
                    status_code=400,
                    detail=f"Row {row_index} is missing required feature column '{col}'.",
                )
            raw_value = row[col]
            if not isinstance(raw_value, (int, float, np.integer, np.floating)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{col}' contains a non-finite or non-numeric value at row {row_index}."
                    ),
                )
            values.append(float(raw_value))
        matrix.append(values)

    return np.asarray(matrix, dtype=float), feature_columns


def _passthrough_projection(
    feature_matrix: np.ndarray, feature_columns: list[str], output_dim: int
) -> tuple[np.ndarray, str, list[str]]:
    """Return the feature matrix as the visual projection without reduction.

    Pads the right-hand columns with a zero axis when there are strictly fewer
    features than ``output_dim`` (so the caller always gets an ``output_dim``-wide
    matrix), and truncates to the first ``output_dim`` columns when there are
    more features than asked for (used by the small-sample fallback).
    """
    n_rows, n_features = feature_matrix.shape

    if n_features >= output_dim:
        projection = feature_matrix[:, :output_dim]
        labels = list(feature_columns[:output_dim])
        return projection, "feature_space", labels

    pad_width = output_dim - n_features
    zero_block = np.zeros((n_rows, pad_width), dtype=float)
    projection = np.hstack([feature_matrix, zero_block])
    labels = list(feature_columns) + ["Zero axis"] * pad_width
    return projection, "feature_space", labels


def _build_visual_projection(
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    random_state: int,
    reduction_method: str,
    output_dim: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    kpca_kernel: str,
    kpca_gamma: float | None,
    kpca_degree: int,
    kpca_coef0: float,
) -> tuple[np.ndarray, str, list[str]]:
    """Project ``feature_matrix`` into ``output_dim`` dimensions for plotting.

    Branching:
      * ``n_features <= output_dim`` — pass-through (pad with a zero axis if
        strictly fewer features).
      * ``n_rows < 5`` — pass-through with the first ``output_dim`` feature
        columns (avoids degenerate TSNE perplexity / UMAP neighbours).
      * Otherwise dispatch on ``reduction_method``.

    Returns:
        Tuple of ``(projection, mode, axis_labels)`` where ``mode`` is one of
        ``"feature_space"``, ``"tsne"``, ``"pca"``, ``"umap"``, ``"kpca"``.
        ``axis_labels`` has length ``output_dim``.
    """
    n_rows, n_features = feature_matrix.shape

    if reduction_method == "none" or n_features <= output_dim:
        return _passthrough_projection(feature_matrix, feature_columns, output_dim)

    if n_rows < 5:
        return _passthrough_projection(feature_matrix, feature_columns, output_dim)

    if reduction_method == "pca":
        projection = PCA(n_components=output_dim, random_state=random_state).fit_transform(
            feature_matrix
        )
        labels = [f"PC {i + 1}" for i in range(output_dim)]
        return projection, "pca", labels

    if reduction_method == "umap":
        effective_neighbors = max(2, min(umap_n_neighbors, n_rows - 1))
        reducer = umap.UMAP(
            n_components=output_dim,
            n_neighbors=effective_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        projection = reducer.fit_transform(feature_matrix)
        labels = [f"UMAP {i + 1}" for i in range(output_dim)]
        return projection, "umap", labels

    if reduction_method == "kpca":
        projection = KernelPCA(
            n_components=output_dim,
            kernel=kpca_kernel,
            gamma=kpca_gamma,
            degree=kpca_degree,
            coef0=kpca_coef0,
            random_state=random_state,
        ).fit_transform(feature_matrix)
        labels = [f"KPC {i + 1}" for i in range(output_dim)]
        return projection, "kpca", labels

    # default: t-SNE
    perplexity = min(30.0, float(n_rows - 1))
    projection = TSNE(
        n_components=output_dim,
        random_state=random_state,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(feature_matrix)
    labels = [f"t-SNE {i + 1}" for i in range(output_dim)]
    return projection, "tsne", labels


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


def _build_estimator(request: ClusterRequest, n_rows: int) -> object:
    """Construct the sklearn estimator selected by ``request.method``.

    Args:
        request: Parsed and validated ``ClusterRequest``.
        n_rows: Number of input rows; used for cluster-count sanity checks.

    Returns:
        A scikit-learn estimator implementing ``fit_predict``.

    Raises:
        HTTPException: 400 for unknown algorithms or impossible cluster counts.
    """
    method = request.method
    if method == "kmeans":
        if request.k > n_rows:
            raise HTTPException(
                status_code=400,
                detail=f"k ({request.k}) cannot be greater than the number of rows ({n_rows}).",
            )
        return KMeans(
            n_clusters=request.k,
            n_init=request.n_init,
            random_state=request.random_state,
        )

    if method == "dbscan":
        return DBSCAN(eps=request.eps, min_samples=request.min_samples)

    if method == "meanshift":
        return MeanShift(bandwidth=request.bandwidth)

    if method == "hdbscan":
        return HDBSCAN(
            min_cluster_size=request.hdbscan_min_cluster_size,
            min_samples=request.hdbscan_min_samples,
        )

    if method == "spectral":
        if request.spectral_n_clusters > n_rows:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"spectral_n_clusters ({request.spectral_n_clusters}) cannot be greater "
                    f"than the number of rows ({n_rows})."
                ),
            )
        kwargs: dict[str, object] = {
            "n_clusters": request.spectral_n_clusters,
            "affinity": request.spectral_affinity,
            "random_state": request.random_state,
            "assign_labels": "kmeans",
        }
        if request.spectral_affinity == "rbf":
            kwargs["gamma"] = request.spectral_gamma
        else:
            # n_neighbors must be < n_samples for nearest_neighbors affinity.
            kwargs["n_neighbors"] = max(2, min(request.spectral_n_neighbors, n_rows - 1))
        return SpectralClustering(**kwargs)

    if method == "hierarchical":
        if request.hierarchical_n_clusters > n_rows:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"hierarchical_n_clusters ({request.hierarchical_n_clusters}) cannot be "
                    f"greater than the number of rows ({n_rows})."
                ),
            )
        return AgglomerativeClustering(
            n_clusters=request.hierarchical_n_clusters,
            linkage=request.hierarchical_linkage,
        )

    raise HTTPException(status_code=400, detail=f"Unknown method: {method}")


def _run_clustering(
    estimator: object,
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    random_state: int,
    reduction_method: str,
    output_dim: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    kpca_kernel: str,
    kpca_gamma: float | None,
    kpca_degree: int,
    kpca_coef0: float,
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    """Fit ``estimator`` on ``feature_matrix`` and build the visual projection.

    Returns:
        Tuple of ``(labels, projection, projection_mode, projection_labels)``.
    """
    labels = estimator.fit_predict(feature_matrix)
    projection, projection_mode, projection_labels = _build_visual_projection(
        feature_matrix=feature_matrix,
        feature_columns=feature_columns,
        random_state=random_state,
        reduction_method=reduction_method,
        output_dim=output_dim,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        kpca_kernel=kpca_kernel,
        kpca_gamma=kpca_gamma,
        kpca_degree=kpca_degree,
        kpca_coef0=kpca_coef0,
    )
    return labels, projection, projection_mode, projection_labels


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_dataframe(request: ClusterRequest) -> ClusterResponse:
    """Cluster the rows in ``request`` and return labels + projection coordinates.

    Args:
        request: Payload selecting the algorithm, hyperparameters, projection
            method, and target output dimensionality. See
            :class:`schemas.ClusterRequest`.

    Returns:
        ClusterResponse with the original rows augmented with ``cluster`` plus
        ``__viz_x`` / ``__viz_y`` (and ``__viz_z`` when ``output_dim == 3``)
        and projection metadata.

    Raises:
        HTTPException: 400 for unknown algorithms or bad inputs; 500 for
            internal sklearn failures.
    """
    rows = request.dataframe
    feature_matrix, feature_columns = _build_feature_matrix(rows, request.feature_columns)
    estimator = _build_estimator(request, n_rows=len(rows))
    output_dim = int(request.output_dim)

    try:
        labels, projection, projection_mode, projection_labels = await run_in_threadpool(
            _run_clustering,
            estimator,
            feature_matrix,
            feature_columns,
            request.random_state,
            request.reduction_method,
            output_dim,
            request.umap_n_neighbors,
            request.umap_min_dist,
            request.kpca_kernel,
            request.kpca_gamma,
            request.kpca_degree,
            request.kpca_coef0,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clustering input: {str(e)}. Features used: {feature_columns}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    viz_columns = list(VIZ_COLS[:output_dim])
    output_rows = [dict(row) for row in rows]
    for row, label, point in zip(output_rows, labels, projection):
        row["cluster"] = int(label)
        for col, value in zip(viz_columns, point):
            row[col] = float(value)

    return ClusterResponse(
        method_used=request.method,
        dataframe=output_rows,
        visualization_mode=projection_mode,
        visualization_columns=viz_columns,
        visualization_labels=projection_labels,
    )
