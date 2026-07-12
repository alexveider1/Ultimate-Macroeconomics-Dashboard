"""Clustering + dim-reduction inference ported from the old clustering service.

GPU-accelerated where RAPIDS cuML has an equivalent (KMeans, DBSCAN, PCA, t-SNE,
UMAP); scikit-learn on CPU for the rest (Mean-Shift, HDBSCAN, Spectral,
Agglomerative, Kernel-PCA). Every cuML path is wrapped so an import/runtime
failure (e.g. cuML lacks 3-component t-SNE) degrades cleanly to the sklearn
implementation instead of failing the request — the numbers may differ slightly
but the contract (labels + 2D/3D projection) is preserved.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VIZ_X_COL = "__viz_x"
VIZ_Y_COL = "__viz_y"
VIZ_Z_COL = "__viz_z"
VIZ_COLS = (VIZ_X_COL, VIZ_Y_COL, VIZ_Z_COL)

_cuml_checked = False
_cuml_available = False


class InputError(ValueError):
    """Raised for bad user input; the adapter maps it to HTTP 400."""


def _has_cuml() -> bool:
    """Return whether cuML is importable (checked once, then cached)."""
    global _cuml_checked, _cuml_available
    if not _cuml_checked:
        try:
            import cuml  # noqa: F401

            _cuml_available = True
        except Exception as exc:  # pragma: no cover - depends on GPU image
            logger.warning("cuML unavailable, using scikit-learn on CPU: %s", exc)
            _cuml_available = False
        _cuml_checked = True
    return _cuml_available


def _f32(matrix: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 copy (the layout cuML/cuDF prefer)."""
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _infer_numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return the keys whose values are numeric in *every* row of ``rows``."""
    first_row = rows[0]
    numeric_columns: list[str] = []
    for col in first_row:
        values = [row.get(col) for row in rows]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            numeric_columns.append(col)
    return numeric_columns


def _build_feature_matrix(
    rows: list[dict[str, Any]], feature_columns: list[str] | None
) -> tuple[np.ndarray, list[str]]:
    """Cast ``rows`` to a 2D float matrix using ``feature_columns``.

    Raises:
        InputError: When no numeric features are available, a feature column is
            missing from a row, or a value isn't a number.
    """
    if feature_columns is None:
        feature_columns = _infer_numeric_columns(rows)

    if len(feature_columns) == 0:
        raise InputError(
            "No numeric features available. Provide numeric columns in 'feature_columns'."
        )

    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        values: list[float] = []
        for col in feature_columns:
            if col not in row:
                raise InputError(f"Row {row_index} is missing required feature column '{col}'.")
            raw_value = row[col]
            if not isinstance(raw_value, (int, float, np.integer, np.floating)):
                raise InputError(
                    f"Column '{col}' contains a non-finite or non-numeric value at row {row_index}."
                )
            values.append(float(raw_value))
        matrix.append(values)

    return np.asarray(matrix, dtype=float), feature_columns


# --------------------------------------------------------------------------- #
# Clustering algorithms
# --------------------------------------------------------------------------- #
def _kmeans(matrix: np.ndarray, k: int, n_init: int, random_state: int) -> np.ndarray:
    if _has_cuml():
        try:
            from cuml.cluster import KMeans as CuKMeans

            estimator = CuKMeans(
                n_clusters=k, n_init=n_init, random_state=random_state, output_type="numpy"
            )
            return np.asarray(estimator.fit_predict(_f32(matrix)))
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("cuML KMeans failed, falling back to sklearn: %s", exc)
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit_predict(matrix)


def _dbscan(matrix: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if _has_cuml():
        try:
            from cuml.cluster import DBSCAN as CuDBSCAN

            estimator = CuDBSCAN(eps=eps, min_samples=min_samples, output_type="numpy")
            return np.asarray(estimator.fit_predict(_f32(matrix)))
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("cuML DBSCAN failed, falling back to sklearn: %s", exc)
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)


def _cluster_labels(request: dict[str, Any], matrix: np.ndarray, n_rows: int) -> np.ndarray:
    """Fit the selected algorithm and return integer cluster labels."""
    method = request.get("method")
    random_state = int(request.get("random_state", 42))

    if method == "kmeans":
        k = int(request.get("k", 3))
        if k > n_rows:
            raise InputError(f"k ({k}) cannot be greater than the number of rows ({n_rows}).")
        return _kmeans(matrix, k, int(request.get("n_init", 10)), random_state)

    if method == "dbscan":
        return _dbscan(matrix, float(request.get("eps", 0.5)), int(request.get("min_samples", 5)))

    if method == "meanshift":
        from sklearn.cluster import MeanShift

        return MeanShift(bandwidth=request.get("bandwidth")).fit_predict(matrix)

    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            min_cluster_size=int(request.get("hdbscan_min_cluster_size", 5)),
            min_samples=request.get("hdbscan_min_samples"),
        ).fit_predict(matrix)

    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        n_clusters = int(request.get("spectral_n_clusters", 4))
        if n_clusters > n_rows:
            raise InputError(
                f"spectral_n_clusters ({n_clusters}) cannot be greater "
                f"than the number of rows ({n_rows})."
            )
        affinity = request.get("spectral_affinity", "rbf")
        kwargs: dict[str, Any] = {
            "n_clusters": n_clusters,
            "affinity": affinity,
            "random_state": random_state,
            "assign_labels": "kmeans",
        }
        if affinity == "rbf":
            kwargs["gamma"] = float(request.get("spectral_gamma", 1.0))
        else:
            kwargs["n_neighbors"] = max(
                2, min(int(request.get("spectral_n_neighbors", 10)), n_rows - 1)
            )
        return SpectralClustering(**kwargs).fit_predict(matrix)

    if method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering

        n_clusters = int(request.get("hierarchical_n_clusters", 4))
        if n_clusters > n_rows:
            raise InputError(
                f"hierarchical_n_clusters ({n_clusters}) cannot be greater "
                f"than the number of rows ({n_rows})."
            )
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=request.get("hierarchical_linkage", "ward"),
        ).fit_predict(matrix)

    raise InputError(f"Unknown method: {method}")


# --------------------------------------------------------------------------- #
# Visual projection
# --------------------------------------------------------------------------- #
def _passthrough_projection(
    feature_matrix: np.ndarray, feature_columns: list[str], output_dim: int
) -> tuple[np.ndarray, str, list[str]]:
    """Return the feature matrix as the projection, padding/truncating to output_dim."""
    n_rows, n_features = feature_matrix.shape
    if n_features >= output_dim:
        projection = feature_matrix[:, :output_dim]
        return projection, "feature_space", list(feature_columns[:output_dim])

    pad_width = output_dim - n_features
    zero_block = np.zeros((n_rows, pad_width), dtype=float)
    projection = np.hstack([feature_matrix, zero_block])
    labels = list(feature_columns) + ["Zero axis"] * pad_width
    return projection, "feature_space", labels


def _pca(matrix: np.ndarray, output_dim: int, random_state: int) -> np.ndarray:
    if _has_cuml():
        try:
            from cuml.decomposition import PCA as CuPCA

            return np.asarray(
                CuPCA(
                    n_components=output_dim, random_state=random_state, output_type="numpy"
                ).fit_transform(_f32(matrix))
            )
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("cuML PCA failed, falling back to sklearn: %s", exc)
    from sklearn.decomposition import PCA

    return PCA(n_components=output_dim, random_state=random_state).fit_transform(matrix)


def _umap(
    matrix: np.ndarray,
    output_dim: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    if _has_cuml():
        try:
            from cuml.manifold import UMAP as CuUMAP

            return np.asarray(
                CuUMAP(
                    n_components=output_dim,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=random_state,
                    output_type="numpy",
                ).fit_transform(_f32(matrix))
            )
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("cuML UMAP failed, falling back to umap-learn: %s", exc)
    import umap

    reducer = umap.UMAP(
        n_components=output_dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(matrix)


def _tsne(matrix: np.ndarray, output_dim: int, perplexity: float, random_state: int) -> np.ndarray:
    # cuML t-SNE only supports 2 components; fall through to sklearn otherwise.
    if _has_cuml() and output_dim == 2:
        try:
            from cuml.manifold import TSNE as CuTSNE

            return np.asarray(
                CuTSNE(
                    n_components=output_dim,
                    perplexity=perplexity,
                    random_state=random_state,
                    output_type="numpy",
                ).fit_transform(_f32(matrix))
            )
        except Exception as exc:  # pragma: no cover - GPU path
            logger.warning("cuML t-SNE failed, falling back to sklearn: %s", exc)
    from sklearn.manifold import TSNE

    return TSNE(
        n_components=output_dim,
        random_state=random_state,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(matrix)


def _build_visual_projection(
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    request: dict[str, Any],
    output_dim: int,
) -> tuple[np.ndarray, str, list[str]]:
    """Project ``feature_matrix`` into ``output_dim`` dims; mirrors the old service."""
    n_rows, n_features = feature_matrix.shape
    reduction_method = request.get("reduction_method", "tsne")
    random_state = int(request.get("random_state", 42))

    if reduction_method == "none" or n_features <= output_dim:
        return _passthrough_projection(feature_matrix, feature_columns, output_dim)

    if n_rows < 5:
        return _passthrough_projection(feature_matrix, feature_columns, output_dim)

    if reduction_method == "pca":
        projection = _pca(feature_matrix, output_dim, random_state)
        return projection, "pca", [f"PC {i + 1}" for i in range(output_dim)]

    if reduction_method == "umap":
        effective_neighbors = max(2, min(int(request.get("umap_n_neighbors", 15)), n_rows - 1))
        projection = _umap(
            feature_matrix,
            output_dim,
            effective_neighbors,
            float(request.get("umap_min_dist", 0.1)),
            random_state,
        )
        return projection, "umap", [f"UMAP {i + 1}" for i in range(output_dim)]

    if reduction_method == "kpca":
        from sklearn.decomposition import KernelPCA

        projection = KernelPCA(
            n_components=output_dim,
            kernel=request.get("kpca_kernel", "rbf"),
            gamma=request.get("kpca_gamma"),
            degree=int(request.get("kpca_degree", 3)),
            coef0=float(request.get("kpca_coef0", 1.0)),
            random_state=random_state,
        ).fit_transform(feature_matrix)
        return projection, "kpca", [f"KPC {i + 1}" for i in range(output_dim)]

    perplexity = min(30.0, float(n_rows - 1))
    projection = _tsne(feature_matrix, output_dim, perplexity, random_state)
    return projection, "tsne", [f"t-SNE {i + 1}" for i in range(output_dim)]


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Cluster ``request`` and return the ClusterResponse-shaped dict.

    Args:
        request: The full clustering request body (already validated by the
            adapter's Pydantic model).

    Raises:
        InputError: For bad inputs (mapped to HTTP 400 by the adapter).
    """
    rows = request["dataframe"]
    feature_matrix, feature_columns = _build_feature_matrix(rows, request.get("feature_columns"))
    output_dim = int(request.get("output_dim", 2))
    n_rows = len(rows)

    labels = _cluster_labels(request, feature_matrix, n_rows)
    projection, projection_mode, projection_labels = _build_visual_projection(
        feature_matrix, feature_columns, request, output_dim
    )

    viz_columns = list(VIZ_COLS[:output_dim])
    output_rows = [dict(row) for row in rows]
    for row, label, point in zip(output_rows, labels, projection):
        row["cluster"] = int(label)
        for col, value in zip(viz_columns, point):
            row[col] = float(value)

    return {
        "method_used": request.get("method"),
        "dataframe": output_rows,
        "visualization_mode": projection_mode,
        "visualization_columns": viz_columns,
        "visualization_labels": projection_labels,
    }
