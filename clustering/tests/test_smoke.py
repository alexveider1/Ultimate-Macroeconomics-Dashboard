"""Smoke tests for the clustering service: feature matrix + projection helpers + /cluster route."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import _build_feature_matrix, _build_visual_projection, app

PROJECTION_DEFAULTS: dict[str, Any] = {
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "kpca_kernel": "rbf",
    "kpca_gamma": None,
    "kpca_degree": 3,
    "kpca_coef0": 1.0,
}


def _project(matrix: np.ndarray, columns: list[str], **overrides: Any):
    """Call ``_build_visual_projection`` with sensible defaults for irrelevant kwargs."""
    kwargs: dict[str, Any] = {
        "random_state": 0,
        "reduction_method": "tsne",
        "output_dim": 2,
        **PROJECTION_DEFAULTS,
        **overrides,
    }
    return _build_visual_projection(matrix, columns, **kwargs)


def _blobs(
    rng: np.random.Generator, centers: list[tuple[float, ...]], per_cluster: int = 10
) -> np.ndarray:
    """Generate well-separated Gaussian blobs around ``centers``."""
    return np.vstack(
        [rng.normal(loc=np.array(c), scale=0.4, size=(per_cluster, len(c))) for c in centers]
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_methods_lists_all_algorithms_and_reductions(client: TestClient) -> None:
    response = client.get("/methods")
    body = response.json()
    assert set(body["available_methods"]) == {
        "kmeans",
        "dbscan",
        "meanshift",
        "hdbscan",
        "spectral",
        "hierarchical",
    }
    assert set(body["available_reductions"]) == {"tsne", "pca", "umap", "kpca"}


def test_build_feature_matrix_infers_numeric_columns() -> None:
    rows = [
        {"a": 1.0, "b": 2.0, "name": "first"},
        {"a": 3.0, "b": 4.0, "name": "second"},
    ]
    matrix, columns = _build_feature_matrix(rows, feature_columns=None)
    assert columns == ["a", "b"]
    assert matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_build_visual_projection_passes_through_two_features() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    projection, mode, labels = _project(matrix, ["a", "b"])
    assert mode == "feature_space"
    assert labels == ["a", "b"]
    assert projection.shape == (3, 2)


def test_build_visual_projection_passthrough_when_features_le_output_dim() -> None:
    """Three features into a 3D plot should skip reduction even if t-SNE is requested."""
    rng = np.random.default_rng(seed=0)
    matrix = rng.normal(size=(10, 3))
    projection, mode, labels = _project(
        matrix, ["x", "y", "z"], reduction_method="tsne", output_dim=3
    )
    assert mode == "feature_space"
    assert labels == ["x", "y", "z"]
    assert projection.shape == (10, 3)


def test_build_visual_projection_passthrough_pads_with_zero_axis() -> None:
    """One feature into a 3D plot should be padded with two zero-valued axes."""
    matrix = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    projection, mode, labels = _project(matrix, ["a"], output_dim=3)
    assert mode == "feature_space"
    assert labels == ["a", "Zero axis", "Zero axis"]
    assert projection.shape == (5, 3)
    assert np.allclose(projection[:, 1:], 0.0)


def test_build_visual_projection_uses_pca_when_requested() -> None:
    rng = np.random.default_rng(seed=42)
    matrix = rng.normal(size=(20, 5))
    projection, mode, labels = _project(matrix, [f"f{i}" for i in range(5)], reduction_method="pca")
    assert mode == "pca"
    assert labels == ["PC 1", "PC 2"]
    assert projection.shape == (20, 2)


def test_build_visual_projection_pca_3d() -> None:
    rng = np.random.default_rng(seed=42)
    matrix = rng.normal(size=(30, 6))
    projection, mode, labels = _project(
        matrix, [f"f{i}" for i in range(6)], reduction_method="pca", output_dim=3
    )
    assert mode == "pca"
    assert labels == ["PC 1", "PC 2", "PC 3"]
    assert projection.shape == (30, 3)


def test_build_visual_projection_umap_3d() -> None:
    rng = np.random.default_rng(seed=42)
    matrix = rng.normal(size=(40, 5))
    projection, mode, labels = _project(
        matrix, [f"f{i}" for i in range(5)], reduction_method="umap", output_dim=3
    )
    assert mode == "umap"
    assert labels == ["UMAP 1", "UMAP 2", "UMAP 3"]
    assert projection.shape == (40, 3)


def test_build_visual_projection_kpca_rbf_2d() -> None:
    rng = np.random.default_rng(seed=42)
    matrix = rng.normal(size=(20, 4))
    projection, mode, labels = _project(
        matrix, [f"f{i}" for i in range(4)], reduction_method="kpca", kpca_kernel="rbf"
    )
    assert mode == "kpca"
    assert labels == ["KPC 1", "KPC 2"]
    assert projection.shape == (20, 2)


def test_build_visual_projection_none_forces_passthrough() -> None:
    """``reduction_method='none'`` must skip reduction even on high-dim input."""
    rng = np.random.default_rng(seed=0)
    matrix = rng.normal(size=(20, 5))
    projection, mode, labels = _project(
        matrix, [f"f{i}" for i in range(5)], reduction_method="none"
    )
    assert mode == "feature_space"
    # Only the first two columns are kept in 2D pass-through.
    assert labels == ["f0", "f1"]
    assert projection.shape == (20, 2)


def test_cluster_endpoint_kmeans_two_features(client: TestClient) -> None:
    rng = np.random.default_rng(seed=0)
    points = _blobs(rng, [(0.0, 0.0), (5.0, 5.0)], per_cluster=8)
    rows = [{"a": float(p[0]), "b": float(p[1])} for p in points]
    response = client.post(
        "/cluster",
        json={"method": "kmeans", "dataframe": rows, "feature_columns": ["a", "b"], "k": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method_used"] == "kmeans"
    assert body["visualization_mode"] == "feature_space"
    assert body["visualization_columns"] == ["__viz_x", "__viz_y"]
    cluster_labels = {row["cluster"] for row in body["dataframe"]}
    assert cluster_labels == {0, 1}


def test_cluster_endpoint_meanshift_auto_bandwidth(client: TestClient) -> None:
    rng = np.random.default_rng(seed=1)
    points = _blobs(rng, [(0.0, 0.0), (10.0, 10.0)], per_cluster=10)
    rows = [{"a": float(p[0]), "b": float(p[1])} for p in points]
    response = client.post(
        "/cluster",
        json={
            "method": "meanshift",
            "dataframe": rows,
            "feature_columns": ["a", "b"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method_used"] == "meanshift"
    cluster_labels = {row["cluster"] for row in body["dataframe"]}
    assert len(cluster_labels) >= 2


def test_cluster_endpoint_hdbscan_default(client: TestClient) -> None:
    rng = np.random.default_rng(seed=2)
    points = _blobs(rng, [(0.0, 0.0), (8.0, 8.0), (0.0, 8.0)], per_cluster=10)
    rows = [{"a": float(p[0]), "b": float(p[1])} for p in points]
    response = client.post(
        "/cluster",
        json={
            "method": "hdbscan",
            "dataframe": rows,
            "feature_columns": ["a", "b"],
            "hdbscan_min_cluster_size": 5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method_used"] == "hdbscan"
    # HDBSCAN should find at least one cluster (label 0+); -1 = noise.
    cluster_labels = {row["cluster"] for row in body["dataframe"]}
    assert any(label >= 0 for label in cluster_labels)


def test_cluster_endpoint_spectral_rbf(client: TestClient) -> None:
    rng = np.random.default_rng(seed=3)
    points = _blobs(rng, [(0.0, 0.0), (6.0, 0.0)], per_cluster=10)
    rows = [{"a": float(p[0]), "b": float(p[1])} for p in points]
    response = client.post(
        "/cluster",
        json={
            "method": "spectral",
            "dataframe": rows,
            "feature_columns": ["a", "b"],
            "spectral_n_clusters": 2,
            "spectral_affinity": "rbf",
            "spectral_gamma": 0.5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method_used"] == "spectral"
    cluster_labels = {row["cluster"] for row in body["dataframe"]}
    assert cluster_labels == {0, 1}


def test_cluster_endpoint_hierarchical_ward(client: TestClient) -> None:
    rng = np.random.default_rng(seed=4)
    points = _blobs(rng, [(0.0, 0.0), (8.0, 0.0)], per_cluster=10)
    rows = [{"a": float(p[0]), "b": float(p[1])} for p in points]
    response = client.post(
        "/cluster",
        json={
            "method": "hierarchical",
            "dataframe": rows,
            "feature_columns": ["a", "b"],
            "hierarchical_n_clusters": 2,
            "hierarchical_linkage": "ward",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method_used"] == "hierarchical"
    cluster_labels = {row["cluster"] for row in body["dataframe"]}
    assert cluster_labels == {0, 1}


def test_cluster_endpoint_3d_output_shape(client: TestClient) -> None:
    """Five features projected to 3D via UMAP should yield __viz_x/y/z keys."""
    rng = np.random.default_rng(seed=5)
    points = rng.normal(size=(20, 5))
    feat_cols = [f"f{i}" for i in range(5)]
    rows = [{c: float(v) for c, v in zip(feat_cols, p)} for p in points]
    response = client.post(
        "/cluster",
        json={
            "method": "kmeans",
            "dataframe": rows,
            "feature_columns": feat_cols,
            "k": 3,
            "reduction_method": "umap",
            "output_dim": 3,
            "umap_n_neighbors": 5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["visualization_columns"] == ["__viz_x", "__viz_y", "__viz_z"]
    assert body["visualization_mode"] == "umap"
    assert all("__viz_z" in row for row in body["dataframe"])
