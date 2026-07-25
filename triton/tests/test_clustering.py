"""CPU-runnable tests for the ported clustering logic (``umd_common``).

cuML isn't importable without a GPU, so ``_has_cuml()`` returns False here and
every path falls back to scikit-learn — which is exactly the fallback the GPU
image relies on when a cuML call is unsupported. That makes these tests a valid
check of the algorithm selection, projection branching, and input validation.
"""

from __future__ import annotations

from typing import Any

import pytest
from umd_common import clustering


def _rows(n: int = 8, features: int = 3) -> list[dict[str, Any]]:
    return [{f"f{j}": float(i * (j + 1)) for j in range(features)} for i in range(n)]


def test_kmeans_labels_and_projection() -> None:
    request = {
        "method": "kmeans",
        "dataframe": _rows(),
        "k": 2,
        "n_init": 5,
        "reduction_method": "pca",
        "output_dim": 2,
    }
    result = clustering.run(request)
    assert result["method_used"] == "kmeans"
    assert result["visualization_mode"] == "pca"
    assert result["visualization_columns"] == ["__viz_x", "__viz_y"]
    assert len(result["dataframe"]) == 8
    assert all("cluster" in row and "__viz_x" in row for row in result["dataframe"])


def test_hierarchical_sklearn_path() -> None:
    request = {
        "method": "hierarchical",
        "dataframe": _rows(),
        "hierarchical_n_clusters": 3,
        "reduction_method": "none",
        "output_dim": 2,
    }
    result = clustering.run(request)
    assert result["method_used"] == "hierarchical"
    assert result["visualization_mode"] == "feature_space"


def test_passthrough_when_features_fit_output_dim() -> None:
    request = {
        "method": "kmeans",
        "dataframe": _rows(features=2),
        "k": 2,
        "reduction_method": "tsne",
        "output_dim": 2,
    }
    result = clustering.run(request)
    # 2 features into a 2D plot needs no reduction.
    assert result["visualization_mode"] == "feature_space"


def test_no_numeric_features_raises() -> None:
    request = {
        "method": "kmeans",
        "dataframe": [{"label": "a"}, {"label": "b"}],
        "k": 2,
    }
    with pytest.raises(clustering.InputError):
        clustering.run(request)


def test_k_greater_than_rows_raises() -> None:
    request = {"method": "kmeans", "dataframe": _rows(n=3), "k": 5}
    with pytest.raises(clustering.InputError):
        clustering.run(request)
