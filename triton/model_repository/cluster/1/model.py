"""Triton python-backend model: clustering + 2D/3D projection (cuML/sklearn)."""

from umd_common.cluster_backend import ClusterModelBase


class TritonPythonModel(ClusterModelBase):
    """KMeans/DBSCAN/PCA/t-SNE/UMAP on cuML; the rest on scikit-learn (CPU)."""
