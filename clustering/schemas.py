"""Pydantic request/response schemas for the clustering FastAPI service."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ClusterRequest(BaseModel):
    """Body accepted by ``POST /cluster``.

    Args:
        method: Clustering algorithm. One of ``kmeans``, ``dbscan``,
            ``meanshift``, ``hdbscan``, ``spectral``, ``hierarchical``.
        dataframe: Tabular input as a list of row dicts; must be non-empty.
        feature_columns: Optional explicit feature list; ``None`` means infer.
        k: Number of clusters (kmeans only).
        n_init: Restart count for kmeans.
        random_state: Seed for deterministic kmeans + dim-reduction.
        eps: Neighborhood radius for dbscan.
        min_samples: Minimum points per neighborhood for dbscan.
        bandwidth: Mean-Shift bandwidth; ``None`` triggers sklearn's
            ``estimate_bandwidth`` helper.
        hdbscan_min_cluster_size: HDBSCAN minimum cluster size.
        hdbscan_min_samples: HDBSCAN sample density threshold; ``None`` reuses
            ``min_cluster_size`` (sklearn default).
        spectral_n_clusters: Number of clusters for spectral clustering.
        spectral_affinity: ``rbf`` (kernelised) or ``nearest_neighbors``.
        spectral_n_neighbors: Neighbourhood size when affinity is
            ``nearest_neighbors``.
        spectral_gamma: Kernel coefficient when affinity is ``rbf``.
        hierarchical_n_clusters: Number of clusters for Agglomerative
            (hierarchical) clustering.
        hierarchical_linkage: Linkage strategy passed to AgglomerativeClustering.
        reduction_method: ``tsne``, ``pca``, ``umap``, ``kpca``, or ``none``.
            Consulted when ``n_features > output_dim``. ``none`` forces the
            pass-through path even on high-dimensional input.
        output_dim: Dimensionality of the visualisation projection — ``2`` or
            ``3``. When ``n_features <= output_dim`` the features are returned
            verbatim (pad with a zero axis if strictly fewer).
        umap_n_neighbors: UMAP neighbourhood size.
        umap_min_dist: UMAP minimum-distance parameter (0–0.99).
        kpca_kernel: Kernel-PCA kernel.
        kpca_gamma: Kernel coefficient (``rbf`` / ``poly`` / ``sigmoid``);
            ``None`` lets sklearn pick the default ``1 / n_features``.
        kpca_degree: Polynomial degree for the ``poly`` kernel.
        kpca_coef0: Independent term for ``poly`` / ``sigmoid``.
    """

    method: Literal["kmeans", "dbscan", "meanshift", "hdbscan", "spectral", "hierarchical"] = Field(
        ..., description="Clustering algorithm to use."
    )
    dataframe: list[dict[str, Any]] = Field(
        ..., description="Tabular data represented as a list of rows (JSON objects)."
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="Optional explicit list of numeric columns to use for clustering.",
    )

    k: int = Field(3, gt=0, description="Number of clusters for kmeans.")
    n_init: int = Field(10, gt=0, description="Number of kmeans initializations.")
    random_state: int = Field(42, description="Random seed for deterministic kmeans behavior.")

    eps: float = Field(0.5, gt=0.0, description="Neighborhood radius for dbscan.")
    min_samples: int = Field(5, gt=0, description="Min points per neighborhood.")

    bandwidth: float | None = Field(
        default=None,
        gt=0.0,
        description="Mean-Shift bandwidth; None triggers sklearn's estimate_bandwidth().",
    )

    hdbscan_min_cluster_size: int = Field(5, gt=1, description="HDBSCAN minimum cluster size.")
    hdbscan_min_samples: int | None = Field(
        default=None,
        gt=0,
        description="HDBSCAN density threshold; None reuses min_cluster_size.",
    )

    spectral_n_clusters: int = Field(
        4, gt=1, description="Number of clusters for spectral clustering."
    )
    spectral_affinity: Literal["rbf", "nearest_neighbors"] = Field(
        default="rbf",
        description="Spectral affinity matrix type.",
    )
    spectral_n_neighbors: int = Field(
        10, gt=1, description="Neighbourhood size for spectral with nearest_neighbors affinity."
    )
    spectral_gamma: float = Field(
        1.0, gt=0.0, description="Kernel coefficient for spectral with rbf affinity."
    )

    hierarchical_n_clusters: int = Field(
        4, gt=1, description="Number of clusters for hierarchical (Agglomerative) clustering."
    )
    hierarchical_linkage: Literal["ward", "complete", "average", "single"] = Field(
        default="ward",
        description="Linkage strategy for AgglomerativeClustering.",
    )

    reduction_method: Literal["tsne", "pca", "umap", "kpca", "none"] = Field(
        default="tsne",
        description=(
            "Dim-reduction method for the visual projection when n_features > output_dim. "
            "'none' forces pass-through. Ignored when n_features <= output_dim."
        ),
    )
    output_dim: Literal[2, 3] = Field(
        default=2,
        description="Number of axes in the visualisation projection (2D or 3D scatter).",
    )

    umap_n_neighbors: int = Field(15, gt=1, description="UMAP neighbourhood size.")
    umap_min_dist: float = Field(
        0.1, ge=0.0, le=0.99, description="UMAP minimum-distance parameter."
    )

    kpca_kernel: Literal["rbf", "poly", "sigmoid", "cosine"] = Field(
        default="rbf",
        description="Kernel for Kernel PCA.",
    )
    kpca_gamma: float | None = Field(
        default=None,
        gt=0.0,
        description="Kernel coefficient (rbf/poly/sigmoid); None defers to sklearn.",
    )
    kpca_degree: int = Field(3, gt=0, description="Polynomial degree for the 'poly' kernel.")
    kpca_coef0: float = Field(
        1.0, description="Independent term for the 'poly' and 'sigmoid' kernels."
    )

    @field_validator("dataframe")
    @classmethod
    def validate_dataframe_not_empty(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reject empty ``dataframe`` payloads at parse time."""
        if len(value) == 0:
            raise ValueError("'dataframe' must contain at least one row.")
        return value

    @model_validator(mode="after")
    def validate_feature_columns(self) -> "ClusterRequest":
        """Cross-field check: ``feature_columns`` must refer to keys in the first row."""
        rows = self.dataframe

        if self.feature_columns is not None and len(self.feature_columns) == 0:
            raise ValueError("'feature_columns' cannot be an empty list.")

        first_row = rows[0]
        available_columns = set(first_row.keys())

        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in available_columns]
            if missing:
                raise ValueError(
                    f"feature_columns contains unknown columns: {missing}. Available columns: {sorted(available_columns)}"
                )

        return self


class ClusterResponse(BaseModel):
    """Response returned by ``POST /cluster``.

    Args:
        method_used: Echo of the algorithm that was actually applied.
        dataframe: Input rows with ``cluster``, ``__viz_x``, ``__viz_y`` and
            (in 3D mode) ``__viz_z`` added.
        visualization_mode: How the projection was produced.
        visualization_columns: Two or three dataframe keys to use as scatter
            coordinates (length matches ``output_dim``).
        visualization_labels: Human-readable axis titles, one per visualisation
            column.
    """

    method_used: str
    dataframe: list[dict[str, Any]]
    visualization_mode: Literal["feature_space", "tsne", "pca", "umap", "kpca"] = Field(
        ..., description="Projection mode used for the visualisation."
    )
    visualization_columns: list[str] = Field(
        ...,
        description="Dataframe columns to use as x/y(/z) coordinates in scatter plots.",
    )
    visualization_labels: list[str] = Field(
        ..., description="Human-readable axis labels for the visualisation."
    )
