from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ClusterRequest(BaseModel):
    method: Literal["kmeans", "dbscan"] = Field(
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
    random_state: int = Field(
        42, description="Random seed for deterministic kmeans behavior."
    )

    eps: float = Field(0.5, gt=0.0, description="Neighborhood radius for dbscan.")
    min_samples: int = Field(5, gt=0, description="Min points per neighborhood.")

    @field_validator("dataframe")
    @classmethod
    def validate_dataframe_not_empty(
        cls, value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(value) == 0:
            raise ValueError("'dataframe' must contain at least one row.")
        return value

    @model_validator(mode="after")
    def validate_feature_columns(self) -> "ClusterRequest":
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
    method_used: str
    dataframe: list[dict[str, Any]]
    visualization_mode: Literal["feature_space", "tsne"] = Field(
        ..., description="Projection mode used for 2D visualization."
    )
    visualization_columns: list[str] = Field(
        ...,
        description="Two dataframe columns to use as x/y coordinates in scatter plots.",
    )
    visualization_labels: list[str] = Field(
        ..., description="Human-readable labels for visualization x/y axes."
    )
