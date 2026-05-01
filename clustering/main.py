import numpy as np
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE

from schemas import ClusterRequest, ClusterResponse

CONFIG_PATH = "config.yaml"
ENV_FILE_PATH = ".env"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)
load_dotenv(ENV_FILE_PATH)

VIZ_X_COL = "__viz_x"
VIZ_Y_COL = "__viz_y"

app = FastAPI(
    title="Clustering API",
    description="A lightweight API for kmeans and dbscan clustering.",
)


def _infer_numeric_columns(rows: list[dict[str, object]]) -> list[str]:
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


def _build_visual_projection(
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    random_state: int,
) -> tuple[np.ndarray, str, list[str]]:
    n_rows, n_features = feature_matrix.shape

    if n_features == 1:
        zero_col = np.zeros((n_rows, 1), dtype=float)
        return (
            np.hstack([feature_matrix, zero_col]),
            "feature_space",
            [
                feature_columns[0],
                "Zero axis",
            ],
        )

    if n_features == 2:
        return feature_matrix, "feature_space", [feature_columns[0], feature_columns[1]]

    if n_rows < 5:
        return (
            feature_matrix[:, :2],
            "feature_space",
            [
                feature_columns[0],
                feature_columns[1],
            ],
        )

    perplexity = min(30.0, float(n_rows - 1))

    projection = TSNE(
        n_components=2,
        random_state=random_state,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(feature_matrix)

    return projection, "tsne", ["t-SNE 1", "t-SNE 2"]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Clustering API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/methods")
def list_methods() -> dict[str, list[str]]:
    return {"available_methods": ["kmeans", "dbscan"]}


@app.post("/cluster", response_model=ClusterResponse)
def cluster_dataframe(request: ClusterRequest) -> ClusterResponse:
    rows = request.dataframe
    feature_matrix, feature_columns = _build_feature_matrix(
        rows, request.feature_columns
    )

    if request.method == "kmeans":
        if request.k > len(rows):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"k ({request.k}) cannot be greater than the number of rows ({len(rows)})."
                ),
            )
        estimator = KMeans(
            n_clusters=request.k,
            n_init=request.n_init,
            random_state=request.random_state,
        )
    elif request.method == "dbscan":
        estimator = DBSCAN(
            eps=request.eps,
            min_samples=request.min_samples,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

    try:
        labels = estimator.fit_predict(feature_matrix)
        projection, projection_mode, projection_labels = _build_visual_projection(
            feature_matrix=feature_matrix,
            feature_columns=feature_columns,
            random_state=request.random_state,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clustering input: {str(e)}. Features used: {feature_columns}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

    output_rows = [dict(row) for row in rows]
    for row, label, point in zip(output_rows, labels, projection):
        row["cluster"] = int(label)
        row[VIZ_X_COL] = float(point[0])
        row[VIZ_Y_COL] = float(point[1])

    return ClusterResponse(
        method_used=request.method,
        dataframe=output_rows,
        visualization_mode=projection_mode,
        visualization_columns=[VIZ_X_COL, VIZ_Y_COL],
        visualization_labels=projection_labels,
    )
