from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
import math


class ForecastRequest(BaseModel):
    model_type: Literal["prophet", "chronos", "arima"] = Field(
        default="prophet", description="Choose the forecasting model."
    )

    dates: List[str] = Field(
        ..., description="Timestamps for the historical data (ISO format)."
    )
    values: List[float] = Field(..., description="Historical time series values.")

    n_prev: int = Field(
        ..., gt=0, description="Number of previous points to consider for fitting."
    )
    n_predict: int = Field(..., gt=0, description="Number of future points to predict.")
    alpha: float = Field(
        0.05,
        ge=0.01,
        le=0.2,
        description="Significance level for CI",
    )

    @field_validator("values")
    def check_lengths_match(cls, v, info):
        if "dates" in info.data and len(v) != len(info.data["dates"]):
            raise ValueError("The number of dates and values must be strictly equal.")
        if len(v) == 0:
            raise ValueError("At least one historical point is required.")
        if any(not math.isfinite(val) for val in v):
            raise ValueError("All values must be finite numbers.")
        return v


class ForecastPoint(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    model_used: str
    forecast: List[ForecastPoint]
