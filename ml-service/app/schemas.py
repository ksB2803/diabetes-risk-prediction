"""Pydantic request/response models for the internal ML API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Algorithm = Literal["logistic_regression", "random_forest", "decision_tree"]
ScoringMetric = Literal["roc_auc", "f1", "recall", "accuracy", "precision"]
TuningLevel = Literal["quick", "balanced", "thorough"]

ALGORITHM_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
}


class TrainConfig(BaseModel):
    """Configuration sent as the `config` form field of POST /ml/train."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    target_column: str = Field(min_length=1, max_length=200)
    feature_columns: list[str] = Field(min_length=1, max_length=100)
    train_fraction: float = Field(ge=0.5, le=0.9)
    algorithms: list[Algorithm] = Field(min_length=1, max_length=3)
    scoring_metric: ScoringMetric = "roc_auc"
    tuning_level: TuningLevel = "balanced"
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)
    task_type: Literal["binary_classification"] = "binary_classification"
    positive_class: Optional[str] = Field(default=None, max_length=200)

    @field_validator("feature_columns")
    @classmethod
    def _unique_features(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("feature_columns contains duplicates")
        return value

    @field_validator("algorithms")
    @classmethod
    def _unique_algorithms(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("algorithms contains duplicates")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    values: dict[str, Any] = Field(min_length=1, max_length=200)


class PredictResponse(BaseModel):
    model_id: str
    predicted_class: str
    is_positive_class: bool
    probability_positive: Optional[float]
    class_probabilities: Optional[dict[str, float]]
    positive_class: str
    negative_class: str
    warnings: list[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    sklearn_version: str
    model_store_writable: bool
