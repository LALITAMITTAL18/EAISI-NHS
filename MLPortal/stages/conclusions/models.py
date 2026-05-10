"""Pydantic models for the Conclusions stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    """Summary card for the best or selected model."""

    model_name: str
    task: str
    best_params: dict
    test_metrics: dict[str, float]
    feature_importances: list[dict] = Field(default_factory=list)
    notes: str = ""


class PerformanceMatrix(BaseModel):
    """All model metrics in one flat structure for export."""

    models: list[str]
    metrics: list[str]
    values: list[list[float | None]]


class PipelineRunSummary(BaseModel):
    """High-level summary of the entire pipeline run."""

    file_name: str
    target_column: str
    task_type: str
    n_rows_raw: int
    n_rows_train: int
    n_rows_test: int
    n_features: int
    n_models_trained: int
    best_model: str
    best_metric_name: str
    best_metric_value: float
    data_preparation_steps: list[str] = Field(default_factory=list)
    key_findings: str = ""
