"""Pydantic models for the Model Comparison stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MetricRow(BaseModel):
    """One row in the comparison table — one model's test metrics."""

    model_name: str
    dataset: str = "default"
    task: str
    # Regression
    test_rmse: float | None = None
    test_mae: float | None = None
    test_r2: float | None = None
    cv_rmse: float | None = None
    # Classification
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    f2: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    # Ordinal
    exact_accuracy: float | None = None
    adjacent_accuracy: float | None = None
    ordinal_mae: float | None = None
    # Derived binary from regression (outcome threshold)
    threshold_precision: float | None = None
    threshold_recall: float | None = None
    threshold_f2: float | None = None


class BlandAltmanResult(BaseModel):
    """Bland-Altman agreement statistics for one model."""

    model_name: str
    bias: float
    upper_loa: float
    lower_loa: float
    pct_within_loa: float


class CalibrationDecile(BaseModel):
    """Calibration check — mean predicted vs mean actual per decile."""

    model_name: str
    decile: int
    mean_predicted: float
    mean_actual: float
    n_samples: int


class SubgroupResult(BaseModel):
    """Per-subgroup performance for one model and one groupby column."""

    model_name: str
    group_column: str
    group_value: str
    metric_name: str
    metric_value: float
    n_samples: int


class ComparisonResult(BaseModel):
    """Full comparison output across all trained models."""

    task: str
    rows: list[MetricRow] = Field(default_factory=list)
    bland_altman: list[BlandAltmanResult] = Field(default_factory=list)
    calibration: list[CalibrationDecile] = Field(default_factory=list)
    subgroup: list[SubgroupResult] = Field(default_factory=list)
    best_model_name: str | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
