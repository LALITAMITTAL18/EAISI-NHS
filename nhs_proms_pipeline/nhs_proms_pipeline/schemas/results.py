"""Pydantic schemas for pipeline results and model metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class CVMetrics(BaseModel):
    """Cross-validation metrics from RepeatedKFold."""

    mae: float = Field(..., description="Mean Absolute Error (CV average)")
    rmse: float = Field(..., description="Root Mean Squared Error (CV average)")
    r2: float = Field(..., description="R² coefficient of determination (CV average)")


class TestMetrics(BaseModel):
    """Hold-out test set regression metrics."""

    mae: float
    rmse: float
    r2: float


class BlandAltmanMetrics(BaseModel):
    """Bland-Altman limits of agreement statistics."""

    bias: float = Field(..., description="Mean prediction bias (positive = over-prediction)")
    loa_upper: float = Field(..., description="Upper limit of agreement (+1.96 SD)")
    loa_lower: float = Field(..., description="Lower limit of agreement (-1.96 SD)")
    loa_width: float = Field(..., description="Total width of limits of agreement")
    within_mcid: bool = Field(
        ...,
        description=(
            "True if both LoA bounds fall within ±MCID — "
            "indicates individual predictions are clinically meaningful."
        ),
    )


class MCIDClassificationMetrics(BaseModel):
    """MCID-based binary classification metrics."""

    threshold: float = Field(..., description="MCID decision threshold used.")
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float  # sensitivity
    specificity: float
    npv: float
    f1: float
    f2: float = Field(..., description="F2 score (recall weighted 2×, reduces FN)")
    auroc: float
    average_precision: float


class ModelRunResult(BaseModel):
    """Complete result for one (dataset x model) training run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    dataset_label: str
    cv_metrics: CVMetrics
    test_metrics: TestMetrics
    bland_altman: Optional[BlandAltmanMetrics] = None
    mcid_metrics: Optional[MCIDClassificationMetrics] = None
    feature_names: list[str] = Field(default_factory=list)
    numeric_cols: list[str] = Field(default_factory=list)
    cat_cols: list[str] = Field(default_factory=list)
    pipeline_path: Optional[Path] = None


class BestModelInfo(BaseModel):
    """Metadata identifying the winning model from a training run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_label: str
    model_name: str
    test_rmse: float
    test_r2: float
    mcid_f2: Optional[float] = None
    pipeline_path: Path


class PredictionResult(BaseModel):
    """Output from the inference predictor for a single patient."""

    predicted_health_gain: float = Field(
        ...,
        description=(
            "Predicted post-op health gain (Post-Op Score − Pre-Op Score). "
            "Higher is better."
        ),
    )
    predicted_benefit: bool = Field(
        ...,
        description=(
            "True if predicted_health_gain ≥ MCID threshold, "
            "indicating the patient is clinically expected to benefit from surgery."
        ),
    )
    mcid_threshold: float = Field(
        ..., description="MCID threshold used for the benefit classification."
    )
    confidence_note: str = Field(
        ...,
        description=(
            "Qualitative note about prediction certainty based on "
            "the model's Bland-Altman limits of agreement."
        ),
    )
    model_name: str
    dataset_label: str
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra metadata (feature values used, etc.).",
    )
