"""Pydantic models for the Explanation stage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExplanationConfig(BaseModel):
    """Configuration for the explanation stage."""

    model_name: str
    max_shap_samples: int = 500
    max_pdp_samples: int = 200
    top_n_features: int = 20


class FeatureImportanceRow(BaseModel):
    """Importance value for a single feature."""

    feature: str
    importance: float
    method: str


class ShapResult(BaseModel):
    """SHAP value output for the selected model."""

    model_name: str
    feature_names: list[str]
    mean_abs_shap: list[float]
    shap_values: list[list[float]] = Field(default_factory=list)
    sample_features: list[dict[str, Any]] = Field(default_factory=list)


class PdpResult(BaseModel):
    """Partial dependence data for one feature."""

    feature: str
    grid_values: list[float]
    avg_predictions: list[float]
