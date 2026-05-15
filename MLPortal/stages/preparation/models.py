"""Pydantic models for the Preparation & Split stage."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from stages.preparation.cleaner import DerivedStep, NullHandlingRule


class OutcomeThresholdConfig(BaseModel):
    """User-defined numeric cutoff that defines a positive outcome.

    Used to derive a binary label from a continuous regression target
    (e.g. health_gain > 8 → Positive outcome = True).
    """

    enabled: bool = False
    threshold: float = 0.0
    direction: Literal["above", "below"] = "above"
    positive_label: str = "Positive"
    negative_label: str = "Negative"
    derived_column_name: str = "outcome_label"


class SplitConfig(BaseModel):
    """Train / test split parameters."""

    method: Literal["random", "time"] = "random"
    test_size: float = 0.20
    random_seed: int = 42
    stratify: bool = True
    time_column: str | None = None
    time_cutoff: str | None = None


class ScalerConfig(BaseModel):
    """Feature scaling configuration (applied post-split, fit on train only)."""

    method: Literal["standard", "minmax", "robust", "quantile", "none"] = "standard"
    apply_to_numeric: bool = True


class EncoderConfig(BaseModel):
    """Encoding for remaining categorical columns (not already handled in Stage 5)."""

    method: Literal["onehot", "ordinal"] = "onehot"
    handle_unknown: Literal["ignore", "error"] = "ignore"
    drop_first: bool = False


class PrepConfig(BaseModel):
    """Full data preparation and split configuration."""

    columns_to_drop: list[str] = Field(default_factory=list)
    null_rules: dict[str, Any] = Field(default_factory=dict)   # col -> NullHandlingRule dict
    derived_steps: list[Any] = Field(default_factory=list)      # list of DerivedStep dicts
    split: SplitConfig = Field(default_factory=SplitConfig)
    scaler: ScalerConfig = Field(default_factory=ScalerConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    outcome_threshold: OutcomeThresholdConfig = Field(
        default_factory=OutcomeThresholdConfig
    )


class SplitResult(BaseModel):
    """Summary of the train/test split."""

    n_train: int
    n_test: int
    n_features: int
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_column: str
    class_balance_train: dict[str, float] = Field(default_factory=dict)
    class_balance_test: dict[str, float] = Field(default_factory=dict)
