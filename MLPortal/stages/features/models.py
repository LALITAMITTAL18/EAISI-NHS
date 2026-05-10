"""Pydantic models for the Feature Engineering stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DerivedFeatureSpec(BaseModel):
    """A new column created by a user-defined formula."""

    name: str
    formula: str
    description: str = ""


class PolynomialSpec(BaseModel):
    """Polynomial / interaction terms configuration."""

    columns: list[str]
    degree: int = 2
    interaction_only: bool = True


class BinningSpec(BaseModel):
    """Bin a continuous column into ordered categories."""

    column: str
    new_column: str
    strategy: Literal["equal_width", "equal_freq", "custom"] = "equal_freq"
    n_bins: int = 4
    custom_bins: list[float] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class TransformSpec(BaseModel):
    """Apply a mathematical transform to reduce skewness."""

    column: str
    method: Literal["log1p", "sqrt", "yeo_johnson", "quantile"] = "yeo_johnson"


class AggregationSpec(BaseModel):
    """Aggregate a numeric column by a categorical groupby column."""

    group_column: str
    value_column: str
    agg_function: Literal["mean", "std", "median", "count"] = "mean"
    new_column: str


class EncodingSpec(BaseModel):
    """Encoding for a categorical column."""

    column: str
    method: Literal["target", "label", "onehot"] = "target"
    target_smoothing: float = 30.0


class FeatureConfig(BaseModel):
    """Full feature engineering configuration."""

    derived: list[DerivedFeatureSpec] = Field(default_factory=list)
    polynomial: list[PolynomialSpec] = Field(default_factory=list)
    binning: list[BinningSpec] = Field(default_factory=list)
    transforms: list[TransformSpec] = Field(default_factory=list)
    aggregations: list[AggregationSpec] = Field(default_factory=list)
    encodings: list[EncodingSpec] = Field(default_factory=list)
    columns_to_drop: list[str] = Field(default_factory=list)
