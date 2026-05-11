"""Pydantic models for the Explore stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnStat(BaseModel):
    """Summary statistics for a single column."""

    name: str
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    top_value: str | None = None
    top_freq: int | None = None


class ExploreConfig(BaseModel):
    """User choices made on the explore page."""

    correlation_method: str = "pearson"
    subgroup_column: str | None = None
    selected_columns: list[str] = Field(default_factory=list)
