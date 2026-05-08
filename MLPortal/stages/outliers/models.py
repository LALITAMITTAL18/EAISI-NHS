"""Pydantic models for the Outlier Detection stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ColumnOutlierSpec(BaseModel):
    """Outlier handling configuration for a single column."""

    column: str
    exempt: bool = False
    action: Literal["keep", "remove", "winsorize"] = "keep"


class OutlierConfig(BaseModel):
    """Full outlier detection configuration."""

    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    require_both_flags: bool = True
    column_specs: list[ColumnOutlierSpec] = Field(default_factory=list)


class ColumnOutlierResult(BaseModel):
    """Outlier detection results for a single column."""

    column: str
    n_iqr_flagged: int
    n_zscore_flagged: int
    n_dual_flagged: int
    lower_bound: float | None
    upper_bound: float | None


class OutlierResult(BaseModel):
    """Aggregated outlier detection results."""

    n_rows_before: int
    n_rows_after: int
    n_removed: int
    column_results: list[ColumnOutlierResult]
