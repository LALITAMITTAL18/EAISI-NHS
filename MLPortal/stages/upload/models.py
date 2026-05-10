"""Pydantic models for the Upload stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadConfig(BaseModel):
    """User configuration captured on the upload page."""

    file_name: str
    target_column: str
    task_type: Literal["regression", "classification", "ordinal"]
    sentinel_values: list[str] = Field(default_factory=lambda: ["*", "", "9", "999"])
    csv_separator: str = ","


class DatasetMeta(BaseModel):
    """Lightweight metadata describing the loaded dataset."""

    n_rows: int
    n_cols: int
    column_names: list[str]
    dtypes: dict[str, str]
    memory_mb_raw: float
    memory_mb_optimised: float
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_counts: dict[str, int]
