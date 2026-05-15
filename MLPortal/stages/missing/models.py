"""Pydantic models for the Missing Data stage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImputerSpec(BaseModel):
    """Imputation strategy for a single column."""

    column: str
    strategy: Literal[
        "mean", "median", "most_frequent", "constant",
        "knn", "mice", "drop_rows", "add_indicator"
    ] = "median"
    constant_value: str | float | None = None
    knn_k: int = 5
    mice_max_iter: int = 10
    add_indicator_flag: bool = False


class MissingConfig(BaseModel):
    """Full missing data configuration."""

    column_specs: list[ImputerSpec] = Field(default_factory=list)
    global_indicator_threshold: float = 0.02
    sentinel_values: list[int | float | str] = Field(
        default_factory=list,
        description="Values treated as missing (e.g. 9 in NHS PROMs Likert columns).",
    )


class ImputationResult(BaseModel):
    """Summary of imputation applied."""

    columns_imputed: int
    rows_dropped: int
    indicator_columns_added: list[str]
    strategy_summary: dict[str, list[str]]
