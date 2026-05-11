"""Pure functions for ingesting data from various file formats."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stages.upload.models import DatasetMeta, UploadConfig


def read_file(
    source: Any,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read a CSV, Excel, Parquet or JSON file into a DataFrame.

    *source* can be a file path (str/Path) or a Streamlit UploadedFile object.
    """
    name = getattr(source, "name", str(source)).lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(source)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    if name.endswith(".json"):
        return pd.read_json(source)
    # Default: CSV
    return pd.read_csv(source, sep=sep, encoding=encoding, low_memory=False)


def apply_sentinels(df: pd.DataFrame, sentinels: list[str]) -> pd.DataFrame:
    """Replace sentinel values with NaN across the entire DataFrame.

    Numeric sentinels are cast to the column dtype before comparison so that
    string "9" correctly matches integer 9.
    """
    df = df.copy()
    for col in df.columns:
        for sv in sentinels:
            try:
                typed_sv: Any = df[col].dtype.type(sv)  # e.g. np.int8("9")
            except (ValueError, TypeError):
                typed_sv = sv
            df[col] = df[col].replace(typed_sv, np.nan)
            if sv != str(typed_sv):
                df[col] = df[col].replace(sv, np.nan)
    return df


def optimise_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, DatasetMeta]:
    """Downcast numeric columns and cast string columns to Categorical.

    Returns the optimised DataFrame and a DatasetMeta summary.
    """
    mem_before = df.memory_usage(deep=True).sum() / 1_048_576

    df = df.copy()
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique()
        if n_unique / max(len(df), 1) < 0.5:
            df[col] = df[col].astype("category")

    mem_after = df.memory_usage(deep=True).sum() / 1_048_576

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    meta = DatasetMeta(
        n_rows=len(df),
        n_cols=df.shape[1],
        column_names=df.columns.tolist(),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        memory_mb_raw=round(mem_before, 2),
        memory_mb_optimised=round(mem_after, 2),
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        missing_counts={
            c: int(df[c].isna().sum())
            for c in df.columns
            if df[c].isna().any()
        },
    )
    return df, meta


def build_config(
    file_name: str,
    target_column: str,
    task_type: str,
    sentinel_values: list[str],
    csv_separator: str = ",",
) -> UploadConfig:
    """Construct and validate an UploadConfig from user inputs."""
    return UploadConfig(
        file_name=file_name,
        target_column=target_column,
        task_type=task_type,
        sentinel_values=sentinel_values,
        csv_separator=csv_separator,
    )
