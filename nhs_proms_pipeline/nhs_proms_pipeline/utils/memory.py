"""Memory optimisation utilities.

These helpers reduce the in-memory footprint of a pandas or polars
DataFrame by downcasting numeric dtypes and converting object columns
to categorical — exactly as done in the NHS PROMs data collection
notebook (1.1-Collect-Data.ipynb).
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def optimise_pandas_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert object columns to category.

    Applies the following reductions in order:
    1. Replace the NHS suppression marker ``'*'`` with ``NaN``.
    2. ``int64`` → smallest unsigned integer type.
    3. ``float64`` → smallest float type.
    4. ``object`` → ``category``.

    Numeric-valued category columns (e.g. Gender coded as '0'/'1'/'2')
    are converted to ``float32`` so that pyarrow can serialise them to
    Parquet without errors.

    Args:
        df: Input DataFrame. A copy is returned; the original is not mutated.

    Returns:
        Memory-optimised copy of *df*.
    """
    df = df.copy()
    initial_mb = df.memory_usage(deep=True).sum() / 1_048_576
    logger.debug("Memory before optimisation: %.1f MB", initial_mb)

    df.replace("*", pd.NA, inplace=True)

    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned")

    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    # Convert numeric-string categories to float32 so pyarrow can handle them
    for col in df.select_dtypes(include="category").columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("float32")
        except (ValueError, TypeError):
            pass  # keep non-numeric categories as-is

    final_mb = df.memory_usage(deep=True).sum() / 1_048_576
    logger.info(
        "Memory optimisation: %.1f MB → %.1f MB (%.0f%% reduction)",
        initial_mb,
        final_mb,
        100 * (1 - final_mb / initial_mb) if initial_mb else 0,
    )
    return df


# ── Polars dtype helpers ───────────────────────────────────────────────────────

POLARS_NUMERIC_DTYPES: frozenset[type] = frozenset(
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]
)

POLARS_STRING_DTYPES: frozenset[type] = frozenset(
    [
        pl.Utf8,
        pl.String,
        pl.Categorical,
    ]
)


def numeric_cols(df: pl.DataFrame) -> list[str]:
    """Return column names whose dtype is a Polars numeric type."""
    return [c for c in df.columns if df[c].dtype in POLARS_NUMERIC_DTYPES]


def string_cols(df: pl.DataFrame) -> list[str]:
    """Return column names whose dtype is a Polars string/categorical type."""
    return [c for c in df.columns if df[c].dtype in POLARS_STRING_DTYPES]
