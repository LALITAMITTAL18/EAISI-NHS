"""Data pre-processing — Step 2.0.

Applies value recoding, null standardisation, label encoding, and column
dropping.  All operations are performed before the train/test split so
they apply uniformly to the full dataset.

Corresponds to notebook: 2.0-Data-Pre-Preparation.ipynb
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nhs_proms_pipeline.config import (
    COLS_TO_LABEL_ENCODE,
    NUMERIC_MISSING_SENTINELS,
    STRING_MISSING_SENTINELS,
    PipelineSettings,
    ProcedureConfig,
)
from nhs_proms_pipeline.utils.io import read_parquet, write_parquet
from nhs_proms_pipeline.utils.logging import get_logger
from nhs_proms_pipeline.utils.memory import POLARS_NUMERIC_DTYPES, POLARS_STRING_DTYPES

logger = get_logger(__name__)

_INPUT_FILENAME = "1.1-Reduced.parquet"
_OUTPUT_FILENAME = "2.0-preprocessing.parquet"


def preprocess_data(settings: PipelineSettings) -> Path:
    """Run the full pre-processing pipeline and persist the result.

    Steps applied:
    1. Recode binary indicators: ``9 → 0`` for columns with exactly 2
       unique values where one is ``9``.
    2. Recode "Assisted By" columns: ``9 → 0`` (not assisted), else ``1``.
    3. Replace missing value sentinels (``9``, ``999``, ``'*'``, ``''``)
       with ``null``.
    4. Label-encode ordinal/nominal categorical columns.
    5. Drop predicted columns, CSVYear, and specific Post-Op columns.

    Args:
        settings: Resolved pipeline settings.

    Returns:
        Path to the written pre-processed parquet file.
    """
    input_path = settings.interim_path(_INPUT_FILENAME)
    df = read_parquet(input_path)
    logger.info("Pre-processing — input shape: %s", df.shape)

    proc_cfg = settings.get_procedure_config()

    df = _recode_binary_indicators(df)
    df = _recode_assisted_by_cols(df)
    df = _replace_missing_sentinels(df)
    df = _label_encode_categoricals(df)
    df = _drop_irrelevant_columns(df, proc_cfg)

    output_path = settings.interim_path(_OUTPUT_FILENAME)
    write_parquet(df, output_path)
    logger.info("Pre-processing complete — output shape: %s, saved to: %s", df.shape, output_path)
    return output_path


# ── Private step functions ────────────────────────────────────────────────────


def _recode_binary_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Replace ``9`` with ``0`` in columns with exactly two unique values
    where one of those values is ``9``.

    In NHS PROMs data, ``9`` signals "not applicable/not reported" in binary
    comorbidity indicator columns (e.g. Arthritis, Diabetes).  A missing
    comorbidity report is treated as absence of that comorbidity.
    """
    cols_to_recode = [
        c
        for c in df.columns
        if _is_binary_with_nine(df[c])
    ]
    if not cols_to_recode:
        logger.debug("No binary indicator columns found to recode.")
        return df

    logger.debug("Recoding binary indicators (9 → 0): %s", cols_to_recode)
    return df.with_columns(
        [
            pl.when(pl.col(c) == 9).then(0).otherwise(pl.col(c)).alias(c)
            for c in cols_to_recode
        ]
    )


def _is_binary_with_nine(series: pl.Series) -> bool:
    """Return True if a series has ≤ 2 unique non-null values and one is 9."""
    unique_vals = set(series.drop_nulls().unique().to_list())
    return len(unique_vals) <= 2 and 9 in unique_vals


def _recode_assisted_by_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Recode 'Pre-Op Q Assisted By' and 'Post-Op Q Assisted By'.

    ``9`` → ``0`` (not assisted), any other value → ``1`` (assisted).
    """
    assisted_cols = [
        c for c in ["Pre-Op Q Assisted By", "Post-Op Q Assisted By"] if c in df.columns
    ]
    if not assisted_cols:
        return df

    exprs = [
        pl.when(pl.col(c) == 9).then(0).otherwise(1).alias(c) for c in assisted_cols
    ]
    logger.debug("Recoding assisted-by columns: %s", assisted_cols)
    return df.with_columns(exprs)


def _replace_missing_sentinels(df: pl.DataFrame) -> pl.DataFrame:
    """Convert sentinel values (``9``, ``999``, ``'*'``, ``''``) to ``null``.

    Numeric and string columns are handled separately because
    ``is_in()`` requires matching types in Polars.
    """
    numeric_exprs = [
        pl.when(pl.col(c).is_in(NUMERIC_MISSING_SENTINELS))
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in df.columns
        if df[c].dtype in POLARS_NUMERIC_DTYPES
    ]
    string_exprs = [
        pl.when(pl.col(c).is_in(STRING_MISSING_SENTINELS))
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in df.columns
        if df[c].dtype in POLARS_STRING_DTYPES
    ]
    logger.debug(
        "Replacing missing sentinels in %d numeric and %d string columns.",
        len(numeric_exprs),
        len(string_exprs),
    )
    return df.with_columns(numeric_exprs + string_exprs)


def _label_encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """Label-encode ordinal/nominal categorical columns as integers (starting from 1).

    Columns encoded:
    - ``Year`` — chronological order
    - ``Age Band`` — natural age order (alphabetical sort is correct for NHS bands)
    - ``Procedure`` — alphabetical
    - ``Provider Code`` — alphabetical

    Encoding starts at 1 so that ``0`` is reserved for "No" in binary columns.
    """
    present_cols = [c for c in COLS_TO_LABEL_ENCODE if c in df.columns]
    if not present_cols:
        logger.debug("No categorical columns to label-encode (none present in DataFrame).")
        return df

    # Build encoding maps from the data; alphabetical sort = correct for Year and Age Band
    encoding_maps: dict[str, dict[str, int]] = {}
    for col in present_cols:
        unique_labels = sorted(df[col].drop_nulls().cast(pl.Utf8).unique().to_list())
        encoding_maps[col] = {label: code for code, label in enumerate(unique_labels, start=1)}

    exprs = [
        pl.col(c)
        .cast(pl.Utf8)
        .replace({k: str(v) for k, v in encoding_maps[c].items()})
        .cast(pl.Int32)
        .alias(c)
        for c in present_cols
    ]
    logger.debug("Label-encoding columns: %s", present_cols)
    return df.with_columns(exprs)


def _drop_irrelevant_columns(df: pl.DataFrame, proc_cfg: ProcedureConfig) -> pl.DataFrame:
    """Drop predicted columns, CSVYear, and other non-feature Post-Op columns.

    Retains only the Post-Op columns that are needed to compute the
    outcome variable or are used as targets.
    """
    cols_to_drop = [
        c
        for c in df.columns
        if c in proc_cfg.predicted_col_substrings or c == "CSVYear"
    ]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    if existing_drops:
        df = df.drop(existing_drops)
        logger.debug("Dropped %d columns: %s", len(existing_drops), existing_drops)
    return df
