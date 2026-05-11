"""Pure functions for outlier detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.outliers.models import (
    ColumnOutlierResult,
    ColumnOutlierSpec,
    OutlierConfig,
    OutlierResult,
)


def flag_iqr(
    series: pd.Series,
    factor: float = 1.5,
) -> tuple[pd.Series, float, float]:
    """Return (boolean flag mask, lower bound, upper bound) using the IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = float(q1 - factor * iqr)
    upper = float(q3 + factor * iqr)
    flagged = (series < lower) | (series > upper)
    return flagged, lower, upper


def flag_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return a boolean mask where |z-score| > threshold."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean).abs() / std
    return z > threshold


def dual_flag(
    series: pd.Series,
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
) -> pd.Series:
    """Return a boolean mask for rows flagged by BOTH the IQR and Z-score rules."""
    iqr_mask, _, _ = flag_iqr(series, iqr_factor)
    z_mask = flag_zscore(series, zscore_threshold)
    return iqr_mask & z_mask


def apply_outlier_config(
    df: pd.DataFrame,
    config: OutlierConfig,
) -> tuple[pd.DataFrame, OutlierResult]:
    """Apply all column-level outlier actions and return (cleaned_df, result)."""
    df = df.copy()
    drop_mask = pd.Series(False, index=df.index)
    col_results: list[ColumnOutlierResult] = []

    spec_map: dict[str, ColumnOutlierSpec] = {
        s.column: s for s in config.column_specs
    }

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        spec = spec_map.get(col, ColumnOutlierSpec(column=col))
        if spec.exempt:
            continue

        clean = df[col].dropna()
        iqr_mask, lower, upper = flag_iqr(clean, config.iqr_factor)
        z_mask = flag_zscore(clean, config.zscore_threshold)

        if config.require_both_flags:
            flagged_idx = clean.index[iqr_mask & z_mask]
        else:
            flagged_idx = clean.index[iqr_mask | z_mask]

        n_dual = int((iqr_mask & z_mask).sum())

        col_results.append(
            ColumnOutlierResult(
                column=col,
                n_iqr_flagged=int(iqr_mask.sum()),
                n_zscore_flagged=int(z_mask.sum()),
                n_dual_flagged=n_dual,
                lower_bound=lower,
                upper_bound=upper,
            )
        )

        if spec.action == "remove":
            drop_mask.loc[flagged_idx] = True
        elif spec.action == "winsorize":
            df.loc[df[col] < lower, col] = lower
            df.loc[df[col] > upper, col] = upper

    n_before = len(df)
    df = df[~drop_mask]

    return df, OutlierResult(
        n_rows_before=n_before,
        n_rows_after=len(df),
        n_removed=int(drop_mask.sum()),
        column_results=col_results,
    )
