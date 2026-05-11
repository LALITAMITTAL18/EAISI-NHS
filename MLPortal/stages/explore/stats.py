"""Pure functions for exploratory data analysis."""

from __future__ import annotations

import pandas as pd
from scipy import stats

from stages.explore.models import ColumnStat, ExploreConfig


def summary_stats(df: pd.DataFrame) -> list[ColumnStat]:
    """Compute summary statistics for every column."""
    rows: list[ColumnStat] = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / max(len(s), 1) * 100, 2)
        is_numeric = pd.api.types.is_numeric_dtype(s)

        stat = ColumnStat(
            name=col,
            dtype=str(s.dtype),
            n_missing=n_missing,
            pct_missing=pct_missing,
            n_unique=int(s.nunique()),
        )
        if is_numeric:
            clean = s.dropna()
            stat = stat.model_copy(
                update=dict(
                    mean=round(float(clean.mean()), 4) if len(clean) else None,
                    std=round(float(clean.std()), 4) if len(clean) else None,
                    min=round(float(clean.min()), 4) if len(clean) else None,
                    q25=round(float(clean.quantile(0.25)), 4) if len(clean) else None,
                    median=round(float(clean.median()), 4) if len(clean) else None,
                    q75=round(float(clean.quantile(0.75)), 4) if len(clean) else None,
                    max=round(float(clean.max()), 4) if len(clean) else None,
                    skewness=round(float(stats.skew(clean)), 4) if len(clean) > 2 else None,
                    kurtosis=round(float(stats.kurtosis(clean)), 4) if len(clean) > 2 else None,
                )
            )
        else:
            vc = s.value_counts()
            if len(vc):
                stat = stat.model_copy(
                    update=dict(
                        top_value=str(vc.index[0]),
                        top_freq=int(vc.iloc[0]),
                    )
                )
        rows.append(stat)
    return rows


def correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    max_cols: int = 50,
) -> pd.DataFrame:
    """Compute pairwise correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number").iloc[:, :max_cols]
    return numeric.corr(method=method).round(3)


def target_correlation(
    df: pd.DataFrame,
    target: str,
    method: str = "pearson",
) -> pd.Series:
    """Return correlation of every numeric column with the target column."""
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        return pd.Series(dtype=float)
    return numeric.corr(method=method)[target].drop(target).sort_values(
        key=abs, ascending=False
    )


def subgroup_means(
    df: pd.DataFrame,
    target: str,
    group_col: str,
) -> pd.DataFrame:
    """Compute mean target value per group and the deviation from the overall mean."""
    overall = df[target].mean()
    grouped = (
        df.groupby(group_col)[target]
        .agg(["mean", "count"])
        .rename(columns={"mean": "group_mean", "count": "n"})
        .reset_index()
    )
    grouped["deviation"] = grouped["group_mean"] - overall
    return grouped


def qq_data(series: pd.Series) -> tuple[list[float], list[float]]:
    """Return (theoretical quantiles, observed quantiles) for a Q-Q plot."""
    import numpy as np
    from scipy.stats import norm

    clean = series.dropna().to_numpy()
    clean.sort()
    n = len(clean)
    quantiles = norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    return quantiles.tolist(), clean.tolist()
