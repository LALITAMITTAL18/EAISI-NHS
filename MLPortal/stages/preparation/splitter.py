"""Pure functions for train/test splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from stages.preparation.models import PrepConfig, SplitResult


def random_split(
    df: pd.DataFrame,
    target: str,
    config: PrepConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified or plain random train/test split."""
    sc = config.split
    stratify = df[target] if sc.stratify and _can_stratify(df[target]) else None
    train, test = train_test_split(
        df,
        test_size=sc.test_size,
        random_state=sc.random_seed,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def time_split(
    df: pd.DataFrame,
    time_column: str,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by a time column — all rows before *cutoff* go to train."""
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff)
    train = df[df[time_column] < cutoff_dt].reset_index(drop=True)
    test = df[df[time_column] >= cutoff_dt].reset_index(drop=True)
    return train, test


def build_split_result(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task: str,
) -> SplitResult:
    """Build a SplitResult summary from the split DataFrames."""
    numeric = train.select_dtypes(include="number").columns.tolist()
    categorical = train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    if target in numeric:
        numeric.remove(target)
    if target in categorical:
        categorical.remove(target)

    train_bal: dict[str, float] = {}
    test_bal: dict[str, float] = {}
    if task in ("classification", "ordinal"):
        vc_train = train[target].value_counts(normalize=True)
        vc_test = test[target].value_counts(normalize=True)
        train_bal = {str(k): round(float(v), 4) for k, v in vc_train.items()}
        test_bal = {str(k): round(float(v), 4) for k, v in vc_test.items()}

    return SplitResult(
        n_train=len(train),
        n_test=len(test),
        n_features=len(numeric) + len(categorical),
        numeric_cols=numeric,
        categorical_cols=categorical,
        target_column=target,
        class_balance_train=train_bal,
        class_balance_test=test_bal,
    )


def _can_stratify(series: pd.Series) -> bool:
    """Return True if stratified split is feasible (categorical / low-cardinality)."""
    return series.nunique() <= 20 and series.nunique() > 1
