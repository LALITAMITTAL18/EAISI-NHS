"""Pure functions for fitting and applying imputers.

Key invariant: fit_imputer() always receives only TRAINING data.
apply_imputer() applies the fitted object to any DataFrame (train or test).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from stages.missing.models import ImputationResult, ImputerSpec, MissingConfig


def fit_imputer(
    spec: ImputerSpec,
    train_series: pd.Series,
) -> Any:
    """Fit and return a sklearn imputer on training data for a single column."""
    strategy = spec.strategy
    arr = train_series.to_frame()

    if strategy in ("mean", "median", "most_frequent"):
        imp = SimpleImputer(strategy=strategy)
    elif strategy == "constant":
        imp = SimpleImputer(
            strategy="constant",
            fill_value=spec.constant_value if spec.constant_value is not None else 0,
        )
    elif strategy == "knn":
        imp = KNNImputer(n_neighbors=spec.knn_k, weights="distance")
    elif strategy == "mice":
        imp = IterativeImputer(
            random_state=42,
            max_iter=spec.mice_max_iter,
        )
    else:
        return None  # drop_rows / add_indicator handled separately

    imp.fit(arr)
    return imp


def apply_imputer(
    fitted_imputer: Any,
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Apply a fitted imputer to *column* in *df*; return a copy."""
    df = df.copy()
    arr = df[[column]].to_numpy()
    df[column] = fitted_imputer.transform(arr).ravel()
    return df


def fit_apply_config(
    config: MissingConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ImputationResult]:
    """Apply the full MissingConfig to train and test DataFrames.

    Imputers are always fit on train_df only, then applied to both.
    Returns (imputed_train, imputed_test, ImputationResult).
    """
    train = train_df.copy()
    test = test_df.copy()

    rows_dropped_train = 0
    rows_dropped_test = 0
    indicators_added: list[str] = []
    strategy_summary: dict[str, list[str]] = {}
    fitted: dict[str, Any] = {}

    spec_map: dict[str, ImputerSpec] = {s.column: s for s in config.column_specs}

    for col, spec in spec_map.items():
        if col not in train.columns:
            continue

        if spec.add_indicator_flag:
            ind = f"{col}_missing"
            train[ind] = train[col].isna().astype("uint8")
            test[ind] = test[col].isna().astype("uint8")
            indicators_added.append(ind)

        if spec.strategy == "drop_rows":
            before = len(train)
            train = train.dropna(subset=[col])
            rows_dropped_train += before - len(train)
            before = len(test)
            test = test.dropna(subset=[col])
            rows_dropped_test += before - len(test)
            strategy_summary.setdefault("drop_rows", []).append(col)
            continue

        if spec.strategy == "add_indicator":
            strategy_summary.setdefault("add_indicator", []).append(col)
            continue

        imp = fit_imputer(spec, train[col])
        if imp is not None:
            train = apply_imputer(imp, train, col)
            test = apply_imputer(imp, test, col)
            fitted[col] = imp
            strategy_summary.setdefault(spec.strategy, []).append(col)

    result = ImputationResult(
        columns_imputed=len(fitted),
        rows_dropped=rows_dropped_train,
        indicator_columns_added=indicators_added,
        strategy_summary=strategy_summary,
    )
    return train, test, result
