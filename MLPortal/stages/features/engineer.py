"""Pure functions for feature engineering."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import PolynomialFeatures

from stages.features.models import (
    AggregationSpec,
    BinningSpec,
    DerivedFeatureSpec,
    EncodingSpec,
    FeatureConfig,
    PolynomialSpec,
    TransformSpec,
)

_FORMULA_ALLOWED = re.compile(r"^[A-Za-z0-9_\s\+\-\*\/\(\)\.\,\<\>\=\!]+$")


def add_derived(
    df: pd.DataFrame,
    specs: list[DerivedFeatureSpec],
) -> pd.DataFrame:
    """Evaluate user-defined formula expressions and add them as new columns."""
    df = df.copy()
    for spec in specs:
        if not _FORMULA_ALLOWED.match(spec.formula):
            raise ValueError(
                f"Formula '{spec.formula}' contains disallowed characters."
            )
        df[spec.name] = df.eval(spec.formula)
    return df


def add_polynomial(
    df: pd.DataFrame,
    specs: list[PolynomialSpec],
) -> pd.DataFrame:
    """Add polynomial / interaction term columns for selected feature sets."""
    df = df.copy()
    for spec in specs:
        cols = [c for c in spec.columns if c in df.columns]
        if not cols:
            continue
        pf = PolynomialFeatures(
            degree=spec.degree,
            interaction_only=spec.interaction_only,
            include_bias=False,
        )
        arr = pf.fit_transform(df[cols].fillna(0))
        feature_names = pf.get_feature_names_out(cols)
        for i, name in enumerate(feature_names):
            if name not in cols:  # skip original columns
                clean_name = name.replace(" ", "_")
                df[clean_name] = arr[:, i]
    return df


def add_binning(
    df: pd.DataFrame,
    specs: list[BinningSpec],
) -> pd.DataFrame:
    """Bin continuous columns into ordered categorical columns."""
    df = df.copy()
    for spec in specs:
        if spec.column not in df.columns:
            continue
        labels = spec.labels if spec.labels else None
        if spec.strategy == "equal_width":
            df[spec.new_column] = pd.cut(
                df[spec.column], bins=spec.n_bins, labels=labels
            )
        elif spec.strategy == "equal_freq":
            df[spec.new_column] = pd.qcut(
                df[spec.column], q=spec.n_bins, labels=labels, duplicates="drop"
            )
        elif spec.strategy == "custom" and spec.custom_bins:
            df[spec.new_column] = pd.cut(
                df[spec.column], bins=spec.custom_bins, labels=labels
            )
    return df


def add_transforms(
    df: pd.DataFrame,
    specs: list[TransformSpec],
) -> pd.DataFrame:
    """Apply mathematical transforms to reduce skewness."""
    df = df.copy()
    for spec in specs:
        if spec.column not in df.columns:
            continue
        col = df[spec.column]
        new_name = f"{spec.column}_{spec.method}"
        if spec.method == "log1p":
            df[new_name] = np.log1p(col.clip(lower=0))
        elif spec.method == "sqrt":
            df[new_name] = np.sqrt(col.clip(lower=0))
        elif spec.method == "yeo_johnson":
            pt = scipy_stats.yeojohnson(col.dropna())
            transformed = np.full(len(df), np.nan)
            mask = col.notna()
            transformed[mask] = scipy_stats.yeojohnson(col[mask])[0]
            df[new_name] = transformed
        elif spec.method == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            qt = QuantileTransformer(output_distribution="normal", random_state=42)
            mask = col.notna()
            arr = col[mask].to_numpy().reshape(-1, 1)
            transformed = np.full(len(df), np.nan)
            transformed[mask] = qt.fit_transform(arr).ravel()
            df[new_name] = transformed
    return df


def add_aggregations(
    df: pd.DataFrame,
    specs: list[AggregationSpec],
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add group-aggregation features.

    If *reference_df* is provided (e.g. training data), aggregation statistics
    are computed from it and then merged — preventing leakage when applied to test.
    """
    df = df.copy()
    ref = reference_df if reference_df is not None else df
    for spec in specs:
        if spec.group_column not in ref.columns or spec.value_column not in ref.columns:
            continue
        agg = (
            ref.groupby(spec.group_column)[spec.value_column]
            .agg(spec.agg_function)
            .rename(spec.new_column)
            .reset_index()
        )
        df = df.merge(agg, on=spec.group_column, how="left")
    return df


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: list[EncodingSpec],
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply target encoding with OOF smoothing on train; apply mapping to test.

    Uses smoothed means: (group_count * group_mean + smoothing * global_mean)
                         / (group_count + smoothing)
    """
    train = train_df.copy()
    test = test_df.copy()
    global_mean = float(train[target].mean())

    for spec in specs:
        if spec.method != "target" or spec.column not in train.columns:
            continue
        smoothing = spec.target_smoothing
        stats = train.groupby(spec.column)[target].agg(["mean", "count"])
        encoded = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        ).rename(f"{spec.column}_target_enc")

        train = train.merge(
            encoded.reset_index(), on=spec.column, how="left"
        )
        test = test.merge(
            encoded.reset_index(), on=spec.column, how="left"
        )
        # Fill unseen categories with global mean
        test[f"{spec.column}_target_enc"].fillna(global_mean, inplace=True)

    return train, test


def apply_feature_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FeatureConfig,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the full FeatureConfig to train and test DataFrames.

    Aggregation and target encoding are always fit on train only.
    """
    for df_part in [train_df, test_df]:
        df_part.drop(
            columns=[c for c in config.columns_to_drop if c in df_part.columns],
            inplace=True,
        )

    train = add_derived(train_df, config.derived)
    test = add_derived(test_df, config.derived)

    train = add_polynomial(train, config.polynomial)
    test = add_polynomial(test, config.polynomial)

    train = add_binning(train, config.binning)
    test = add_binning(test, config.binning)

    train = add_transforms(train, config.transforms)
    test = add_transforms(test, config.transforms)

    train = add_aggregations(train, config.aggregations, reference_df=train)
    test = add_aggregations(test, config.aggregations, reference_df=train)

    train, test = add_target_encoding(train, test, config.encodings, target)

    return train, test
