"""Pure functions for building sklearn ColumnTransformer pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from stages.preparation.models import EncoderConfig, PrepConfig, ScalerConfig


def _get_scaler(config: ScalerConfig):
    method = config.method
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()
    if method == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    return None  # "none"


def _get_encoder(config: EncoderConfig):
    if config.method == "onehot":
        return OneHotEncoder(
            handle_unknown=config.handle_unknown,
            drop="first" if config.drop_first else None,
            sparse_output=False,
        )
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def build_column_transformer(
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: PrepConfig,
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes + scales numeric and encodes categorical.

    This transformer must be fit on training data only.
    """
    scaler = _get_scaler(config.scaler)
    encoder = _get_encoder(config.encoder)

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler is not None and config.scaler.apply_to_numeric:
        num_steps.append(("scaler", scaler))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def apply_outcome_threshold(
    df: pd.DataFrame,
    target: str,
    config,
) -> pd.DataFrame:
    """Derive a binary label from a continuous target using the outcome threshold.

    The derived column is added alongside the original target — it is not a
    replacement.
    """
    if not config.enabled:
        return df
    df = df.copy()
    if config.direction == "above":
        df[config.derived_column_name] = (
            df[target] > config.threshold
        ).map({True: config.positive_label, False: config.negative_label})
    else:
        df[config.derived_column_name] = (
            df[target] < config.threshold
        ).map({True: config.positive_label, False: config.negative_label})
    return df
