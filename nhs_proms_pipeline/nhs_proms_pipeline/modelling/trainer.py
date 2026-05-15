"""Model trainer — builds, cross-validates, and evaluates regression pipelines.

Implements the ``build_regression_pipeline`` factory from notebook
3.1-Data-Modelling-Regression.ipynb as a clean, reusable function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nhs_proms_pipeline.schemas.results import CVMetrics, ModelRunResult, TestMetrics
from nhs_proms_pipeline.utils.logging import get_logger
from nhs_proms_pipeline.utils.memory import POLARS_NUMERIC_DTYPES, POLARS_STRING_DTYPES

logger = get_logger(__name__)


@dataclass
class TrainingOutput:
    """Container for one (dataset × model) training run.

    Separates the Pydantic-serialisable :class:`ModelRunResult` from the
    non-serialisable sklearn ``Pipeline`` and numpy arrays so both can be
    stored and passed around cleanly.
    """

    result: ModelRunResult
    fitted_pipeline: Pipeline
    y_test: np.ndarray
    y_pred: np.ndarray


def build_regression_pipeline(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target: str,
    model,
    model_name: str = "Model",
    dataset_label: str = "dataset",
    drop_cols: list[str] | None = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> TrainingOutput:
    """Cross-validate and evaluate one sklearn regression model on a Polars dataset.

    Steps:
    1. Separate features from the target column.
    2. Detect numeric vs. categorical columns via Polars dtype.
    3. Build a ``ColumnTransformer`` (median imputation + scaling for
       numeric; constant-fill + one-hot for categorical).
    4. Run ``RepeatedKFold`` cross-validation and capture CV metrics.
    5. Fit on the full training set and evaluate on the hold-out test set.

    The conversion from Polars to pandas happens only at the sklearn
    boundary to keep the upstream pipeline Polars-native.

    Args:
        train_df:      Training Polars DataFrame (must contain *target* column).
        test_df:       Test Polars DataFrame (must contain *target* column).
        target:        Name of the outcome column (``'health_gain'``).
        model:         An unfitted sklearn estimator.
        model_name:    Label used in result objects and logging.
        dataset_label: Label for the dataset variant (e.g. ``'2.1-Manual'``).
        drop_cols:     Additional columns to exclude from feature set.
        n_splits:      Number of KFold splits for cross-validation.
        n_repeats:     Number of KFold repeats for cross-validation.
        random_state:  Random seed for KFold shuffle.

    Returns:
        :class:`TrainingOutput` containing the :class:`ModelRunResult` metrics,
        the fitted sklearn ``Pipeline``, and the test-set targets/predictions as
        numpy arrays.
    """
    logger.info("Training: %s × %s", dataset_label, model_name)

    # ── 1. Feature / target separation ───────────────────────────────────────
    all_drop = [c for c in ([target] + (drop_cols or [])) if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in all_drop]

    x_train = train_df.select(feature_cols)
    y_train = train_df[target].to_numpy()

    # Align test columns to training schema (add nulls for missing columns)
    x_test = _align_test_schema(test_df, feature_cols, x_train)
    y_test = test_df[target].to_numpy()

    # ── 2. Column type detection ──────────────────────────────────────────────
    num_cols = [c for c in x_train.columns if x_train[c].dtype in POLARS_NUMERIC_DTYPES]
    cat_cols = [c for c in x_train.columns if x_train[c].dtype in POLARS_STRING_DTYPES]
    logger.debug("  Features: %d numeric, %d categorical", len(num_cols), len(cat_cols))

    # ── 3. Convert to pandas (sklearn boundary) ───────────────────────────────
    x_train_pd = x_train.to_pandas()
    x_test_pd = x_test.to_pandas()

    # ── 4. Preprocessor ───────────────────────────────────────────────────────
    preprocessor = _build_preprocessor(num_cols, cat_cols)

    # ── 5. Cross-validation ───────────────────────────────────────────────────
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    cv_pipe = Pipeline([("pre", preprocessor), ("model", model)])
    cv_res = cross_validate(
        cv_pipe,
        x_train_pd,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    cv_metrics = CVMetrics(
        mae=float(-cv_res["test_MAE"].mean()),
        rmse=float(-cv_res["test_RMSE"].mean()),
        r2=float(cv_res["test_R2"].mean()),
    )
    logger.info(
        "  CV  — MAE=%.4f  RMSE=%.4f  R²=%.4f",
        cv_metrics.mae,
        cv_metrics.rmse,
        cv_metrics.r2,
    )

    # ── 6. Fit on full training set ───────────────────────────────────────────
    fitted_pipeline = Pipeline([("pre", preprocessor), ("model", model)])
    fitted_pipeline.fit(x_train_pd, y_train)

    # ── 7. Test-set evaluation ────────────────────────────────────────────────
    y_pred = fitted_pipeline.predict(x_test_pd)
    test_metrics = TestMetrics(
        mae=float(mean_absolute_error(y_test, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        r2=float(r2_score(y_test, y_pred)),
    )
    logger.info(
        "  Test — MAE=%.4f  RMSE=%.4f  R²=%.4f",
        test_metrics.mae,
        test_metrics.rmse,
        test_metrics.r2,
    )

    # ── 8. Feature names after one-hot expansion ──────────────────────────────
    onehot_names: list[str] = []
    if cat_cols:
        try:
            enc = (
                fitted_pipeline.named_steps["pre"]
                .named_transformers_["cat"]
                .named_steps["onehot"]
            )
            onehot_names = enc.get_feature_names_out(cat_cols).tolist()
        except (AttributeError, KeyError):
            logger.warning("Could not extract one-hot feature names.")
    feature_names = num_cols + onehot_names

    result = ModelRunResult(
        model_name=model_name,
        dataset_label=dataset_label,
        cv_metrics=cv_metrics,
        test_metrics=test_metrics,
        feature_names=feature_names,
        numeric_cols=num_cols,
        cat_cols=cat_cols,
    )

    return TrainingOutput(
        result=result,
        fitted_pipeline=fitted_pipeline,
        y_test=y_test,
        y_pred=y_pred,
    )


# ── Private helpers ────────────────────────────────────────────────────────────


def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Construct the ``ColumnTransformer`` for numeric and categorical columns."""
    transformers = []

    if num_cols:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, num_cols))

    if cat_cols:
        cat_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_transformer, cat_cols))

    if not transformers:
        raise ValueError("No numeric or categorical columns found — cannot build preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _align_test_schema(
    test_df: pl.DataFrame,
    feature_cols: list[str],
    train_schema_df: pl.DataFrame,
) -> pl.DataFrame:
    """Ensure the test DataFrame has exactly the columns in *feature_cols*.

    Columns present in training but absent in test are added as null columns
    with the same dtype as in training.  Extra columns are dropped.
    """
    aligned = pl.DataFrame()
    for col in feature_cols:
        if col in test_df.columns:
            aligned = aligned.with_columns(test_df[col])
        else:
            null_series = pl.Series(
                col, [None] * test_df.shape[0], dtype=train_schema_df[col].dtype
            )
            aligned = aligned.with_columns(null_series)
    return aligned.select(feature_cols)
