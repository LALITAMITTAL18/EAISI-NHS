"""Data preparation — Step 2.1.

Implements the full data preparation pipeline:
  * Listwise deletion for critical columns
  * Derived score calculation (Oxford score, EQ-5D index)
  * Train / test split (80 / 20, reproducible)
  * Post-split imputation (mode and constant fills)
  * Outcome variable creation (health_gain)

All imputation values are fit on the training set and applied identically
to the test set to prevent data leakage.

Corresponds to notebook: 2.1-data-preparation-Manual.ipynb
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nhs_proms_pipeline.config import (
    CONSTANT_FILL_COLS,
    EQ5D_PRE_OP_COLS,
    MODE_IMPUTE_COLS,
    VAS_DROP_COLS,
    PipelineSettings,
    ProcedureConfig,
)
from nhs_proms_pipeline.features.eq5d import calculate_eq5d_index
from nhs_proms_pipeline.utils.io import read_parquet, write_parquet
from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_INPUT_FILENAME = "2.0-preprocessing.parquet"
_OUTPUT_TRAIN = "2.1-train.parquet"
_OUTPUT_TEST = "2.1-test.parquet"


@dataclass
class SplitDataset:
    """Container for the train/test split after preparation."""

    train: pl.DataFrame
    test: pl.DataFrame
    train_path: Path
    test_path: Path
    imputation_values: dict[str, float | int]


def prepare_data(settings: PipelineSettings) -> SplitDataset:
    """Run the full data preparation pipeline and persist train/test sets.

    Args:
        settings: Resolved pipeline settings.

    Returns:
        :class:`SplitDataset` with train/test DataFrames and their paths.
    """
    input_path = settings.interim_path(_INPUT_FILENAME)
    df = read_parquet(input_path)
    logger.info("Data preparation — input shape: %s", df.shape)

    proc_cfg = settings.get_procedure_config()

    # ── Steps applied BEFORE the train/test split ─────────────────────────────
    df = _step_verify_and_fill_indicators(df)
    df = _step_listwise_delete_pre_op_dims(df, proc_cfg)
    df = _step_calculate_pre_op_score(df, proc_cfg)
    df = _step_calculate_eq5d_index(df)
    df = _step_listwise_delete_eq5d(df)
    df = _step_listwise_delete_symptom_period(df)
    df = _step_remove_post_op_cols(df, proc_cfg)
    df = _step_listwise_delete_demographics(df)
    df = _step_listwise_delete_post_op_score(df, proc_cfg)

    logger.info("Shape after pre-split cleaning: %s", df.shape)

    # ── Train / test split ────────────────────────────────────────────────────
    train_df, test_df = _split(df, settings)

    # ── Steps applied AFTER the split (imputation fit on train only) ──────────
    train_df, test_df, imputation_values = _step_impute(train_df, test_df)
    train_df, test_df = _step_create_outcome(train_df, test_df, proc_cfg)

    # ── Persist ───────────────────────────────────────────────────────────────
    train_path = settings.interim_path(_OUTPUT_TRAIN)
    test_path = settings.interim_path(_OUTPUT_TEST)
    write_parquet(train_df, train_path)
    write_parquet(test_df, test_path)

    logger.info("Train: %s → %s", train_df.shape, train_path)
    logger.info("Test:  %s → %s", test_df.shape, test_path)

    return SplitDataset(
        train=train_df,
        test=test_df,
        train_path=train_path,
        test_path=test_path,
        imputation_values=imputation_values,
    )


# ── Private step functions ────────────────────────────────────────────────────


def _step_verify_and_fill_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Verify comorbidity indicators — nulls at this point are already 0.

    The previous step (preprocessor) replaces all ``9``s with ``0`` for
    binary indicator columns.  This step logs any residual nulls to make
    the pipeline auditable.
    """
    from nhs_proms_pipeline.config import COMORBIDITY_INDICATOR_COLS

    present = [c for c in COMORBIDITY_INDICATOR_COLS if c in df.columns]
    null_check = df.select(present).null_count()
    total_nulls = sum(null_check[c][0] for c in present)
    if total_nulls:
        logger.warning(
            "Residual nulls in comorbidity indicators: %d — check preprocessing step.", total_nulls
        )
    else:
        logger.debug("All comorbidity indicators clean (zero nulls).")
    return df


def _step_listwise_delete_pre_op_dims(
    df: pl.DataFrame, proc_cfg: ProcedureConfig
) -> pl.DataFrame:
    """Delete rows with any missing Oxford Knee/Hip Score pre-op dimension."""
    pre_op_dim_cols = [
        c
        for c in df.columns
        if c.startswith(proc_cfg.pre_op_q_prefix)
        and c != proc_cfg.pre_op_score_col
    ]
    if not pre_op_dim_cols:
        logger.warning("No pre-op Q dimension columns found for prefix '%s'.", proc_cfg.pre_op_q_prefix)
        return df

    initial_rows = df.shape[0]
    df = df.drop_nulls(subset=pre_op_dim_cols)
    logger.info(
        "Listwise deletion (pre-op dims): removed %d rows.", initial_rows - df.shape[0]
    )
    return df


def _step_calculate_pre_op_score(
    df: pl.DataFrame, proc_cfg: ProcedureConfig
) -> pl.DataFrame:
    """Fill missing Oxford Knee/Hip Score by summing the 12 dimension columns."""
    score_col = proc_cfg.pre_op_score_col
    dim_cols = [
        c
        for c in df.columns
        if c.startswith(proc_cfg.pre_op_q_prefix) and c != score_col
    ]
    if len(dim_cols) != 12:
        logger.warning(
            "Expected 12 pre-op Q dimension columns, found %d.  Skipping score calculation.",
            len(dim_cols),
        )
        return df

    if score_col not in df.columns:
        logger.warning("Score column '%s' not found — skipping.", score_col)
        return df

    df = df.with_columns(
        pl.when(pl.col(score_col).is_null())
        .then(pl.sum_horizontal([pl.col(c) for c in dim_cols]))
        .otherwise(pl.col(score_col))
        .alias(score_col)
    )
    remaining_nulls = df[score_col].null_count()
    logger.debug("Pre-op score nulls after calculation: %d", remaining_nulls)
    return df


def _step_calculate_eq5d_index(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing EQ-5D index and profile from the five dimension columns."""
    # Build profile string from dimensions if missing
    if "Pre-Op Q EQ5D Index Profile" in df.columns:
        present_eq5d = [c for c in EQ5D_PRE_OP_COLS if c in df.columns]
        if present_eq5d:
            df = df.with_columns(
                pl.when(pl.col("Pre-Op Q EQ5D Index Profile").is_null())
                .then(
                    pl.concat_str([pl.col(c).cast(pl.Utf8) for c in present_eq5d], separator="")
                )
                .otherwise(pl.col("Pre-Op Q EQ5D Index Profile"))
                .alias("Pre-Op Q EQ5D Index Profile")
            )

    # Calculate index from profile
    if "Pre-Op Q EQ5D Index" in df.columns and "Pre-Op Q EQ5D Index Profile" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("Pre-Op Q EQ5D Index").is_null())
            .then(
                pl.col("Pre-Op Q EQ5D Index Profile").map_elements(
                    calculate_eq5d_index, return_dtype=pl.Float64
                )
            )
            .otherwise(pl.col("Pre-Op Q EQ5D Index"))
            .alias("Pre-Op Q EQ5D Index")
        )
        logger.debug(
            "EQ-5D index nulls remaining: %d", df["Pre-Op Q EQ5D Index"].null_count()
        )
    return df


def _step_listwise_delete_eq5d(df: pl.DataFrame) -> pl.DataFrame:
    """Delete rows with any missing EQ-5D dimension (valid range 1–3)."""
    present_cols = [c for c in EQ5D_PRE_OP_COLS if c in df.columns]
    if not present_cols:
        return df
    initial_rows = df.shape[0]
    df = df.drop_nulls(subset=present_cols)
    logger.info(
        "Listwise deletion (EQ-5D dims): removed %d rows.", initial_rows - df.shape[0]
    )
    return df


def _step_listwise_delete_symptom_period(df: pl.DataFrame) -> pl.DataFrame:
    """Delete rows missing 'Pre-Op Q Symptom Period' (critical predictor)."""
    col = "Pre-Op Q Symptom Period"
    if col not in df.columns:
        return df
    initial_rows = df.shape[0]
    df = df.drop_nulls(subset=[col])
    logger.info(
        "Listwise deletion (symptom period): removed %d rows.", initial_rows - df.shape[0]
    )
    return df


def _step_remove_post_op_cols(
    df: pl.DataFrame, proc_cfg: ProcedureConfig
) -> pl.DataFrame:
    """Remove Post-Op question columns that would cause target leakage.

    Retains:
    - ``Post-Op Q EQ5D Index Profile``
    - ``Post-Op Q EQ5D Index``
    - ``Post-Op Q EQ VAS``
    - Procedure-specific post-op score column (used to compute health_gain)
    """
    removed = [
        c
        for c in df.columns
        if ("Post-Op" in c and c not in proc_cfg.keep_post_op_cols)
        or "Predicted" in c
        or c == "CSVYear"
    ]
    # Also drop VAS columns that are not useful features
    vas_present = [c for c in VAS_DROP_COLS if c in df.columns and c not in removed]
    removed.extend(vas_present)

    existing = [c for c in removed if c in df.columns]
    if existing:
        df = df.drop(existing)
        logger.debug("Removed %d Post-Op/VAS/Predicted columns.", len(existing))
    return df


def _step_listwise_delete_demographics(df: pl.DataFrame) -> pl.DataFrame:
    """Delete rows where Age Band or Gender is missing."""
    subset = [c for c in ["Age Band", "Gender"] if c in df.columns]
    if not subset:
        return df
    initial_rows = df.shape[0]
    df = df.drop_nulls(subset=subset)
    logger.info(
        "Listwise deletion (demographics): removed %d rows.", initial_rows - df.shape[0]
    )
    return df


def _step_listwise_delete_post_op_score(
    df: pl.DataFrame, proc_cfg: ProcedureConfig
) -> pl.DataFrame:
    """Delete rows where the post-op score is null (non-responders).

    Without a valid post-op score, ``health_gain`` cannot be computed,
    so these rows cannot be used in training or evaluation.
    """
    score_col = proc_cfg.post_op_score_col
    if score_col not in df.columns:
        logger.warning("Post-op score column '%s' not found.", score_col)
        return df
    initial_rows = df.shape[0]
    df = df.drop_nulls(subset=[score_col])
    logger.info(
        "Listwise deletion (post-op score): removed %d rows.", initial_rows - df.shape[0]
    )
    return df


def _split(
    df: pl.DataFrame, settings: PipelineSettings
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Shuffle and split the dataset into train and test sets.

    The split is applied after all listwise deletions and derived column
    calculations but before any imputation.
    """
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=settings.random_seed)
    split_n = int(len(df_shuffled) * settings.train_test_ratio)
    train_df = df_shuffled[:split_n]
    test_df = df_shuffled[split_n:]
    logger.info(
        "Train/test split: %d train (%.0f%%), %d test (%.0f%%)",
        len(train_df),
        100 * settings.train_test_ratio,
        len(test_df),
        100 * (1 - settings.train_test_ratio),
    )
    return train_df, test_df


def _step_impute(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float | int]]:
    """Impute remaining nulls — values learned from the training set only.

    Columns imputed:
    - ``Pre-Op Q Assisted`` → mode (most frequent value from train)
    - ``Pre-Op Q Previous Surgery`` → mode
    - ``Pre-Op Q Living Arrangements`` → constant 9 (Unknown)
    - ``Pre-Op Q Disability`` → constant 9 (Not Disclosed)
    """
    imputation_values: dict[str, float | int] = {}

    # Mode-imputation: fit on train, apply to both
    mode_exprs_train = []
    mode_exprs_test = []
    for col in MODE_IMPUTE_COLS:
        if col not in train_df.columns:
            continue
        mode_val = int(train_df[col].drop_nulls().mode()[0])
        imputation_values[col] = mode_val
        logger.debug("Mode for '%s' (from train): %d", col, mode_val)
        expr = pl.col(col).fill_null(mode_val)
        mode_exprs_train.append(expr)
        mode_exprs_test.append(expr)

    if mode_exprs_train:
        train_df = train_df.with_columns(mode_exprs_train)
        test_df = test_df.with_columns(mode_exprs_test)

    # Constant fills
    const_exprs = []
    for col, fill_value in CONSTANT_FILL_COLS.items():
        if col not in train_df.columns:
            continue
        imputation_values[col] = fill_value
        const_exprs.append(pl.col(col).fill_null(fill_value))

    if const_exprs:
        train_df = train_df.with_columns(const_exprs)
        test_df = test_df.with_columns(const_exprs)

    logger.info("Imputation complete. Remaining nulls (train): %s", train_df.null_count().sum_horizontal().to_list())
    return train_df, test_df, imputation_values


def _step_create_outcome(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    proc_cfg: ProcedureConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add ``health_gain`` and drop the post-op score column.

    ``health_gain = Post-Op Q Score − Pre-Op Q Score``

    After computing the outcome, the post-op score column is dropped to
    prevent target leakage.
    """
    pre_col = proc_cfg.pre_op_score_col
    post_col = proc_cfg.post_op_score_col

    health_gain_expr = (pl.col(post_col) - pl.col(pre_col)).alias("health_gain")
    train_df = train_df.with_columns(health_gain_expr)
    test_df = test_df.with_columns(health_gain_expr)

    # Drop post-op score — it directly determines health_gain (target leakage)
    for col in [post_col]:
        if col in train_df.columns:
            train_df = train_df.drop(col)
            test_df = test_df.drop(col)
            logger.debug("Dropped '%s' (target leakage prevention).", col)

    logger.info(
        "health_gain: train mean=%.2f, test mean=%.2f",
        float(train_df["health_gain"].mean() or 0),
        float(test_df["health_gain"].mean() or 0),
    )
    return train_df, test_df
