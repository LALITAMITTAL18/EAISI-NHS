"""Column-level cleaning for the Preparation stage.

Provides:
- Per-column null-handling rules: listwise deletion, mean / median / mode,
  constant fill, or a custom Python expression evaluated per-row.
- Derived-column steps: arbitrary Python code executed with ``df`` in scope,
  with optional drop of helper columns used for the computation.
- Application helpers that respect the train-only fitting convention so no
  leakage occurs across the split boundary.
"""

from __future__ import annotations

import traceback
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────


class NullHandlingRule(BaseModel):
    """Per-column strategy for handling missing values."""

    strategy: Literal[
        "none",             # leave untouched
        "listwise_delete",  # drop rows where this column is null (before split)
        "mean",             # fill nulls with train mean (fit on train, apply to both)
        "median",           # fill nulls with train median
        "mode",             # fill nulls with train mode (most frequent value)
        "constant",         # fill nulls with a user-supplied constant value
        "python_expr",      # fill nulls via a per-row Python expression on other columns
    ] = "none"

    constant_value: str = ""
    # Expression used when strategy == "python_expr".
    # The expression is evaluated with `row` as a dict of the current row values.
    # Example:  "row['col_a'] + row['col_b']"
    python_expr: str = ""


class DerivedStep(BaseModel):
    """Custom Python code step that computes or modifies columns.

    The code block is executed with ``df`` (a pandas DataFrame) available in
    its local namespace.  The code must assign back to ``df``::

        df['health_gain'] = df['Post-Op Score'] - df['Pre-Op Score']

    After execution, any columns listed in ``drop_after`` are removed.
    """

    name: str = "Unnamed step"
    code: str = ""
    drop_after: list[str] = Field(default_factory=list)
    apply_when: Literal["before_split", "after_split"] = "before_split"


# ─────────────────────────────────────────────────────────────────────────────
# Pure transformation helpers
# ─────────────────────────────────────────────────────────────────────────────


def apply_column_drops(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop columns that exist in the DataFrame (silently skip missing ones)."""
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


def apply_listwise_deletions(
    df: pd.DataFrame,
    rules: dict[str, NullHandlingRule],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows where a column marked ``listwise_delete`` is null.

    Returns the cleaned DataFrame and a dict of {column: rows_removed}.
    """
    removed: dict[str, int] = {}
    for col, rule in rules.items():
        if rule.strategy != "listwise_delete":
            continue
        if col not in df.columns:
            continue
        before = len(df)
        df = df.dropna(subset=[col])
        removed[col] = before - len(df)
    return df.reset_index(drop=True), removed


def apply_imputation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rules: dict[str, NullHandlingRule],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit imputation fill-values on *train only*, then apply to both sets.

    Returns ``(train, test, fill_values)`` where ``fill_values`` records the
    exact value used for each column (useful for audit / logging).
    """
    fill_values: dict[str, Any] = {}
    train = train.copy()
    test = test.copy()

    for col, rule in rules.items():
        if col not in train.columns:
            continue
        if rule.strategy in ("none", "listwise_delete", "python_expr"):
            continue

        if rule.strategy == "mean":
            val: Any = train[col].mean()
        elif rule.strategy == "median":
            val = train[col].median()
        elif rule.strategy == "mode":
            modes = train[col].mode()
            val = modes.iloc[0] if not modes.empty else np.nan
        elif rule.strategy == "constant":
            raw = rule.constant_value
            try:
                val = float(raw) if "." in str(raw) else int(raw)
            except (ValueError, TypeError):
                val = raw
        else:
            continue

        fill_values[col] = val
        train[col] = train[col].fillna(val)
        test[col] = test[col].fillna(val)

    return train, test, fill_values


def apply_python_expr_rules(
    df: pd.DataFrame,
    rules: dict[str, NullHandlingRule],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fill null values using per-row Python expressions.

    The expression is evaluated with a ``row`` dict of the current row values.
    Only null cells are updated — non-null values are unchanged.

    Returns ``(df, errors)`` where ``errors`` is ``{col: traceback_str}``.
    """
    errors: dict[str, str] = {}
    df = df.copy()

    for col, rule in rules.items():
        if rule.strategy != "python_expr":
            continue
        if col not in df.columns or not rule.python_expr.strip():
            continue

        null_mask = df[col].isna()
        if not null_mask.any():
            continue

        expr = rule.python_expr.strip()
        try:
            filled = df[null_mask].apply(
                lambda row: eval(expr, {"__builtins__": {}}, {"row": row.to_dict(), "np": np}),  # noqa: S307
                axis=1,
            )
            df.loc[null_mask, col] = filled
        except Exception:
            errors[col] = traceback.format_exc()

    return df, errors


def apply_derived_step(
    df: pd.DataFrame,
    step: DerivedStep,
) -> tuple[pd.DataFrame, str | None]:
    """Execute a ``DerivedStep`` code block against *df*.

    The code block runs with ``df``, ``pd``, and ``np`` in scope.  The code
    should assign back to ``df``::

        df['health_gain'] = df['Post-Op Score'] - df['Pre-Op Score']

    After execution, columns listed in ``step.drop_after`` are removed.

    Returns ``(df, error_message_or_None)``.
    """
    if not step.code.strip():
        return df, None

    local_ns: dict[str, Any] = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(step.code, {}, local_ns)  # noqa: S102
        result = local_ns.get("df", df)
        if not isinstance(result, pd.DataFrame):
            return df, "Code did not produce a DataFrame assigned to `df`."
        result = apply_column_drops(result, step.drop_after)
        return result.reset_index(drop=True), None
    except Exception:
        return df, traceback.format_exc()


def apply_derived_steps_timed(
    df: pd.DataFrame,
    steps: list[DerivedStep],
    when: Literal["before_split", "after_split"],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply all DerivedSteps with the given ``apply_when`` value.

    Returns ``(df, list_of_error_messages)``.
    """
    errors: list[str] = []
    for step in steps:
        if step.apply_when != when:
            continue
        df, err = apply_derived_step(df, step)
        if err:
            errors.append(f"[{step.name}] {err}")
    return df, errors


def summarise_null_rules(
    df: pd.DataFrame,
    rules: dict[str, NullHandlingRule],
) -> pd.DataFrame:
    """Build a summary table of null counts and configured strategies."""
    rows = []
    for col in df.columns:
        null_ct = int(df[col].isna().sum())
        rule = rules.get(col, NullHandlingRule())
        rows.append(
            {
                "Column": col,
                "Null Count": null_ct,
                "Null %": round(null_ct / max(len(df), 1) * 100, 1),
                "Strategy": rule.strategy,
                "Fill Value / Expr": rule.constant_value or rule.python_expr or "—",
            }
        )
    return pd.DataFrame(rows)
