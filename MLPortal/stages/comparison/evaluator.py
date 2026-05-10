"""Pure functions for model evaluation and comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from stages.comparison.models import (
    BlandAltmanResult,
    CalibrationDecile,
    ComparisonResult,
    MetricRow,
    SubgroupResult,
)
from stages.modelling.models import TrainResult


def _fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    denom = beta**2 * precision + recall
    return (1 + beta**2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    outcome_threshold: float | None = None,
) -> MetricRow:
    """Compute regression metrics (and optional outcome-threshold binary metrics)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    row = MetricRow(
        model_name=model_name,
        task="regression",
        test_rmse=round(rmse, 4),
        test_mae=round(mae, 4),
        test_r2=round(r2, 4),
    )

    if outcome_threshold is not None:
        y_bin_true = (y_true > outcome_threshold).astype(int)
        y_bin_pred = (pd.Series(y_pred) > outcome_threshold).astype(int)
        prec = float(precision_score(y_bin_true, y_bin_pred, zero_division=0))
        rec = float(recall_score(y_bin_true, y_bin_pred, zero_division=0))
        row = row.model_copy(
            update={
                "threshold_precision": round(prec, 4),
                "threshold_recall": round(rec, 4),
                "threshold_f2": round(_fbeta(prec, rec, beta=2.0), 4),
            }
        )
    return row


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    model_name: str,
    task: str = "classification",
) -> MetricRow:
    """Compute classification metrics."""
    average = "macro" if task == "ordinal" else "binary"
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    f2 = _fbeta(prec, rec, beta=2.0)

    roc = pr = None
    if y_prob is not None:
        try:
            multi = "ovr" if task == "ordinal" else "raise"
            roc = float(roc_auc_score(y_true, y_prob, multi_class=multi if task == "ordinal" else None))
            if task != "ordinal":
                pr = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    return MetricRow(
        model_name=model_name,
        task=task,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        f2=round(f2, 4),
        roc_auc=round(roc, 4) if roc else None,
        pr_auc=round(pr, 4) if pr else None,
    )


def compare_models(
    results: list[TrainResult],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    outcome_threshold: float | None = None,
    dataset: str = "default",
) -> ComparisonResult:
    """Evaluate all trained models on the held-out test set."""
    rows: list[MetricRow] = []

    for result in results:
        if result.pipeline is None:
            continue
        y_pred = result.pipeline.predict(X_test)

        if task == "regression":
            row = evaluate_regression(
                y_test, y_pred, result.model_name, outcome_threshold
            )
        else:
            y_prob = None
            if hasattr(result.pipeline, "predict_proba"):
                try:
                    prob = result.pipeline.predict_proba(X_test)
                    y_prob = prob[:, 1] if prob.shape[1] == 2 else prob
                except Exception:
                    pass
            row = evaluate_classification(y_test, y_pred, y_prob, result.model_name, task)

        row = row.model_copy(update={"dataset": dataset})
        result.test_metrics = {k: v for k, v in row.model_dump().items() if v is not None}
        rows.append(row)

    best_name, best_metric, best_value = _pick_best(rows, task)

    return ComparisonResult(
        task=task,
        rows=rows,
        best_model_name=best_name,
        best_metric_name=best_metric,
        best_metric_value=best_value,
    )


def _pick_best(rows: list[MetricRow], task: str):
    if not rows:
        return None, None, None
    if task == "regression":
        key, direction = "test_rmse", min
    elif task == "ordinal":
        key, direction = "ordinal_mae", min
    else:
        key, direction = "f2", max

    valid = [(r, getattr(r, key)) for r in rows if getattr(r, key) is not None]
    if not valid:
        return None, None, None
    best_row, best_val = direction(valid, key=lambda x: x[1])
    return best_row.model_name, key, best_val


def bland_altman_stats(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> BlandAltmanResult:
    """Compute Bland-Altman limits of agreement."""
    diff = np.array(y_pred) - np.array(y_true)
    bias = float(diff.mean())
    sd = float(diff.std())
    pct = float(((diff.abs() if hasattr(diff, "abs") else np.abs(diff)) <= 1.96 * sd).mean() * 100)
    return BlandAltmanResult(
        model_name=model_name,
        bias=round(bias, 4),
        upper_loa=round(bias + 1.96 * sd, 4),
        lower_loa=round(bias - 1.96 * sd, 4),
        pct_within_loa=round(pct, 2),
    )


def calibration_by_decile(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> list[CalibrationDecile]:
    """Compute mean predicted vs mean actual for each decile of predicted values."""
    df = pd.DataFrame({"true": y_true.values, "pred": y_pred})
    df["decile"] = pd.qcut(df["pred"], q=10, labels=False, duplicates="drop")
    rows: list[CalibrationDecile] = []
    for dec, grp in df.groupby("decile"):
        rows.append(
            CalibrationDecile(
                model_name=model_name,
                decile=int(dec) + 1,
                mean_predicted=round(float(grp["pred"].mean()), 4),
                mean_actual=round(float(grp["true"].mean()), 4),
                n_samples=len(grp),
            )
        )
    return rows


def subgroup_eval(
    y_true: pd.Series,
    y_pred: np.ndarray,
    group_series: pd.Series,
    model_name: str,
    metric_name: str = "rmse",
) -> list[SubgroupResult]:
    """Compute per-group performance metrics."""
    results: list[SubgroupResult] = []
    group_col = group_series.name or "group"
    for val, idx in group_series.groupby(group_series).groups.items():
        yt = y_true.iloc[idx]
        yp = np.array(y_pred)[idx]
        if metric_name == "rmse":
            value = float(np.sqrt(mean_squared_error(yt, yp)))
        elif metric_name == "mae":
            value = float(mean_absolute_error(yt, yp))
        else:
            value = float(r2_score(yt, yp))
        results.append(
            SubgroupResult(
                model_name=model_name,
                group_column=str(group_col),
                group_value=str(val),
                metric_name=metric_name,
                metric_value=round(value, 4),
                n_samples=len(idx),
            )
        )
    return results
