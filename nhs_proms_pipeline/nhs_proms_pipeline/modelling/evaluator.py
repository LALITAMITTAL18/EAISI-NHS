"""Model evaluator — clinical and statistical evaluation of trained pipelines.

Implements the five evaluation tiers from notebook 4.1-Clinical-Model-Evaluation.ipynb:
1. Bland-Altman limits of agreement
2. MCID-based confusion matrix + threshold optimisation
3. AUROC and Precision-Recall curves
4. Calibration by predicted score decile
5. Summary reporting

All tier functions accept raw numpy arrays and return Pydantic schemas so
they can be serialised and compared across training runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from nhs_proms_pipeline.schemas.results import BlandAltmanMetrics, MCIDClassificationMetrics
from nhs_proms_pipeline.utils.logging import get_logger

matplotlib.use("Agg")  # headless backend — safe for server/CI environments

logger = get_logger(__name__)


# ── Tier 1: Bland-Altman ─────────────────────────────────────────────────────


def bland_altman_stats(
    y_true: np.ndarray, y_pred: np.ndarray, mcid: float
) -> BlandAltmanMetrics:
    """Compute Bland-Altman limits of agreement for one model.

    Args:
        y_true: Array of actual health gain values.
        y_pred: Array of predicted health gain values.
        mcid:   Minimum Clinically Important Difference threshold.

    Returns:
        :class:`~nhs_proms_pipeline.schemas.results.BlandAltmanMetrics`.
    """
    diff = y_pred - y_true  # positive = model over-predicts
    bias = float(diff.mean())
    loa_sd = float(diff.std())
    loa_upper = bias + 1.96 * loa_sd
    loa_lower = bias - 1.96 * loa_sd
    return BlandAltmanMetrics(
        bias=round(bias, 4),
        loa_upper=round(loa_upper, 4),
        loa_lower=round(loa_lower, 4),
        loa_width=round(loa_upper - loa_lower, 4),
        within_mcid=(loa_upper <= mcid and loa_lower >= -mcid),
    )


def plot_bland_altman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mcid: float,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Save a Bland-Altman scatter plot to *save_path*.

    Args:
        y_true:    Actual outcomes.
        y_pred:    Predicted outcomes.
        mcid:      MCID threshold to draw as reference lines.
        title:     Chart title.
        save_path: If provided, the figure is saved here (PNG).
    """
    ba = bland_altman_stats(y_true, y_pred, mcid)
    mean_pair = (y_pred + y_true) / 2
    diff = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mean_pair, diff, alpha=0.35, s=18, color="steelblue", edgecolors="none")
    ax.axhline(ba.bias, color="crimson", linewidth=1.5, label=f"Bias={ba.bias:+.2f}")
    ax.axhline(ba.loa_upper, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"+1.96SD={ba.loa_upper:+.2f}")
    ax.axhline(ba.loa_lower, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"−1.96SD={ba.loa_lower:+.2f}")
    ax.axhline(mcid, color="green", linestyle=":", linewidth=1.0, alpha=0.6, label=f"MCID=+{mcid}")
    ax.axhline(-mcid, color="green", linestyle=":", linewidth=1.0, alpha=0.6, label=f"MCID=−{mcid}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Mean of Predicted & Actual health gain")
    ax.set_ylabel("Predicted − Actual (model error)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Bland-Altman plot saved: %s", save_path)
    plt.close(fig)


# ── Tier 2: MCID classification metrics ──────────────────────────────────────


def mcid_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mcid: float,
) -> MCIDClassificationMetrics:
    """Binarise predictions at *mcid* and compute classification metrics.

    A patient is predicted to benefit (positive class) if the predicted
    health gain ≥ MCID.

    Args:
        y_true: True health gain values.
        y_pred: Predicted health gain values.
        mcid:   Threshold (e.g. 5.0 for Knee, 6.0 for Hip).

    Returns:
        :class:`~nhs_proms_pipeline.schemas.results.MCIDClassificationMetrics`.
    """
    y_true_bin = (y_true >= mcid).astype(int)
    y_pred_bin = (y_pred >= mcid).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    f2 = float(fbeta_score(y_true_bin, y_pred_bin, beta=2, zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true_bin, y_pred))
        ap = float(average_precision_score(y_true_bin, y_pred))
    except ValueError:
        auroc, ap = 0.0, 0.0

    return MCIDClassificationMetrics(
        threshold=mcid,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        precision=round(prec, 4),
        recall=round(rec, 4),
        specificity=round(spec, 4),
        npv=round(npv, 4),
        f1=round(f1, 4),
        f2=round(f2, 4),
        auroc=round(auroc, 4),
        average_precision=round(ap, 4),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mcid: float,
    n_points: int = 400,
) -> dict[str, Any]:
    """Find the decision threshold that maximises F₂ score.

    Sweeps from ``y_pred.min()`` to ``y_pred.max()`` and returns the
    threshold at which F₂ is maximised.

    Args:
        y_true:   True health gain values.
        y_pred:   Predicted health gain values.
        mcid:     MCID threshold (used to define the positive class).
        n_points: Number of candidate thresholds to evaluate.

    Returns:
        Dict with keys ``optimal_threshold``, ``fn_saved``, ``fp_added``,
        and per-threshold sweep DataFrame.
    """
    y_true_bin = (y_true >= mcid).astype(int)
    thresholds = np.linspace(float(y_pred.min()), float(y_pred.max()), n_points)

    rows = []
    for t in thresholds:
        y_pred_bin = (y_pred >= t).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm.ravel()
        rec_t = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        prec_t = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        f2_t = float(fbeta_score(y_true_bin, y_pred_bin, beta=2, zero_division=0))
        rows.append(
            {
                "Threshold": t,
                "TP": tp_t, "TN": tn_t, "FP": fp_t, "FN": fn_t,
                "Recall": rec_t, "Precision": prec_t, "F2": f2_t,
            }
        )

    sweep_df = pd.DataFrame(rows)
    best_idx = sweep_df["F2"].idxmax()
    opt_row = sweep_df.loc[best_idx]
    mcid_row = sweep_df.loc[(sweep_df["Threshold"] - mcid).abs().idxmin()]

    return {
        "optimal_threshold": float(opt_row["Threshold"]),
        "optimal_f2": float(opt_row["F2"]),
        "fn_saved": int(mcid_row["FN"] - opt_row["FN"]),
        "fp_added": int(opt_row["FP"] - mcid_row["FP"]),
        "sweep_df": sweep_df,
    }


# ── Tier 3: Calibration ───────────────────────────────────────────────────────


def calibration_by_decile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare mean predicted vs mean actual health gain by predicted score decile.

    Args:
        y_true:  True health gain values.
        y_pred:  Predicted health gain values.
        n_bins:  Number of equal-frequency bins (deciles).

    Returns:
        DataFrame with columns ``Decile``, ``Mean Predicted``, ``Mean Actual``, ``Bias``.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["decile"] = pd.qcut(df["y_pred"], q=n_bins, labels=False, duplicates="drop")
    calibration = (
        df.groupby("decile")
        .agg(
            mean_predicted=("y_pred", "mean"),
            mean_actual=("y_true", "mean"),
            count=("y_true", "count"),
        )
        .reset_index()
    )
    calibration["bias"] = calibration["mean_predicted"] - calibration["mean_actual"]
    return calibration


def plot_calibration(
    calibration_df: pd.DataFrame,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Save a calibration scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        calibration_df["mean_predicted"],
        calibration_df["mean_actual"],
        "o-",
        color="steelblue",
        linewidth=1.5,
        markersize=7,
        label="Mean actual by decile",
    )
    min_val = min(
        calibration_df["mean_predicted"].min(),
        calibration_df["mean_actual"].min(),
    )
    max_val = max(
        calibration_df["mean_predicted"].max(),
        calibration_df["mean_actual"].max(),
    )
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted health gain (by decile)")
    ax.set_ylabel("Mean Actual health gain")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved: %s", save_path)
    plt.close(fig)


# ── Summary helper ────────────────────────────────────────────────────────────


def build_comparison_table(
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Flatten the nested ``all_results`` dict into a sortable comparison table.

    Args:
        results: Nested dict ``{dataset_label: {model_name: result_dict}}``.

    Returns:
        DataFrame sorted by ``Test RMSE`` ascending.
    """
    rows = []
    for ds_label, models_dict in results.items():
        for mdl_name, res in models_dict.items():
            rows.append(
                {
                    "Dataset": ds_label,
                    "Model": mdl_name,
                    "CV RMSE": round(res["test_metrics"].rmse, 4),
                    "CV R²": round(res["cv_metrics"].r2, 4),
                    "Test MAE": round(res["test_metrics"].mae, 4),
                    "Test RMSE": round(res["test_metrics"].rmse, 4),
                    "Test R²": round(res["test_metrics"].r2, 4),
                }
            )
    return pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)
