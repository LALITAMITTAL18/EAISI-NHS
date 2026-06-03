"""Plots for the Model Comparison stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.viz import PALETTE, add_reference_line, annotate_message, base_fig, highlight_at
from stages.comparison.models import (
    BlandAltmanResult,
    CalibrationDecile,
    ComparisonResult,
    SubgroupResult,
)


def metric_comparison_bar(
    comparison: ComparisonResult,
    metric: str,
    lower_is_better: bool = False,
) -> go.Figure:
    """Horizontal bar of a chosen metric across models; best model highlighted."""
    rows = [r for r in comparison.rows if getattr(r, metric) is not None]
    if not rows:
        return base_fig(title=f"No data for metric: {metric}")

    if lower_is_better:
        best_idx = min(range(len(rows)), key=lambda i: getattr(rows[i], metric))
    else:
        best_idx = max(range(len(rows)), key=lambda i: getattr(rows[i], metric))

    names = [r.model_name for r in rows]
    values = [getattr(r, metric) for r in rows]
    colors = highlight_at(len(names), best_idx)

    fig = base_fig(
        title=f"Model comparison — {metric}",
        xlabel=metric,
        ylabel="",
        height=max(300, len(names) * 32),
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    direction = "lowest" if lower_is_better else "highest"
    subtitle = f"<i><span style='color:{PALETTE.highlight};font-size:12px'>'{names[best_idx]}' has the {direction} {metric}</span></i>"
    fig.update_layout(
        title=dict(
            text=f"<b>Model comparison — {metric}</b><br>{subtitle}",
        ),
        margin=dict(t=75, r=90),
        xaxis=dict(range=[0, max(values) * 1.18]),
    )
    return fig


def roc_overlay(
    roc_data: dict[str, tuple[list, list]],
    best_model: str | None = None,
) -> go.Figure:
    """Overlaid ROC curves — best model highlighted, others gray."""
    fig = base_fig(
        title="ROC curves (all models)",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    for model_name, (fpr, tpr) in roc_data.items():
        is_best = model_name == best_model
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(
                    color=PALETTE.highlight if is_best else PALETTE.gray,
                    width=2.5 if is_best else 1.0,
                ),
                name=model_name,
                hovertemplate=f"{model_name}<br>FPR: %{{x:.3f}} TPR: %{{y:.3f}}<extra></extra>",
            )
        )
    add_reference_line(fig, 0, label="")
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color=PALETTE.gray, dash="dot", width=1),
    )
    return fig


def bland_altman_plot(results: list[BlandAltmanResult], best_model: str | None = None) -> go.Figure:
    """Bland-Altman limits of agreement for all models.

    Best model bars highlighted; others gray.
    """
    names = [r.model_name for r in results]
    biases = [r.bias for r in results]
    colors = [PALETTE.highlight if n == best_model else PALETTE.gray for n in names]

    fig = base_fig(
        title="Bland-Altman bias per model",
        xlabel="",
        ylabel="Bias (mean difference)",
        height=max(300, len(names) * 32),
    )
    fig.add_trace(
        go.Bar(
            x=names,
            y=biases,
            marker_color=colors,
            text=[f"{b:+.3f}" for b in biases],
            textposition="outside",
            hovertemplate="%{x} bias: %{y:+.3f}<extra></extra>",
        )
    )
    add_reference_line(fig, 0, label="Zero bias")
    if best_model:
        best_res = next(r for r in results if r.model_name == best_model)
        annotate_message(
            fig,
            f"'{best_model}' bias = {best_res.bias:+.3f} (LoA: {best_res.lower_loa:.2f} to {best_res.upper_loa:.2f})",
        )
    return fig


def calibration_plot(deciles: list[CalibrationDecile], best_model: str) -> go.Figure:
    """Calibration plot (mean predicted vs mean actual by decile) for best model."""
    model_dec = [d for d in deciles if d.model_name == best_model]
    if not model_dec:
        return base_fig(title="No calibration data")

    predicted = [d.mean_predicted for d in model_dec]
    actual = [d.mean_actual for d in model_dec]
    sizes = [d.n_samples for d in model_dec]
    max_size = max(sizes)

    fig = base_fig(
        title=f"Calibration by decile — {best_model}",
        xlabel="Mean predicted",
        ylabel="Mean actual",
    )
    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=actual,
            mode="markers",
            marker=dict(
                color=PALETTE.highlight,
                size=[12 * s / max_size + 6 for s in sizes],
                opacity=0.8,
            ),
            hovertemplate="Decile %{pointNumber+1}<br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
        )
    )
    lo = min(min(predicted), min(actual))
    hi = max(max(predicted), max(actual))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color=PALETTE.gray, dash="dash"),
            name="Perfect calibration",
        )
    )
    annotate_message(fig, "Bubble size ∝ n samples in decile; ideal = diagonal line")
    return fig


def equity_bar(
    subgroup_results: list[SubgroupResult],
    model_name: str,
    metric_name: str = "rmse",
    acceptable_gap: float | None = None,
) -> go.Figure:
    """Bar of per-subgroup metric; groups with large gaps from median are highlighted."""
    data = [r for r in subgroup_results if r.model_name == model_name and r.metric_name == metric_name]
    if not data:
        return base_fig(title="No subgroup data")

    groups = [r.group_value for r in data]
    values = [r.metric_value for r in data]
    median_val = float(np.median(values))
    gap = acceptable_gap or (float(np.std(values)) * 1.5)
    colors = [PALETTE.highlight if abs(v - median_val) > gap else PALETTE.gray for v in values]

    fig = base_fig(
        title=f"Equity analysis — {metric_name} by subgroup ({model_name})",
        xlabel=metric_name,
        ylabel="",
        height=max(300, len(groups) * 28),
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=groups,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        )
    )
    add_reference_line(fig, median_val, orientation="v", label=f"Median = {median_val:.3f}")
    highlighted = sum(abs(v - median_val) > gap for v in values)
    if highlighted:
        annotate_message(fig, f"{highlighted} subgroup(s) show notable performance gaps")
    return fig


def threshold_explorer_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: float,
    model_name: str,
) -> go.Figure:
    """Interactive 4-panel classification threshold explorer.

    Panels:
    - Left:         KDE density curves + jittered scatter for each class; vertical
                    threshold line; TP / FN / FP / TN annotations.
    - Middle top:   TP and FN counts as a function of threshold k; current k marked.
    - Middle bottom:FP and TN counts as a function of threshold k; current k marked.
    - Right top:    ROC curve with current k highlighted; Recall / FPR / AUC readout.
    - Right bottom: Precision-Recall curve with current k highlighted; metrics readout.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    pos_probs = y_prob[pos_mask]
    neg_probs = y_prob[neg_mask]

    # ── Confusion values at current k ────────────────────────────────────────
    y_pred_k = (y_prob >= k).astype(int)
    TP = int(((y_pred_k == 1) & pos_mask).sum())
    FN = int(((y_pred_k == 0) & pos_mask).sum())
    FP = int(((y_pred_k == 1) & neg_mask).sum())
    TN = int(((y_pred_k == 0) & neg_mask).sum())

    recall_k = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr_k = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    prec_k = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # ── Threshold-sweep curves ────────────────────────────────────────────────
    thresholds = np.linspace(0.0, 1.0, 200)
    tp_arr, fn_arr, fp_arr, tn_arr = [], [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp_arr.append(int(((yp == 1) & pos_mask).sum()))
        fn_arr.append(int(((yp == 0) & pos_mask).sum()))
        fp_arr.append(int(((yp == 1) & neg_mask).sum()))
        tn_arr.append(int(((yp == 0) & neg_mask).sum()))

    # ── ROC & PR ──────────────────────────────────────────────────────────────
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr_arr, tpr_arr)
    roc_idx = int(np.argmin(np.abs(roc_thresh - k)))
    fpr_at_k = float(fpr_arr[roc_idx])
    tpr_at_k = float(tpr_arr[roc_idx])

    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_arr, prec_arr)
    pr_idx = int(np.argmin(np.abs(pr_thresh - k))) if len(pr_thresh) > 0 else 0
    prec_at_k = float(prec_arr[pr_idx])
    rec_at_k = float(rec_arr[pr_idx])

    # ── KDE ───────────────────────────────────────────────────────────────────
    x_grid = np.linspace(0.0, 1.0, 300)
    kde_pos = gaussian_kde(pos_probs)(x_grid) if len(pos_probs) > 2 else np.zeros(300)
    kde_neg = gaussian_kde(neg_probs)(x_grid) if len(neg_probs) > 2 else np.zeros(300)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"rowspan": 2}, {}, {}],
            [None,           {}, {}],
        ],
        subplot_titles=[
            model_name,
            "Cases (TP / FN) vs threshold",
            "ROC",
            None,
            "Non-cases (FP / TN) vs threshold",
            "Precision / Recall",
        ],
        column_widths=[0.38, 0.30, 0.32],
        row_heights=[0.50, 0.50],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # ── Panel 1: KDE + scatter ────────────────────────────────────────────────
    # Non-case density (orange)
    fig.add_trace(go.Scatter(
        x=x_grid, y=kde_neg,
        fill="tozeroy",
        fillcolor="rgba(255,165,0,0.25)",
        line=dict(color="rgba(255,165,0,0.7)", width=1),
        name="Non-case density",
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)
    # Case density (teal)
    fig.add_trace(go.Scatter(
        x=x_grid, y=kde_pos,
        fill="tozeroy",
        fillcolor="rgba(42,157,143,0.25)",
        line=dict(color="rgba(42,157,143,0.8)", width=1.5),
        name="Case density",
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # Jittered scatter — cases (y ≈ 1)
    rng = np.random.default_rng(42)
    case_y = 1.0 + rng.uniform(-0.03, 0.03, len(pos_probs))
    nc_y = rng.uniform(-0.03, 0.03, len(neg_probs))
    fig.add_trace(go.Scatter(
        x=pos_probs, y=case_y,
        mode="markers",
        marker=dict(
            color=[PALETTE.highlight if p >= k else PALETTE.gray for p in pos_probs],
            size=5, opacity=0.5,
        ),
        name="Cases",
        showlegend=False,
        hovertemplate="p = %{x:.3f}<extra>Case</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=neg_probs, y=nc_y,
        mode="markers",
        marker=dict(
            color=[PALETTE.highlight if p >= k else PALETTE.gray for p in neg_probs],
            size=5, opacity=0.5,
        ),
        name="Non-cases",
        showlegend=False,
        hovertemplate="p = %{x:.3f}<extra>Non-case</extra>",
    ), row=1, col=1)

    # Threshold vertical line
    fig.add_shape(
        type="line", x0=k, y0=-0.1, x1=k, y1=1.1,
        xref="x", yref="y",
        line=dict(color=PALETTE.secondary, dash="dash", width=1.5),
    )
    # TP / FN / FP / TN annotations
    mid_right = (k + 1) / 2
    mid_left = k / 2
    for text, x_pos, y_pos in [
        (f"FN: {FN}", mid_left, 1.07),
        (f"TP: {TP}", mid_right, 1.07),
        (f"TN: {TN}", mid_left, -0.07),
        (f"FP: {FP}", mid_right, -0.07),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=f"<b>{text}</b>",
            xref="x", yref="y",
            showarrow=False,
            font=dict(size=10, color=PALETTE.text),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=PALETTE.axis_line,
            borderwidth=1,
        )
    # k label on the line
    fig.add_annotation(
        x=k, y=-0.12, text=f"k = {k:.2f}",
        xref="x", yref="y",
        showarrow=False, font=dict(size=9, color=PALETTE.secondary),
    )
    fig.update_xaxes(title_text="Predicted probability", row=1, col=1)
    fig.update_yaxes(title_text="Probability of being a case", row=1, col=1)

    # ── Panel 2: Cases vs threshold ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=thresholds, y=fn_arr,
        fill="tozeroy",
        fillcolor="rgba(191,191,191,0.30)",
        line=dict(color=PALETTE.gray, width=1),
        showlegend=False, name="FN",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=thresholds, y=tp_arr,
        fill="tozeroy",
        fillcolor="rgba(42,157,143,0.35)",
        line=dict(color=PALETTE.secondary, width=1.5),
        showlegend=False, name="TP",
    ), row=1, col=2)
    fig.add_shape(
        type="line", x0=k, y0=0, x1=k, y1=max(tp_arr + fn_arr) * 1.1,
        xref="x2", yref="y2",
        line=dict(color=PALETTE.highlight, dash="dash", width=1.5),
    )
    fig.add_trace(go.Scatter(
        x=[k], y=[TP],
        mode="markers+text",
        marker=dict(color=PALETTE.highlight, size=9),
        text=[f"  TP: {TP}"], textposition="middle right",
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Threshold (k)", row=1, col=2)
    fig.update_yaxes(title_text="Cases", row=1, col=2)

    # ── Panel 3: Non-cases vs threshold ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=thresholds, y=tn_arr,
        fill="tozeroy",
        fillcolor="rgba(69,123,157,0.25)",
        line=dict(color=PALETTE.secondary, width=1.5),
        showlegend=False, name="TN",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=thresholds, y=fp_arr,
        fill="tozeroy",
        fillcolor="rgba(191,191,191,0.30)",
        line=dict(color=PALETTE.gray, width=1),
        showlegend=False, name="FP",
    ), row=2, col=2)
    fig.add_shape(
        type="line", x0=k, y0=0, x1=k, y1=max(fp_arr + tn_arr) * 1.1,
        xref="x3", yref="y3",
        line=dict(color=PALETTE.highlight, dash="dash", width=1.5),
    )
    fig.add_trace(go.Scatter(
        x=[k], y=[FP],
        mode="markers+text",
        marker=dict(color=PALETTE.highlight, size=9),
        text=[f"  FP: {FP}"], textposition="middle right",
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text="Threshold (k)", row=2, col=2)
    fig.update_yaxes(title_text="Non-cases", row=2, col=2)

    # ── Panel 4: ROC ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=fpr_arr, y=tpr_arr,
        fill="tozeroy",
        fillcolor="rgba(191,191,191,0.20)",
        line=dict(color=PALETTE.secondary, width=2),
        showlegend=False,
        hovertemplate="FPR: %{x:.3f} TPR: %{y:.3f}<extra></extra>",
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=PALETTE.gray, dash="dot", width=1),
        showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[fpr_at_k], y=[tpr_at_k],
        mode="markers",
        marker=dict(color=PALETTE.highlight, size=10, symbol="circle"),
        showlegend=False,
        hovertemplate=f"k = {k:.2f}<br>FPR: {fpr_at_k:.3f}<br>TPR: {tpr_at_k:.3f}<extra></extra>",
    ), row=1, col=3)
    fig.add_annotation(
        x=0.98, y=0.35,
        xref="x4 domain", yref="y4 domain",
        text=(
            f"Recall = {recall_k:.2f}<br>"
            f"FPR = {fpr_k:.2f}<br>"
            f"AUC = {roc_auc:.2f}"
        ),
        align="right", showarrow=False,
        font=dict(size=10, color=PALETTE.text),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=PALETTE.axis_line, borderwidth=1,
    )
    fig.add_annotation(
        x=fpr_at_k, y=tpr_at_k + 0.06,
        xref="x4", yref="y4",
        text=f"k = {k:.2f}",
        showarrow=False, font=dict(size=9, color=PALETTE.highlight),
    )
    fig.update_xaxes(title_text="FPR = FP / (TN + FP)", row=1, col=3)
    fig.update_yaxes(title_text="Recall = TP / (TP + FN)", row=1, col=3)

    # ── Panel 5: Precision / Recall ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=rec_arr, y=prec_arr,
        fill="tozeroy",
        fillcolor="rgba(191,191,191,0.20)",
        line=dict(color=PALETTE.secondary, width=2),
        showlegend=False,
        hovertemplate="Recall: %{x:.3f} Precision: %{y:.3f}<extra></extra>",
    ), row=2, col=3)
    fig.add_trace(go.Scatter(
        x=[rec_at_k], y=[prec_at_k],
        mode="markers",
        marker=dict(color=PALETTE.highlight, size=10, symbol="circle"),
        showlegend=False,
        hovertemplate=f"k = {k:.2f}<br>Recall: {rec_at_k:.3f}<br>Precision: {prec_at_k:.3f}<extra></extra>",
    ), row=2, col=3)
    fig.add_annotation(
        x=0.98, y=0.35,
        xref="x5 domain", yref="y5 domain",
        text=(
            f"Precision = {prec_at_k:.2f}<br>"
            f"Recall = {rec_at_k:.2f}<br>"
            f"AUC = {pr_auc:.2f}"
        ),
        align="right", showarrow=False,
        font=dict(size=10, color=PALETTE.text),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=PALETTE.axis_line, borderwidth=1,
    )
    fig.add_annotation(
        x=rec_at_k, y=prec_at_k + 0.06,
        xref="x5", yref="y5",
        text=f"k = {k:.2f}",
        showarrow=False, font=dict(size=9, color=PALETTE.highlight),
    )
    fig.update_xaxes(title_text="Recall = TP / (TP + FN)", row=2, col=3)
    fig.update_yaxes(title_text="Precision = TP / (TP + FP)", row=2, col=3)

    # ── Global style ──────────────────────────────────────────────────────────
    fig.update_layout(
        height=620,
        plot_bgcolor=PALETTE.background,
        paper_bgcolor=PALETTE.background,
        font=dict(color=PALETTE.text, size=11),
        margin=dict(l=50, r=30, t=70, b=50),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=PALETTE.axis_line)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=PALETTE.axis_line)
    return fig


def pr_curves_grid_plot(
    pr_summary: list[dict],
    target_precision: float,
    mcid: float,
) -> go.Figure:
    """Grid of Precision-Recall curves, one subplot per dataset+model combination."""
    import math

    n = len(pr_summary)
    if n == 0:
        return base_fig("No results to display")

    ncols = max(1, min(3, math.ceil(math.sqrt(n))))
    nrows = math.ceil(n / ncols)
    subplot_titles = [f"{r['Dataset']}<br>{r['Model']}" for r in pr_summary]

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    p_label = int(target_precision * 100)

    for i, row in enumerate(pr_summary):
        r_idx = i // ncols + 1
        c_idx = i % ncols + 1

        prec_c = row["_prec_c"]
        rec_c = row["_rec_c"]
        prev = row["_prev"]

        fig.add_trace(
            go.Scatter(
                x=rec_c, y=prec_c,
                mode="lines",
                line=dict(color=PALETTE.highlight, width=2),
                showlegend=False,
            ),
            row=r_idx, col=c_idx,
        )

        # No-skill baseline (prevalence)
        fig.add_hline(
            y=prev, line_dash="dash", line_color=PALETTE.gray,
            row=r_idx, col=c_idx,
        )

        # Target precision reference line
        fig.add_hline(
            y=target_precision, line_dash="dot", line_color="gold",
            row=r_idx, col=c_idx,
        )

        # Operating point at MCID threshold (diamond)
        op_prec = row["_op_prec"]
        op_rec = row["_op_rec"]
        fig.add_trace(
            go.Scatter(
                x=[op_rec], y=[op_prec],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color=PALETTE.highlight),
                showlegend=False,
            ),
            row=r_idx, col=c_idx,
        )

        # Tuned threshold point (star)
        tp_prec = row["_tp_prec"]
        tp_rec = row["_tp_rec"]
        if tp_prec is not None and tp_rec is not None:
            fig.add_trace(
                go.Scatter(
                    x=[tp_rec], y=[tp_prec],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color="gold"),
                    showlegend=False,
                ),
                row=r_idx, col=c_idx,
            )

    row_height = 300
    fig.update_layout(
        height=max(350, nrows * row_height),
        plot_bgcolor=PALETTE.background,
        paper_bgcolor=PALETTE.background,
        font=dict(color=PALETTE.text, size=10),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        range=[0, 1], showgrid=False, zeroline=False,
        linecolor=PALETTE.axis_line, title_text="Recall",
    )
    fig.update_yaxes(
        range=[0, 1.05], showgrid=False, zeroline=False,
        linecolor=PALETTE.axis_line, title_text="Precision",
    )
    return fig


def confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mcid: float,
    tuned_thr: float | None,
    model_name: str,
    dataset_name: str,
) -> go.Figure:
    """Two side-by-side confusion matrices: at MCID threshold and at tuned threshold."""
    from sklearn.metrics import confusion_matrix as sk_cm

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~np.isnan(y_true)
    y_true, y_pred = y_true[valid], y_pred[valid]

    y_t_bin = (y_true < mcid).astype(int)

    def _cm_annotations(cm: np.ndarray):
        total = cm.sum()
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        denom = 4 * prec + rec
        f2 = 5 * prec * rec / denom if denom > 0 else 0.0
        text = [
            [f"TN<br>{TN}<br>({100*TN/total:.1f}%)", f"FP<br>{FP}<br>({100*FP/total:.1f}%)"],
            [f"FN<br>{FN}<br>({100*FN/total:.1f}%)", f"TP<br>{TP}<br>({100*TP/total:.1f}%)"],
        ]
        return text, prec, rec, f2

    y_bin_mcid = (y_pred < mcid).astype(int)
    cm_mcid = sk_cm(y_t_bin, y_bin_mcid)
    text_mcid, prec_mcid, rec_mcid, f2_mcid = _cm_annotations(cm_mcid)

    subtitles = [
        f"Default MCID ({mcid:.3g} pts)  P={prec_mcid:.3f}  R={rec_mcid:.3f}  F2={f2_mcid:.3f}",
    ]

    if tuned_thr is not None:
        y_bin_thr = (y_pred < tuned_thr).astype(int)
        cm_thr = sk_cm(y_t_bin, y_bin_thr)
        text_thr, prec_thr, rec_thr, f2_thr = _cm_annotations(cm_thr)
        subtitles.append(
            f"Tuned ({tuned_thr:.3f} pts)  P={prec_thr:.3f}  R={rec_thr:.3f}  F2={f2_thr:.3f}"
        )
    else:
        cm_thr = None
        subtitles.append("No solution — target precision not achievable")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=subtitles,
        horizontal_spacing=0.14,
    )

    colorscale = [[0, PALETTE.background], [1, PALETTE.highlight]]

    def _add_heatmap(cm_data: np.ndarray, text: list, col: int) -> None:
        fig.add_trace(
            go.Heatmap(
                z=cm_data,
                x=["Pred: No benefit", "Pred: Benefit"],
                y=["Actual: No benefit", "Actual: Benefit"],
                text=text,
                texttemplate="%{text}",
                colorscale=colorscale,
                showscale=False,
            ),
            row=1, col=col,
        )

    _add_heatmap(cm_mcid, text_mcid, 1)

    if cm_thr is not None:
        _add_heatmap(cm_thr, text_thr, 2)
    else:
        fig.add_annotation(
            text="Target precision not achievable<br>with any threshold",
            xref="x2 domain", yref="y2 domain",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=PALETTE.text),
        )

    fig.update_layout(
        title=f"Confusion Matrices — {model_name} on {dataset_name}",
        height=420,
        plot_bgcolor=PALETTE.background,
        paper_bgcolor=PALETTE.background,
        font=dict(color=PALETTE.text, size=11),
        margin=dict(l=40, r=20, t=110, b=40),
    )
    return fig


def regression_outcome_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    model_name: str,
    positive_below: bool = False,
    positive_label: str = "Class 1 (positive)",
    negative_label: str = "Class 0 (negative)",
) -> go.Figure:
    """5-panel outcome explorer for regression models.

    *positive_below* controls which side of the threshold is class 1:
    - False (default): actual >= threshold → class 1
    - True:            actual <  threshold → class 1  (e.g. NHS "no benefit" = gain < 7)

    For ROC/PR the continuous y_pred is used as the ranking score; when positive_below
    is True the score is negated so that lower predictions rank as more likely positive.
    """
    from scipy.stats import gaussian_kde
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Binarise according to user-defined direction
    if positive_below:
        pos_mask = y_true < threshold
        y_pred_bin = (y_pred < threshold).astype(int)
        roc_score = -y_pred          # lower prediction → more likely class 1
        _pos_side = f"< {threshold:.3g}"
        _neg_side = f"≥ {threshold:.3g}"
    else:
        pos_mask = y_true >= threshold
        y_pred_bin = (y_pred >= threshold).astype(int)
        roc_score = y_pred
        _pos_side = f"≥ {threshold:.3g}"
        _neg_side = f"< {threshold:.3g}"

    neg_mask = ~pos_mask
    pos_preds = y_pred[pos_mask]
    neg_preds = y_pred[neg_mask]

    y_true_bin = pos_mask.astype(int)
    TP = int(((y_pred_bin == 1) & pos_mask).sum())
    FN = int(((y_pred_bin == 0) & pos_mask).sum())
    FP = int(((y_pred_bin == 1) & neg_mask).sum())
    TN = int(((y_pred_bin == 0) & neg_mask).sum())

    recall_k = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr_k    = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    prec_k   = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Threshold-sweep curves
    pred_range = np.linspace(y_pred.min(), y_pred.max(), 200)
    tp_arr, fn_arr, fp_arr, tn_arr = [], [], [], []
    for t in pred_range:
        yp = (y_pred < t).astype(int) if positive_below else (y_pred >= t).astype(int)
        tp_arr.append(int(((yp == 1) & pos_mask).sum()))
        fn_arr.append(int(((yp == 0) & pos_mask).sum()))
        fp_arr.append(int(((yp == 1) & neg_mask).sum()))
        tn_arr.append(int(((yp == 0) & neg_mask).sum()))

    # ROC & PR — roc_score already accounts for direction
    fpr_arr, tpr_arr, roc_thresh = roc_curve(y_true_bin, roc_score)
    roc_auc_val = auc(fpr_arr, tpr_arr)
    # Find the point on the ROC curve closest to our threshold
    roc_score_at_thresh = -threshold if positive_below else threshold
    roc_idx = int(np.argmin(np.abs(roc_thresh - roc_score_at_thresh)))
    fpr_at_k = float(fpr_arr[roc_idx])
    tpr_at_k = float(tpr_arr[roc_idx])

    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true_bin, roc_score)
    pr_auc_val = auc(rec_arr, prec_arr)
    pr_score_at_thresh = -threshold if positive_below else threshold
    pr_idx = int(np.argmin(np.abs(pr_thresh - pr_score_at_thresh))) if len(pr_thresh) > 0 else 0
    prec_at_k = float(prec_arr[pr_idx])
    rec_at_k  = float(rec_arr[pr_idx])

    # KDE
    x_grid = np.linspace(y_pred.min(), y_pred.max(), 300)
    kde_pos = gaussian_kde(pos_preds)(x_grid) if len(pos_preds) > 2 else np.zeros(300)
    kde_neg = gaussian_kde(neg_preds)(x_grid) if len(neg_preds) > 2 else np.zeros(300)

    fig = make_subplots(
        rows=2, cols=3,
        specs=[[{"rowspan": 2}, {}, {}], [None, {}, {}]],
        subplot_titles=[
            model_name,
            f"{positive_label} (actual {_pos_side}) vs threshold",
            "ROC",
            None,
            f"{negative_label} (actual {_neg_side}) vs threshold",
            "Precision / Recall",
        ],
        column_widths=[0.38, 0.30, 0.32],
        row_heights=[0.50, 0.50],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    # Colour for predicted-correct vs predicted-wrong scatter dots
    def _dot_color(preds, is_positive_class):
        correct_color = PALETTE.highlight
        wrong_color   = PALETTE.gray
        if positive_below:
            return [correct_color if p < threshold else wrong_color for p in preds] if is_positive_class \
                   else [wrong_color if p < threshold else correct_color for p in preds]
        else:
            return [correct_color if p >= threshold else wrong_color for p in preds] if is_positive_class \
                   else [wrong_color if p >= threshold else correct_color for p in preds]

    # Panel 1: KDE + scatter
    fig.add_trace(go.Scatter(
        x=x_grid, y=kde_neg, fill="tozeroy",
        fillcolor="rgba(255,165,0,0.25)",
        line=dict(color="rgba(255,165,0,0.7)", width=1),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_grid, y=kde_pos, fill="tozeroy",
        fillcolor="rgba(42,157,143,0.25)",
        line=dict(color="rgba(42,157,143,0.8)", width=1.5),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    rng = np.random.default_rng(42)
    case_y = 1.0 + rng.uniform(-0.03, 0.03, len(pos_preds))
    nc_y   = rng.uniform(-0.03, 0.03, len(neg_preds))
    fig.add_trace(go.Scatter(
        x=pos_preds, y=case_y, mode="markers",
        marker=dict(color=_dot_color(pos_preds, True), size=5, opacity=0.5),
        showlegend=False,
        hovertemplate="predicted = %{x:.3f}<extra>Positive outcome</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=neg_preds, y=nc_y, mode="markers",
        marker=dict(
            color=[PALETTE.highlight if p >= threshold else PALETTE.gray for p in neg_preds],
            size=5, opacity=0.5,
        ),
        showlegend=False,
        hovertemplate="predicted = %{x:.3f}<extra>Negative outcome</extra>",
    ), row=1, col=1)

    fig.add_shape(
        type="line", x0=threshold, y0=-0.1, x1=threshold, y1=1.1,
        xref="x", yref="y",
        line=dict(color=PALETTE.secondary, dash="dash", width=1.5),
    )
    mid_right = (threshold + y_pred.max()) / 2
    mid_left = (threshold + y_pred.min()) / 2
    for text, x_pos, y_pos in [
        (f"FN: {FN}", mid_left, 1.07), (f"TP: {TP}", mid_right, 1.07),
        (f"TN: {TN}", mid_left, -0.07), (f"FP: {FP}", mid_right, -0.07),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=f"<b>{text}</b>",
            xref="x", yref="y", showarrow=False,
            font=dict(size=10, color=PALETTE.text),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=PALETTE.axis_line, borderwidth=1,
        )
    fig.add_annotation(
        x=threshold, y=-0.13,
        text=f"t = {threshold:.3g}",
        xref="x", yref="y", showarrow=False,
        font=dict(size=9, color=PALETTE.secondary),
    )
    fig.update_xaxes(title_text="Predicted value", row=1, col=1)
    fig.update_yaxes(
        title_text=f"← {negative_label} | {positive_label} →",
        row=1, col=1,
    )

    # Panel 2: positive outcomes vs threshold sweep
    fig.add_trace(go.Scatter(
        x=pred_range, y=fn_arr, fill="tozeroy",
        fillcolor="rgba(191,191,191,0.30)",
        line=dict(color=PALETTE.gray, width=1),
        showlegend=False, name="FN",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=pred_range, y=tp_arr, fill="tozeroy",
        fillcolor="rgba(42,157,143,0.35)",
        line=dict(color=PALETTE.secondary, width=1.5),
        showlegend=False, name="TP",
    ), row=1, col=2)
    fig.add_shape(
        type="line", x0=threshold, y0=0, x1=threshold, y1=max(tp_arr + fn_arr) * 1.1,
        xref="x2", yref="y2",
        line=dict(color=PALETTE.highlight, dash="dash", width=1.5),
    )
    fig.add_trace(go.Scatter(
        x=[threshold], y=[TP], mode="markers+text",
        marker=dict(color=PALETTE.highlight, size=9),
        text=[f"  TP: {TP}"], textposition="middle right",
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text=positive_label, row=1, col=2)

    # Panel 3: negative outcomes vs threshold sweep
    fig.add_trace(go.Scatter(
        x=pred_range, y=tn_arr, fill="tozeroy",
        fillcolor="rgba(69,123,157,0.25)",
        line=dict(color=PALETTE.secondary, width=1.5),
        showlegend=False, name="TN",
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=pred_range, y=fp_arr, fill="tozeroy",
        fillcolor="rgba(191,191,191,0.30)",
        line=dict(color=PALETTE.gray, width=1),
        showlegend=False, name="FP",
    ), row=2, col=2)
    fig.add_shape(
        type="line", x0=threshold, y0=0, x1=threshold, y1=max(fp_arr + tn_arr) * 1.1,
        xref="x3", yref="y3",
        line=dict(color=PALETTE.highlight, dash="dash", width=1.5),
    )
    fig.add_trace(go.Scatter(
        x=[threshold], y=[FP], mode="markers+text",
        marker=dict(color=PALETTE.highlight, size=9),
        text=[f"  FP: {FP}"], textposition="middle right",
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text="Threshold", row=2, col=2)
    fig.update_yaxes(title_text=negative_label, row=2, col=2)

    # Panel 4: ROC
    fig.add_trace(go.Scatter(
        x=fpr_arr, y=tpr_arr, fill="tozeroy",
        fillcolor="rgba(191,191,191,0.20)",
        line=dict(color=PALETTE.secondary, width=2),
        showlegend=False,
        hovertemplate="FPR: %{x:.3f} TPR: %{y:.3f}<extra></extra>",
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=PALETTE.gray, dash="dot", width=1),
        showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[fpr_at_k], y=[tpr_at_k], mode="markers",
        marker=dict(color=PALETTE.highlight, size=10, symbol="circle"),
        showlegend=False,
        hovertemplate=f"t={threshold:.3g}<br>FPR: {fpr_at_k:.3f}<br>TPR: {tpr_at_k:.3f}<extra></extra>",
    ), row=1, col=3)
    fig.add_annotation(
        x=0.98, y=0.35, xref="x4 domain", yref="y4 domain",
        text=f"Recall = {recall_k:.2f}<br>FPR = {fpr_k:.2f}<br>AUC = {roc_auc_val:.2f}",
        align="right", showarrow=False,
        font=dict(size=10, color=PALETTE.text),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=PALETTE.axis_line, borderwidth=1,
    )
    fig.add_annotation(
        x=fpr_at_k, y=tpr_at_k + 0.06, xref="x4", yref="y4",
        text=f"t = {threshold:.3g}", showarrow=False,
        font=dict(size=9, color=PALETTE.highlight),
    )
    fig.update_xaxes(title_text="FPR = FP / (TN + FP)", row=1, col=3)
    fig.update_yaxes(title_text="Recall = TP / (TP + FN)", row=1, col=3)

    # Panel 5: Precision / Recall
    fig.add_trace(go.Scatter(
        x=rec_arr, y=prec_arr, fill="tozeroy",
        fillcolor="rgba(191,191,191,0.20)",
        line=dict(color=PALETTE.secondary, width=2),
        showlegend=False,
        hovertemplate="Recall: %{x:.3f} Precision: %{y:.3f}<extra></extra>",
    ), row=2, col=3)
    fig.add_trace(go.Scatter(
        x=[rec_at_k], y=[prec_at_k], mode="markers",
        marker=dict(color=PALETTE.highlight, size=10, symbol="circle"),
        showlegend=False,
        hovertemplate=f"t={threshold:.3g}<br>Recall: {rec_at_k:.3f}<br>Precision: {prec_at_k:.3f}<extra></extra>",
    ), row=2, col=3)
    fig.add_annotation(
        x=0.98, y=0.35, xref="x5 domain", yref="y5 domain",
        text=f"Precision = {prec_at_k:.2f}<br>Recall = {rec_at_k:.2f}<br>AUC = {pr_auc_val:.2f}",
        align="right", showarrow=False,
        font=dict(size=10, color=PALETTE.text),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=PALETTE.axis_line, borderwidth=1,
    )
    fig.add_annotation(
        x=rec_at_k, y=prec_at_k + 0.06, xref="x5", yref="y5",
        text=f"t = {threshold:.3g}", showarrow=False,
        font=dict(size=9, color=PALETTE.highlight),
    )
    fig.update_xaxes(title_text="Recall = TP / (TP + FN)", row=2, col=3)
    fig.update_yaxes(title_text="Precision = TP / (TP + FP)", row=2, col=3)

    fig.update_layout(
        height=620,
        plot_bgcolor=PALETTE.background,
        paper_bgcolor=PALETTE.background,
        font=dict(color=PALETTE.text, size=11),
        margin=dict(l=50, r=30, t=70, b=50),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=PALETTE.axis_line)
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=PALETTE.axis_line)
    return fig
