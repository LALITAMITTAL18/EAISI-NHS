"""Plots for the Explanation stage."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, gray_series, highlight_at
from stages.explanation.models import PdpResult, ShapResult


def shap_summary_bar(result: ShapResult, top_n: int = 20) -> go.Figure:
    """Horizontal bar of mean |SHAP| values; top feature highlighted."""
    names = result.feature_names[:top_n]
    values = result.mean_abs_shap[:top_n]
    if not names:
        return base_fig(title="No SHAP data")

    colors = highlight_at(len(names), 0)
    fig = base_fig(
        title=f"Feature importance (SHAP) — {result.model_name}",
        xlabel="Mean |SHAP value|",
        ylabel="",
        height=max(320, len(names) * 22),
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
    annotate_message(fig, f"'{names[0]}' has the greatest impact on predictions")
    return fig


def waterfall_plot(
    feature_names: list[str],
    shap_values: list[float],
    base_value: float,
    prediction: float,
) -> go.Figure:
    """SHAP waterfall chart for a single prediction."""
    pairs = sorted(zip(shap_values, feature_names), key=lambda x: abs(x[0]), reverse=True)
    values = [p[0] for p in pairs]
    names = [p[1] for p in pairs]
    colors = [PALETTE.highlight if v > 0 else PALETTE.secondary for v in values]

    fig = base_fig(
        title="SHAP waterfall — individual prediction",
        xlabel="SHAP value",
        ylabel="Feature",
        height=max(350, len(names) * 24),
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    from shared.viz import add_reference_line
    add_reference_line(fig, 0, orientation="v")
    annotate_message(fig, f"Base value: {base_value:.3f} → Prediction: {prediction:.3f}")
    return fig


def pdp_plot(result: PdpResult) -> go.Figure:
    """Partial dependence plot for a single feature."""
    fig = base_fig(
        title=f"Partial dependence — {result.feature}",
        xlabel=result.feature,
        ylabel="Average prediction",
    )
    fig.add_trace(
        go.Scatter(
            x=result.grid_values,
            y=result.avg_predictions,
            mode="lines+markers",
            line=dict(color=PALETTE.highlight, width=2),
            marker=dict(color=PALETTE.highlight, size=5),
            hovertemplate=f"{result.feature}: %{{x:.3f}}<br>Avg prediction: %{{y:.3f}}<extra></extra>",
        )
    )
    return fig
