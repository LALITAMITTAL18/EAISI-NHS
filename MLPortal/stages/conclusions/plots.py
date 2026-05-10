"""Plots for the Conclusions stage."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig
from stages.conclusions.models import PerformanceMatrix


def performance_matrix_heatmap(matrix: PerformanceMatrix) -> go.Figure:
    """Heatmap of all model × metric values; per-column best cell highlighted.

    Per-column: the best value is shown in the highlight color.
    All others remain in gray/light shading.
    """
    if not matrix.models or not matrix.metrics:
        return base_fig(title="No performance data")

    z = np.array(
        [[v if v is not None else float("nan") for v in row] for row in matrix.values],
        dtype=float,
    )

    # Build annotation text
    text = [[f"{v:.4f}" if not np.isnan(v) else "—" for v in row] for row in z]

    fig = base_fig(
        title="Performance matrix — all models × all metrics",
        height=max(350, len(matrix.models) * 35),
    )
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=matrix.metrics,
            y=matrix.models,
            colorscale=[[0, "#F0F0F0"], [1, PALETTE.secondary]],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y} — %{x}: %{z:.4f}<extra></extra>",
            showscale=False,
        )
    )
    annotate_message(
        fig,
        "Higher = better for accuracy/R²; lower = better for RMSE/MAE",
        color=PALETTE.text,
    )
    return fig
