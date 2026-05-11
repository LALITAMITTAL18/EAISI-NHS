"""Plots for the Outlier Detection stage."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, gray_series, highlight_where
from stages.outliers.models import ColumnOutlierResult


def scatter_with_outliers(
    series: pd.Series,
    lower: float,
    upper: float,
) -> go.Figure:
    """Scatter of all values — dual-flagged outliers highlighted in red."""
    flagged = (series < lower) | (series > upper)
    colors = highlight_where(series.tolist(), lambda v: v < lower or v > upper)

    fig = base_fig(
        title=f"Outliers in '{series.name}'",
        xlabel="Row index",
        ylabel=str(series.name),
    )
    # Normal points
    normal_idx = series.index[~flagged]
    fig.add_trace(
        go.Scatter(
            x=normal_idx,
            y=series[~flagged],
            mode="markers",
            marker=dict(color=PALETTE.gray, size=4, opacity=0.6),
            name="Normal",
            hovertemplate="Index %{x}: %{y:.3f}<extra></extra>",
        )
    )
    # Outlier points
    out_idx = series.index[flagged]
    if len(out_idx):
        fig.add_trace(
            go.Scatter(
                x=out_idx,
                y=series[flagged],
                mode="markers",
                marker=dict(color=PALETTE.highlight, size=6, symbol="x"),
                name="Flagged outlier",
                hovertemplate="Index %{x}: %{y:.3f}<extra></extra>",
            )
        )
    # Bounds
    from shared.viz import add_reference_line
    add_reference_line(fig, lower, label=f"Lower bound = {lower:.2f}")
    add_reference_line(fig, upper, label=f"Upper bound = {upper:.2f}")

    n_out = int(flagged.sum())
    annotate_message(fig, f"{n_out} flagged outliers ({n_out / max(len(series), 1) * 100:.1f}%)")
    return fig


def flag_summary_bar(col_results: list[ColumnOutlierResult]) -> go.Figure:
    """Horizontal bar showing n_dual_flagged per column; top column highlighted."""
    if not col_results:
        return base_fig(title="No numeric columns analysed")

    sorted_res = sorted(col_results, key=lambda r: r.n_dual_flagged, reverse=True)
    names = [r.column for r in sorted_res]
    counts = [r.n_dual_flagged for r in sorted_res]
    from shared.viz import highlight_at
    colors = highlight_at(len(names), 0)

    fig = base_fig(
        title="Dual-flagged outliers per column",
        xlabel="Rows flagged",
        ylabel="",
    )
    fig.add_trace(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=colors,
            text=counts,
            textposition="outside",
            hovertemplate="%{y}: %{x} dual-flagged rows<extra></extra>",
        )
    )
    if counts:
        annotate_message(fig, f"'{names[0]}' has the most outliers ({counts[0]})")
    return fig
