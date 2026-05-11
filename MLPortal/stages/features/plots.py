"""Plots for the Feature Engineering stage."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, highlight_at, highlight_where


def feature_correlation_bar(corr_series: pd.Series, top_n: int = 20) -> go.Figure:
    """Horizontal bar: engineered feature correlations with the target.

    The highest-correlation feature is highlighted.
    """
    top = corr_series.abs().nlargest(top_n)
    names = top.index.tolist()
    values = [corr_series[n] for n in names]
    colors = highlight_at(len(names), 0)

    fig = base_fig(
        title="Feature — target correlations (after engineering)",
        xlabel="Correlation with target",
        ylabel="",
        height=max(320, len(names) * 22),
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    from shared.viz import add_reference_line
    add_reference_line(fig, 0, orientation="v")
    if names:
        annotate_message(fig, f"'{names[0]}' is most correlated with the target")
    return fig


def distribution_before_after(
    before: pd.Series,
    after: pd.Series,
    column: str,
) -> go.Figure:
    """Overlaid histograms comparing distribution before and after a transform."""
    fig = base_fig(
        title=f"Distribution before / after transform: {column}",
        xlabel=column,
        ylabel="Count",
    )
    fig.add_trace(
        go.Histogram(
            x=before.dropna(),
            name="Before",
            marker_color=PALETTE.gray,
            opacity=0.6,
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=after.dropna(),
            name="After",
            marker_color=PALETTE.highlight,
            opacity=0.6,
            nbinsx=40,
        )
    )
    fig.update_layout(barmode="overlay")
    annotate_message(fig, "Red = after transform", color=PALETTE.highlight)
    return fig
