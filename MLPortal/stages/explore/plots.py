"""Plots for the Explore stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.viz import (
    PALETTE,
    add_reference_line,
    annotate_message,
    base_fig,
    gray_series,
    highlight_at,
    highlight_where,
)


def distribution_plot(series: pd.Series, target: str | None = None) -> go.Figure:
    """Histogram + KDE for a single numeric series.

    If *target* matches the series name, it is highlighted; otherwise gray.
    """
    clean = series.dropna()
    is_target = series.name == target
    bar_color = PALETTE.highlight if is_target else PALETTE.gray

    fig = base_fig(title=f"Distribution: {series.name}", xlabel=series.name, ylabel="Count")
    fig.add_trace(
        go.Histogram(
            x=clean,
            marker_color=bar_color,
            opacity=0.8,
            name=str(series.name),
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    return fig


def correlation_heatmap(corr_df: pd.DataFrame, target: str | None = None) -> go.Figure:
    """Heatmap of pairwise correlations.

    Values near ±1 are visually strongest; all others fade toward gray.
    """
    z = corr_df.values
    cols = corr_df.columns.tolist()

    fig = base_fig(title="Correlation matrix", height=520)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=cols,
            y=cols,
            colorscale=[
                [0.0, PALETTE.secondary],
                [0.5, "#F0F0F0"],
                [1.0, PALETTE.highlight],
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_layout(height=max(400, len(cols) * 30))
    return fig


def target_correlation_bar(
    corr_series: pd.Series,
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar of feature-target correlations.

    The top correlated feature is highlighted; others are gray.
    """
    top = corr_series.abs().nlargest(top_n)
    names = top.index.tolist()
    values = [corr_series[n] for n in names]
    colors = highlight_at(len(names), 0)

    fig = base_fig(
        title=f"Top {top_n} features by correlation with target",
        xlabel="Correlation",
        ylabel="",
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
    add_reference_line(fig, 0, orientation="v")
    annotate_message(fig, f"'{names[0]}' has the strongest correlation")
    return fig


def subgroup_bar(subgroup_df: pd.DataFrame, group_col: str, target: str) -> go.Figure:
    """Bar chart of mean target per group; groups far from overall mean are highlighted."""
    overall_mean = subgroup_df["group_mean"].mean()
    threshold = subgroup_df["deviation"].abs().mean()
    colors = highlight_where(
        subgroup_df["deviation"].tolist(),
        lambda v: abs(v) > threshold,
    )

    fig = base_fig(
        title=f"Mean {target} by {group_col}",
        xlabel=group_col,
        ylabel=f"Mean {target}",
    )
    fig.add_trace(
        go.Bar(
            x=subgroup_df[group_col].astype(str),
            y=subgroup_df["group_mean"],
            marker_color=colors,
            customdata=subgroup_df[["n", "deviation"]].values,
            hovertemplate=(
                f"{group_col}: %{{x}}<br>"
                "Mean: %{y:.3f}<br>"
                "n: %{customdata[0]}<br>"
                "Deviation: %{customdata[1]:+.3f}<extra></extra>"
            ),
        )
    )
    add_reference_line(fig, overall_mean, label=f"Overall mean = {overall_mean:.2f}")
    return fig


def qq_plot(theoretical: list[float], observed: list[float], col_name: str) -> go.Figure:
    """Q-Q plot for normality assessment."""
    lo = min(min(theoretical), min(observed))
    hi = max(max(theoretical), max(observed))

    fig = base_fig(title=f"Q-Q plot: {col_name}", xlabel="Theoretical quantiles", ylabel="Observed quantiles")
    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=observed,
            mode="markers",
            marker=dict(color=PALETTE.gray, size=4),
            name="Data points",
            hovertemplate="Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color=PALETTE.highlight, dash="dash", width=1.5),
            name="Normal reference",
        )
    )
    return fig
