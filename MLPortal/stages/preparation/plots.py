"""Plots for the Preparation & Split stage."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, two_tone


def class_balance_bar(balance: dict[str, float], title: str = "Class balance") -> go.Figure:
    """Bar chart of class distribution; minority class highlighted."""
    if not balance:
        return base_fig(title=title)
    labels = list(balance.keys())
    values = list(balance.values())
    min_val = min(values)
    colors = two_tone([v == min_val for v in values])

    fig = base_fig(title=title, xlabel="Class", ylabel="Proportion")
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )
    if len(values) == 2:
        ratio = max(values) / max(min(values), 1e-9)
        annotate_message(fig, f"Class imbalance ratio {ratio:.1f}:1")
    return fig


def split_summary_bar(n_train: int, n_test: int) -> go.Figure:
    """Simple two-bar chart showing train vs test sizes."""
    fig = base_fig(title="Train / test split", xlabel="", ylabel="Rows")
    fig.add_trace(
        go.Bar(
            x=["Train", "Test"],
            y=[n_train, n_test],
            marker_color=[PALETTE.gray, PALETTE.highlight],
            text=[f"{n_train:,}", f"{n_test:,}"],
            textposition="outside",
        )
    )
    pct = n_test / max(n_train + n_test, 1) * 100
    annotate_message(fig, f"{pct:.0f}% held out for testing")
    fig.update_layout(height=320)
    return fig
