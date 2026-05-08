"""Plots for the Upload stage."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, gray_series, highlight_where


def dtype_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar showing count of each dtype family.

    All bars are gray; the most common dtype is highlighted.
    """
    families = (
        df.dtypes.astype(str)
        .str.replace(r"\d+", "", regex=True)
        .value_counts()
    )
    names = families.index.tolist()
    counts = families.values.tolist()
    colors = highlight_where(counts, lambda v: v == max(counts))

    fig = base_fig(title="Column data types", xlabel="Count", ylabel="")
    fig.add_trace(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=colors,
            text=counts,
            textposition="outside",
            hovertemplate="%{y}: %{x} columns<extra></extra>",
        )
    )
    annotate_message(
        fig, f"Most columns are {names[0]} ({counts[0]})", color=PALETTE.highlight
    )
    return fig


def memory_bar(mem_raw: float, mem_opt: float) -> go.Figure:
    """Two-bar chart comparing raw vs optimised memory usage."""
    labels = ["Raw", "Optimised"]
    values = [mem_raw, mem_opt]
    colors = [PALETTE.gray, PALETTE.highlight]

    fig = base_fig(title="Memory usage after dtype optimisation", ylabel="MB")
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f} MB" for v in values],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f} MB<extra></extra>",
        )
    )
    saving_pct = (1 - mem_opt / mem_raw) * 100 if mem_raw else 0
    annotate_message(fig, f"{saving_pct:.0f}% memory saved after optimisation")
    fig.update_layout(height=320)
    return fig
