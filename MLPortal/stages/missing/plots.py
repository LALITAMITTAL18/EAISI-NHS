"""Plots for the Missing Data stage."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, highlight_where


def missingness_bar(summary_df: pd.DataFrame, threshold: float = 10.0) -> go.Figure:
    """Horizontal bar of % missing per column; columns > threshold are highlighted."""
    df = summary_df[summary_df["pct_missing"] > 0].copy()
    if df.empty:
        fig = base_fig(title="Missing data — no missing values found")
        return fig

    colors = highlight_where(
        df["pct_missing"].tolist(), lambda v: v > threshold
    )
    fig = base_fig(
        title="Missing data by column",
        xlabel="% Missing",
        ylabel="",
        height=max(300, len(df) * 22),
    )
    fig.add_trace(
        go.Bar(
            x=df["pct_missing"],
            y=df["column"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in df["pct_missing"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}% missing (%{customdata} rows)<extra></extra>",
            customdata=df["n_missing"],
        )
    )
    from shared.viz import add_reference_line
    add_reference_line(fig, threshold, orientation="v", label=f"{threshold}% threshold")
    high = df[df["pct_missing"] > threshold]
    subtitle = (
        f"<i><span style='color:{PALETTE.highlight};font-size:12px'>"
        f"{len(high)} columns exceed {threshold}% missing</span></i>"
        if len(high) else ""
    )
    title_text = "<b>Missing data by column</b>"
    if subtitle:
        title_text += f"<br>{subtitle}"
    fig.update_layout(
        title=dict(text=title_text),
        margin=dict(t=75 if subtitle else 60, r=90),
        xaxis=dict(range=[0, df["pct_missing"].max() * 1.2]),
    )
    return fig


def co_missing_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of co-missingness rates between column pairs."""
    cols = matrix.columns.tolist()
    fig = base_fig(title="Co-missingness heatmap (% rows where both are missing)", height=max(400, len(cols) * 25))
    fig.add_trace(
        go.Heatmap(
            z=matrix.values,
            x=cols,
            y=cols,
            colorscale=[[0.0, "#FFFFFF"], [1.0, PALETTE.highlight]],
            hovertemplate="%{y} & %{x}: %{z:.1f}%<extra></extra>",
            showscale=True,
        )
    )
    return fig


def umap_scatter(umap_df: pd.DataFrame, labels: pd.Series | None = None) -> go.Figure:
    """2D UMAP scatter of missingness patterns.

    Each point is a row; color shows cluster label (or all gray if no labels).
    """
    fig = base_fig(
        title="UMAP — missingness pattern clusters",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        height=480,
    )
    if labels is None or labels.nunique() <= 1:
        fig.add_trace(
            go.Scatter(
                x=umap_df["umap_1"],
                y=umap_df["umap_2"],
                mode="markers",
                marker=dict(color=PALETTE.gray, size=4, opacity=0.5),
                hovertemplate="Row %{pointIndex}<extra></extra>",
            )
        )
    else:
        for label in sorted(labels.unique()):
            mask = labels == label
            color = PALETTE.highlight if label == labels.value_counts().idxmax() else PALETTE.gray
            fig.add_trace(
                go.Scatter(
                    x=umap_df.loc[mask, "umap_1"],
                    y=umap_df.loc[mask, "umap_2"],
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.6),
                    name=f"Cluster {label}",
                )
            )
    annotate_message(fig, "Each point is a row; position reflects missing-data pattern")
    return fig
