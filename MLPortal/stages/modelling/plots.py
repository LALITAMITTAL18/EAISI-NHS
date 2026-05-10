"""Plots for the Modelling stage."""

from __future__ import annotations

import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig, gray_series, highlight_at


def optuna_history_plot(history: list[dict], model_name: str) -> go.Figure:
    """Line chart of Optuna trial values.

    The best trial is highlighted with a red marker; all others are gray dots.
    """
    if not history:
        return base_fig(title=f"Optuna history — {model_name}")

    trials = [h["trial"] for h in history]
    values = [h["value"] for h in history]
    best_idx = max(range(len(values)), key=lambda i: values[i])

    fig = base_fig(
        title=f"Optuna HPO — {model_name}",
        xlabel="Trial",
        ylabel="CV score",
    )
    fig.add_trace(
        go.Scatter(
            x=trials,
            y=values,
            mode="markers",
            marker=dict(color=PALETTE.gray, size=5),
            name="Trial",
            hovertemplate="Trial %{x}: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trials[best_idx]],
            y=[values[best_idx]],
            mode="markers",
            marker=dict(color=PALETTE.highlight, size=12, symbol="star"),
            name=f"Best (trial {trials[best_idx]})",
            hovertemplate=f"Best: %{{y:.4f}}<extra></extra>",
        )
    )
    annotate_message(fig, f"Best score: {values[best_idx]:.4f} at trial {trials[best_idx]}")
    return fig


def training_summary_bar(
    model_names: list[str],
    cv_scores: list[float],
    metric_label: str = "CV score",
) -> go.Figure:
    """Bar chart of CV scores across all trained models; best is highlighted."""
    if not model_names:
        return base_fig(title="Training summary")

    best_idx = max(range(len(cv_scores)), key=lambda i: cv_scores[i])
    colors = highlight_at(len(model_names), best_idx)

    fig = base_fig(
        title="Cross-validation scores by model",
        xlabel=metric_label,
        ylabel="",
        height=max(320, len(model_names) * 32),
    )
    fig.add_trace(
        go.Bar(
            x=cv_scores,
            y=model_names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in cv_scores],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    annotate_message(
        fig,
        f"'{model_names[best_idx]}' achieves the best CV score ({cv_scores[best_idx]:.4f})",
    )
    return fig
