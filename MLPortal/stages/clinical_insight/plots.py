"""Visualizations for the Clinical Insight stage."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig
from stages.clinical_insight.models import KL_DESCRIPTIONS, ShapResult


def shap_summary_bar(result: ShapResult, top_n: int = 15) -> go.Figure:
    """Horizontal bar of mean |SHAP| values; most important feature highlighted."""
    names = result.feature_names[:top_n]
    values = result.mean_abs_shap[:top_n]
    if not names:
        return base_fig(title="No SHAP data available")

    colors = [PALETTE.highlight if i == 0 else PALETTE.secondary for i in range(len(names))]
    fig = base_fig(
        title=f"Feature impact on predicted health gain — {result.model_name}",
        xlabel="Mean |SHAP value|",
        ylabel="",
        height=max(320, len(names) * 24),
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
    if names:
        annotate_message(fig, f"'{names[0]}' has the greatest influence on this prediction")
    return fig


def shap_waterfall(
    feature_names: list[str],
    shap_values: list[float],
    base_value: float,
    prediction: float,
    top_n: int = 12,
) -> go.Figure:
    """SHAP waterfall chart for a single patient's prediction."""
    pairs = sorted(zip(shap_values, feature_names), key=lambda x: abs(x[0]), reverse=True)[:top_n]
    values = [p[0] for p in pairs]
    names = [p[1] for p in pairs]
    colors = [PALETTE.highlight if v > 0 else PALETTE.secondary for v in values]

    fig = base_fig(
        title=f"SHAP waterfall — this patient's prediction (base: {base_value:.2f} → {prediction:.2f})",
        xlabel="SHAP contribution",
        ylabel="Feature",
        height=max(350, len(names) * 26),
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
    fig.add_vline(x=0, line_width=1, line_color="gray")
    annotate_message(
        fig,
        f"Positive (blue) = pushes prediction UP | Negative = pushes DOWN",
    )
    return fig


def kl_grade_gauge(kl_grade: int, confidence: float) -> go.Figure:
    """Gauge chart showing the predicted KL grade."""
    grade_colors = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c", 4: "#922b21"}
    color = grade_colors.get(kl_grade, "#95a5a6")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=kl_grade,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"KL Grade (Confidence: {confidence:.0%})<br>"
                f"<span style='font-size:0.8em'>{KL_DESCRIPTIONS.get(kl_grade, '')}</span>",
                "font": {"size": 14},
            },
            gauge={
                "axis": {"range": [0, 4], "tickvals": [0, 1, 2, 3, 4]},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 1], "color": "#d5f5e3"},
                    {"range": [1, 2], "color": "#fef9e7"},
                    {"range": [2, 3], "color": "#fef0e6"},
                    {"range": [3, 4], "color": "#fdedec"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 3,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def class_probability_bar(probabilities: list[float], class_names: list[str]) -> go.Figure:
    """Horizontal bar chart of class probabilities from the DL model."""
    grade_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#922b21"]
    colors = [grade_colors[i] if i < len(grade_colors) else "#95a5a6" for i in range(len(class_names))]
    predicted = int(np.argmax(probabilities))
    opacities = [1.0 if i == predicted else 0.45 for i in range(len(class_names))]

    fig = base_fig(
        title="KL Grade probability distribution",
        xlabel="Probability",
        ylabel="Grade",
        height=220,
    )
    fig.add_trace(
        go.Bar(
            x=probabilities,
            y=class_names,
            orientation="h",
            marker=dict(
                color=colors,
                opacity=opacities,
            ),
            text=[f"{p:.1%}" for p in probabilities],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    return fig


def health_gain_meter(prediction: float, reference_low: float = 0, reference_high: float = 20) -> go.Figure:
    """Gauge showing the predicted health gain score."""
    clamped = max(reference_low, min(reference_high, prediction))
    pct = (clamped - reference_low) / max(reference_high - reference_low, 1)

    if pct < 0.25:
        color, label = "#e74c3c", "Low benefit predicted"
    elif pct < 0.5:
        color, label = "#e67e22", "Moderate benefit predicted"
    elif pct < 0.75:
        color, label = "#f1c40f", "Good benefit predicted"
    else:
        color, label = "#2ecc71", "High benefit predicted"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prediction, 2),
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"Predicted Health Gain (OKS improvement)<br>"
                f"<span style='font-size:0.85em;color:{color}'>{label}</span>",
                "font": {"size": 14},
            },
            number={"suffix": " pts", "font": {"size": 28}},
            gauge={
                "axis": {"range": [reference_low, reference_high]},
                "bar": {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [reference_low, reference_low + (reference_high - reference_low) * 0.25], "color": "#fdedec"},
                    {"range": [reference_low + (reference_high - reference_low) * 0.25, reference_low + (reference_high - reference_low) * 0.5], "color": "#fef0e6"},
                    {"range": [reference_low + (reference_high - reference_low) * 0.5, reference_low + (reference_high - reference_low) * 0.75], "color": "#fef9e7"},
                    {"range": [reference_low + (reference_high - reference_low) * 0.75, reference_high], "color": "#d5f5e3"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(t=60, b=20, l=20, r=20))
    return fig
