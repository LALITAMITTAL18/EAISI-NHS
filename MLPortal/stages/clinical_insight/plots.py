"""Visualizations for the Clinical Insight stage."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from shared.viz import PALETTE, annotate_message, base_fig
from stages.clinical_insight.models import KL_DESCRIPTIONS, ShapResult


def _clean_name(name: str) -> str:
    """Strip sklearn ColumnTransformer prefixes and underscores for display."""
    for prefix in ("numeric__", "categorical__", "remainder__", "cat__", "num__", "passthrough__"):
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace("_", " ").strip()


def shap_summary_bar(result: ShapResult, top_n: int = 15) -> go.Figure:
    """Horizontal bar of mean |SHAP| values; most important feature highlighted."""
    names = [_clean_name(n) for n in result.feature_names[:top_n]]
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


def shap_force_plot(
    feature_names: list[str],
    shap_values: list[float],
    base_value: float,
    prediction: float,
    top_n: int = 12,
) -> go.Figure:
    """Diverging SHAP impact bar chart — one bar per feature, sorted by impact.

    ▶ Green bars  = feature INCREASES predicted health gain.
    ◀ Red bars    = feature DECREASES predicted health gain.
    Bar length    = magnitude of influence (SHAP value).
    """
    pairs = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    # Sort for display: most negative at top, most positive at bottom
    # (gives natural "waterfall" reading order)
    pairs = sorted(pairs, key=lambda x: x[1])

    names_clean = [_clean_name(n) for n, _ in pairs]
    values = [v for _, v in pairs]
    raw_names = [n for n, _ in pairs]

    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in values]
    labels = [f"{v:+.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names_clean,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="white", width=0.8),
        ),
        text=labels,
        textposition="outside",
        textfont=dict(size=10, color="#2c3e50"),
        cliponaxis=False,
        customdata=raw_names,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "SHAP impact: %{x:+.4f} pts<br>"
            "<i>Positive = increases predicted benefit</i><extra></extra>"
        ),
    ))

    # Zero-line to show direction clearly
    fig.add_vline(x=0, line_color="#bdc3c7", line_width=1.5)

    # Summary annotation in top-right corner
    direction = "▲ increases" if prediction >= base_value else "▼ decreases"
    net = prediction - base_value
    fig.add_annotation(
        xref="paper", yref="paper", x=1.0, y=1.02,
        text=(
            f"<b>Prediction: {prediction:.2f} pts</b>  ·  "
            f"Base: {base_value:.2f}  ·  "
            f"Net effect: {net:+.2f} pts"
        ),
        showarrow=False,
        font=dict(size=11, color="#2c3e50"),
        xanchor="right",
        bgcolor="rgba(240,244,255,0.9)",
        bordercolor="#c7d2fe",
        borderwidth=1,
    )

    n = len(pairs)
    fig.update_layout(
        title=dict(
            text=(
                "Feature impact on this patient's prediction  "
                "<span style='font-size:11px;color:#7f8c8d'>"
                "🟢 increases benefit  🔴 decreases benefit</span>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(
            title="SHAP value (OKS pts)",
            zeroline=False,
            showgrid=True,
            gridcolor="#f0f0f0",
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
            automargin=True,
        ),
        showlegend=False,
        height=max(320, n * 32 + 100),
        margin=dict(l=10, r=160, t=80, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def shap_waterfall(
    feature_names: list[str],
    shap_values: list[float],
    base_value: float,
    prediction: float,
    top_n: int = 12,
) -> go.Figure:
    """SHAP waterfall chart — vertical step-by-step path from base to prediction."""
    pairs = sorted(zip(shap_values, feature_names), key=lambda x: abs(x[0]), reverse=True)[:top_n]
    values = [p[0] for p in pairs]
    names = [_clean_name(p[1]) for p in pairs]

    # Build a true Plotly Waterfall
    labels = [n[:22] + "…" if len(n) > 24 else n for n in names] + [f"f(x) = {prediction:.2f}"]
    measures = ["relative"] * len(values) + ["total"]
    y_vals = values + [0]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=y_vals,
            base=base_value,
            textposition="outside",
            text=[f"{v:+.3f}" for v in values] + [f"{prediction:.2f}"],
            connector=dict(line=dict(color="#bdc3c7", dash="dot")),
            increasing=dict(marker=dict(color="#e74c3c")),
            decreasing=dict(marker=dict(color="#3498db")),
            totals=dict(marker=dict(color="#2ecc71")),
            hovertemplate="%{x}<br>SHAP: %{y:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"SHAP Waterfall — base: {base_value:.2f} → f(x): {prediction:.2f}",
            font=dict(size=13),
        ),
        xaxis_title="Feature",
        yaxis_title="Predicted health gain (OKS pts)",
        showlegend=False,
        height=max(360, len(values) * 28),
        margin=dict(l=10, r=10, t=60, b=120),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-35)
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
