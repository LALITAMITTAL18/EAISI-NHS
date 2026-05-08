"""Shared visualization utilities — single source of truth for all chart styling.

Rules (Storytelling with Data):
- Data that is NOT the message → gray (#BFBFBF)
- The message / primary highlight → highlight (#E63946)
- Secondary comparison series → secondary (#457B9D)
- No gridlines, no chart junk, minimal axis decoration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import plotly.graph_objects as go


@dataclass(frozen=True)
class StoryPalette:
    """Immutable color palette for all charts."""

    gray: str = "#BFBFBF"
    highlight: str = "#E63946"
    secondary: str = "#457B9D"
    positive: str = "#2A9D8F"
    background: str = "#FFFFFF"
    text: str = "#2D2D2D"
    light_bg: str = "#F9F9F9"
    axis_line: str = "#DEDEDE"


PALETTE = StoryPalette()


# ── Color helpers ─────────────────────────────────────────────────────────────


def gray_series(n: int) -> list[str]:
    """Return *n* gray color strings — the default for non-message data."""
    return [PALETTE.gray] * n


def highlight_at(n: int, idx: int, color: str | None = None) -> list[str]:
    """Gray list with a single accent at position *idx*."""
    colors = gray_series(n)
    colors[idx] = color or PALETTE.highlight
    return colors


def highlight_min(values: Sequence[float], color: str | None = None) -> list[str]:
    """Highlight the minimum value; all others gray."""
    min_idx = int(min(range(len(values)), key=lambda i: values[i]))
    return highlight_at(len(values), min_idx, color)


def highlight_max(values: Sequence[float], color: str | None = None) -> list[str]:
    """Highlight the maximum value; all others gray."""
    max_idx = int(max(range(len(values)), key=lambda i: values[i]))
    return highlight_at(len(values), max_idx, color)


def highlight_where(
    values: Sequence[Any],
    condition_fn: Callable[[Any], bool],
    highlight_color: str | None = None,
    default_color: str | None = None,
) -> list[str]:
    """Return per-value colors: accent where condition_fn is True, gray elsewhere."""
    hc = highlight_color or PALETTE.highlight
    dc = default_color or PALETTE.gray
    return [hc if condition_fn(v) else dc for v in values]


def two_tone(
    condition: Sequence[bool],
    true_color: str | None = None,
    false_color: str | None = None,
) -> list[str]:
    """Map a boolean sequence to two colors (highlight / secondary)."""
    tc = true_color or PALETTE.highlight
    fc = false_color or PALETTE.secondary
    return [tc if b else fc for b in condition]


# ── Figure factory ────────────────────────────────────────────────────────────


def base_fig(
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    height: int = 420,
) -> go.Figure:
    """Clean Plotly figure — no gridlines, no chart junk."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, color=PALETTE.text),
            x=0,
            xref="paper",
        ),
        xaxis=dict(
            title=dict(text=xlabel, font=dict(color=PALETTE.text, size=12)),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=PALETTE.axis_line,
            tickfont=dict(color=PALETTE.text, size=11),
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color=PALETTE.text, size=12)),
            showgrid=False,
            zeroline=False,
            showline=False,
            tickfont=dict(color=PALETTE.text, size=11),
        ),
        plot_bgcolor=PALETTE.background,
        paper_bgcolor=PALETTE.background,
        font=dict(color=PALETTE.text),
        height=height,
        margin=dict(l=50, r=20, t=60 if title else 20, b=50),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=PALETTE.text, size=11),
        ),
        hoverlabel=dict(
            bgcolor=PALETTE.background,
            font_color=PALETTE.text,
            bordercolor=PALETTE.axis_line,
        ),
    )
    return fig


def annotate_message(
    fig: go.Figure,
    text: str,
    x: float = 0.0,
    y: float = 1.05,
    color: str | None = None,
    size: int = 13,
) -> go.Figure:
    """Add a 'so-what' text annotation — the takeaway message on the chart."""
    fig.add_annotation(
        x=x,
        y=y,
        text=f"<i>{text}</i>",
        showarrow=False,
        font=dict(color=color or PALETTE.highlight, size=size),
        xref="paper",
        yref="paper",
        xanchor="left",
    )
    return fig


def add_reference_line(
    fig: go.Figure,
    value: float,
    orientation: str = "h",
    label: str = "",
    color: str | None = None,
) -> go.Figure:
    """Add a horizontal or vertical reference line."""
    line_color = color or PALETTE.secondary
    if orientation == "h":
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=line_color,
            line_width=1.5,
            annotation_text=label,
            annotation_font_color=line_color,
        )
    else:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=line_color,
            line_width=1.5,
            annotation_text=label,
            annotation_font_color=line_color,
        )
    return fig
