"""Pure functions for generating reports and exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stages.comparison.models import ComparisonResult
from stages.conclusions.models import ModelCard, PerformanceMatrix, PipelineRunSummary
from stages.modelling.models import TrainResult


def build_model_card(
    result: TrainResult,
    feature_importances: list[dict] | None = None,
    notes: str = "",
) -> ModelCard:
    """Build a ModelCard from a TrainResult."""
    return ModelCard(
        model_name=result.model_name,
        task=result.task,
        best_params=result.best_params,
        test_metrics=result.test_metrics,
        feature_importances=feature_importances or [],
        notes=notes,
    )


def build_performance_matrix(comparison: ComparisonResult) -> PerformanceMatrix:
    """Flatten ComparisonResult into a PerformanceMatrix for export."""
    if not comparison.rows:
        return PerformanceMatrix(models=[], metrics=[], values=[])

    first = comparison.rows[0].model_dump()
    metric_keys = [k for k, v in first.items() if isinstance(v, float) and k != "model_name"]

    models = [r.model_name for r in comparison.rows]
    values: list[list[float | None]] = []
    for row in comparison.rows:
        d = row.model_dump()
        values.append([d.get(m) for m in metric_keys])

    return PerformanceMatrix(models=models, metrics=metric_keys, values=values)


def build_run_summary(
    upload_cfg: dict[str, Any],
    split_result: Any,
    comparison: ComparisonResult,
    n_rows_raw: int,
    prep_steps: list[str],
    notes: str = "",
) -> PipelineRunSummary:
    """Assemble the final PipelineRunSummary."""
    return PipelineRunSummary(
        file_name=upload_cfg.get("file_name", "unknown"),
        target_column=upload_cfg.get("target_column", "unknown"),
        task_type=upload_cfg.get("task_type", "unknown"),
        n_rows_raw=n_rows_raw,
        n_rows_train=getattr(split_result, "n_train", 0),
        n_rows_test=getattr(split_result, "n_test", 0),
        n_features=getattr(split_result, "n_features", 0),
        n_models_trained=len(comparison.rows),
        best_model=comparison.best_model_name or "—",
        best_metric_name=comparison.best_metric_name or "—",
        best_metric_value=comparison.best_metric_value or 0.0,
        data_preparation_steps=prep_steps,
        key_findings=notes,
    )


def export_comparison_csv(
    comparison: ComparisonResult,
    path: str | Path,
) -> Path:
    """Write the comparison table to a CSV file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in comparison.rows])
    df.to_csv(dest, index=False)
    return dest


def export_html_report(
    summary: PipelineRunSummary,
    comparison: ComparisonResult,
    path: str | Path,
) -> Path:
    """Write a minimal HTML report containing the run summary and comparison table."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    rows_html = "\n".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row.model_dump().values()) + "</tr>"
        for row in comparison.rows
    )
    headers_html = "".join(
        f"<th>{k}</th>" for k in comparison.rows[0].model_dump().keys()
    ) if comparison.rows else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Pipeline Report — {summary.file_name}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; color: #2D2D2D; }}
    h1 {{ color: #E63946; }} h2 {{ color: #457B9D; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #457B9D; color: white; padding: 8px; text-align: left; }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #DEDEDE; }}
    tr:nth-child(even) {{ background: #F9F9F9; }}
    .highlight {{ color: #E63946; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>ML Pipeline Report</h1>
  <p><b>File:</b> {summary.file_name} &nbsp;|&nbsp;
     <b>Target:</b> {summary.target_column} &nbsp;|&nbsp;
     <b>Task:</b> {summary.task_type}</p>
  <p><b>Train rows:</b> {summary.n_rows_train:,} &nbsp;|&nbsp;
     <b>Test rows:</b> {summary.n_rows_test:,} &nbsp;|&nbsp;
     <b>Features:</b> {summary.n_features}</p>
  <p><b>Best model:</b> <span class="highlight">{summary.best_model}</span>
     ({summary.best_metric_name} = {summary.best_metric_value:.4f})</p>
  <h2>Model Comparison</h2>
  <table><thead><tr>{headers_html}</tr></thead>
  <tbody>{rows_html}</tbody></table>
  <h2>Key Findings</h2>
  <pre>{summary.key_findings}</pre>
</body>
</html>"""
    dest.write_text(html, encoding="utf-8")
    return dest
