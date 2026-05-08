"""Stage 10 — Conclusions & Export."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_joblib, load_parquet
from shared.nav import render_sidebar
from shared.state import get_state, mark_stage_complete, project_models_dir, project_reports_dir, update_state
from stages.comparison.models import ComparisonResult, MetricRow
from stages.conclusions.models import PipelineRunSummary
from stages.conclusions.plots import performance_matrix_heatmap
from stages.conclusions.reporter import (
    build_model_card,
    build_performance_matrix,
    export_comparison_csv,
    export_html_report,
)

st.set_page_config(page_title="10 — Conclusions", page_icon="🏁", layout="wide")
render_sidebar()

st.title("Stage 10 — Conclusions & Export")
st.caption("Review the best model, compare all results, document your findings and export reports.")

state = get_state()
_BASE = Path(__file__).parent.parent
task = state.upload_cfg.get("task_type", "regression")
target = state.upload_cfg.get("target_column", "")

# ── Best model card ───────────────────────────────────────────────────────────
best_model = state.best_model_name
if best_model:
    st.subheader(f"🏆 Best model: {best_model}")
    metrics = state.metrics_summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best metric", metrics.get("best_metric", "—"))
    with c2:
        st.metric("Best value", f"{metrics.get('best_value', 0):.4f}" if metrics.get("best_value") else "—")
    with c3:
        st.metric("Task", task.replace("_", " ").title())

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("Performance matrix")
comp_table = state.comparison_table
if comp_table:
    comp_df = pd.DataFrame(comp_table).dropna(axis=1, how="all")
    st.dataframe(comp_df.set_index("model_name"), use_container_width=True)

    # Heatmap
    from stages.comparison.models import ComparisonResult, MetricRow
    rows_pydantic = [MetricRow.model_validate(r) for r in comp_table]
    dummy_comparison = ComparisonResult(task=task, rows=rows_pydantic, best_model_name=best_model)
    perf_matrix = build_performance_matrix(dummy_comparison)
    st.plotly_chart(performance_matrix_heatmap(perf_matrix), use_container_width=True)

# ── Analyst notes ─────────────────────────────────────────────────────────────
st.subheader("Key findings & notes")
notes = st.text_area(
    "Document your key findings, limitations and recommendations",
    value=state.notes,
    height=200,
    placeholder="e.g. CatBoost achieved the lowest RMSE. The main predictor of outcome was pre-op score. "
                "Class imbalance (85/15) required careful threshold tuning…",
)
if notes != state.notes:
    update_state({"notes": notes})

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export")
col_e1, col_e2, col_e3 = st.columns(3)

out_dir = project_reports_dir()

with col_e1:
    if st.button("📥 Export CSV comparison table"):
        if comp_table:
            rows_pydantic = [MetricRow.model_validate(r) for r in comp_table]
            csv_path = export_comparison_csv(
                ComparisonResult(task=task, rows=rows_pydantic),
                out_dir / "comparison.csv",
            )
            with open(csv_path, "rb") as f:
                st.download_button("Download CSV", f, file_name="comparison.csv", mime="text/csv")

with col_e2:
    if st.button("📄 Export HTML report"):
        if comp_table and state.results_cache_path:
            rows_pydantic = [MetricRow.model_validate(r) for r in comp_table]
            comparison = ComparisonResult(task=task, rows=rows_pydantic, best_model_name=best_model)
            summary = PipelineRunSummary(
                file_name=state.upload_cfg.get("file_name", "—"),
                target_column=target,
                task_type=task,
                n_rows_raw=0,
                n_rows_train=0,
                n_rows_test=0,
                n_features=0,
                n_models_trained=len(comp_table),
                best_model=best_model or "—",
                best_metric_name=state.metrics_summary.get("best_metric", "—"),
                best_metric_value=state.metrics_summary.get("best_value", 0.0),
                key_findings=notes,
            )
            html_path = export_html_report(summary, comparison, out_dir / "report.html")
            with open(html_path, "rb") as f:
                st.download_button("Download HTML", f, file_name="report.html", mime="text/html")

with col_e3:
    if state.pipeline_config_path:
        cfg_path = project_models_dir() / state.pipeline_config_path
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                st.download_button(
                    "📋 Download pipeline config (JSON)",
                    f,
                    file_name="pipeline_config.json",
                    mime="application/json",
                )

st.divider()
if not state.stage_complete.get("conclusions"):
    if st.button("Mark pipeline complete ✓", type="primary"):
        mark_stage_complete("conclusions")
        st.rerun()
else:
    st.success("🎉 Pipeline complete! All stages finished.")
