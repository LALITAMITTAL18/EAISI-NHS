"""Stage 8 — Model Comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_joblib, load_parquet
from shared.nav import render_sidebar
from shared.state import get_state, mark_stage_complete, project_datasets_dir, project_models_dir, update_state
from stages.comparison.evaluator import (
    bland_altman_stats,
    calibration_by_decile,
    compare_models,
    subgroup_eval,
)
from stages.comparison.plots import (
    bland_altman_plot,
    calibration_plot,
    equity_bar,
    metric_comparison_bar,
    roc_overlay,
    threshold_explorer_plot,
)

st.set_page_config(page_title="8 — Comparison", page_icon="📊", layout="wide")
render_sidebar()

st.title("Stage 8 — Model Comparison")
st.caption("Evaluate all trained models on the held-out test set and compare across metrics.")

state = get_state()
_BASE = Path(__file__).parent.parent
task = state.upload_cfg.get("task_type", "regression")
target = state.upload_cfg.get("target_column", "")

if not state.results_cache_path:
    st.warning("⚠️ Complete Stage 7 (Modelling) first.")
    st.stop()

results = load_joblib(project_models_dir() / state.results_cache_path)
test = load_parquet(project_datasets_dir() / state.test_data_path)
X_test = test.drop(columns=[target], errors="ignore")
y_test = test[target]

prep_cfg_raw = state.prep_cfg
outcome_thresh = (
    prep_cfg_raw.get("outcome_threshold", {}).get("threshold")
    if prep_cfg_raw and prep_cfg_raw.get("outcome_threshold", {}).get("enabled")
    else None
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
with st.spinner("Evaluating models…"):
    comparison = compare_models(results, X_test, y_test, task, outcome_thresh)

# Persist best model name
best_name = comparison.best_model_name
update_state({
    "best_model_name": best_name,
    "comparison_table": [r.model_dump() for r in comparison.rows],
    "metrics_summary": {
        "best_model": best_name,
        "best_metric": comparison.best_metric_name,
        "best_value": comparison.best_metric_value,
    },
})

st.success(
    f"🏆 Best model: **{best_name}** "
    f"({comparison.best_metric_name} = {comparison.best_metric_value:.4f})"
)

# ── Metric comparison ─────────────────────────────────────────────────────────
st.subheader("Metric comparison")
metric_options = {
    "regression": [("test_rmse", True), ("test_mae", True), ("test_r2", False)],
    "classification": [("f2", False), ("roc_auc", False), ("pr_auc", False), ("recall", False)],
    "ordinal": [("ordinal_mae", True), ("exact_accuracy", False), ("adjacent_accuracy", False)],
}
for metric, lower_is_better in metric_options.get(task, []):
    fig = metric_comparison_bar(comparison, metric, lower_is_better)
    st.plotly_chart(fig, use_container_width=True)

# ── Full comparison table ─────────────────────────────────────────────────────
st.subheader("Full results table")
comp_df = pd.DataFrame([r.model_dump() for r in comparison.rows])
comp_df = comp_df.dropna(axis=1, how="all")
st.dataframe(comp_df.set_index("model_name"), use_container_width=True)

# ── Regression-specific ───────────────────────────────────────────────────────
if task == "regression" and best_name:
    st.divider()
    st.subheader("Bland-Altman (best model)")
    best_result = next((r for r in results if r.model_name == best_name), None)
    if best_result and best_result.pipeline:
        import numpy as np
        y_pred_best = best_result.pipeline.predict(X_test)
        ba = bland_altman_stats(y_test, y_pred_best, best_name)
        cal = calibration_by_decile(y_test, y_pred_best, best_name)

        col1, col2 = st.columns(2)
        with col1:
            ba_all = [bland_altman_stats(y_test, r.pipeline.predict(X_test), r.model_name)
                      for r in results if r.pipeline]
            st.plotly_chart(bland_altman_plot(ba_all, best_name), use_container_width=True)
        with col2:
            st.plotly_chart(calibration_plot(cal, best_name), use_container_width=True)

# ── Subgroup equity analysis ──────────────────────────────────────────────────
cat_cols = test.select_dtypes(include=["object", "category"]).columns.tolist()

# ── Classification threshold explorer ─────────────────────────────────────────
if task in ("classification", "ordinal") and best_name:
    st.divider()
    st.subheader("Threshold explorer")
    st.caption(
        "Drag the threshold **k** to see how TP / FN / FP / TN change in real time. "
        "The ROC and Precision-Recall curves update to show where the current threshold sits."
    )

    model_choices = [r.model_name for r in results if r.pipeline is not None]
    th_model = st.selectbox(
        "Model to explore",
        model_choices,
        index=model_choices.index(best_name) if best_name in model_choices else 0,
        key="thresh_model",
    )
    th_result = next(r for r in results if r.model_name == th_model and r.pipeline)

    # Compute predicted probabilities
    import numpy as np
    y_prob_all = th_result.pipeline.predict_proba(X_test)
    # Take the probability of the positive class
    if y_prob_all.shape[1] == 2:
        y_prob_pos = y_prob_all[:, 1]
    else:
        # Multiclass — show one-vs-rest for the majority positive class
        y_prob_pos = y_prob_all.max(axis=1)

    k_val = st.slider(
        "Classification threshold k",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        format="%.2f",
        key="threshold_k",
    )

    st.plotly_chart(
        threshold_explorer_plot(
            y_true=np.asarray(y_test),
            y_prob=y_prob_pos,
            k=k_val,
            model_name=th_model,
        ),
        use_container_width=True,
    )

if cat_cols and best_name:
    st.divider()
    st.subheader("Equity / subgroup analysis")
    group_col = st.selectbox("Groupby column for equity analysis", cat_cols)
    metric_eq = st.selectbox("Metric", ["rmse", "mae"] if task == "regression" else ["f1", "accuracy"])
    best_result = next((r for r in results if r.model_name == best_name and r.pipeline), None)
    if best_result:
        y_pred = best_result.pipeline.predict(X_test)
        sg_results = subgroup_eval(y_test, y_pred, test[group_col].reset_index(drop=True), best_name, metric_eq)
        st.plotly_chart(equity_bar(sg_results, best_name, metric_eq), use_container_width=True)

st.divider()
if st.button("Continue to Explanation →", type="primary"):
    mark_stage_complete("comparison")
    st.switch_page("pages/9_explanation.py")
