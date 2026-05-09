"""Stage 8 — Model Comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_parquet
from shared.nav import render_sidebar
from shared.state import get_state, list_trained_variants, load_variant_results, mark_stage_complete, project_datasets_dir, project_models_dir, update_state
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

# ── Variant selector ──────────────────────────────────────────────────────────
trained_variants = list_trained_variants()
if not trained_variants:
    st.warning("⚠️ No trained model results found. Complete Stage 7 (Modelling) first.")
    st.stop()

variant_names = [v["name"] for v in trained_variants]
# Default to the variant that matches the current active dataset, otherwise first
_active_slug = state.active_dataset
_default_idx = next((i for i, v in enumerate(trained_variants) if v["slug"] == _active_slug), 0)

st.subheader("Select trained variant to compare")
_sel_name = st.selectbox(
    "Dataset variant",
    options=variant_names,
    index=_default_idx,
    help="Each trained variant has its own saved results. Select which one to evaluate.",
)
_sel_variant = next(v for v in trained_variants if v["name"] == _sel_name)

task = _sel_variant["task_type"]
target = _sel_variant["target_column"]
_cache_file = _sel_variant["cache_path"]
_test_path = _sel_variant["test_path"]

if not _test_path:
    st.error("No test data path found for this variant.")
    st.stop()

results = load_variant_results(_sel_variant["slug"], project_models_dir(), _sel_variant["task_type"])
test = load_parquet(project_datasets_dir() / _test_path)
X_test = test.drop(columns=[target], errors="ignore")
y_test = test[target]

prep_cfg_raw = state.prep_cfg
outcome_thresh = (
    prep_cfg_raw.get("outcome_threshold", {}).get("threshold")
    if prep_cfg_raw and prep_cfg_raw.get("outcome_threshold", {}).get("enabled")
    else None
)

# ── Evaluate selected variant ─────────────────────────────────────────────────
with st.spinner("Evaluating models…"):
    comparison = compare_models(
        results, X_test, y_test, task, outcome_thresh, dataset=_sel_variant["name"]
    )

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

# ── Cross-variant full results table (all datasets × all models) ──────────────
st.subheader("Full results — all datasets × all models")
st.caption(
    "Every variant that has been trained is evaluated here against its own test set. "
    "Use the 'Dataset variant' selector above to focus the charts below on one variant."
)

all_rows: list[dict] = []
with st.spinner("Loading results from all trained variants…"):
    for _tv in trained_variants:
        try:
            _tv_results = load_variant_results(_tv["slug"], project_models_dir(), _tv["task_type"])
            _tv_test_path = _tv["test_path"]
            if not _tv_test_path:
                continue
            _tv_test = load_parquet(project_datasets_dir() / _tv_test_path)
            _tv_target = _tv["target_column"]
            _tv_task = _tv["task_type"]
            if _tv_target not in _tv_test.columns:
                continue
            _tv_X = _tv_test.drop(columns=[_tv_target], errors="ignore")
            _tv_y = _tv_test[_tv_target]
            _tv_comp = compare_models(
                _tv_results, _tv_X, _tv_y, _tv_task, dataset=_tv["name"]
            )
            all_rows.extend([r.model_dump() for r in _tv_comp.rows])
        except Exception as _e:
            st.warning(f"Could not load results for **{_tv['name']}**: {_e}")

if all_rows:
    all_df = pd.DataFrame(all_rows)
    _front = [c for c in ["dataset", "model_name", "task"] if c in all_df.columns]
    _rest = [c for c in all_df.columns if c not in _front]
    all_df = all_df[_front + _rest].dropna(axis=1, how="all")
    st.dataframe(all_df.set_index(["dataset", "model_name"]), use_container_width=True)
else:
    st.info("No cross-variant results available.")

# ── Metric comparison (selected variant) ──────────────────────────────────────
st.divider()
st.subheader(f"Metric comparison — {_sel_variant['name']}")
metric_options = {
    "regression": [("test_rmse", True), ("test_mae", True), ("test_r2", False)],
    "classification": [("f2", False), ("roc_auc", False), ("pr_auc", False), ("recall", False)],
    "ordinal": [("ordinal_mae", True), ("exact_accuracy", False), ("adjacent_accuracy", False)],
}
for metric, lower_is_better in metric_options.get(task, []):
    fig = metric_comparison_bar(comparison, metric, lower_is_better)
    st.plotly_chart(fig, use_container_width=True)

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
