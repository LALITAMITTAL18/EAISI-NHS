"""Stage 9 — Model Explanation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_joblib, load_parquet
from shared.nav import render_sidebar
from shared.state import get_state, list_trained_variants, mark_stage_complete, project_datasets_dir, project_models_dir
from stages.explanation.explainer import (
    compute_native_importance,
    compute_pdp,
    compute_permutation_importance,
    compute_shap,
    predict_single,
)
from stages.explanation.models import ExplanationConfig
from stages.explanation.plots import pdp_plot, shap_summary_bar, waterfall_plot

st.set_page_config(page_title="9 — Explanation", page_icon="🔬", layout="wide")
render_sidebar()

st.title("Stage 9 — Model Explanation")
st.caption("Understand why the model makes its predictions using SHAP, permutation importance and PDP.")

state = get_state()
_BASE = Path(__file__).parent.parent

# ── Variant selector ──────────────────────────────────────────────────────────
trained_variants = list_trained_variants()
if not trained_variants:
    st.warning("⚠️ No trained model results found. Complete Stage 7 (Modelling) first.")
    st.stop()

variant_names = [v["name"] for v in trained_variants]
_active_slug = state.active_dataset
_default_idx = next((i for i, v in enumerate(trained_variants) if v["slug"] == _active_slug), 0)

st.subheader("Select trained variant to explain")
_sel_name = st.selectbox(
    "Dataset variant",
    options=variant_names,
    index=_default_idx,
    help="Each trained variant has its own saved results. Select which one to explain.",
)
_sel_variant = next(v for v in trained_variants if v["name"] == _sel_name)

target = _sel_variant["target_column"]
_cache_file = _sel_variant["cache_path"]
_test_path = _sel_variant["test_path"]

if not _test_path:
    st.error("No test data path found for this variant.")
    st.stop()

results = load_joblib(project_models_dir() / _cache_file)
test = load_parquet(project_datasets_dir() / _test_path)
X_test = test.drop(columns=[target], errors="ignore")
y_test = test[target]

model_names = [r.model_name for r in results if r.pipeline is not None]
default_model = state.best_model_name if state.best_model_name in model_names else model_names[0]

# ── Model selector ────────────────────────────────────────────────────────────
selected_model = st.selectbox("Select model to explain", model_names, index=model_names.index(default_model))
result = next(r for r in results if r.model_name == selected_model)
pipeline = result.pipeline

cfg = ExplanationConfig(model_name=selected_model, max_shap_samples=500)

# ── SHAP values ───────────────────────────────────────────────────────────────
st.subheader("SHAP feature importance")
try:
    import shap  # noqa: F401
    top_n = st.slider("Top N features", 5, 30, 20)
    cfg = cfg.model_copy(update={"top_n_features": top_n})

    with st.spinner("Computing SHAP values…"):
        shap_result = compute_shap(pipeline, X_test, cfg)

    st.plotly_chart(shap_summary_bar(shap_result, top_n), use_container_width=True)

    # Waterfall for a single row
    st.subheader("Waterfall — single prediction")
    row_idx = st.slider("Row index", 0, len(X_test) - 1, 0)
    row_shap = shap_result.shap_values[row_idx] if row_idx < len(shap_result.shap_values) else []
    if row_shap:
        y_pred_single = pipeline.predict(X_test.iloc[[row_idx]])[0]
        base_val = float(y_test.mean())
        st.plotly_chart(
            waterfall_plot(shap_result.feature_names, row_shap, base_val, float(y_pred_single)),
            use_container_width=True,
        )
except ImportError:
    st.info("Install `shap` to enable SHAP explanations: `pip install shap`")

# ── Native / permutation importance ──────────────────────────────────────────
st.subheader("Feature importance (native & permutation)")
tab_nat, tab_perm = st.tabs(["Native (model built-in)", "Permutation (model-agnostic)"])

with tab_nat:
    nat = compute_native_importance(pipeline, X_test)
    if nat:
        imp_df = pd.DataFrame([r.model_dump() for r in nat[:20]])
        st.bar_chart(imp_df.set_index("feature")["importance"])
    else:
        st.info("This model does not expose native feature importances.")

with tab_perm:
    if st.button("Compute permutation importance (slow for large datasets)"):
        with st.spinner("Computing…"):
            perm = compute_permutation_importance(pipeline, X_test, y_test)
        perm_df = pd.DataFrame([r.model_dump() for r in perm[:20]])
        st.bar_chart(perm_df.set_index("feature")["importance"])

# ── Partial dependence ────────────────────────────────────────────────────────
st.subheader("Partial Dependence Plot (PDP)")
numeric_features = X_test.select_dtypes(include="number").columns.tolist()
if numeric_features:
    pdp_feature = st.selectbox("Feature for PDP", numeric_features)
    with st.spinner("Computing PDP…"):
        pdp_result = compute_pdp(pipeline, X_test, pdp_feature)
    st.plotly_chart(pdp_plot(pdp_result), use_container_width=True)

# ── Live prediction explorer ──────────────────────────────────────────────────
st.subheader("Live prediction explorer")
st.caption("Enter feature values and get an instant prediction.")
with st.form("predict_form"):
    row_input: dict = {}
    cols = st.columns(min(4, len(X_test.columns)))
    for i, col in enumerate(X_test.columns[:20]):
        with cols[i % len(cols)]:
            val = st.text_input(col, value=str(X_test[col].median() if pd.api.types.is_numeric_dtype(X_test[col]) else X_test[col].mode()[0]), key=f"pred_{col}")
            try:
                row_input[col] = float(val)
            except ValueError:
                row_input[col] = val

    if st.form_submit_button("Predict", type="primary"):
        pred = predict_single(pipeline, row_input)
        st.metric("Prediction", f"{pred:.4f}" if isinstance(pred, float) else str(pred))

st.divider()
if st.button("Continue to Conclusions →", type="primary"):
    mark_stage_complete("explanation")
    st.switch_page("pages/10_conclusions.py")
