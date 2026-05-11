"""Stage 5 — Feature Engineering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_parquet, save_parquet
from shared.nav import render_sidebar
from shared.state import get_state, mark_stage_complete, project_datasets_dir, update_state
from stages.explore.stats import target_correlation
from stages.features.engineer import add_binning, add_derived, add_transforms
from stages.features.models import (
    BinningSpec,
    DerivedFeatureSpec,
    FeatureConfig,
    TransformSpec,
)
from stages.features.plots import distribution_before_after, feature_correlation_bar

st.set_page_config(page_title="5 — Feature Engineering", page_icon="⚙️", layout="wide")
render_sidebar()

st.title("Stage 5 — Feature Engineering")
st.caption(
    "Create derived features, polynomial interactions, binning, transforms, "
    "aggregations and target encoding. All train-only operations are applied in Stage 6."
)

state = get_state()
_BASE = Path(__file__).parent.parent
target = state.upload_cfg.get("target_column", "")

prev_path = state.imputed_data_path or state.outlier_data_path or state.raw_data_path
if not prev_path:
    st.warning("⚠️ Complete Stage 1 first.")
    st.stop()

df = load_parquet(project_datasets_dir() / prev_path)
numeric_cols = df.select_dtypes(include="number").columns.tolist()

feature_cfg = FeatureConfig()

# ── Columns to drop ───────────────────────────────────────────────────────────
st.subheader("Drop columns")
cols_to_drop = st.multiselect(
    "Select columns to remove (ID columns, admin columns, post-outcome leakage)",
    options=[c for c in df.columns if c != target],
)
feature_cfg = feature_cfg.model_copy(update={"columns_to_drop": cols_to_drop})

# ── Derived features ──────────────────────────────────────────────────────────
st.subheader("Derived features (formula editor)")
st.caption("Use column names directly. Example: `col_a - col_b` or `48 - pre_op_score`")

n_derived = st.number_input("Number of derived features", 0, 10, 0, step=1)
derived_specs: list[DerivedFeatureSpec] = []
for i in range(int(n_derived)):
    c1, c2 = st.columns([1, 2])
    with c1:
        name = st.text_input(f"Feature {i+1} name", key=f"dn_{i}")
    with c2:
        formula = st.text_input(f"Formula {i+1}", key=f"df_{i}")
    if name and formula:
        derived_specs.append(DerivedFeatureSpec(name=name, formula=formula))
feature_cfg = feature_cfg.model_copy(update={"derived": derived_specs})

# ── Binning ───────────────────────────────────────────────────────────────────
st.subheader("Binning")
n_bins = st.number_input("Number of columns to bin", 0, 5, 0, step=1)
bin_specs: list[BinningSpec] = []
for i in range(int(n_bins)):
    c1, c2, c3 = st.columns(3)
    with c1:
        bcol = st.selectbox(f"Column {i+1}", numeric_cols, key=f"bc_{i}")
    with c2:
        bstrat = st.selectbox("Strategy", ["equal_width", "equal_freq", "custom"], key=f"bs_{i}")
    with c3:
        bn = st.number_input("Bins", 2, 20, 4, key=f"bn_{i}")
    if bcol:
        bin_specs.append(
            BinningSpec(column=bcol, new_column=f"{bcol}_bin", strategy=bstrat, n_bins=int(bn))
        )
feature_cfg = feature_cfg.model_copy(update={"binning": bin_specs})

# ── Transforms ────────────────────────────────────────────────────────────────
st.subheader("Skewness transforms")
skewed = [
    c for c in numeric_cols
    if c != target and df[c].skew() is not None and abs(df[c].skew()) > 1.0
]
if skewed:
    cols_to_transform = st.multiselect(
        f"Columns with |skewness| > 1 (suggested)", skewed
    )
    transform_method = st.selectbox("Transform method", ["yeo_johnson", "log1p", "sqrt", "quantile"])
    transform_specs = [
        TransformSpec(column=c, method=transform_method) for c in cols_to_transform
    ]
    feature_cfg = feature_cfg.model_copy(update={"transforms": transform_specs})

    # Preview
    if cols_to_transform:
        preview_col = cols_to_transform[0]
        try:
            preview_df = add_transforms(df[[preview_col]], transform_specs[:1])
            new_col = f"{preview_col}_{transform_method}"
            if new_col in preview_df.columns:
                st.plotly_chart(
                    distribution_before_after(df[preview_col], preview_df[new_col], preview_col),
                    use_container_width=True,
                )
        except Exception:
            pass

# ── Target correlation ────────────────────────────────────────────────────────
st.divider()
st.subheader("Feature — target correlations")
if target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
    df_preview = df.copy()
    if derived_specs:
        try:
            df_preview = add_derived(df_preview, derived_specs)
        except Exception:
            pass
    tc = target_correlation(df_preview, target)
    if len(tc):
        st.plotly_chart(feature_correlation_bar(tc), use_container_width=True)

# ── Save ──────────────────────────────────────────────────────────────────────
st.divider()
if st.button("Save configuration & Continue →", type="primary"):
    out_path = project_datasets_dir() / "4_features_cfg.parquet"
    save_parquet(df, out_path)
    update_state({
        "features_data_path": out_path.name,
        "feature_cfg": feature_cfg.model_dump(),
    })
    mark_stage_complete("features")
    st.switch_page("pages/6_preparation.py")
