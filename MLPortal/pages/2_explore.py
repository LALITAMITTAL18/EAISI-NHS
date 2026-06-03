"""Stage 2 — Data Exploration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from shared.io import load_parquet
from shared.nav import render_sidebar
from shared.state import get_state, mark_stage_complete, project_datasets_dir
from stages.explore.plots import (
    correlation_heatmap,
    distribution_plot,
    grouped_count_plot,
    grouped_distribution_plot,
    qq_plot,
    subgroup_bar,
    target_correlation_bar,
)
from stages.explore.stats import (
    correlation_matrix,
    qq_data,
    subgroup_means,
    summary_stats,
    target_correlation,
)

st.set_page_config(page_title="2 — Explore", page_icon="🔍", layout="wide")
render_sidebar()

st.title("Stage 2 — Data Exploration")
st.caption("Understand distributions, correlations, subgroup effects and data quality.")

state = get_state()
_BASE = Path(__file__).parent.parent

if not state.stage_complete.get("upload"):
    st.warning("⚠️ Complete Stage 1 (Upload) first.")
    st.stop()

df = load_parquet(project_datasets_dir() / state.raw_data_path)
_target_cfg = state.upload_cfg.get("target_column", "")
target = _target_cfg if _target_cfg in df.columns else df.columns[0]

# ── Calculated columns ────────────────────────────────────────────────────────
if "calc_cols" not in st.session_state:
    st.session_state["calc_cols"] = []

with st.expander("➕ Add calculated column", expanded=False):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        calc_name = st.text_input("Column name", placeholder="e.g. health_gain")
        calc_op = st.selectbox(
            "Operation",
            ["Difference  (A − B)", "Threshold  (A − B) < value → 0/1", "Ratio  (A / B)"],
        )
    with cc2:
        col_a = st.selectbox("Column A", options=numeric_cols, key="calc_col_a")
        col_b = st.selectbox("Column B", options=numeric_cols, key="calc_col_b")
    with cc3:
        threshold_val = st.number_input("Threshold value", value=7.0, key="calc_thresh")
        threshold_op = st.radio("Comparison", ["<", "<=", ">", ">="], horizontal=True, key="calc_op")

    if st.button("Add column"):
        if calc_name and calc_name not in df.columns:
            st.session_state["calc_cols"].append(
                {"name": calc_name, "op": calc_op, "col_a": col_a, "col_b": col_b,
                 "threshold": threshold_val, "threshold_op": threshold_op}
            )
            st.success(f"Added '{calc_name}'")
        elif not calc_name:
            st.error("Enter a column name.")
        else:
            st.error(f"Column '{calc_name}' already exists.")

    if st.session_state["calc_cols"]:
        st.write("**Defined columns:**")
        for i, cc in enumerate(st.session_state["calc_cols"]):
            col_l, col_r = st.columns([4, 1])
            with col_l:
                st.text(f"{cc['name']}  ({cc['op']}  |  {cc['col_a']} , {cc['col_b']})")
            with col_r:
                if st.button("Remove", key=f"rm_calc_{i}"):
                    st.session_state["calc_cols"].pop(i)
                    st.rerun()

# Apply calculated columns to a working copy of df
df = df.copy()
for cc in st.session_state["calc_cols"]:
    a, b = df[cc["col_a"]], df[cc["col_b"]]
    op = cc["op"]
    if op.startswith("Difference"):
        df[cc["name"]] = a - b
    elif op.startswith("Ratio"):
        df[cc["name"]] = a / b.replace(0, pd.NA)
    elif op.startswith("Threshold"):
        diff = a - b
        tv = cc["threshold"]
        cmp = cc["threshold_op"]
        if cmp == "<":
            df[cc["name"]] = (diff < tv).astype(int)
        elif cmp == "<=":
            df[cc["name"]] = (diff <= tv).astype(int)
        elif cmp == ">":
            df[cc["name"]] = (diff > tv).astype(int)
        else:
            df[cc["name"]] = (diff >= tv).astype(int)

# ── Summary statistics ────────────────────────────────────────────────────────
st.subheader("Summary statistics")
stats = summary_stats(df)
import pandas as pd
stats_df = pd.DataFrame([s.model_dump() for s in stats])
st.dataframe(stats_df, use_container_width=True)

st.divider()

# ── Distribution ──────────────────────────────────────────────────────────────
st.subheader("Column distributions")
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
with ctrl1:
    cols = df.columns.tolist()
    default_idx = cols.index(target) if target in cols else 0
    col_choice = st.selectbox("Select column", options=cols, index=default_idx)
with ctrl2:
    group_options = ["(none)"] + [c for c in df.columns if c != col_choice]
    group_choice = st.selectbox("Group by", options=group_options)
with ctrl3:
    if group_choice != "(none)":
        view_mode = st.radio("View as", ["Counts", "Distribution"], horizontal=True)
    else:
        view_mode = "Distribution"

if group_choice != "(none)":
    if view_mode == "Counts":
        normalize = st.toggle("Show as % of group", value=False)
        st.plotly_chart(
            grouped_count_plot(df, col_choice, group_choice, normalize=normalize),
            use_container_width=True,
        )
    else:
        plot_type = st.radio("Chart type", ["box", "violin"], horizontal=True)
        st.plotly_chart(
            grouped_distribution_plot(df, col_choice, group_choice, plot_type=plot_type),
            use_container_width=True,
        )
else:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.plotly_chart(distribution_plot(df[col_choice], target=target), use_container_width=True)
    with col_d2:
        if pd.api.types.is_numeric_dtype(df[col_choice]):
            th, obs = qq_data(df[col_choice])
            st.plotly_chart(qq_plot(th, obs, col_choice), use_container_width=True)

st.divider()

# ── Correlation ───────────────────────────────────────────────────────────────
st.subheader("Correlations")
tab_corr, tab_target = st.tabs(["Matrix", "Feature–target"])
with tab_corr:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    corr = correlation_matrix(df, method=method)
    highlight = target if target in df.select_dtypes(include="number").columns else None
    st.plotly_chart(correlation_heatmap(corr, target=highlight), use_container_width=True)
with tab_target:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for correlation analysis.")
    else:
        proxy_default = target if target in numeric_cols else numeric_cols[0]
        if target not in numeric_cols:
            st.info(
                f"The configured target column **{target!r}** is not numeric or not yet "
                "available in the raw data. Select a numeric column below to explore "
                "correlations — useful for identifying potential outcome variables."
            )
        proxy_col = st.selectbox(
            "Correlate features against",
            options=numeric_cols,
            index=numeric_cols.index(proxy_default),
            key="proxy_target",
        )
        tc_method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="tc_method")
        tc = target_correlation(df, proxy_col, method=tc_method)
        if len(tc):
            st.plotly_chart(target_correlation_bar(tc), use_container_width=True)
        else:
            st.info("No numeric features found to correlate.")

st.divider()

# ── Subgroup analysis ─────────────────────────────────────────────────────────
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
if cat_cols and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
    st.subheader("Subgroup analysis")
    group_col = st.selectbox("Group by", options=cat_cols)
    sg_df = subgroup_means(df, target, group_col)
    st.plotly_chart(subgroup_bar(sg_df, group_col, target), use_container_width=True)

st.divider()
if st.button("Mark as reviewed & Continue →", type="primary"):
    mark_stage_complete("explore")
    st.switch_page("pages/3_outliers.py")
