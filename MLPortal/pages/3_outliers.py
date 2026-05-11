"""Stage 3 — Outlier Detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_parquet, save_parquet
from shared.nav import render_sidebar
from shared.state import get_state, mark_stage_complete, project_datasets_dir, update_state
from stages.outliers.detector import apply_outlier_config
from stages.outliers.models import ColumnOutlierSpec, OutlierConfig
from stages.outliers.plots import flag_summary_bar, scatter_with_outliers

st.set_page_config(page_title="3 — Outliers", page_icon="🔎", layout="wide")
render_sidebar()

st.title("Stage 3 — Outlier Detection")
st.caption(
    "Rows flagged by **both** the IQR and Z-score rules are confirmed outliers. "
    "Choose to keep, remove or winsorize per column."
)

state = get_state()
_BASE = Path(__file__).parent.parent

if not state.stage_complete.get("upload"):
    st.warning("⚠️ Complete Stage 1 first.")
    st.stop()

df = load_parquet(project_datasets_dir() / state.raw_data_path)
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# ── Global config ─────────────────────────────────────────────────────────────
st.subheader("Detection parameters")
col_a, col_b = st.columns(2)
with col_a:
    iqr_factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
with col_b:
    z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)

require_both = st.checkbox(
    "Require both flags (recommended — reduces false positives)", value=True
)

# ── Per-column table ──────────────────────────────────────────────────────────
st.subheader("Per-column actions")
st.caption("Configure handling for each numeric column. Exempt columns with known valid extreme values.")

col_specs: list[ColumnOutlierSpec] = []
actions: dict[str, str] = {}
exemptions: set[str] = set()

with st.form("outlier_config"):
    for col in numeric_cols:
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            st.markdown(f"**{col}**")
        with c2:
            action = st.selectbox(
                "Action", ["keep", "remove", "winsorize"],
                key=f"act_{col}", label_visibility="collapsed"
            )
            actions[col] = action
        with c3:
            exempt = st.checkbox("Exempt", key=f"ex_{col}")
            if exempt:
                exemptions.add(col)
        col_specs.append(ColumnOutlierSpec(column=col, action=action, exempt=exempt))

    submitted = st.form_submit_button("Apply & Preview", type="primary")

if submitted:
    config = OutlierConfig(
        iqr_factor=iqr_factor,
        zscore_threshold=z_thresh,
        require_both_flags=require_both,
        column_specs=col_specs,
    )

    df_clean, result = apply_outlier_config(df, config)

    st.info(
        f"**{result.n_removed:,} rows removed** "
        f"({result.n_rows_before:,} → {result.n_rows_after:,})"
    )

    # Summary chart
    st.plotly_chart(flag_summary_bar(result.column_results), use_container_width=True)

    # Per-column scatter
    st.subheader("Inspect a column")
    col_inspect = st.selectbox("Column", numeric_cols, key="inspect_col")
    cr = next((r for r in result.column_results if r.column == col_inspect), None)
    if cr and cr.lower_bound is not None:
        st.plotly_chart(
            scatter_with_outliers(df[col_inspect], cr.lower_bound, cr.upper_bound),
            use_container_width=True,
        )

    if st.button("Save & Continue →", type="primary"):
        out_path = project_datasets_dir() / "2_outliers.parquet"
        save_parquet(df_clean, out_path)
        update_state({
            "outlier_data_path": out_path.name,
            "outlier_cfg": config.model_dump(),
        })
        mark_stage_complete("outliers")
        st.switch_page("pages/4_missing.py")
