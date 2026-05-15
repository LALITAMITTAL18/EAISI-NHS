"""Stage 4 — Missing Data Analysis & Imputation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import load_parquet, save_parquet
from shared.nav import render_sidebar
from shared.state import get_state, mark_stage_complete, project_datasets_dir, update_state
from stages.missing.analyser import add_missing_indicators, missingness_summary, replace_sentinels
from stages.missing.models import ImputerSpec, MissingConfig
from stages.missing.plots import co_missing_heatmap, missingness_bar

st.set_page_config(page_title="4 — Missing Data", page_icon="🕳️", layout="wide")
render_sidebar()

st.title("Stage 4 — Missing Data")
st.caption("Analyse missingness patterns and configure imputation strategies per column.")

state = get_state()
_BASE = Path(__file__).parent.parent

prev_path = state.outlier_data_path or state.raw_data_path
if not prev_path:
    st.warning("⚠️ Complete Stage 1 first.")
    st.stop()

df = load_parquet(project_datasets_dir() / prev_path)
target = state.upload_cfg.get("target_column", df.columns[0])

# ── Sentinel values ───────────────────────────────────────────────────────────
st.subheader("Sentinel / coded missing values")
st.caption(
    "NHS PROMs uses **9** (and sometimes **99**) to encode *'unknown / not answered'* "
    "in Likert-scale columns. These are stored as real integers, so pandas does not "
    "detect them as missing. List any such sentinel values here — they will be treated "
    "as NaN throughout this stage and in Stage 6 imputation."
)
sentinel_raw = st.text_input(
    "Sentinel values (comma-separated integers/floats)",
    value="9",
    help="e.g. '9' or '9, 99, -1'",
)
sentinel_values: list[int | float] = []
for tok in sentinel_raw.split(","):
    tok = tok.strip()
    if tok:
        try:
            sentinel_values.append(int(tok) if "." not in tok else float(tok))
        except ValueError:
            st.warning(f"Ignoring non-numeric sentinel value: '{tok}'")

# Replace sentinels in the working copy used for analysis
df_analysis = replace_sentinels(df, sentinel_values)

st.divider()

# ── Missingness overview ──────────────────────────────────────────────────────
summary = missingness_summary(df_analysis)
n_missing_cols = int((summary["n_missing"] > 0).sum())

st.subheader("Missingness overview")
st.metric("Columns with missing values", n_missing_cols, delta=None)
st.plotly_chart(missingness_bar(summary), use_container_width=True)

# ── Co-missingness ────────────────────────────────────────────────────────────
with st.expander("Co-missingness heatmap"):
    from stages.missing.analyser import co_missing_matrix
    mat = co_missing_matrix(df_analysis)
    st.plotly_chart(co_missing_heatmap(mat), use_container_width=True)

# ── UMAP (optional) ──────────────────────────────────────────────────────────
with st.expander("Missingness pattern visualisation (UMAP — optional)"):
    try:
        from stages.missing.analyser import missingness_umap_data
        from stages.missing.plots import umap_scatter

        if st.button("Compute UMAP (may take a moment)"):
            with st.spinner("Computing UMAP embedding…"):
                umap_df = missingness_umap_data(df_analysis)
            st.plotly_chart(umap_scatter(umap_df), use_container_width=True)
    except ImportError:
        st.info(
            "Install `umap-learn` to enable this visualisation: "
            "`pip install umap-learn`"
        )

st.divider()

# ── Per-column imputation config ──────────────────────────────────────────────
st.subheader("Imputation strategies")
st.caption(
    "Configure how each column's missing values are handled. "
    "All imputers are fit on training data only (applied in Stage 6)."
)

missing_cols = summary[summary["n_missing"] > 0]["column"].tolist()
indicator_threshold = st.slider(
    "Auto-add missingness indicator flag for columns with > x% missing",
    0, 50, 2,
)

col_specs: list[ImputerSpec] = []
STRATEGIES = ["median", "mean", "most_frequent", "constant", "knn", "mice", "drop_rows", "add_indicator"]

with st.form("impute_form"):
    for col in missing_cols:
        pct = float(summary[summary["column"] == col]["pct_missing"].values[0])
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1:
            st.markdown(f"**{col}** — {pct:.1f}% missing")
        with c2:
            strategy = st.selectbox(
                "Strategy", STRATEGIES, key=f"strat_{col}", label_visibility="collapsed"
            )
        with c3:
            constant_val = st.text_input("Constant", value="", key=f"const_{col}", label_visibility="collapsed")
        with c4:
            add_ind = st.checkbox("Indicator", key=f"ind_{col}", value=pct > indicator_threshold)

        col_specs.append(
            ImputerSpec(
                column=col,
                strategy=strategy,
                constant_value=constant_val if strategy == "constant" and constant_val else None,
                add_indicator_flag=add_ind,
            )
        )

    submitted = st.form_submit_button("Save configuration & Continue →", type="primary")

if submitted:
    config = MissingConfig(
        column_specs=col_specs,
        global_indicator_threshold=indicator_threshold / 100,
        sentinel_values=sentinel_values,
    )
    out_path = project_datasets_dir() / "3_missing_cfg.parquet"
    # Save the sentinel-replaced DataFrame so downstream stages see NaN
    save_parquet(df_analysis, out_path)
    update_state({
        "imputed_data_path": out_path.name,
        "missing_cfg": config.model_dump(),
    })
    mark_stage_complete("missing")
    st.switch_page("pages/5_features.py")
