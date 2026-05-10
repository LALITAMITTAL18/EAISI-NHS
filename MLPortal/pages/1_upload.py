"""Stage 1 — Upload Data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from shared.io import save_parquet
from shared.nav import render_sidebar
from shared.state import (
    get_state,
    mark_stage_complete,
    project_datasets_dir,
    update_state,
)
from stages.upload.loader import apply_sentinels, build_config, optimise_dtypes, read_file
from stages.upload.plots import dtype_bar_chart, memory_bar

st.set_page_config(page_title="1 — Upload Data", page_icon="📤", layout="wide")

render_sidebar()

st.title("Stage 1 — Upload Data")
st.caption(
    "Bring your own dataset (CSV, Excel, Parquet or JSON). "
    "Configure sentinel values, define or derive the outcome variable, and select the task type."
)

state = get_state()
_OUT = project_datasets_dir()

# ── Resume banner ─────────────────────────────────────────────────────────────
if state.stage_complete.get("upload") and state.raw_data_path:
    fname = state.upload_cfg.get("file_name", "—")
    st.success(
        f"✅ Resuming from previous session — dataset: **{fname}**. "
        "Re-upload below to replace it."
    )
    with st.expander("Previous upload config"):
        st.json(state.upload_cfg)

# ── File upload ───────────────────────────────────────────────────────────────
st.subheader("Upload dataset")
uploaded = st.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "xls", "parquet", "json"],
    help="Supported: CSV, Excel (.xlsx/.xls), Parquet, JSON",
)

col_sep, _ = st.columns(2)
with col_sep:
    sep = st.text_input("CSV separator", value=",", max_chars=3, help="Usually , or ;")

if uploaded is not None:
    with st.spinner("Reading file…"):
        try:
            df_raw = read_file(uploaded, sep=sep)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()

    st.success(f"Loaded **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**")

    # ── Sentinel configuration ────────────────────────────────────────────────
    st.subheader("Sentinel values")
    st.caption("These values will be replaced with NaN across all columns.")
    sentinels_str = st.text_input(
        "Values to treat as missing (comma-separated)",
        value="*, , 9, 999",
    )
    sentinel_values = [s.strip() for s in sentinels_str.split(",") if s.strip()]

    # Apply sentinels for preview purposes
    df_clean = apply_sentinels(df_raw, sentinel_values)

    # ── Outcome variable builder ──────────────────────────────────────────────
    st.subheader("Outcome variable")
    st.caption(
        "Choose an existing column as your outcome, or derive a new one from a formula. "
        "You can experiment with different outcome definitions within the same project."
    )

    outcome_mode = st.radio(
        "Outcome source",
        ["Use existing column", "Derive from formula"],
        horizontal=True,
    )

    derived_col_name: str | None = None
    derived_formula: str | None = None

    if outcome_mode == "Use existing column":
        target_col = st.selectbox(
            "Target column",
            options=df_clean.columns.tolist(),
            help="The column you want to predict.",
        )
    else:
        st.markdown(
            "Define a new outcome column using a Python/pandas expression. "
            "Reference other columns by name. Examples:  \n"
            "- `48 - pre_op_oxford_score` → change score  \n"
            "- `post_op_score - pre_op_score` → absolute improvement  \n"
            "- `(post_op_score - pre_op_score) / pre_op_score * 100` → % change"
        )
        col_fn, col_ff = st.columns([1, 2])
        with col_fn:
            derived_col_name = st.text_input(
                "New column name",
                value="outcome",
                help="This column will be added to the dataset.",
            )
        with col_ff:
            derived_formula = st.text_input(
                "Formula (use column names directly)",
                placeholder="post_op_score - pre_op_score",
            )

        # Live preview
        if derived_formula and derived_col_name:
            try:
                df_clean = df_clean.copy()
                df_clean[derived_col_name] = df_clean.eval(derived_formula)
                target_col = derived_col_name
                col_preview1, col_preview2 = st.columns(2)
                with col_preview1:
                    st.caption(f"Preview of **{derived_col_name}**:")
                    st.dataframe(df_clean[[derived_col_name]].describe().T, use_container_width=True)
                with col_preview2:
                    import plotly.graph_objects as go
                    from shared.viz import PALETTE
                    fig = go.Figure(go.Histogram(
                        x=df_clean[derived_col_name].dropna(),
                        marker_color=PALETTE.highlight,
                    ))
                    fig.update_layout(
                        height=220, margin=dict(l=10, r=10, t=30, b=10),
                        title_text=f"Distribution of {derived_col_name}",
                        paper_bgcolor=PALETTE.background,
                        plot_bgcolor=PALETTE.background,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Formula error: {exc}")
                target_col = df_clean.columns[0]
        else:
            st.info("Enter a formula above to preview the derived outcome.")
            target_col = df_clean.columns[0]

    # ── Task type ─────────────────────────────────────────────────────────────
    st.subheader("Task type")
    task_map = {
        "Regression": "regression",
        "Classification": "classification",
        "Ordinal Classification": "ordinal",
    }
    # Auto-suggest based on target dtype & cardinality
    if target_col in df_clean.columns:
        n_unique = df_clean[target_col].nunique()
        dtype = df_clean[target_col].dtype
        suggested = "regression"
        if not pd.api.types.is_numeric_dtype(dtype) or n_unique <= 10:
            suggested = "classification"
        task_default = list(task_map.keys()).index(
            next(k for k, v in task_map.items() if v == suggested)
        )
    else:
        task_default = 0

    task_label = st.radio(
        "Task type",
        options=list(task_map.keys()),
        index=task_default,
        horizontal=True,
        help="Auto-suggested based on the target column's dtype and cardinality.",
    )
    task_type = task_map[task_label]

    # ── Data preview ──────────────────────────────────────────────────────────
    st.subheader("Data preview")
    st.dataframe(df_clean.head(50), use_container_width=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(dtype_bar_chart(df_clean), use_container_width=True)

    # ── Save & continue ───────────────────────────────────────────────────────
    st.divider()
    if st.button("Save & Continue →", type="primary"):
        with st.spinner("Applying sentinels and optimising dtypes…"):
            df_opt, meta = optimise_dtypes(df_clean)
            cfg = build_config(
                file_name=uploaded.name,
                target_column=target_col,
                task_type=task_type,
                sentinel_values=sentinel_values,
                csv_separator=sep,
            )
            _OUT.mkdir(parents=True, exist_ok=True)
            out_path = _OUT / "1_raw.parquet"
            save_parquet(df_opt, out_path)

        # Store derived formula info in cfg so it can be reproduced
        cfg_dict = cfg.model_dump()
        if derived_formula and derived_col_name:
            cfg_dict["derived_outcome_formula"] = derived_formula
            cfg_dict["derived_outcome_column"] = derived_col_name

        update_state({
            "upload_cfg": cfg_dict,
            "raw_data_path": str(out_path.relative_to(_OUT.parent)),
        })
        mark_stage_complete("upload")

        with col_c2:
            st.plotly_chart(
                memory_bar(meta.memory_mb_raw, meta.memory_mb_optimised),
                use_container_width=True,
            )

        st.success("✅ Data saved!")
        st.switch_page("pages/2_explore.py")

