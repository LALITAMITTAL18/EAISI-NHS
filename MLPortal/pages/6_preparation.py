"""Stage 6 — Data Preparation & Train/Test Split.

Supports multiple named dataset variants per project. Each variant is a fully
prepared train/test pair stored under datasets/<slug>_train.parquet.

Three ways to create a variant:
  • Pipeline mode  — data flows from Stages 1-5; app applies imputation/encoding/split.
  • Bypass mode    — upload your own pre-split train/test files directly.
  • From notebook  — import an already-prepared pair from an existing file path
                     (used when bootstrapping from pre-existing experiments).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import re

import pandas as pd
import streamlit as st

from shared.io import load_parquet, save_parquet
from shared.nav import render_sidebar
from shared.state import (
    get_state,
    list_project_datasets,
    mark_stage_complete,
    project_datasets_dir,
    register_dataset,
    set_active_dataset,
    update_state,
)
from stages.features.engineer import apply_feature_config
from stages.features.models import FeatureConfig
from stages.missing.imputer import fit_apply_config
from stages.missing.models import MissingConfig
from stages.preparation.encoder import apply_outcome_threshold, build_column_transformer
from stages.preparation.cleaner import (
    DerivedStep,
    NullHandlingRule,
    apply_column_drops,
    apply_derived_steps_timed,
    apply_imputation,
    apply_listwise_deletions,
    apply_python_expr_rules,
)
from stages.preparation.models import (
    EncoderConfig,
    OutcomeThresholdConfig,
    PrepConfig,
    ScalerConfig,
    SplitConfig,
)
from stages.preparation.plots import class_balance_bar, split_summary_bar
from stages.preparation.splitter import build_split_result, random_split, time_split
from stages.upload.loader import read_file

st.set_page_config(page_title="6 — Preparation", page_icon="✂️", layout="wide")
render_sidebar()

st.title("Stage 6 — Data Preparation & Split")
st.caption(
    "Create one or more **dataset variants** — each is a named, fully-prepared "
    "train/test pair. Different variants can use different imputation strategies, "
    "feature sets, or splits. Stage 7 lets you choose which variant to model."
)

state = get_state()
_BASE = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Existing dataset variants
# ─────────────────────────────────────────────────────────────────────────────
variants = list_project_datasets()

st.subheader("Dataset variants")
if not variants:
    st.info("No dataset variants yet. Create one below.")
else:
    active_slug = state.active_dataset
    for v in variants:
        is_active = v["slug"] == active_slug
        col_info, col_stats, col_act = st.columns([3, 2, 1])
        with col_info:
            badge = " 🟢 **active**" if is_active else ""
            st.markdown(f"**{v['name']}**{badge}")
            if v.get("description"):
                st.caption(v["description"])
            if v.get("pipeline_method"):
                st.caption(f"Method: {v['pipeline_method']}")
        with col_stats:
            st.caption(
                f"Train: {v.get('n_train', '?'):,} rows | "
                f"Test: {v.get('n_test', '?'):,} rows | "
                f"{v.get('n_features', '?')} features"
            )
            st.caption(f"Created: {v.get('created_at', '—')}")
        with col_act:
            if not is_active:
                if st.button("Use this", key=f"use_{v['slug']}", use_container_width=True, type="primary"):
                    set_active_dataset(v["slug"])
                    st.rerun()
        if v.get("notes"):
            with st.expander("Notes"):
                st.markdown(v["notes"])
        st.divider()

# If there is already an active prepared variant, let the user proceed directly.
active_variant = next((v for v in variants if v["slug"] == state.active_dataset), None)
if active_variant and state.train_data_path and state.test_data_path:
    mark_stage_complete("preparation")
    st.success(
        f"✅ Active variant **{active_variant['name']}** is already prepared — "
        f"**{active_variant.get('n_train', '?'):,} train** / "
        f"**{active_variant.get('n_test', '?'):,} test** rows."
    )
    if st.button("Continue to Modelling →", type="primary"):
        st.switch_page("pages/7_modelling.py")
    st.divider()

_show_create = not (active_variant and state.train_data_path) or st.session_state.get("_show_create_form", False)
if active_variant and state.train_data_path:
    if st.button("➕ Create a different variant", key="_toggle_create"):
        st.session_state["_show_create_form"] = True
        st.rerun()

if not _show_create:
    st.stop()

st.subheader("Create a new dataset variant")

# ── Variant metadata (shared across all modes) ────────────────────────────────
col_vn, col_vd = st.columns([1, 2])
with col_vn:
    variant_name = st.text_input(
        "Variant name *",
        placeholder="e.g. MICE Imputation",
        help="A short memorable name. Each variant must have a unique name.",
    )
with col_vd:
    variant_desc = st.text_input(
        "Description",
        placeholder="What makes this preparation different?",
    )

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    "How to create this variant?",
    options=[
        "Pipeline (use data from Stages 1–5)",
        "Upload my own pre-split train / test files",
        "Upload a single (unsplit) dataset — split here",
    ],
    horizontal=True,
)
bypass = mode.startswith("Upload my own pre-split")
single_upload = mode.startswith("Upload a single")

# ─────────────────────────────────────────────────────────────────────────────
# BYPASS MODE — user uploads pre-split files directly
# ─────────────────────────────────────────────────────────────────────────────
if bypass:
    st.caption(
        "Upload your own pre-prepared train and test files. "
        "Both files must have the **same columns** and the target column must be present in both."
    )

    col_train, col_test = st.columns(2)
    with col_train:
        train_file = st.file_uploader(
            "Training file (CSV / Excel / Parquet / JSON)",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            key="bypass_train",
        )
    with col_test:
        test_file = st.file_uploader(
            "Test file (CSV / Excel / Parquet / JSON)",
            type=["csv", "xlsx", "xls", "parquet", "json"],
            key="bypass_test",
        )

    if train_file and test_file:
        bypass_sep = st.text_input("CSV separator (if applicable)", value=",", max_chars=3)
        try:
            train_df = read_file(train_file, sep=bypass_sep)
            test_df = read_file(test_file, sep=bypass_sep)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()

        st.success(
            f"Train: **{train_df.shape[0]:,} rows × {train_df.shape[1]} cols** | "
            f"Test: **{test_df.shape[0]:,} rows × {test_df.shape[1]} cols**"
        )

        missing_in_test = set(train_df.columns) - set(test_df.columns)
        if missing_in_test:
            st.error(f"Columns in train but not in test: {missing_in_test}")
            st.stop()

        col_t, col_k = st.columns(2)
        with col_t:
            bypass_target = st.selectbox("Target column", train_df.columns.tolist())
        with col_k:
            task_map = {"Regression": "regression", "Classification": "classification", "Ordinal": "ordinal"}
            bypass_task = task_map[st.radio("Task type", list(task_map.keys()), horizontal=True, key="bypass_task")]

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.dataframe(train_df.head(10), use_container_width=True)
        with col_p2:
            st.dataframe(test_df.head(10), use_container_width=True)

        st.divider()
        if st.button("Save as variant & Continue →", type="primary", disabled=not variant_name.strip()):
            out_dir = project_datasets_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-z0-9_-]", "_", variant_name.lower().strip())[:40]
            train_path = out_dir / f"{slug}_train.parquet"
            test_path = out_dir / f"{slug}_test.parquet"
            save_parquet(train_df, train_path)
            save_parquet(test_df, test_path)

            if not state.upload_cfg:
                update_state({
                    "upload_cfg": {
                        "file_name": f"{train_file.name} + {test_file.name}",
                        "target_column": bypass_target,
                        "task_type": bypass_task,
                        "sentinel_values": [],
                        "csv_separator": bypass_sep,
                    }
                })

            n_features = len(train_df.columns) - 1
            split_result = build_split_result(train_df, test_df, bypass_target, bypass_task)
            register_dataset(
                name=variant_name.strip(),
                train_path=train_path.name,
                test_path=test_path.name,
                description=variant_desc.strip(),
                n_train=train_df.shape[0],
                n_test=test_df.shape[0],
                n_features=n_features,
                pipeline_method="Uploaded pre-split files",
                set_active=True,
            )
            for s in ["upload", "explore", "outliers", "missing", "features", "preparation"]:
                mark_stage_complete(s)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.plotly_chart(split_summary_bar(split_result.n_train, split_result.n_test), use_container_width=True)
            with col_s2:
                if split_result.class_balance_train:
                    st.plotly_chart(class_balance_bar(split_result.class_balance_train, "Train balance"), use_container_width=True)
            st.success(f"✅ Variant **{variant_name}** saved.")
            st.switch_page("pages/7_modelling.py")
    else:
        st.info("Upload both a training file and a test file above to continue.")

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE C — user uploads a single (unsplit) dataset; split happens here
# ─────────────────────────────────────────────────────────────────────────────
if single_upload:
    st.caption(
        "Upload a single combined dataset. You configure the target, task type, "
        "split, imputation, scaling and encoding below — the split is applied here "
        "in a leakage-free order."
    )

    single_sep = st.text_input("CSV separator (if applicable)", value=",", max_chars=3, key="single_sep")
    single_file = st.file_uploader(
        "Dataset file (CSV / Excel / Parquet / JSON)",
        type=["csv", "xlsx", "xls", "parquet", "json"],
        key="single_upload_file",
    )

    if single_file is None:
        st.info("Upload your dataset above to continue.")
        st.stop()

    try:
        df_single = read_file(single_file, sep=single_sep)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        st.stop()

    st.success(f"Loaded **{df_single.shape[0]:,} rows × {df_single.shape[1]} columns**")

    # Column overview with null detection
    null_cts = df_single.isnull().sum()
    col_ov = pd.DataFrame({
        "Column": df_single.columns,
        "Type": df_single.dtypes.astype(str).values,
        "Non-Null": (len(df_single) - null_cts).values,
        "Null Count": null_cts.values,
        "Null %": (null_cts / len(df_single) * 100).round(1).values,
    })

    def _hl(row):
        c = "background-color: #fff3cd" if row["Null Count"] > 0 else ""
        return [c] * len(row)

    st.subheader("Column overview")
    st.dataframe(col_ov.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)

    col_st, col_sk = st.columns(2)
    with col_st:
        single_target = st.selectbox("Target column", df_single.columns.tolist(), key="single_target")
    with col_sk:
        task_map_s = {"Regression": "regression", "Classification": "classification", "Ordinal": "ordinal"}
        single_task = task_map_s[st.radio("Task type", list(task_map_s.keys()), horizontal=True, key="single_task")]

    # ── Split configuration ───────────────────────────────────────────────────
    st.subheader("Train / test split")
    split_method_s = st.radio("Split method", ["random", "time"], horizontal=True, key="single_split_method")
    sc_s = st.columns(3)
    with sc_s[0]:
        test_size_s = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05, key="single_test_size")
    with sc_s[1]:
        seed_s = st.number_input("Random seed", value=42, step=1, key="single_seed")
    with sc_s[2]:
        stratify_s = st.checkbox("Stratify split", value=single_task in ("classification", "ordinal"), key="single_stratify")

    time_col_s = cutoff_s = None
    if split_method_s == "time":
        time_col_s = st.selectbox("Time column", df_single.columns.tolist(), key="single_time_col")
        cutoff_s = st.text_input("Cutoff date (YYYY-MM-DD or year)", key="single_cutoff")

    split_cfg_s = SplitConfig(
        method=split_method_s,
        test_size=test_size_s,
        random_seed=int(seed_s),
        stratify=stratify_s,
        time_column=time_col_s,
        time_cutoff=cutoff_s,
    )

    # ── Scaling & encoding ────────────────────────────────────────────────────
    st.subheader("Scaling & encoding")
    col_sc_s, col_enc_s = st.columns(2)
    with col_sc_s:
        scaler_method_s = st.selectbox("Scaler", ["standard", "minmax", "robust", "quantile", "none"], key="single_scaler")
    with col_enc_s:
        enc_method_s = st.selectbox("Categorical encoder", ["onehot", "ordinal"], key="single_enc")

    # ── Imputation strategy ───────────────────────────────────────────────────
    st.subheader("Imputation strategy")
    imputation_choice_s = st.radio(
        "Missing value strategy",
        ["Median / mode", "MICE (IterativeImputer, slow)", "Drop rows with any missing"],
        help="Split is applied first; imputation is fitted on train only.",
        key="single_impute",
    )

    # ── Feature selection ─────────────────────────────────────────────────────
    st.subheader("Feature selection")
    all_feature_cols_s = [c for c in df_single.columns if c != single_target]
    with st.expander("Choose features to include (default: all)", expanded=False):
        selected_features_s = st.multiselect(
            "Features",
            options=all_feature_cols_s,
            default=all_feature_cols_s,
            key="single_features",
        )
    if not selected_features_s:
        st.warning("Select at least one feature.")
        st.stop()

    st.divider()
    if st.button("Create variant & Continue →", type="primary", key="single_save", disabled=not variant_name.strip()):
        prep_cfg_s = PrepConfig(
            split=split_cfg_s,
            scaler=ScalerConfig(method=scaler_method_s),
            encoder=EncoderConfig(method=enc_method_s),
        )

        with st.spinner("Splitting and preparing data…"):
            df_work_s = df_single[selected_features_s + [single_target]].copy()

            if split_method_s == "time" and time_col_s and cutoff_s:
                train_s, test_s = time_split(df_work_s, time_col_s, cutoff_s)
            else:
                train_s, test_s = random_split(df_work_s, single_target, prep_cfg_s)

            if imputation_choice_s == "Median / mode":
                from sklearn.impute import SimpleImputer
                num_cols_s = train_s.select_dtypes(include="number").columns.difference([single_target])
                cat_cols_s = train_s.select_dtypes(include=["object", "category"]).columns
                for col in list(num_cols_s) + list(cat_cols_s):
                    flag = f"{col}_missing"
                    train_s[flag] = train_s[col].isna().astype("uint8")
                    test_s[flag] = test_s[col].isna().astype("uint8")
                if len(num_cols_s):
                    ni = SimpleImputer(strategy="median")
                    train_s[num_cols_s] = ni.fit_transform(train_s[num_cols_s])
                    test_s[num_cols_s] = ni.transform(test_s[num_cols_s])
                if len(cat_cols_s):
                    ci = SimpleImputer(strategy="most_frequent")
                    train_s[cat_cols_s] = ci.fit_transform(train_s[cat_cols_s])
                    test_s[cat_cols_s] = ci.transform(test_s[cat_cols_s])
            elif imputation_choice_s == "MICE (IterativeImputer, slow)":
                from sklearn.experimental import enable_iterative_imputer  # noqa
                from sklearn.impute import IterativeImputer, SimpleImputer
                num_cols_s = train_s.select_dtypes(include="number").columns.difference([single_target])
                cat_cols_s = train_s.select_dtypes(include=["object", "category"]).columns
                for col in list(num_cols_s) + list(cat_cols_s):
                    flag = f"{col}_missing"
                    train_s[flag] = train_s[col].isna().astype("uint8")
                    test_s[flag] = test_s[col].isna().astype("uint8")
                if len(num_cols_s):
                    mice_s = IterativeImputer(random_state=42, max_iter=10)
                    train_s[num_cols_s] = mice_s.fit_transform(train_s[num_cols_s])
                    test_s[num_cols_s] = mice_s.transform(test_s[num_cols_s])
                if len(cat_cols_s):
                    ci = SimpleImputer(strategy="most_frequent")
                    train_s[cat_cols_s] = ci.fit_transform(train_s[cat_cols_s])
                    test_s[cat_cols_s] = ci.transform(test_s[cat_cols_s])
            elif imputation_choice_s == "Drop rows with any missing":
                train_s = train_s.dropna()
                test_s = test_s.dropna()

        split_result_s = build_split_result(train_s, test_s, single_target, single_task)

        out_dir_s = project_datasets_dir()
        out_dir_s.mkdir(parents=True, exist_ok=True)
        slug_s = re.sub(r"[^a-z0-9_-]", "_", variant_name.lower().strip())[:40]
        train_path_s = out_dir_s / f"{slug_s}_train.parquet"
        test_path_s = out_dir_s / f"{slug_s}_test.parquet"
        save_parquet(train_s, train_path_s)
        save_parquet(test_s, test_path_s)

        method_label_s = (
            f"{imputation_choice_s.split('(')[0].strip()} | "
            f"split={test_size_s:.0%} | scaler={scaler_method_s} | "
            f"features={len(selected_features_s)}/{len(all_feature_cols_s)}"
        )

        if not state.upload_cfg:
            update_state({
                "upload_cfg": {
                    "file_name": single_file.name,
                    "target_column": single_target,
                    "task_type": single_task,
                    "sentinel_values": [],
                    "csv_separator": single_sep,
                }
            })

        register_dataset(
            name=variant_name.strip(),
            train_path=train_path_s.name,
            test_path=test_path_s.name,
            description=variant_desc.strip(),
            n_train=split_result_s.n_train,
            n_test=split_result_s.n_test,
            n_features=split_result_s.n_features,
            pipeline_method=method_label_s,
            set_active=True,
        )
        for s in ["upload", "explore", "outliers", "missing", "features", "preparation"]:
            mark_stage_complete(s)

        update_state({"prep_cfg": prep_cfg_s.model_dump()})

        col1_s, col2_s = st.columns(2)
        with col1_s:
            st.plotly_chart(split_summary_bar(split_result_s.n_train, split_result_s.n_test), use_container_width=True)
        with col2_s:
            if split_result_s.class_balance_train:
                st.plotly_chart(class_balance_bar(split_result_s.class_balance_train, "Train class balance"), use_container_width=True)

        st.success(
            f"✅ Variant **{variant_name}** saved — "
            f"**{split_result_s.n_train:,} train** / **{split_result_s.n_test:,} test** rows, "
            f"**{split_result_s.n_features}** features."
        )
        st.switch_page("pages/7_modelling.py")

    st.stop()


task = state.upload_cfg.get("task_type", "regression")
target = state.upload_cfg.get("target_column", "")

prev_path = (
    state.features_data_path
    or state.imputed_data_path
    or state.outlier_data_path
    or state.raw_data_path
)
if not prev_path:
    st.warning("⚠️ Complete Stage 1 first.")
    st.stop()

df = load_parquet(project_datasets_dir() / prev_path)

all_cols = [c for c in df.columns if c != target]
null_counts_ser = df[all_cols].isnull().sum()
cols_with_nulls = null_counts_ser[null_counts_ser > 0].index.tolist()

# Session-state null rules: {col: {"strategy":..., "constant_value":..., "python_expr":...}}
if "_prep_null_rules" not in st.session_state:
    st.session_state._prep_null_rules = {}

# ── Two-column layout: config (left) | preview (right) ──────────────────────
cfg_col, preview_col = st.columns([1, 1], gap="large")

# ─── RIGHT — Dataset Preview ─────────────────────────────────────────────────
with preview_col:
    st.subheader("Dataset Preview")
    total_nulls = int(df[all_cols].isnull().sum().sum())
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Rows", f"{len(df):,}")
    _m2.metric("Features", str(len(all_cols)))
    _m3.metric("Missing cells", f"{total_nulls:,}")
    _m4.metric("Cols w/ missing", str(len(cols_with_nulls)))

    show_missing_only = st.toggle(
        "Only columns with missing values",
        value=bool(cols_with_nulls),
        key="preview_missing_toggle",
    )
    _preview_show = cols_with_nulls if show_missing_only and cols_with_nulls else all_cols
    _display_cols = [c for c in _preview_show if c in df.columns]
    if target in df.columns and target not in _display_cols:
        _display_cols = _display_cols + [target]

    def _style_null(val: object) -> str:
        return "background-color: #fff3cd; color: #856404;" if pd.isna(val) else ""

    st.dataframe(
        df[_display_cols].head(200).style.map(_style_null),
        use_container_width=True,
        height=420,
    )
    if total_nulls:
        st.caption("Amber cells = missing value")

    if cols_with_nulls:
        st.markdown("**Missing value summary**")
        _configured_preview = st.session_state._prep_null_rules
        st.dataframe(
            pd.DataFrame({
                "Column": cols_with_nulls,
                "Type": [str(df[c].dtype) for c in cols_with_nulls],
                "Missing": [int(null_counts_ser[c]) for c in cols_with_nulls],
                "Missing %": [
                    f"{null_counts_ser[c] / len(df) * 100:.1f}%"
                    for c in cols_with_nulls
                ],
                "Rule": [
                    _configured_preview.get(c, {}).get("strategy", "—")
                    for c in cols_with_nulls
                ],
            }),
            hide_index=True,
            use_container_width=True,
            height=250,
        )

# ─── LEFT — Configuration ────────────────────────────────────────────────────
with cfg_col:

    # ── Step 1 — Remove Columns ───────────────────────────────────────────────
    with st.expander("Step 1 — Remove Columns", expanded=True):
        _dc1, _dc2 = st.columns([2, 1])
        with _dc1:
            col_pattern = st.text_input(
                "Filter by name…",
                placeholder="e.g. Post-Op, Predicted",
                key="col_drop_filter",
            )
        with _dc2:
            dtype_filter = st.selectbox(
                "Type",
                ["All", "Numeric", "Categorical"],
                key="col_drop_dtype",
            )
        _filtered_drop = all_cols
        if col_pattern:
            _filtered_drop = [c for c in _filtered_drop if col_pattern.lower() in c.lower()]
        if dtype_filter == "Numeric":
            _filtered_drop = [c for c in _filtered_drop if pd.api.types.is_numeric_dtype(df[c])]
        elif dtype_filter == "Categorical":
            _filtered_drop = [c for c in _filtered_drop if not pd.api.types.is_numeric_dtype(df[c])]

        cols_to_drop = st.multiselect(
            "Columns to drop",
            _filtered_drop,
            default=[],
            key="cols_to_drop_select",
            help="Removed before any other operation runs.",
        )
        if cols_to_drop:
            st.info(
                f"Will drop **{len(cols_to_drop)}** column(s): "
                + ", ".join(cols_to_drop[:5])
                + ("…" if len(cols_to_drop) > 5 else "")
            )

    remaining_feature_cols = [c for c in all_cols if c not in cols_to_drop]
    active_null_cols = [c for c in cols_with_nulls if c not in cols_to_drop]

    # ── Step 2 — Handle Missing Values (action-first UI) ─────────────────────
    _null_label = f"Step 2 — Handle Missing Values  ({len(active_null_cols)} columns have nulls)"
    with st.expander(_null_label, expanded=bool(active_null_cols)):

        _STRAT_LABELS = {
            "none": "Leave as-is",
            "listwise_delete": "Delete rows (before split)",
            "mean": "Fill — Mean (numeric, train-fitted)",
            "median": "Fill — Median (numeric, train-fitted)",
            "mode": "Fill — Mode / most frequent (train-fitted)",
            "constant": "Fill — Constant value",
            "python_expr": "Fill — Python expression (per-row, before split)",
        }
        _STRAT_DESC = {
            "none": "No action — values remain. Handled by the global fallback below if configured.",
            "listwise_delete": "Drops entire rows where this column is null. Runs **before** the train/test split.",
            "mean": "Fills nulls with the training-set mean. Best for symmetric numeric distributions.",
            "median": "Fills nulls with the training-set median. Robust to outliers; ideal for skewed numeric data.",
            "mode": "Fills with the most frequent value from training. Works for numeric and categorical columns.",
            "constant": "Fills nulls with a fixed value you supply (e.g. `0`, `Unknown`, `9` for NHS sentinel values).",
            "python_expr": "Fills via a per-row Python expression using `row` dict, e.g. `row['a'] + row['b']`. Before split.",
        }

        if active_null_cols:
            _num_null = [c for c in active_null_cols if pd.api.types.is_numeric_dtype(df[c])]

            st.markdown("**Select action → select columns → Apply.**")

            action = st.radio(
                "Action:",
                list(_STRAT_LABELS.keys()),
                format_func=lambda k: _STRAT_LABELS[k],
                key="mv_action",
            )
            st.caption(f"*{_STRAT_DESC[action]}*")

            _eligible = _num_null if action in ("mean", "median") else active_null_cols
            _elig_note = (
                f"{len(_eligible)} numeric column(s) with missing values"
                if action in ("mean", "median")
                else f"{len(_eligible)} column(s) with missing values"
            )

            _cv_input = ""
            _ex_input = ""
            if action == "constant":
                _cv_input = st.text_input(
                    "Constant fill value:",
                    key="mv_const",
                    placeholder="e.g. 0, Unknown, 9",
                )
            elif action == "python_expr":
                _ex_input = st.text_input(
                    "Expression (`row` dict):",
                    key="mv_expr",
                    placeholder="row['col_a'] + row['col_b']",
                )

            if _eligible:
                _ca, _cb = st.columns([3, 1])
                with _ca:
                    _selected_for_action = st.multiselect(
                        f"Columns ({_elig_note}):",
                        _eligible,
                        key="mv_selected_cols",
                    )
                with _cb:
                    st.write("")
                    _apply_all_btn = st.button(
                        "All eligible",
                        key="mv_apply_all",
                        use_container_width=True,
                    )

                _apply_btn = st.button(
                    "Apply to selected →",
                    key="mv_apply",
                    type="primary",
                    disabled=not _selected_for_action,
                )

                if _apply_btn or _apply_all_btn:
                    _target_cols = _selected_for_action if _apply_btn else _eligible
                    for _col in _target_cols:
                        st.session_state._prep_null_rules[_col] = {
                            "strategy": action,
                            "constant_value": _cv_input,
                            "python_expr": _ex_input,
                        }
                    st.rerun()
            elif action in ("mean", "median"):
                st.warning("No numeric columns with missing values available for this action.")

            st.divider()

            # Current rules summary
            _set_rules = {
                c: r
                for c, r in st.session_state._prep_null_rules.items()
                if c in active_null_cols and r.get("strategy", "none") != "none"
            }
            _unset = [
                c for c in active_null_cols
                if c not in st.session_state._prep_null_rules
                or st.session_state._prep_null_rules[c].get("strategy", "none") == "none"
            ]

            if _set_rules:
                st.markdown("**Applied rules:**")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Column": _c,
                            "Missing": int(null_counts_ser[_c]),
                            "Strategy": _STRAT_LABELS.get(_r["strategy"], _r["strategy"]),
                            "Value": _r.get("constant_value") or _r.get("python_expr") or "—",
                        }
                        for _c, _r in _set_rules.items()
                    ]),
                    hide_index=True,
                    use_container_width=True,
                )
                if st.button("Reset all rules", key="mv_reset"):
                    st.session_state._prep_null_rules = {}
                    st.rerun()

            if _unset:
                st.caption(
                    f"**{len(_unset)} column(s) without a rule** → covered by global fallback: "
                    + ", ".join(_unset[:6])
                    + ("…" if len(_unset) > 6 else "")
                )
        else:
            st.info("No columns with missing values (or all are being dropped).")

        st.markdown("**Global fallback** for any remaining nulls:")
        imputation_fallback = st.radio(
            "Fallback strategy:",
            [
                "Use config from Stage 4",
                "Median / mode",
                "MICE (IterativeImputer, slow)",
                "Drop rows with any missing",
            ],
            help="Applied after per-column rules above. Fitted on training data only.",
            key="mv_fallback",
        )

    # ── Step 3 — Derived Columns ──────────────────────────────────────────────
    with st.expander("Step 3 — Derived Columns & Custom Python Code", expanded=False):
        st.caption(
            "Add custom Python steps to compute or transform columns. "
            "Code runs with `df`, `pd`, and `np` in scope. "
            "Drop helper columns after each step."
        )
        st.markdown(
            "**Example:**\n"
            "```python\n"
            "df['health_gain'] = df['Post-Op Score'] - df['Pre-Op Score']\n"
            "```"
        )
        n_steps = st.number_input("Number of steps", 0, 10, 0, 1, key="n_derived_steps")
        derived_steps: list[DerivedStep] = []
        for _i in range(int(n_steps)):
            st.markdown(f"---\n**Step {_i + 1}**")
            _sn, _sw = st.columns([3, 1])
            with _sn:
                _step_name = st.text_input(
                    "Step name", key=f"ds_name_{_i}", placeholder="e.g. Compute health_gain"
                )
            with _sw:
                _when = st.selectbox(
                    "When",
                    ["before_split", "after_split"],
                    key=f"ds_when_{_i}",
                    help="before_split: on full dataset; after_split: on train & test separately",
                )
            _step_code = st.text_area(
                "Python code (`df` is the DataFrame)",
                key=f"ds_code_{_i}",
                height=100,
                placeholder="df['health_gain'] = df['Post-Op Score'] - df['Pre-Op Score']",
            )
            _step_drop = st.multiselect(
                "Drop columns after this step",
                df.columns.tolist(),
                key=f"ds_drop_{_i}",
                help="Columns to remove once the step has run.",
            )
            derived_steps.append(
                DerivedStep(
                    name=_step_name or f"Step {_i + 1}",
                    code=_step_code,
                    drop_after=_step_drop,
                    apply_when=_when,
                )
            )

    # ── Outcome threshold ─────────────────────────────────────────────────────
    with st.expander("Outcome Threshold (optional)", expanded=False):
        use_thresh = st.checkbox("Enable outcome threshold", key="ot_enable")
        otc = OutcomeThresholdConfig()
        if use_thresh:
            _o1, _o2, _o3 = st.columns(3)
            with _o1:
                thresh_val = st.number_input("Threshold value", value=0.0, key="ot_thresh")
            with _o2:
                _ot_direction = st.selectbox("Direction", ["above", "below"], key="ot_dir")
            with _o3:
                _ot_pos = st.text_input("Positive label", "Positive", key="ot_pos")
            _ot_neg = st.text_input("Negative label", "Negative", key="ot_neg")
            _ot_col = st.text_input("Derived column name", "outcome_label", key="ot_col")
            otc = OutcomeThresholdConfig(
                enabled=True,
                threshold=thresh_val,
                direction=_ot_direction,
                positive_label=_ot_pos,
                negative_label=_ot_neg,
                derived_column_name=_ot_col,
            )

    # ── Step 4 — Row-filter Variants ──────────────────────────────────────────
    with st.expander("Step 4 — Row-filter Variants", expanded=False):
        st.caption(
            "Generate multiple named variants by excluding rows with specific values. "
            "Each filter produces a separate train/test pair."
        )
        _filter_opts = [c for c in remaining_feature_cols if c not in cols_to_drop]
        fv_col = st.selectbox(
            "Column to filter on",
            ["— none —"] + _filter_opts,
            key="fv_col",
            help="Typically an age band or grouping column whose extreme values you want to exclude.",
        )
        row_filter_variants: list[dict] = []
        if fv_col and fv_col != "— none —":
            _unique_vals = sorted(df[fv_col].dropna().unique().tolist())
            st.caption(f"Unique values: {_unique_vals}")
            _fp1, _fp2, _fp3, _fp4 = st.columns(4)
            with _fp1:
                add_lowest = st.checkbox("Exclude lowest", key="fv_lowest")
            with _fp2:
                add_highest = st.checkbox("Exclude highest", key="fv_highest")
            with _fp3:
                add_both = st.checkbox("Exclude both extremes", key="fv_both")
            with _fp4:
                add_custom = st.checkbox("Custom exclusion", key="fv_custom")
            if _unique_vals:
                _lv = _unique_vals[0]
                _hv = _unique_vals[-1]
                _slug_col = fv_col.lower().replace(" ", "-")[:15]
                if add_lowest:
                    row_filter_variants.append({"label": f"no-lowest-{_slug_col}", "col": fv_col, "exclude": [_lv]})
                if add_highest:
                    row_filter_variants.append({"label": f"no-highest-{_slug_col}", "col": fv_col, "exclude": [_hv]})
                if add_both:
                    row_filter_variants.append({"label": f"no-extreme-{_slug_col}", "col": fv_col, "exclude": [_lv, _hv]})
                if add_custom:
                    _custom_excl = st.multiselect("Values to exclude", _unique_vals, key="fv_custom_vals")
                    _custom_label = st.text_input("Label suffix", "custom-filter", key="fv_custom_label")
                    if _custom_excl:
                        row_filter_variants.append({
                            "label": re.sub(r"[^a-z0-9_-]", "-", _custom_label.lower())[:30],
                            "col": fv_col,
                            "exclude": _custom_excl,
                        })
            if row_filter_variants:
                st.info(
                    f"Will create **{len(row_filter_variants) + 1}** variant(s): "
                    f"1 base + {', '.join(v['label'] for v in row_filter_variants)}"
                )

    # ── Split, Scaling & Encoding ─────────────────────────────────────────────
    with st.expander("Split, Scaling & Encoding", expanded=True):
        split_method = st.radio("Split method", ["random", "time"], horizontal=True, key="split_method")
        _sc_cols = st.columns(3)
        with _sc_cols[0]:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05, key="test_size_slider")
        with _sc_cols[1]:
            seed = st.number_input("Random seed", value=42, step=1, key="rand_seed")
        with _sc_cols[2]:
            stratify = st.checkbox(
                "Stratify split",
                value=True,
                help="For regression targets, the target is binned into quantiles before stratification.",
                key="stratify_ck",
            )
        stratify_bins = 5
        if stratify and task == "regression":
            stratify_bins = st.slider("Stratification bins", 2, 20, 5, key="strat_bins")
        time_col = cutoff = None
        if split_method == "time":
            time_col = st.selectbox("Time column", df.columns.tolist(), key="time_col_sel")
            cutoff = st.text_input("Cutoff date (YYYY-MM-DD or year)", key="time_cutoff")
        _sc2, _enc2 = st.columns(2)
        with _sc2:
            scaler_method = st.selectbox(
                "Scaler", ["standard", "minmax", "robust", "quantile", "none"], key="scaler_method"
            )
        with _enc2:
            enc_method = st.selectbox("Categorical encoder", ["onehot", "ordinal"], key="enc_method")

split_cfg = SplitConfig(
    method=split_method,
    test_size=test_size,
    random_seed=int(seed),
    stratify=stratify,
    time_column=time_col,
    time_cutoff=cutoff,
)

# Selected features = remaining feature cols after drops
selected_features = remaining_feature_cols
if not selected_features:
    st.warning("No feature columns remain — adjust Step 1.")
    st.stop()

# ── Apply & save ──────────────────────────────────────────────────────────────
st.divider()
if st.button("Create variant & Continue →", type="primary", disabled=not variant_name.strip()):
    # Build NullHandlingRule objects from session-state accumulator
    null_rules: dict[str, NullHandlingRule] = {
        col: NullHandlingRule(
            strategy=rule.get("strategy", "none"),
            constant_value=rule.get("constant_value", ""),
            python_expr=rule.get("python_expr", ""),
        )
        for col, rule in st.session_state._prep_null_rules.items()
        if col in remaining_feature_cols
    }

    prep_cfg = PrepConfig(
        columns_to_drop=cols_to_drop,
        null_rules={c: r.model_dump() for c, r in null_rules.items()},
        derived_steps=[s.model_dump() for s in derived_steps],
        split=split_cfg,
        scaler=ScalerConfig(method=scaler_method),
        encoder=EncoderConfig(method=enc_method),
        outcome_threshold=otc,
    )

    def _prepare_slice(df_slice: pd.DataFrame, variant_suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame] | None:
        dw = df_slice.copy()

        dw = apply_column_drops(dw, cols_to_drop)

        dw, _errs = apply_derived_steps_timed(dw, derived_steps, "before_split")
        for e in _errs:
            st.error(f"[{variant_suffix}] Derived step error (before split): {e}")

        dw, _expr_errs = apply_python_expr_rules(dw, null_rules)
        for col_e, e in _expr_errs.items():
            st.error(f"[{variant_suffix}] Python expression error for '{col_e}': {e}")

        dw, _deleted = apply_listwise_deletions(dw, null_rules)
        for col_d, n in _deleted.items():
            st.info(f"[{variant_suffix}] Listwise deletion on '{col_d}': removed **{n:,}** rows.")

        if len(dw) == 0:
            st.error(f"[{variant_suffix}] No rows remain after listwise deletion — skipping.")
            return None

        dw = apply_outcome_threshold(dw, target, otc)

        if split_method == "time" and time_col and cutoff:
            tr, te = time_split(dw, time_col, cutoff)
        elif stratify and task == "regression":
            import numpy as np
            bins = min(stratify_bins, dw[target].nunique())
            try:
                strat_col = pd.qcut(dw[target], q=bins, labels=False, duplicates="drop")
            except Exception:
                strat_col = None
            from sklearn.model_selection import train_test_split as _tts
            tr, te = _tts(dw, test_size=test_size, random_state=int(seed), stratify=strat_col)
            tr = tr.reset_index(drop=True)
            te = te.reset_index(drop=True)
        else:
            tr, te = random_split(dw, target, prep_cfg)

        tr, te, fill_vals = apply_imputation(tr, te, null_rules)
        if fill_vals and not variant_suffix:
            with st.expander("Imputation fill values (from training set)", expanded=False):
                st.json({k: (float(v) if hasattr(v, "item") else v) for k, v in fill_vals.items()})

        miss_cfg_raw = state.missing_cfg
        if imputation_fallback == "Use config from Stage 4" and miss_cfg_raw:
            miss_cfg = MissingConfig.model_validate(miss_cfg_raw)
            tr, te, _ = fit_apply_config(miss_cfg, tr, te)
        elif imputation_fallback == "Median / mode":
            from sklearn.impute import SimpleImputer
            num_c = tr.select_dtypes(include="number").columns.difference([target])
            cat_c = tr.select_dtypes(include=["object", "category"]).columns
            for col_i in list(num_c) + list(cat_c):
                tr[f"{col_i}_missing"] = tr[col_i].isna().astype("uint8")
                te[f"{col_i}_missing"] = te[col_i].isna().astype("uint8")
            if len(num_c):
                ni = SimpleImputer(strategy="median")
                tr[num_c] = ni.fit_transform(tr[num_c])
                te[num_c] = ni.transform(te[num_c])
            if len(cat_c):
                ci = SimpleImputer(strategy="most_frequent")
                tr[cat_c] = ci.fit_transform(tr[cat_c])
                te[cat_c] = ci.transform(te[cat_c])
        elif imputation_fallback == "MICE (IterativeImputer, slow)":
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer, SimpleImputer
            num_c = tr.select_dtypes(include="number").columns.difference([target])
            cat_c = tr.select_dtypes(include=["object", "category"]).columns
            for col_i in list(num_c) + list(cat_c):
                tr[f"{col_i}_missing"] = tr[col_i].isna().astype("uint8")
                te[f"{col_i}_missing"] = te[col_i].isna().astype("uint8")
            if len(num_c):
                mice = IterativeImputer(random_state=42, max_iter=10)
                tr[num_c] = mice.fit_transform(tr[num_c])
                te[num_c] = mice.transform(te[num_c])
            if len(cat_c):
                ci = SimpleImputer(strategy="most_frequent")
                tr[cat_c] = ci.fit_transform(tr[cat_c])
                te[cat_c] = ci.transform(te[cat_c])
        elif imputation_fallback == "Drop rows with any missing":
            tr = tr.dropna()
            te = te.dropna()

        feat_cfg_raw = state.feature_cfg
        if feat_cfg_raw:
            feat_cfg = FeatureConfig.model_validate(feat_cfg_raw)
            tr, te = apply_feature_config(tr, te, feat_cfg, target)

        tr, _errs2 = apply_derived_steps_timed(tr, derived_steps, "after_split")
        te, _errs3 = apply_derived_steps_timed(te, derived_steps, "after_split")
        for e in _errs2 + _errs3:
            st.error(f"[{variant_suffix}] Derived step error (after split): {e}")

        return tr, te

    all_col_features = [c for c in df.columns if c != target]
    df_base = df[selected_features + [target]].copy()

    variants_to_create: list[tuple[str, str, pd.DataFrame]] = [
        (variant_name.strip(), variant_desc.strip(), df_base)
    ]
    for fv in row_filter_variants:
        filtered_df = df_base.copy()
        if fv["col"] in filtered_df.columns:
            filtered_df = filtered_df[~filtered_df[fv["col"]].isin(fv["exclude"])].reset_index(drop=True)
        fv_name = f"{variant_name.strip()} — {fv['label']}"
        variants_to_create.append((fv_name, f"Row filter: {fv['col']} ∉ {fv['exclude']}", filtered_df))

    out_dir = project_datasets_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    method_label = (
        f"{imputation_fallback.split('(')[0].strip()} | "
        f"split={test_size:.0%} | stratify={stratify} | "
        f"features={len(selected_features)}/{len(all_col_features)}"
    )

    first_slug = None
    with st.spinner(f"Creating {len(variants_to_create)} variant(s)…"):
        for v_name, v_desc, v_df in variants_to_create:
            result = _prepare_slice(v_df, variant_suffix=v_name)
            if result is None:
                continue
            v_train, v_test = result
            split_result = build_split_result(v_train, v_test, target, task)

            slug = re.sub(r"[^a-z0-9_-]", "_", v_name.lower())[:40]
            if first_slug is None:
                first_slug = slug
            tr_path = out_dir / f"{slug}_train.parquet"
            te_path = out_dir / f"{slug}_test.parquet"
            save_parquet(v_train, tr_path)
            save_parquet(v_test, te_path)

            register_dataset(
                name=v_name,
                train_path=tr_path.name,
                test_path=te_path.name,
                description=v_desc,
                n_train=split_result.n_train,
                n_test=split_result.n_test,
                n_features=split_result.n_features,
                pipeline_method=method_label + (f" | filter={v_desc}" if v_desc.startswith("Row filter") else ""),
                set_active=(v_name == variants_to_create[0][0]),
            )

            st.markdown(f"**{v_name}** — {split_result.n_train:,} train / {split_result.n_test:,} test")
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                st.plotly_chart(split_summary_bar(split_result.n_train, split_result.n_test), use_container_width=True)
            with _rc2:
                if split_result.class_balance_train:
                    st.plotly_chart(class_balance_bar(split_result.class_balance_train, "Train target balance"), use_container_width=True)
                else:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=v_train[target], name="Train", opacity=0.7, nbinsx=30))
                    fig.add_trace(go.Histogram(x=v_test[target], name="Test", opacity=0.7, nbinsx=30))
                    fig.update_layout(barmode="overlay", title="Target distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)

    update_state({"prep_cfg": prep_cfg.model_dump()})
    mark_stage_complete("preparation")

    st.success(f"✅ Created **{len(variants_to_create)}** variant(s) successfully.")
    st.switch_page("pages/7_modelling.py")
