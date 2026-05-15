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

# ── Split configuration ───────────────────────────────────────────────────────
st.subheader("Train / test split")
split_method = st.radio("Split method", ["random", "time"], horizontal=True)

sc = st.columns(3)
with sc[0]:
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
with sc[1]:
    seed = st.number_input("Random seed", value=42, step=1)
with sc[2]:
    stratify = st.checkbox(
        "Stratify split",
        value=True,
        help=(
            "For regression targets, the target is binned into quantiles and used for "
            "stratification so train and test have equal target distributions."
        ),
    )

stratify_bins = 5
if stratify and task == "regression":
    stratify_bins = st.slider(
        "Stratification bins (quantile bins on target)",
        min_value=2, max_value=20, value=5,
        help="The target column is divided into this many equal-frequency bins before stratification.",
    )

time_col = cutoff = None
if split_method == "time":
    time_col = st.selectbox("Time column", df.columns.tolist())
    cutoff = st.text_input("Cutoff date (YYYY-MM-DD or year)")

split_cfg = SplitConfig(
    method=split_method,
    test_size=test_size,
    random_seed=int(seed),
    stratify=stratify,
    time_column=time_col,
    time_cutoff=cutoff,
)

# ── Scaling & encoding ────────────────────────────────────────────────────────
st.subheader("Scaling & encoding")
col_sc, col_enc = st.columns(2)
with col_sc:
    scaler_method = st.selectbox("Scaler", ["standard", "minmax", "robust", "quantile", "none"])
with col_enc:
    enc_method = st.selectbox("Categorical encoder", ["onehot", "ordinal"])

# ── Imputation strategy (affects variant identity) ────────────────────────────
st.subheader("Imputation strategy")
imputation_choice = st.radio(
    "Missing value strategy",
    ["Use config from Stage 4", "Median / mode (override)", "MICE (IterativeImputer, slow)", "Drop rows with any missing"],
    help="Each strategy creates a distinct data variant.",
)

# ── Data Cleaning Pipeline ────────────────────────────────────────────────────
st.subheader("Data Cleaning Pipeline")
st.caption(
    "Configure column removal, missing-value handling, and custom derived-column "
    "steps. Operations marked **Before split** run on the full dataset; "
    "**After split** operations (imputation) are fit on the training set only."
)

all_cols = [c for c in df.columns if c != target]
null_counts_df = df[all_cols].isnull().sum()
cols_with_nulls = null_counts_df[null_counts_df > 0].index.tolist()

# ── Step 1 — Column removal ───────────────────────────────────────────────────
with st.expander("Step 1 — Remove Columns", expanded=True):
    st.caption(
        "Select columns to **drop** from the dataset. Use the pattern filter to "
        "quickly find columns by name fragment."
    )
    col_filter_a, col_filter_b = st.columns([2, 1])
    with col_filter_a:
        col_pattern = st.text_input(
            "Filter by name contains…",
            key="col_drop_filter",
            placeholder="e.g. Post-Op, Predicted, CSVYear",
        )
    with col_filter_b:
        dtype_filter = st.selectbox(
            "Filter by dtype",
            ["All", "numeric", "object / categorical"],
            key="col_drop_dtype",
        )

    filtered_cols = all_cols
    if col_pattern:
        filtered_cols = [c for c in filtered_cols if col_pattern.lower() in c.lower()]
    if dtype_filter == "numeric":
        filtered_cols = [c for c in filtered_cols if pd.api.types.is_numeric_dtype(df[c])]
    elif dtype_filter == "object / categorical":
        filtered_cols = [c for c in filtered_cols if not pd.api.types.is_numeric_dtype(df[c])]

    cols_to_drop = st.multiselect(
        "Columns to drop",
        options=filtered_cols,
        default=[],
        help="These columns are removed before anything else runs.",
        key="cols_to_drop_select",
    )
    if cols_to_drop:
        st.info(f"Will drop **{len(cols_to_drop)}** column(s): {', '.join(cols_to_drop)}")

# Features remaining after drops
remaining_feature_cols = [c for c in all_cols if c not in cols_to_drop]

# ── Step 2 — Missing value rules ─────────────────────────────────────────────
with st.expander(
    f"Step 2 — Missing Value Rules  ({len(cols_with_nulls)} columns have nulls)",
    expanded=bool(cols_with_nulls),
):
    st.caption(
        "Configure how to handle missing values per column. "
        "**Listwise deletion** runs *before* the split; all other strategies "
        "are fit on training data only to prevent leakage."
    )

    null_rules: dict[str, NullHandlingRule] = {}

    STRATEGY_OPTIONS = [
        "none",
        "listwise_delete",
        "mean",
        "median",
        "mode",
        "constant",
        "python_expr",
    ]
    STRATEGY_LABELS = {
        "none": "Leave as-is",
        "listwise_delete": "Listwise deletion (drop rows, before split)",
        "mean": "Fill — Mean (train only)",
        "median": "Fill — Median (train only)",
        "mode": "Fill — Mode / most frequent (train only)",
        "constant": "Fill — Custom constant",
        "python_expr": "Fill — Python expression (per-row, before split)",
    }

    # Show only columns that are not being dropped and have nulls
    active_null_cols = [c for c in cols_with_nulls if c not in cols_to_drop]

    if not active_null_cols:
        st.info("No columns with missing values (or all are being dropped).")
    else:
        # Bulk action
        bulk_col1, bulk_col2 = st.columns([2, 1])
        with bulk_col1:
            bulk_strategy = st.selectbox(
                "Apply to ALL columns below",
                ["— pick one to apply —"] + STRATEGY_OPTIONS,
                format_func=lambda x: STRATEGY_LABELS.get(x, x),
                key="bulk_null_strategy",
            )
        with bulk_col2:
            apply_bulk = st.button("Apply bulk strategy", key="apply_bulk_null")

        st.divider()

        for col in active_null_cols:
            null_ct = int(null_counts_df[col])
            pct = round(null_ct / len(df) * 100, 1)
            col_a, col_b, col_c = st.columns([3, 2, 3])

            with col_a:
                st.markdown(f"**{col}**")
                st.caption(f"{null_ct:,} nulls ({pct}%) — dtype: {df[col].dtype}")

            default_strategy = "none"
            if apply_bulk and bulk_strategy != "— pick one to apply —":
                default_strategy = bulk_strategy
            prev_key = f"null_rule_{col}"
            if prev_key in st.session_state:
                default_strategy = st.session_state[prev_key]

            with col_b:
                chosen = st.selectbox(
                    "Strategy",
                    options=STRATEGY_OPTIONS,
                    format_func=lambda x: STRATEGY_LABELS.get(x, x),
                    index=STRATEGY_OPTIONS.index(default_strategy),
                    key=f"null_strategy_{col}",
                    label_visibility="collapsed",
                )
                st.session_state[prev_key] = chosen

            with col_c:
                const_val = ""
                expr_val = ""
                if chosen == "constant":
                    const_val = st.text_input(
                        "Constant value",
                        key=f"null_const_{col}",
                        label_visibility="collapsed",
                        placeholder="e.g. 0, Unknown, 9",
                    )
                elif chosen == "python_expr":
                    expr_val = st.text_input(
                        "Expression (row dict available as `row`)",
                        key=f"null_expr_{col}",
                        label_visibility="collapsed",
                        placeholder="row['col_a'] + row['col_b']",
                    )
                else:
                    st.empty()

            null_rules[col] = NullHandlingRule(
                strategy=chosen,
                constant_value=const_val,
                python_expr=expr_val,
            )

# ── Step 3 — Derived columns / custom code ───────────────────────────────────
with st.expander("Step 3 — Derived Columns & Custom Python Code", expanded=False):
    st.caption(
        "Add custom Python steps to compute or transform columns. "
        "Each step runs with `df` (pandas DataFrame), `pd`, and `np` in scope. "
        "You can mark columns to **drop after** the step — useful when a helper "
        "column was only needed to compute another (e.g. drop Post-Op Score after "
        "computing health_gain)."
    )

    st.markdown(
        """**Example — compute health_gain then drop source columns:**
```python
df['health_gain'] = df['Knee Replacement Post-Op Q Score'] - df['Knee Replacement Pre-Op Q Score']
```
*Drop after:* `Knee Replacement Post-Op Q Score`, `Knee Replacement Pre-Op Q Score`
"""
    )

    n_steps = st.number_input(
        "Number of custom steps", min_value=0, max_value=10, value=0, step=1, key="n_derived_steps"
    )

    derived_steps: list[DerivedStep] = []
    for i in range(int(n_steps)):
        st.markdown(f"---\n**Step {i + 1}**")
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            step_name = st.text_input(
                "Step name", key=f"ds_name_{i}", placeholder=f"e.g. Compute health_gain"
            )
        with sc2:
            when = st.selectbox(
                "Apply when",
                ["before_split", "after_split"],
                key=f"ds_when_{i}",
                help="before_split: on the full dataset; after_split: on train & test separately",
            )
        step_code = st.text_area(
            "Python code (`df` is the DataFrame)",
            key=f"ds_code_{i}",
            height=120,
            placeholder="df['health_gain'] = df['Post-Op Score'] - df['Pre-Op Score']",
        )
        step_drop = st.multiselect(
            "Drop columns after this step",
            options=df.columns.tolist(),
            key=f"ds_drop_{i}",
            help="Columns to remove once the step has run (e.g. helper columns used to compute a derived feature).",
        )
        derived_steps.append(
            DerivedStep(
                name=step_name or f"Step {i + 1}",
                code=step_code,
                drop_after=step_drop,
                apply_when=when,
            )
        )

# Final feature selection after drops & derived steps
selected_features = remaining_feature_cols
if not selected_features:
    st.warning("No feature columns remain — adjust the column removal step.")
    st.stop()

# ── Outcome threshold ─────────────────────────────────────────────────────────
st.subheader("Outcome Threshold (optional)")
use_thresh = st.checkbox("Enable Outcome Threshold")
otc = OutcomeThresholdConfig()
if use_thresh:
    col_ot1, col_ot2, col_ot3 = st.columns(3)
    with col_ot1:
        thresh_val = st.number_input("Threshold value", value=0.0)
    with col_ot2:
        direction = st.selectbox("Direction", ["above", "below"])
    with col_ot3:
        pos_label = st.text_input("Positive label", "Positive")
    neg_label = st.text_input("Negative label", "Negative")
    derived_col = st.text_input("Derived column name", "outcome_label")
    otc = OutcomeThresholdConfig(
        enabled=True,
        threshold=thresh_val,
        direction=direction,
        positive_label=pos_label,
        negative_label=neg_label,
        derived_column_name=derived_col,
    )

# ── Step 4 — Row-filter Variants ─────────────────────────────────────────────
with st.expander("Step 4 — Row-filter Variants (generate multiple datasets at once)", expanded=False):
    st.caption(
        "Define filters to automatically create **multiple named dataset variants** "
        "in one click — e.g. remove the lowest age band, highest age band, or both. "
        "Each filter produces a separate train/test pair saved alongside the base variant. "
        "Leave empty to only create the base (unfiltered) variant."
    )

    # Pick the column to filter on
    filter_col_options = [c for c in remaining_feature_cols if c not in cols_to_drop]
    fv_col = st.selectbox(
        "Column to filter on",
        options=["— none —"] + filter_col_options,
        key="fv_col",
        help="Typically 'Age Band' or any grouping column whose extreme values you want to exclude.",
    )

    row_filter_variants: list[dict] = []

    if fv_col and fv_col != "— none —":
        # Show the unique values present
        unique_vals = sorted(df[fv_col].dropna().unique().tolist())
        st.caption(f"Unique values in **{fv_col}**: {unique_vals}")

        # Preset buttons for common NHS patterns
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            add_lowest = st.checkbox("Exclude lowest value", value=False, key="fv_lowest")
        with col_p2:
            add_highest = st.checkbox("Exclude highest value", value=False, key="fv_highest")
        with col_p3:
            add_both = st.checkbox("Exclude both extremes", value=False, key="fv_both")
        with col_p4:
            add_custom = st.checkbox("Custom value exclusion", value=False, key="fv_custom")

        if unique_vals:
            lowest_val = unique_vals[0]
            highest_val = unique_vals[-1]

            if add_lowest:
                row_filter_variants.append({
                    "label": f"no-lowest-{fv_col.lower().replace(' ', '-')}",
                    "col": fv_col,
                    "exclude": [lowest_val],
                })
            if add_highest:
                row_filter_variants.append({
                    "label": f"no-highest-{fv_col.lower().replace(' ', '-')}",
                    "col": fv_col,
                    "exclude": [highest_val],
                })
            if add_both:
                row_filter_variants.append({
                    "label": f"no-extreme-{fv_col.lower().replace(' ', '-')}",
                    "col": fv_col,
                    "exclude": [lowest_val, highest_val],
                })
            if add_custom:
                custom_excl = st.multiselect(
                    "Values to exclude",
                    options=unique_vals,
                    key="fv_custom_vals",
                )
                custom_label = st.text_input(
                    "Custom variant label suffix",
                    value="custom-filter",
                    key="fv_custom_label",
                )
                if custom_excl:
                    row_filter_variants.append({
                        "label": re.sub(r"[^a-z0-9_-]", "-", custom_label.lower())[:30],
                        "col": fv_col,
                        "exclude": custom_excl,
                    })

        if row_filter_variants:
            st.info(
                f"Will create **{len(row_filter_variants) + 1}** variant(s): "
                f"1 base + {', '.join(v['label'] for v in row_filter_variants)}"
            )

# ── Apply & save ──────────────────────────────────────────────────────────────
st.divider()
if st.button("Create variant & Continue →", type="primary", disabled=not variant_name.strip()):
    prep_cfg = PrepConfig(
        columns_to_drop=cols_to_drop,
        null_rules={c: r.model_dump() for c, r in null_rules.items()},
        derived_steps=[s.model_dump() for s in derived_steps],
        split=split_cfg,
        scaler=ScalerConfig(method=scaler_method),
        encoder=EncoderConfig(method=enc_method),
        outcome_threshold=otc,
    )

    # Helper: run the full cleaning + split + imputation pipeline on a given df slice
    def _prepare_slice(df_slice: pd.DataFrame, variant_suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame] | None:
        dw = df_slice.copy()

        # 1. Drop columns
        dw = apply_column_drops(dw, cols_to_drop)

        # 2. Before-split derived steps
        dw, _errs = apply_derived_steps_timed(dw, derived_steps, "before_split")
        for e in _errs:
            st.error(f"[{variant_suffix}] Derived step error (before split): {e}")

        # 3. Python-expression null fills
        dw, _expr_errs = apply_python_expr_rules(dw, null_rules)
        for col_e, e in _expr_errs.items():
            st.error(f"[{variant_suffix}] Python expression error for '{col_e}': {e}")

        # 4. Listwise deletion
        dw, _deleted = apply_listwise_deletions(dw, null_rules)
        for col_d, n in _deleted.items():
            st.info(f"[{variant_suffix}] Listwise deletion on '{col_d}': removed **{n:,}** rows.")

        if len(dw) == 0:
            st.error(f"[{variant_suffix}] No rows remain after listwise deletion — skipping.")
            return None

        # 5. Outcome threshold
        dw = apply_outcome_threshold(dw, target, otc)

        # 6. Stratified split on target
        if split_method == "time" and time_col and cutoff:
            tr, te = time_split(dw, time_col, cutoff)
        elif stratify and task == "regression":
            # Bin the target into quantiles for stratification
            import numpy as np
            bins = min(stratify_bins, dw[target].nunique())
            try:
                strat_col = pd.qcut(dw[target], q=bins, labels=False, duplicates="drop")
            except Exception:
                strat_col = None
            from sklearn.model_selection import train_test_split as _tts
            tr, te = _tts(
                dw,
                test_size=test_size,
                random_state=int(seed),
                stratify=strat_col,
            )
            tr = tr.reset_index(drop=True)
            te = te.reset_index(drop=True)
        else:
            tr, te = random_split(dw, target, prep_cfg)

        # 7. Constant / mean / median / mode imputation (fit on train only)
        tr, te, fill_vals = apply_imputation(tr, te, null_rules)
        if fill_vals and not variant_suffix:
            with st.expander("Imputation fill values (from training set)", expanded=False):
                st.json({k: (float(v) if hasattr(v, "item") else v) for k, v in fill_vals.items()})

        # 8. Stage-4 imputation config
        miss_cfg_raw = state.missing_cfg
        if imputation_choice == "Use config from Stage 4" and miss_cfg_raw:
            miss_cfg = MissingConfig.model_validate(miss_cfg_raw)
            tr, te, _ = fit_apply_config(miss_cfg, tr, te)
        elif imputation_choice == "Median / mode (override)":
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
        elif imputation_choice == "MICE (IterativeImputer, slow)":
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
        elif imputation_choice == "Drop rows with any missing":
            tr = tr.dropna()
            te = te.dropna()

        # 9. Stage-5 feature engineering
        feat_cfg_raw = state.feature_cfg
        if feat_cfg_raw:
            feat_cfg = FeatureConfig.model_validate(feat_cfg_raw)
            tr, te = apply_feature_config(tr, te, feat_cfg, target)

        # 10. After-split derived steps
        tr, _errs2 = apply_derived_steps_timed(tr, derived_steps, "after_split")
        te, _errs3 = apply_derived_steps_timed(te, derived_steps, "after_split")
        for e in _errs2 + _errs3:
            st.error(f"[{variant_suffix}] Derived step error (after split): {e}")

        return tr, te

    # Build the list of all (name, df_slice) to process
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
        f"{imputation_choice.split('(')[0].strip()} | "
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

            # Show distribution chart for this variant
            st.markdown(f"**{v_name}** — {split_result.n_train:,} train / {split_result.n_test:,} test")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(split_summary_bar(split_result.n_train, split_result.n_test), use_container_width=True)
            with c2:
                if split_result.class_balance_train:
                    st.plotly_chart(class_balance_bar(split_result.class_balance_train, "Train target balance"), use_container_width=True)
                else:
                    # For regression: show target distribution comparison
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
