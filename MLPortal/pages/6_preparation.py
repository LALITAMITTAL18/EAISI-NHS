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
    ],
    horizontal=True,
)
bypass = mode.startswith("Upload")

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
# PIPELINE MODE — data flows from Stages 1-5
# ─────────────────────────────────────────────────────────────────────────────
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
    stratify = st.checkbox("Stratify split", value=task in ("classification", "ordinal"))

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

# ── Feature selection ─────────────────────────────────────────────────────────
st.subheader("Feature selection")
all_feature_cols = [c for c in df.columns if c != target]
default_selected = all_feature_cols
with st.expander("Choose features to include (default: all)", expanded=False):
    selected_features = st.multiselect(
        "Features",
        options=all_feature_cols,
        default=default_selected,
        help="Remove features to create a reduced-feature variant.",
    )
if not selected_features:
    st.warning("Select at least one feature.")
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

# ── Apply & save ──────────────────────────────────────────────────────────────
st.divider()
if st.button("Create variant & Continue →", type="primary", disabled=not variant_name.strip()):
    prep_cfg = PrepConfig(
        split=split_cfg,
        scaler=ScalerConfig(method=scaler_method),
        encoder=EncoderConfig(method=enc_method),
        outcome_threshold=otc,
    )

    with st.spinner("Applying feature engineering & imputation…"):
        df_work = df[selected_features + [target]].copy()

        # Outcome threshold
        df_work = apply_outcome_threshold(df_work, target, otc)

        # Split first (prevent data leakage in imputation)
        if split_method == "time" and time_col and cutoff:
            train, test = time_split(df_work, time_col, cutoff)
        else:
            train, test = random_split(df_work, target, prep_cfg)

        # Feature engineering from Stage 5 config
        feat_cfg_raw = state.feature_cfg
        if feat_cfg_raw:
            feat_cfg = FeatureConfig.model_validate(feat_cfg_raw)
            train, test = apply_feature_config(train, test, feat_cfg, target)

        # Imputation
        miss_cfg_raw = state.missing_cfg
        if imputation_choice == "Use config from Stage 4" and miss_cfg_raw:
            miss_cfg = MissingConfig.model_validate(miss_cfg_raw)
            train, test, _ = fit_apply_config(miss_cfg, train, test)
        elif imputation_choice == "Median / mode (override)":
            from sklearn.impute import SimpleImputer
            import numpy as np
            num_cols = train.select_dtypes(include="number").columns.difference([target])
            cat_cols = train.select_dtypes(include=["object", "category"]).columns
            # Add missingness indicator flags
            for col in list(num_cols) + list(cat_cols):
                flag = f"{col}_missing"
                train[flag] = train[col].isna().astype("uint8")
                test[flag] = test[col].isna().astype("uint8")
            num_imp = SimpleImputer(strategy="median")
            if len(num_cols):
                train[num_cols] = num_imp.fit_transform(train[num_cols])
                test[num_cols] = num_imp.transform(test[num_cols])
            cat_imp = SimpleImputer(strategy="most_frequent")
            if len(cat_cols):
                train[cat_cols] = cat_imp.fit_transform(train[cat_cols])
                test[cat_cols] = cat_imp.transform(test[cat_cols])
        elif imputation_choice == "MICE (IterativeImputer, slow)":
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer, SimpleImputer
            import numpy as np
            num_cols = train.select_dtypes(include="number").columns.difference([target])
            cat_cols = train.select_dtypes(include=["object", "category"]).columns
            for col in list(num_cols) + list(cat_cols):
                flag = f"{col}_missing"
                train[flag] = train[col].isna().astype("uint8")
                test[flag] = test[col].isna().astype("uint8")
            if len(num_cols):
                mice = IterativeImputer(random_state=42, max_iter=10)
                train[num_cols] = mice.fit_transform(train[num_cols])
                test[num_cols] = mice.transform(test[num_cols])
            if len(cat_cols):
                cat_imp = SimpleImputer(strategy="most_frequent")
                train[cat_cols] = cat_imp.fit_transform(train[cat_cols])
                test[cat_cols] = cat_imp.transform(test[cat_cols])
        elif imputation_choice == "Drop rows with any missing":
            train = train.dropna()
            test = test.dropna()

    split_result = build_split_result(train, test, target, task)

    # Save to uniquely named files
    out_dir = project_datasets_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9_-]", "_", variant_name.lower().strip())[:40]
    train_path = out_dir / f"{slug}_train.parquet"
    test_path = out_dir / f"{slug}_test.parquet"
    save_parquet(train, train_path)
    save_parquet(test, test_path)

    method_label = (
        f"{imputation_choice.split('(')[0].strip()} | "
        f"split={test_size:.0%} | scaler={scaler_method} | "
        f"features={len(selected_features)}/{len(all_feature_cols)}"
    )

    register_dataset(
        name=variant_name.strip(),
        train_path=train_path.name,
        test_path=test_path.name,
        description=variant_desc.strip(),
        n_train=split_result.n_train,
        n_test=split_result.n_test,
        n_features=split_result.n_features,
        pipeline_method=method_label,
        set_active=True,
    )

    update_state({"prep_cfg": prep_cfg.model_dump()})
    mark_stage_complete("preparation")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(split_summary_bar(split_result.n_train, split_result.n_test), use_container_width=True)
    with col2:
        if split_result.class_balance_train:
            st.plotly_chart(class_balance_bar(split_result.class_balance_train, "Train class balance"), use_container_width=True)

    st.success(
        f"✅ Variant **{variant_name}** saved — "
        f"**{split_result.n_train:,} train** / **{split_result.n_test:,} test** rows, "
        f"**{split_result.n_features}** features."
    )
    st.switch_page("pages/7_modelling.py")
