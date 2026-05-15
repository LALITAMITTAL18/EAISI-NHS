"""Stage 7 — Model Configuration & Training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from shared.io import load_joblib, load_parquet, save_joblib, save_json
from shared.nav import render_sidebar
from shared.state import (
    get_state,
    list_project_datasets,
    mark_stage_complete,
    project_datasets_dir,
    project_models_dir,
    set_active_dataset,
    update_state,
)
from stages.modelling.models import OptunaConfig, TrainConfig
from stages.modelling.plots import optuna_history_plot, training_summary_bar
from stages.modelling.trainer import load_registry, persist_results, train_all
from stages.preparation.models import PrepConfig

st.set_page_config(page_title="7 — Modelling", page_icon="🤖", layout="wide")
render_sidebar()

st.title("Stage 7 — Model Training")
st.caption(
    "Select a dataset variant, choose models, configure Optuna hyperparameter "
    "optimisation and run training."
)

state = get_state()
_BASE = Path(__file__).parent.parent
_project_task = state.upload_cfg.get("task_type", "regression")
target = state.upload_cfg.get("target_column", "")

# ── Dataset variant selector ──────────────────────────────────────────────────
variants = list_project_datasets()
if not variants:
    st.warning("⚠️ Complete Stage 6 (Preparation & Split) first to create a dataset variant.")
    st.stop()

st.subheader("Dataset variants")
variant_names = [v["name"] for v in variants]
active_slug = state.active_dataset
active_name = next((v["name"] for v in variants if v["slug"] == active_slug), variant_names[0])

selected_names_variants = st.multiselect(
    "Select one or more dataset variants to train on",
    options=variant_names,
    default=[active_name],
    help="Models will be trained independently on each selected variant.",
)

if not selected_names_variants:
    st.warning("⚠️ Select at least one dataset variant.")
    st.stop()

selected_variants = [v for v in variants if v["name"] in selected_names_variants]

# Show info table for all selected variants
info_cols = st.columns(len(selected_variants))
for col, sv in zip(info_cols, selected_variants):
    with col:
        _n_tr = sv.get("n_train", "?")
        _n_te = sv.get("n_test", "?")
        st.caption(f"**{sv['name']}**")
        st.caption(f"Train: {_n_tr:,} rows" if isinstance(_n_tr, int) else f"Train: {_n_tr} rows")
        st.caption(f"Test: {_n_te:,} rows" if isinstance(_n_te, int) else f"Test: {_n_te} rows")
        st.caption(f"Features: {sv.get('n_features', '?')}")
        if sv.get("pipeline_method"):
            st.caption(f"Method: {sv['pipeline_method']}")

# Use the first selected variant as the reference for task/target config
_ref_variant = selected_variants[0]
if _ref_variant["slug"] != active_slug:
    set_active_dataset(_ref_variant["slug"])
    state = get_state()

if not state.train_data_path:
    st.warning("⚠️ Reference variant has no data paths — re-create it in Stage 6.")
    st.stop()

_ref_train = load_parquet(project_datasets_dir() / state.train_data_path)

# ── Task & target (applies to all selected variants) ─────────────────────────
_variant_target = _ref_variant.get("target_column") or target
_variant_task   = _ref_variant.get("task_type")   or _project_task

if _variant_target in _ref_train.columns and _ref_train[_variant_target].nunique() == 2 and _variant_task == "regression":
    _variant_task = "classification"

st.subheader("Task & target (applied to all selected variants)")
col_task_a, col_task_b = st.columns(2)
with col_task_a:
    _task_labels = ["regression", "classification", "ordinal"]
    task = st.selectbox(
        "Task type",
        options=_task_labels,
        index=_task_labels.index(_variant_task) if _variant_task in _task_labels else 0,
        help="Applied to every selected variant.",
    )
with col_task_b:
    _all_cols = _ref_train.columns.tolist()
    _default_target_idx = _all_cols.index(_variant_target) if _variant_target in _all_cols else 0
    target = st.selectbox(
        "Target column",
        options=_all_cols,
        index=_default_target_idx,
        help="Applied to every selected variant.",
    )

# Reference columns used for model registry filtering
_ref_X = _ref_train.drop(columns=[target], errors="ignore")
numeric_cols = _ref_X.select_dtypes(include="number").columns.tolist()
categorical_cols = _ref_X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# ── Model registry ────────────────────────────────────────────────────────────
registry = load_registry(_BASE)
available = registry.list_for_task(task)

if not available:
    st.error(f"No models available for task '{task}'. Check models.json and installed packages.")
    st.stop()

st.subheader("Select models")
st.caption(
    "💡 Add custom models by editing `website/data/custom_models.json` — "
    "no code changes needed."
)

tag_filter = st.multiselect(
    "Filter by tag",
    options=sorted({t for spec in available for t in spec.tags}),
)
if tag_filter:
    available = [s for s in available if any(t in s.tags for t in tag_filter)]

selected_names = []
for spec in available:
    checked = st.checkbox(
        f"**{spec.display_name}**  `{', '.join(spec.tags)}`",
        value=True,
        key=f"sel_{spec.name}",
    )
    if checked:
        selected_names.append(spec.name)

# ── Optuna config ─────────────────────────────────────────────────────────────
st.subheader("Optuna HPO configuration")
col1, col2, col3 = st.columns(3)
with col1:
    n_trials = st.slider("Trials per model", 5, 200, 40)
with col2:
    cv_folds = st.slider("CV folds", 3, 10, 5)
with col3:
    sampler = st.selectbox("Sampler", ["tpe", "cmaes", "random"])

metric_options = {
    "regression":     ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
    "classification": ["roc_auc", "average_precision", "f1", "f1_weighted"],
    "ordinal":        ["f1_weighted", "accuracy"],
}
metric = st.selectbox("Optimise metric", metric_options.get(task, ["neg_root_mean_squared_error"]))
timeout = st.number_input("Timeout per model (seconds, 0 = no limit)", 0, 3600, 0, step=30)

optuna_cfg = OptunaConfig(
    n_trials=n_trials,
    cv_folds=cv_folds,
    metric=metric,
    sampler=sampler,
    timeout_seconds=timeout if timeout > 0 else None,
)
prep_cfg_raw = state.prep_cfg
prep_cfg = PrepConfig.model_validate(prep_cfg_raw) if prep_cfg_raw else PrepConfig()

train_cfg = TrainConfig(
    task=task,
    target_column=target,
    selected_models=selected_names,
    optuna=optuna_cfg,
)

# ── Training ──────────────────────────────────────────────────────────────────
_n_variants = len(selected_variants)
_btn_label = (
    f"🚀 Train selected models on {_n_variants} variant{'s' if _n_variants > 1 else ''}"
)
if selected_names and st.button(_btn_label, type="primary"):
    out_dir = project_models_dir()
    _total_trials = train_cfg.optuna.n_trials
    _total_models = len(selected_names)

    all_variant_results: dict[str, list] = {}  # slug → list[TrainResult]

    for _vi, _sv in enumerate(selected_variants):
        vslug = _sv["slug"]
        st.markdown(f"#### Variant {_vi + 1}/{_n_variants}: **{_sv['name']}**")

        # Resolve data paths from the variant metadata directly
        _v_train_path = _sv.get("train_path") or _sv.get("train_data_path")
        _v_test_path  = _sv.get("test_path")  or _sv.get("test_data_path")

        if not _v_train_path:
            st.warning(f"⚠️ No data path for variant **{_sv['name']}** — skipping.")
            continue

        _v_train = load_parquet(project_datasets_dir() / _v_train_path)
        _v_X_train = _v_train.drop(columns=[target], errors="ignore")
        _v_y_train = _v_train[target]
        _v_num_cols = _v_X_train.select_dtypes(include="number").columns.tolist()
        _v_cat_cols = _v_X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        progress = st.progress(0)
        status = st.empty()
        trial_status = st.empty()

        def on_trial_done(study, trial, _tt=_total_trials, _ts=trial_status) -> None:
            best = study.best_value if study.best_trial else None
            best_str = f"  |  best so far: {best:.4f}" if best is not None else ""
            _ts.caption(f"Trial {trial.number + 1}/{_tt}{best_str}")

        def on_model_done(name: str, done: int, total: int,
                          _p=progress, _s=status, _ts=trial_status,
                          _sn=selected_names) -> None:
            _p.progress(done / total)
            _s.info(f"✅ Completed {done}/{total}: **{name}**")
            _ts.empty()
            if done < total:
                _s.info(f"⏳ {done}/{total} done — training **{_sn[done]}**…")

        status.info(f"⏳ 0/{_total_models} done — training **{selected_names[0]}**…")

        with st.spinner(f"Training on {_sv['name']}…"):
            results = train_all(
                selected_names=selected_names,
                registry=registry,
                X_train=_v_X_train,
                y_train=_v_y_train,
                config=train_cfg,
                numeric_cols=_v_num_cols,
                categorical_cols=_v_cat_cols,
                prep_config=prep_cfg,
                progress_callback=on_trial_done,
                model_done_callback=on_model_done,
                output_dir=out_dir,
                variant_slug=vslug,
            )
        progress.progress(1.0)
        status.success(f"✅ All {len(results)} models trained on **{_sv['name']}**!")
        all_variant_results[vslug] = results

        # Save results cache and config per variant
        cache_path = out_dir / f"{vslug}_results_cache.joblib"
        save_joblib([r.model_dump(mode="python") for r in results], cache_path)
        cfg_path = out_dir / f"{vslug}_pipeline_config.json"
        save_json(train_cfg.model_dump(mode="json"), cfg_path)

    # Update state using the last-trained variant's paths (reference variant)
    _last_slug = selected_variants[-1]["slug"]
    update_state({
        "results_cache_path": f"{_last_slug}_results_cache.joblib",
        "pipeline_config_path": f"{_last_slug}_pipeline_config.json",
        "train_cfg": train_cfg.model_dump(mode="json"),
    })
    mark_stage_complete("modelling")

    # ── Results summary per variant ───────────────────────────────────────────
    for vslug, results in all_variant_results.items():
        _sv_name = next((v["name"] for v in selected_variants if v["slug"] == vslug), vslug)
        st.markdown(f"#### Results — {_sv_name}")
        cv_scores = [r.cv_score for r in results if r.cv_score is not None]
        model_names_res = [r.model_name for r in results if r.cv_score is not None]
        if cv_scores:
            st.plotly_chart(
                training_summary_bar(model_names_res, cv_scores, metric),
                use_container_width=True,
            )
        for result in results:
            with st.expander(f"Optuna history — {result.model_name}"):
                st.plotly_chart(
                    optuna_history_plot(result.optuna_history, result.model_name),
                    use_container_width=True,
                )

    st.switch_page("pages/8_comparison.py")
