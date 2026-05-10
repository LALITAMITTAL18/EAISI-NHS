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

st.subheader("Dataset variant")
variant_names = [v["name"] for v in variants]
active_slug = state.active_dataset
active_idx = next((i for i, v in enumerate(variants) if v["slug"] == active_slug), 0)

col_vs, col_vi = st.columns([2, 3])
with col_vs:
    selected_name = st.selectbox(
        "Select dataset variant to use for training",
        options=variant_names,
        index=active_idx,
    )
selected_variant = next(v for v in variants if v["name"] == selected_name)

with col_vi:
    st.caption(f"**Train:** {selected_variant.get('n_train', '?'):,} rows")
    st.caption(f"**Test:** {selected_variant.get('n_test', '?'):,} rows")
    st.caption(f"**Features:** {selected_variant.get('n_features', '?')}")
    if selected_variant.get("pipeline_method"):
        st.caption(f"**Method:** {selected_variant['pipeline_method']}")

# Switch active dataset if user changed the selector
if selected_variant["slug"] != active_slug:
    set_active_dataset(selected_variant["slug"])
    state = get_state()

if not state.train_data_path:
    st.warning("⚠️ Selected variant has no data paths — re-create it in Stage 6.")
    st.stop()

train = load_parquet(project_datasets_dir() / state.train_data_path)
test = load_parquet(project_datasets_dir() / state.test_data_path)

# ── Per-variant task & target override ───────────────────────────────────────
_variant_target = selected_variant.get("target_column") or target
_variant_task   = selected_variant.get("task_type")   or _project_task

# Auto-detect: binary target with project task=regression -> likely classification
if _variant_target in train.columns and train[_variant_target].nunique() == 2 and _variant_task == "regression":
    _variant_task = "classification"

st.subheader("Task & target for this variant")
col_task_a, col_task_b = st.columns(2)
with col_task_a:
    _task_labels = ["regression", "classification", "ordinal"]
    task = st.selectbox(
        "Task type",
        options=_task_labels,
        index=_task_labels.index(_variant_task) if _variant_task in _task_labels else 0,
        help="Override the project-level task type for this variant.",
    )
with col_task_b:
    _all_cols = train.columns.tolist()
    _default_target_idx = _all_cols.index(_variant_target) if _variant_target in _all_cols else 0
    target = st.selectbox(
        "Target column",
        options=_all_cols,
        index=_default_target_idx,
        help="Override the project-level target column for this variant.",
    )

X_train = train.drop(columns=[target], errors="ignore")
y_train = train[target]
X_test = test.drop(columns=[target], errors="ignore")
y_test = test[target]

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

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
if selected_names and st.button("🚀 Train selected models", type="primary"):
    progress = st.progress(0)
    status = st.empty()
    trial_status = st.empty()

    _total_trials = train_cfg.optuna.n_trials
    _total_models = len(selected_names)

    def on_trial_done(study, trial) -> None:
        """Called by Optuna after every trial — keeps the WebSocket alive."""
        best = study.best_value if study.best_trial else None
        best_str = f"  |  best so far: {best:.4f}" if best is not None else ""
        trial_status.caption(
            f"Trial {trial.number + 1}/{_total_trials}{best_str}"
        )

    def on_model_done(name: str, done: int, total: int) -> None:
        """Called after each model finishes — advances the progress bar."""
        progress.progress(done / total)
        status.info(f"✅ Completed {done}/{total}: **{name}**")
        trial_status.empty()
        # Show which model is up next
        if done < total:
            next_name = selected_names[done]  # done is 1-based, so index=done
            status.info(f"⏳ {done}/{total} done — training **{next_name}**…")

    # Show immediately which model is starting first
    status.info(f"⏳ 0/{_total_models} done — training **{selected_names[0]}**…")

    vslug = selected_variant["slug"]
    with st.spinner("Training…"):
        results = train_all(
            selected_names=selected_names,
            registry=registry,
            X_train=X_train,
            y_train=y_train,
            config=train_cfg,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            prep_config=prep_cfg,
            progress_callback=on_trial_done,
            model_done_callback=on_model_done,
            output_dir=project_models_dir(),
            variant_slug=vslug,
        )
    progress.progress(1.0)
    status.success(f"✅ All {len(results)} models trained!")

    # Individual model files already saved inside train_all (one per model as it completes).
    # Save the results cache as plain dicts — avoids Pydantic class-identity pickling errors
    # that occur when Streamlit re-executes the page script and reimports the class.
    # Pipelines are NOT stored here; load_variant_results() restores them from individual files.
    out_dir = project_models_dir()
    cache_path = out_dir / f"{vslug}_results_cache.joblib"
    save_joblib([r.model_dump(mode="python") for r in results], cache_path)

    # Config JSON for reproducibility
    cfg_path = out_dir / f"{vslug}_pipeline_config.json"
    save_json(train_cfg.model_dump(mode="json"), cfg_path)

    update_state({
        "results_cache_path": cache_path.name,
        "pipeline_config_path": cfg_path.name,
        "train_cfg": train_cfg.model_dump(mode="json"),
    })
    mark_stage_complete("modelling")

    # Show CV summary
    cv_scores = [r.cv_score for r in results if r.cv_score is not None]
    model_names = [r.model_name for r in results if r.cv_score is not None]
    st.plotly_chart(training_summary_bar(model_names, cv_scores, metric), use_container_width=True)

    # Per-model Optuna history
    for result in results:
        with st.expander(f"Optuna history — {result.model_name}"):
            st.plotly_chart(optuna_history_plot(result.optuna_history, result.model_name), use_container_width=True)

    st.switch_page("pages/8_comparison.py")
