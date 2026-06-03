"""Stage 8 — Model Comparison (Precision-Recall analysis across multiple variants)."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

# Locate MLPortal root by finding the directory that contains 'shared/'.
# Using a walk rather than a fixed .parent.parent depth makes this robust
# against Streamlit hot-reload resolving __file__ to a .pyc path
# (pages/__pycache__/...) which would make .parent.parent point to pages/
# instead of MLPortal/.
_here = Path(__file__).resolve()
for _candidate in (_here.parent, *_here.parents):
    if (_candidate / "shared").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import numpy as np
import pandas as pd
import streamlit as st

from shared.io import load_joblib, load_parquet, save_joblib
from shared.nav import render_sidebar
from shared.state import (
    get_state,
    list_trained_variants,
    load_variant_results,
    mark_stage_complete,
    project_comparison_dir,
    project_datasets_dir,
    project_models_dir,
    update_state,
)
from stages.comparison.evaluator import (
    bland_altman_stats,
    calibration_by_decile,
    compare_models,
    compute_pr_analysis,
    subgroup_eval,
)
from stages.comparison.plots import (
    bland_altman_plot,
    calibration_plot,
    confusion_matrix_heatmap,
    equity_bar,
    metric_comparison_bar,
    pr_curves_grid_plot,
    regression_outcome_plot,
)

st.set_page_config(page_title="8 — Comparison", page_icon="📊", layout="wide")
render_sidebar()

st.title("Stage 8 — Model Comparison")
st.caption(
    "Compare trained models across multiple dataset variants. "
    "Set a precision target, rank combinations, then drill into the best one."
)

state = get_state()

trained_variants = list_trained_variants()
if not trained_variants:
    st.warning("⚠️ No trained model results found. Complete Stage 7 (Modelling) first.")
    st.stop()

# ── Cache paths ───────────────────────────────────────────────────────────────
_cmp_dir = project_comparison_dir()
_PRED_CACHE = _cmp_dir / "pr_predictions.joblib"   # {dataset__model → entry}
_RES_CACHE = _cmp_dir / "pr_results.joblib"        # last run config + pr_summary

# ── Session state: restore from disk on first load ────────────────────────────
_KEY = "pr_analysis"
if _KEY not in st.session_state:
    if _RES_CACHE.exists():
        try:
            st.session_state[_KEY] = load_joblib(_RES_CACHE)
        except Exception:
            st.session_state[_KEY] = None
    else:
        st.session_state[_KEY] = None

# Tracks the user's explicit table selection as "Dataset | Model" string
if "inv_combo" not in st.session_state:
    st.session_state["inv_combo"] = None

# ── Section A — Configuration ─────────────────────────────────────────────────
with st.expander("⚙️ Analysis configuration", expanded=True):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        mcid = st.number_input(
            "MCID threshold (pts)",
            min_value=0.1, max_value=100.0,
            value=7.0, step=0.5,
            help="Minimum Clinically Important Difference. Predictions below this are 'no clinical benefit'.",
        )
    with col_b:
        target_precision = st.slider(
            "Target precision",
            min_value=0.50, max_value=1.00,
            value=0.80, step=0.01, format="%.2f",
            help="Find the threshold achieving at least this precision with the highest possible recall.",
        )
    with col_c:
        p_label = int(target_precision * 100)
        _sort_options = [
            f"Recall@P{p_label}", f"Thr@P{p_label}", f"Prec@P{p_label}",
            "AP", "Recall@MCID", "Precision@MCID", "F2@MCID",
        ]
        sort_by = st.multiselect(
            "Sort results by",
            options=_sort_options,
            default=[f"Recall@P{p_label}", f"Thr@P{p_label}"],
            help="Sort columns (descending, nulls last). Updating this is instant — no re-run needed.",
        )
    with col_d:
        top_n = st.number_input(
            "Top N results",
            min_value=1, max_value=50,
            value=10,
            help="How many top-ranked combinations to show in the table and PR grid.",
        )

# ── Section B — Dataset & Model multi-select ──────────────────────────────────
st.subheader("Select datasets and models")

variant_name_map: dict[str, dict] = {v["name"]: v for v in trained_variants}

selected_dataset_names: list[str] = st.multiselect(
    "Datasets",
    options=list(variant_name_map.keys()),
    default=list(variant_name_map.keys()),
)

if not selected_dataset_names:
    st.info("Select at least one dataset to continue.")
    st.stop()

# Collect all model names available for selected datasets (read from disk glob)
def _available_models(slugs: list[str]) -> list[str]:
    names: set[str] = set()
    mdir = project_models_dir()
    for slug in slugs:
        for f in mdir.glob(f"{slug}__*.joblib"):
            names.add(f.name.split("__")[1].replace(".joblib", ""))
    return sorted(names)

_selected_slugs = [variant_name_map[n]["slug"] for n in selected_dataset_names]
all_model_names = _available_models(_selected_slugs)

selected_model_names: list[str] = st.multiselect(
    "Models",
    options=all_model_names,
    default=all_model_names,
)

if not selected_model_names:
    st.info("Select at least one model to continue.")
    st.stop()

# ── Stale detection & Run button ──────────────────────────────────────────────
_current_config = {
    "mcid": float(mcid),
    "target_precision": float(target_precision),
    "datasets": sorted(selected_dataset_names),
    "models": sorted(selected_model_names),
}

_stored = st.session_state.get(_KEY)
_stored_config = _stored.get("config", {}) if _stored else {}
_is_stale = (
    _stored is None
    or {k: _stored_config.get(k) for k in ("mcid", "target_precision", "datasets", "models")}
    != _current_config
)

col_btn, col_status = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("▶ Run Analysis", type="primary", use_container_width=True)
with col_status:
    if _stored is None:
        st.info("No results yet — click **Run Analysis** to get started.")
    elif _is_stale:
        _last_ts = _stored_config.get("run_at", "unknown time")
        st.warning(
            f"⚠️ Settings changed since last run ({_last_ts}). "
            "Click **Run Analysis** to update, or keep browsing previous results below."
        )
    else:
        st.success(f"✓ Results from {_stored_config.get('run_at', '')}")

# ── Helper: load predictions for a single dataset+model ──────────────────────
def _load_single_entry(variant: dict, model_name: str) -> dict | None:
    """Load test data + run predictions for one dataset+model pair."""
    mdir = project_models_dir()
    ddir = project_datasets_dir()
    slug = variant["slug"]
    target = variant.get("target_column") or get_state().upload_cfg.get("target_column", "")
    test_path = variant.get("test_path")
    if not test_path:
        return None
    try:
        test_df = load_parquet(ddir / test_path)
    except Exception:
        return None
    if target not in test_df.columns:
        return None

    y_test = test_df[target].values
    X_test = test_df.drop(columns=[target], errors="ignore")
    task = variant.get("task_type", "regression")

    # Try individual model file first, then fall back to results cache
    model_file = mdir / f"{slug}__{model_name}.joblib"
    if model_file.exists():
        try:
            pipeline = load_joblib(model_file)
        except Exception:
            pipeline = None
    else:
        results = load_variant_results(slug, mdir, task)
        pipeline = next(
            (r.pipeline for r in results if r.model_name == model_name and r.pipeline),
            None,
        )

    if pipeline is None:
        return None

    try:
        Xa = (
            X_test.reindex(columns=pipeline.feature_names_in_, fill_value=np.nan)
            if hasattr(pipeline, "feature_names_in_")
            else X_test
        )
        y_pred = pipeline.predict(Xa)
    except Exception:
        return None

    return {
        "dataset": variant.get("name", slug),
        "slug": slug,
        "model_name": model_name,
        "task": task,
        "target": target,
        "test_path": test_path,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ── Run analysis (incremental) ────────────────────────────────────────────────
if run_clicked:
    # Load existing predictions cache
    pred_cache: dict = {}
    if _PRED_CACHE.exists():
        try:
            pred_cache = load_joblib(_PRED_CACHE)
        except Exception:
            pred_cache = {}

    all_entries: list[dict] = []
    new_count = 0
    skip_count = 0
    total_combos = len(selected_dataset_names) * len(selected_model_names)
    _prog = st.progress(0, text="Preparing…")

    for i, ds_name in enumerate(selected_dataset_names):
        variant = variant_name_map[ds_name]
        for j, mdl_name in enumerate(selected_model_names):
            combo_idx = i * len(selected_model_names) + j + 1
            _prog.progress(
                combo_idx / total_combos,
                text=f"[{combo_idx}/{total_combos}] {ds_name} | {mdl_name}",
            )
            key = f"{ds_name}__{mdl_name}"
            if key in pred_cache:
                all_entries.append(pred_cache[key])
            else:
                entry = _load_single_entry(variant, mdl_name)
                if entry:
                    pred_cache[key] = entry
                    all_entries.append(entry)
                    new_count += 1
                else:
                    skip_count += 1

    _prog.empty()

    # Persist predictions cache so future runs are fast
    save_joblib(pred_cache, _PRED_CACHE)

    # Compute PR metrics for regression entries
    regression_entries = [e for e in all_entries if e["task"] == "regression"]
    if regression_entries:
        with st.spinner("Computing precision-recall metrics…"):
            pr_summary = compute_pr_analysis(
                regression_entries, mcid=float(mcid), target_precision=float(target_precision)
            )
    else:
        pr_summary = []

    # Persist results
    _result = {
        "config": {
            **_current_config,
            "run_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "pr_summary": pr_summary,
    }
    save_joblib(_result, _RES_CACHE)
    st.session_state[_KEY] = _result

    _msg = f"✓ Done. {new_count} new combinations computed"
    if skip_count:
        _msg += f", {skip_count} skipped (model file not found)"
    if len(all_entries) - new_count > 0:
        _msg += f", {len(all_entries) - new_count} loaded from cache"
    st.success(_msg)
    st.rerun()

# ── Section C — PR analysis table + PR grid ───────────────────────────────────
st.divider()
st.subheader("Precision-Recall analysis")

_stored_now = st.session_state.get(_KEY)
pr_summary_display: list[dict] = _stored_now.get("pr_summary", []) if _stored_now else []
pr_summary_top: list[dict] = []
pr_summary_sorted: list[dict] = []

if not pr_summary_display:
    if _stored_now is None:
        st.info("Click **▶ Run Analysis** above to compute results.")
    else:
        st.info("No precision-recall results in last run (no regression models selected?).")
else:
    # Build display DataFrame (strip private _ keys)
    display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in pr_summary_display]
    pr_df = pd.DataFrame(display_rows)

    # Sort by user-chosen columns (descending, nulls last)
    if sort_by:
        valid_sort = [c for c in sort_by if c in pr_df.columns]
        if valid_sort:
            pr_df = pr_df.sort_values(valid_sort, ascending=False, na_position="last").reset_index(drop=True)

    # Re-align pr_summary to sorted order
    key_to_row = {(r["Dataset"], r["Model"]): r for r in pr_summary_display}
    pr_summary_sorted = [
        key_to_row[(row["Dataset"], row["Model"])]
        for _, row in pr_df.iterrows()
        if (row["Dataset"], row["Model"]) in key_to_row
    ]

    top_n_int = int(top_n)
    pr_df_top = pr_df.head(top_n_int)
    pr_summary_top = pr_summary_sorted[:top_n_int]

    # Clickable table — clicking a row sets it as the investigation target
    try:
        _sel_event = st.dataframe(
            pr_df_top.set_index(["Dataset", "Model"]),
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="pr_table",
        )
        # on_select stores the row index; look up the actual combo string
        if _sel_event.selection.rows:
            _row_idx = _sel_event.selection.rows[0]
            _ds = pr_df_top.iloc[_row_idx]["Dataset"]
            _mdl = pr_df_top.iloc[_row_idx]["Model"]
            st.session_state["inv_combo"] = f"{_ds} | {_mdl}"
    except Exception:
        # Fallback for Streamlit versions that don't support on_select
        st.dataframe(pr_df_top.set_index(["Dataset", "Model"]), use_container_width=True)

    st.caption(
        f"**Click a row** to select a model for investigation below.  "
        f"Diamond = MCID operating point ({mcid:.3g} pts).  "
        f"Star = tuned P{p_label} threshold.  "
        f"Sorting and Top-N update instantly."
    )

    # Show which combo is currently selected
    if st.session_state["inv_combo"]:
        # Check if the stored selection is still in the current top-N table
        _combo_in_table = any(
            f"{r['Dataset']} | {r['Model']}" == st.session_state["inv_combo"]
            for r in pr_summary_top
        )
        _badge = "✓" if _combo_in_table else "↓"
        st.success(
            f"{_badge} Selected for investigation: **{st.session_state['inv_combo']}** "
            + ("" if _combo_in_table else "*(not in current top-N view, but still loaded below)*")
        )
    else:
        st.info("Click a row in the table to select a model and generate investigation plots.")

    st.plotly_chart(
        pr_curves_grid_plot(pr_summary_top, target_precision=float(target_precision), mcid=float(mcid)),
        use_container_width=True,
    )

# ── Section E — Detailed investigation (only when user has selected a row) ─────
if st.session_state["inv_combo"]:
    _inv_dataset, _inv_model = [s.strip() for s in st.session_state["inv_combo"].split("|", 1)]

    st.divider()
    st.subheader(f"Investigation — {_inv_model} on {_inv_dataset}")

    # Find PR row for the tuned threshold value
    inv_pr_row = next(
        (r for r in pr_summary_display if r["Dataset"] == _inv_dataset and r["Model"] == _inv_model),
        None,
    )
    tuned_thr = inv_pr_row["_tp_thr"] if inv_pr_row else None

    # Load predictions: check cache first, then compute fresh
    _pred_cache_inv: dict = {}
    if _PRED_CACHE.exists():
        try:
            _pred_cache_inv = load_joblib(_PRED_CACHE)
        except Exception:
            _pred_cache_inv = {}

    _inv_key = f"{_inv_dataset}__{_inv_model}"
    _inv_variant = variant_name_map.get(_inv_dataset)

    inv_entry = _pred_cache_inv.get(_inv_key)
    if inv_entry is None and _inv_variant:
        with st.spinner(f"Loading {_inv_model} predictions for {_inv_dataset}…"):
            inv_entry = _load_single_entry(_inv_variant, _inv_model)
            if inv_entry:
                _pred_cache_inv[_inv_key] = inv_entry
                save_joblib(_pred_cache_inv, _PRED_CACHE)

    if inv_entry is None:
        st.warning(
            f"Could not load predictions for **{_inv_model}** on **{_inv_dataset}**. "
            "Run Analysis first to compute and cache this combination."
        )
    else:
        y_test_inv = np.asarray(inv_entry["y_test"], dtype=float)
        y_pred_inv = inv_entry["y_pred"]
        task_inv = inv_entry["task"]
        inv_slug = inv_entry["slug"]
        inv_target = inv_entry["target"]
        inv_test_path = inv_entry["test_path"]

        # Load full variant (all models) for Bland-Altman and metric comparison bars
        with st.spinner(f"Evaluating all models in {_inv_dataset}…"):
            inv_results = load_variant_results(inv_slug, project_models_dir(), task_inv)
            inv_test_df = load_parquet(project_datasets_dir() / inv_test_path)
            inv_X_test = inv_test_df.drop(columns=[inv_target], errors="ignore")
            inv_y_series = inv_test_df[inv_target]
            inv_comparison = compare_models(inv_results, inv_X_test, inv_y_series, task_inv, dataset=_inv_dataset)

        # Persist evaluated metrics so Stage 10 can read them
        _inv_cache_path = project_models_dir() / f"{inv_slug}_results_cache.joblib"
        save_joblib([r.model_dump() for r in inv_results], _inv_cache_path)
        update_state({
            "best_model_name": _inv_model,
            "comparison_table": [r.model_dump() for r in inv_comparison.rows],
            "metrics_summary": {
                "best_model": _inv_model,
                "best_dataset": _inv_dataset,
                "tuned_threshold": tuned_thr,
            },
        })

        # E1: Confusion matrices (regression)
        if task_inv == "regression":
            st.markdown("#### Confusion matrices at MCID and tuned threshold")
            st.plotly_chart(
                confusion_matrix_heatmap(
                    y_test_inv, y_pred_inv,
                    mcid=float(mcid),
                    tuned_thr=tuned_thr,
                    model_name=_inv_model,
                    dataset_name=_inv_dataset,
                ),
                use_container_width=True,
            )

            # E2: 5-panel outcome distribution + curves
            st.markdown("#### Outcome distribution & threshold curves")

            _thr_options = {f"Default MCID ({mcid:.3g} pts)": float(mcid)}
            if tuned_thr is not None:
                _thr_options[f"Optimised P{p_label} ({tuned_thr:.3f} pts)"] = float(tuned_thr)

            _thr_label = st.radio(
                "Threshold",
                options=list(_thr_options.keys()),
                horizontal=True,
                key="outcome_thr_choice",
                help=(
                    "Default MCID uses the fixed clinical threshold. "
                    f"Optimised P{p_label} uses the threshold that achieves "
                    f"{target_precision:.0%} precision with the highest recall for this model."
                    if tuned_thr is not None
                    else "No optimised threshold found — target precision may not be achievable."
                ),
            )
            _active_thr = _thr_options[_thr_label]

            st.plotly_chart(
                regression_outcome_plot(
                    y_test_inv, y_pred_inv,
                    threshold=_active_thr,
                    model_name=_inv_model,
                    positive_below=True,
                    positive_label=f"No benefit (gain < {_active_thr:.3g})",
                    negative_label=f"Benefit (gain ≥ {_active_thr:.3g})",
                ),
                use_container_width=True,
            )

            # E3: Bland-Altman + calibration
            st.markdown("#### Bland-Altman & calibration")
            ba_all = [
                bland_altman_stats(inv_y_series, r.pipeline.predict(inv_X_test), r.model_name)
                for r in inv_results
                if r.pipeline is not None
            ]
            cal = calibration_by_decile(inv_y_series, y_pred_inv, _inv_model)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bland_altman_plot(ba_all, _inv_model), use_container_width=True)
            with col2:
                st.plotly_chart(calibration_plot(cal, _inv_model), use_container_width=True)

        # E4: Metric comparison bars (all models in variant)
        st.markdown("#### Metric comparison — all models in selected dataset")
        _metric_options = {
            "regression": [("test_rmse", True), ("test_mae", True), ("test_r2", False)],
            "classification": [("f2", False), ("roc_auc", False), ("pr_auc", False), ("recall", False)],
            "ordinal": [("ordinal_mae", True), ("exact_accuracy", False), ("adjacent_accuracy", False)],
        }
        for metric, lower_is_better in _metric_options.get(task_inv, []):
            st.plotly_chart(metric_comparison_bar(inv_comparison, metric, lower_is_better), use_container_width=True)

        # E5: Equity / subgroup analysis
        _MAX_GROUPS = 50
        inv_group_cols = [
            c for c in inv_test_df.columns
            if c != inv_target and inv_test_df[c].nunique() <= _MAX_GROUPS
        ]
        if inv_group_cols:
            st.markdown("#### Equity / subgroup analysis")
            group_col = st.selectbox("Groupby column", inv_group_cols, key="inv_group_col")
            metric_eq = st.selectbox(
                "Metric",
                ["rmse", "mae"] if task_inv == "regression" else ["f1", "accuracy"],
                key="inv_metric_eq",
            )
            sg = subgroup_eval(
                inv_y_series, y_pred_inv,
                inv_test_df[group_col].reset_index(drop=True),
                _inv_model, metric_eq,
            )
            st.plotly_chart(equity_bar(sg, _inv_model, metric_eq), use_container_width=True)

st.divider()
if st.button("Continue to Clinical Insight →", type="primary"):
    mark_stage_complete("comparison")
    st.switch_page("pages/9_clinical_insight.py")
