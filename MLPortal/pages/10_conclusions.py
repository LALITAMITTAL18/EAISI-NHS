"""Stage 10 — Conclusions & Export."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import shared  # pre-register parent package before submodule imports (Python 3.13 hot-reload guard)
import streamlit as st

from shared.io import load_joblib, load_parquet
from shared.nav import render_sidebar
from shared.state import (
    get_state,
    list_trained_variants,
    mark_stage_complete,
    project_models_dir,
    project_reports_dir,
    update_state,
)
from stages.comparison.models import ComparisonResult, MetricRow
from stages.comparison.plots import regression_outcome_plot, threshold_explorer_plot
from stages.conclusions.models import PipelineRunSummary
from stages.conclusions.reporter import (
    export_comparison_csv,
    export_html_report,
)

st.set_page_config(page_title="10 — Conclusions", page_icon="🏁", layout="wide")
render_sidebar()

st.title("Stage 10 — Conclusions & Export")
st.caption("Review the best model, compare all results, document your findings and export reports.")

state = get_state()
_BASE = Path(__file__).parent.parent
task = state.upload_cfg.get("task_type", "regression")
target = state.upload_cfg.get("target_column", "")

# ── Load cached metrics from ALL trained variants ─────────────────────────────
_mdir = project_models_dir()
_trained_variants = list_trained_variants()

_PRIMARY: dict[str, tuple[str, bool]] = {
    "regression": ("test_rmse", True),
    "classification": ("roc_auc", False),
    "ordinal": ("exact_accuracy", False),
}
_pk, _lower = _PRIMARY.get(task, ("test_rmse", True))

_all_metric_rows: list[dict] = []
_all_entries_full: list[dict] = []  # includes best_params for hyperparameter display

for _v in _trained_variants:
    _slug = _v["slug"]
    _cf = _mdir / f"{_slug}_results_cache.joblib"
    if not _cf.exists():
        continue
    try:
        _raw = load_joblib(_cf)
    except Exception:
        continue
    for _item in _raw:
        if isinstance(_item, dict):
            _mn = _item.get("model_name", "")
            _tm = _item.get("test_metrics", {}) or {}
            _bp = _item.get("best_params", {}) or {}
        elif hasattr(_item, "model_name"):
            _mn = _item.model_name
            _tm = getattr(_item, "test_metrics", {}) or {}
            _bp = getattr(_item, "best_params", {}) or {}
        else:
            continue
        if _mn:
            _all_entries_full.append({
                "dataset": _v["name"],
                "model_name": _mn,
                "best_params": _bp,
                "test_metrics": _tm,
            })
            if _tm:
                _row = {"dataset": _v["name"], "model_name": _mn, "task": _v["task_type"]}
                _row.update(_tm)
                _all_metric_rows.append(_row)

# Sort by primary metric (best first)
def _metric_val(row: dict) -> float:
    v = row.get(_pk)
    if v is None:
        return float("inf") if _lower else float("-inf")
    return v if _lower else -v

_all_metric_rows.sort(key=_metric_val)

# Global best model (across all datasets)
_global_best = _all_metric_rows[0] if _all_metric_rows else None

# ── Global best model card ────────────────────────────────────────────────────
if _global_best:
    st.subheader(
        f"🏆 Best model across all datasets: "
        f"{_global_best['model_name']}  [{_global_best['dataset']}]"
    )
    _metric_cols = [c for c in _global_best if c not in ("dataset", "model_name", "task")]
    _mc = st.columns(min(4, max(1, len(_metric_cols))))
    for _i, _mk in enumerate(_metric_cols[:4]):
        _mv = _global_best[_mk]
        _mc[_i].metric(_mk, f"{_mv:.4f}" if isinstance(_mv, float) else str(_mv))
    st.caption(f"Task: {_global_best.get('task', task).replace('_', ' ').title()}")
elif state.best_model_name:
    # Fallback: metrics not yet cached — Stage 8 hasn't been run for all variants.
    st.subheader(f"🏆 Best model (last Stage 8 run): {state.best_model_name}")
    metrics = state.metrics_summary
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best metric", metrics.get("best_metric", "—"))
    with c2:
        st.metric("Best value", f"{metrics.get('best_value', 0):.4f}" if metrics.get("best_value") else "—")
    with c3:
        st.metric("Task", task.replace("_", " ").title())
    st.caption("Run Stage 8 (Comparison) for all dataset variants to see the global best model here.")

# ── Performance matrix — all datasets × all models ────────────────────────────
st.subheader("Performance matrix — all datasets × all models")
st.caption(
    f"Results from all trained variants, sorted by **{_pk}** "
    f"({'lower = better' if _lower else 'higher = better'})."
)

if _all_metric_rows:
    _all_df = pd.DataFrame(_all_metric_rows)
    _front = [c for c in ["dataset", "model_name", "task"] if c in _all_df.columns]
    _rest = [c for c in _all_df.columns if c not in _front]
    _all_df = _all_df[_front + _rest].dropna(axis=1, how="all")

    # Highlight best value per metric column
    _metric_cols_df = [c for c in _rest if _all_df[c].dtype in ("float64", "float32")]

    def _highlight_best(s: pd.Series) -> list[str]:
        lower_better = _pk in s.name and _lower
        try:
            best_val = s.min() if lower_better else s.max()
            return ["font-weight: bold; background-color: #d4edda" if v == best_val else "" for v in s]
        except Exception:
            return [""] * len(s)

    styled = _all_df.set_index(["dataset", "model_name"])
    if _metric_cols_df:
        try:
            styled = styled.style.apply(_highlight_best, subset=_metric_cols_df)
        except Exception:
            pass

    st.dataframe(styled, use_container_width=True)

elif state.comparison_table:
    # Fallback: use the last run saved to state
    st.caption("(Showing last run from Stage 8 — run Stage 8 for all variants to populate this table.)")
    comp_df = pd.DataFrame(state.comparison_table).dropna(axis=1, how="all")
    st.dataframe(comp_df.set_index("model_name"), use_container_width=True)
else:
    st.info("No evaluation results found. Complete Stage 8 (Comparison) first.")

# ── Precision-constrained recall ranking — all models × all datasets ──────────
st.divider()
st.subheader("Best recall at target precision — all models × all datasets")
st.caption(
    "Set a minimum precision you need to achieve. "
    "All model + dataset combinations are evaluated; those that can meet the precision target "
    "are ranked from highest to lowest recall."
)

if _trained_variants:
    import numpy as _np_rank
    from sklearn.metrics import precision_recall_curve as _prc
    from shared.state import project_datasets_dir as _pdd

    # ── Inputs ───────────────────────────────────────────────────────────────
    _rk_col_prec, _rk_col_mcid, _rk_col_sort = st.columns([1, 1, 2])
    with _rk_col_prec:
        _target_prec = st.number_input(
            "Target precision",
            min_value=0.01, max_value=1.0, value=0.80, step=0.01, format="%.2f",
            key="conc_target_prec",
            help="The model must achieve at least this precision at some threshold. "
                 "We find the threshold that maximises recall while meeting this constraint.",
        )
    with _rk_col_mcid:
        if task == "regression":
            _mcid_val = st.number_input(
                "MCID (clinical boundary)",
                min_value=0.0, max_value=200.0,
                value=float(st.session_state.get("conc_mcid", 7.0)),
                step=0.5, format="%.1f",
                key="conc_mcid",
                help=(
                    "The minimum health gain considered clinically meaningful. "
                    "Patients with predicted gain below the MCID are 'No Benefit'. "
                    "For NHS knee replacement the MCID is 7."
                ),
            )
        else:
            _mcid_val = None
    with _rk_col_sort:
        if task == "regression":
            _rank_sort_mode = st.radio(
                "Rank models by",
                options=[
                    "Highest recall  (catch the most 'No Benefit' patients)",
                    "Threshold closest to MCID  (decision boundary near clinical rule)",
                ],
                index=0,
                key="conc_rank_sort",
                help=(
                    "**Highest recall**: finds models that detect the most 'No Benefit' patients at target precision. "
                    "May show negative thresholds (model is conservative).  \n"
                    "**Closest to MCID**: finds models whose decision boundary is nearest the clinical threshold of 7. "
                    "Matches the notebook Thr@P80 ranking — threshold stays positive and near MCID."
                ),
                horizontal=False,
            )
            _sort_by_threshold = _rank_sort_mode.startswith("Threshold")
        else:
            _sort_by_threshold = False

    # Regression needs an outcome class definition for binarisation
    if task == "regression":
        st.markdown("**Outcome class definition** (applies to all regression models)")
        _rk_c1, _rk_c2, _rk_c3, _rk_c4 = st.columns(4)
        with _rk_c1:
            _rk_dir = st.radio(
                "Class 1 = value",
                ["Below threshold  (< t)", "At or above  (≥ t)"],
                index=0,
                key="conc_rank_dir",
                horizontal=True,
            )
            _rk_positive_below = _rk_dir.startswith("Below")
        with _rk_c2:
            _rk_outcome_thresh = st.number_input(
                "Threshold (t)",
                value=float(st.session_state.get("conc_thresh_reg_val", 0.0)),
                min_value=float(st.session_state.get("conc_range_min", 0.0)),
                max_value=float(st.session_state.get("conc_range_max", 1e6)),
                step=0.001, format="%.3f",
                key="conc_rank_thresh",
            )
        with _rk_c3:
            _rk_pos_label = st.text_input(
                "Class 1 name", key="conc_rank_pos_lbl",
                value=st.session_state.get("conc_pos_lbl", "No benefit"),
            )
        with _rk_c4:
            _rk_neg_label = st.text_input(
                "Class 0 name", key="conc_rank_neg_lbl",
                value=st.session_state.get("conc_neg_lbl", "Benefit"),
            )

        # Acceptable range row — mirrors the one in the threshold explorer
        st.markdown("**Acceptable threshold range**")
        st.caption(
            "Must match the range set in the Outcome Threshold Explorer above. "
            "Values outside this range will not appear in the search."
        )
        _rk_ra, _rk_rb = st.columns(2)
        with _rk_ra:
            _rk_range_min = st.number_input(
                "Range minimum",
                value=float(st.session_state.get("conc_range_min", 0.0)),
                step=0.01, format="%.3f", key="conc_rank_range_min",
            )
        with _rk_rb:
            _rk_range_max = st.number_input(
                "Range maximum",
                value=float(st.session_state.get("conc_range_max", 100.0)),
                step=0.01, format="%.3f", key="conc_rank_range_max",
            )
    else:
        _rk_positive_below = False
        _rk_outcome_thresh = None
        _rk_pos_label = "Case"
        _rk_neg_label = "Non-case"
        _rk_range_min = None
        _rk_range_max = None

    # ── Compute button ────────────────────────────────────────────────────────
    if st.button("⚡ Rank models by recall at target precision", key="conc_rank_btn"):
        _rank_rows: list[dict] = []
        _cached_mcid_now = st.session_state.get("conc_mcid") if task == "regression" else None
        st.session_state["conc_rank_mcid"] = _cached_mcid_now
        st.session_state["conc_rank_sort_by_thr"] = _sort_by_threshold
        _all_pf = [
            (_v, pf)
            for _v in _trained_variants
            for pf in sorted(_mdir.glob(f"{_v['slug']}__*.joblib"))
        ]
        _prog = st.progress(0.0, text="Loading models…")

        for _i, (_v, _pf) in enumerate(_all_pf):
            _mn = _pf.name[len(f"{_v['slug']}__"):-len(".joblib")]
            _prog.progress((_i + 1) / max(len(_all_pf), 1), text=f"{_mn}  [{_v['name']}]")

            _tp = _v.get("test_path")
            if not _tp:
                continue
            try:
                import warnings
                _rv_test = load_parquet(_pdd() / _tp)
                _rv_tgt = _v["target_column"]
                if _rv_tgt not in _rv_test.columns:
                    continue
                _rv_y = _np_rank.asarray(_rv_test[_rv_tgt], dtype=float)
                _rv_X = _rv_test.drop(columns=[_rv_tgt], errors="ignore")

                if task == "regression":
                    _rv_y_bin = (
                        (_rv_y < _rk_outcome_thresh).astype(int)
                        if _rk_positive_below
                        else (_rv_y >= _rk_outcome_thresh).astype(int)
                    )
                else:
                    _rv_y_bin = _rv_y.astype(int)

                if _rv_y_bin.sum() == 0 or _rv_y_bin.sum() == len(_rv_y_bin):
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _rv_pipe = load_joblib(_pf)

                if task == "regression":
                    _rv_score = _np_rank.asarray(_rv_pipe.predict(_rv_X), dtype=float)
                    if _rk_positive_below:
                        _rv_score = -_rv_score
                else:
                    if not hasattr(_rv_pipe, "predict_proba"):
                        continue
                    _proba = _rv_pipe.predict_proba(_rv_X)
                    _rv_score = _proba[:, 1] if _proba.shape[1] == 2 else _proba.max(axis=1)

                _prec_c, _rec_c, _thr_c = _prc(_rv_y_bin, _rv_score)

                # Convert thresholds to original prediction scale.
                # For positive_below=True the score was negated (-y_pred), so
                # thresholds are in that negated space; negate back to get real y_pred values.
                # Matches notebook pattern: tp_pred_thr = -tp_score_thr
                _thr_c_orig = -_thr_c if _rk_positive_below else _thr_c

                # Exclude trivial last point (no threshold) — matches notebook: prec_c[:-1]
                _prec_inner = _prec_c[:-1]
                _rec_inner  = _rec_c[:-1]

                _valid_prec = _prec_inner >= _target_prec

                if task == "regression" and _rk_range_min is not None:
                    _in_range = (
                        (_thr_c_orig >= _rk_range_min) & (_thr_c_orig <= _rk_range_max)
                    )
                    _valid = _valid_prec & _in_range
                    _range_str = f"[{_rk_range_min:.3g}, {_rk_range_max:.3g}]"
                else:
                    _valid = _valid_prec
                    _range_str = None

                if not _valid.any():
                    _reason = (
                        "cannot reach target precision"
                        if not _valid_prec.any()
                        else f"threshold outside range {_range_str}"
                    )
                    _rank_rows.append({
                        "Model": _mn, "Dataset": _v["name"],
                        "achieves": False, "reason": _reason,
                        "Recall": None, "Precision": None, "Threshold": None,
                    })
                else:
                    if _sort_by_threshold and _cached_mcid_now is not None:
                        # Notebook-style: highest threshold (closest to MCID) at target precision
                        _bi = int(_np_rank.argmax(_thr_c_orig * _valid))
                    else:
                        # Portal default: highest recall at target precision
                        _bi = int(_np_rank.argmax(_rec_inner * _valid))
                    _rank_rows.append({
                        "Model": _mn, "Dataset": _v["name"],
                        "achieves": True,
                        "Recall": float(_rec_inner[_bi]),
                        "Precision": float(_prec_inner[_bi]),
                        "Threshold": float(_thr_c_orig[_bi]),
                    })
            except Exception:
                pass

        _prog.empty()
        st.session_state["conc_rank_rows"] = _rank_rows
        st.session_state["conc_rank_target"] = _target_prec
        st.session_state["conc_rank_pos_label"] = _rk_pos_label
        st.session_state["conc_rank_neg_label"] = _rk_neg_label

    # ── Display cached results ────────────────────────────────────────────────
    if "conc_rank_rows" in st.session_state:
        _cached_target = st.session_state.get("conc_rank_target", _target_prec)
        _cached_mcid        = st.session_state.get("conc_rank_mcid")
        _cached_sort_by_thr = st.session_state.get("conc_rank_sort_by_thr", False)
        _cached_pos         = st.session_state.get("conc_rank_pos_label", _rk_pos_label)
        _cached_neg         = st.session_state.get("conc_rank_neg_label", _rk_neg_label)
        if _cached_target != _target_prec:
            st.warning(
                f"Table shows results for target precision = **{_cached_target:.2f}**. "
                "Click the button above to recompute for the new target."
            )

        if _cached_sort_by_thr and _cached_mcid is not None:
            # Notebook-style: highest threshold (closest to MCID, not penalising negatives explicitly)
            # Sort descending by threshold value — models with threshold near/above MCID come first
            _achievers = sorted(
                [r for r in st.session_state["conc_rank_rows"] if r["achieves"]],
                key=lambda r: r["Threshold"] if r["Threshold"] is not None else float("-inf"),
                reverse=True,
            )
        else:
            _achievers = sorted(
                [r for r in st.session_state["conc_rank_rows"] if r["achieves"]],
                key=lambda r: r["Recall"] if r["Recall"] is not None else 0.0,
                reverse=True,
            )
        _non_achievers = [r for r in st.session_state["conc_rank_rows"] if not r["achieves"]]
        _top10 = _achievers[:10]

        if _achievers:
            # ── Threshold / MCID explanation ─────────────────────────────────
            if task == "regression" and _cached_mcid is not None:
                _mcid_lbl = f"{_cached_mcid:.1f}"
                with st.expander(
                    f"What does 'Pred. Health Gain Cutoff' mean? "
                    f"(and why can it be negative?)"
                ):
                    st.markdown(
                        f"**How the threshold works**  \n"
                        f"Each model produces a predicted health gain score for every patient. "
                        f"To classify patients as *{_cached_pos}* or *{_cached_neg}*, the model "
                        f"uses a cutoff — patients predicted below that value are flagged as "
                        f"**{_cached_pos}** (class 1).  \n\n"
                        f"The column **'Pred. Health Gain Cutoff'** shows that cutoff in the "
                        f"original health gain units. The **Gap to MCID** column shows how far "
                        f"this cutoff is from the clinical boundary of **{_mcid_lbl}**.\n\n"
                        f"| Cutoff value | What it means |\n"
                        f"|---|---|\n"
                        f"| **< 0** (negative) | Model only flags patients it predicts will *get worse* (lose health). "
                        f"Very conservative — most true '{_cached_pos}' patients (gain 0–{_mcid_lbl}) are missed. |\n"
                        f"| **0 – {_mcid_lbl}** | Conservative — only flags patients with very low predicted gain, "
                        f"missing those near the clinical boundary. |\n"
                        f"| **≈ {_mcid_lbl} (MCID)** | Ideal alignment — the model's decision boundary matches "
                        f"the clinical definition of benefit. |\n"
                        f"| **> {_mcid_lbl}** | Liberal — model also flags some patients who would clinically "
                        f"benefit, increasing recall but potentially reducing precision. |\n\n"
                        f"**A negative cutoff does NOT mean the model is wrong** — it means the model has "
                        f"set a very high bar for flagging 'No Benefit', resulting in high precision but "
                        f"low recall. Most patients with gain 0–7 will be missed."
                    )

            _sort_label = (
                f"closest threshold to MCID={_cached_mcid:.0f}"
                if _cached_sort_by_thr and _cached_mcid is not None
                else "highest recall"
            )
            st.success(
                f"**{len(_achievers)}** model(s) achieve precision ≥ {_cached_target:.2f} "
                f"(class 1 = **{_cached_pos}**). "
                f"Showing top {min(10, len(_achievers))} by **{_sort_label}**:"
            )

            # ── Build ranked table ────────────────────────────────────────────
            _rank_df = pd.DataFrame(_top10).drop(columns=["achieves"])
            _rank_df.insert(0, "Rank", range(1, len(_rank_df) + 1))

            # Add MCID-relative columns before formatting to strings
            if task == "regression" and _cached_mcid is not None:
                def _gap_label(x: float | None) -> str:
                    if x is None:
                        return "—"
                    gap = x - _cached_mcid
                    return f"{gap:+.2f}"

                def _context_label(x: float | None) -> str:
                    if x is None:
                        return "—"
                    gap = x - _cached_mcid
                    if x < 0:
                        return f"⚠ Only flags patients predicted to get WORSE (gain < {x:.1f})"
                    elif gap < -3:
                        return f"Conservative — {abs(gap):.1f} pts below clinical boundary"
                    elif abs(gap) <= 1.5:
                        return f"✓ Near MCID (gap {gap:+.2f})"
                    elif gap > 0:
                        return f"Liberal — {gap:.1f} pts above MCID, flags some 'Benefit' patients"
                    else:
                        return f"{gap:+.1f} pts from MCID"

                _rank_df["Gap to MCID"] = _rank_df["Threshold"].apply(_gap_label)
                _rank_df["Context"] = _rank_df["Threshold"].apply(_context_label)
                _rank_df = _rank_df.rename(
                    columns={"Threshold": f"Pred. Health Gain Cutoff (MCID={_cached_mcid:.0f})"}
                )
                _thresh_col = f"Pred. Health Gain Cutoff (MCID={_cached_mcid:.0f})"
            else:
                _thresh_col = "Threshold"

            _rank_df["Recall"]    = _rank_df["Recall"].apply(lambda x: f"{x:.3f}")
            _rank_df["Precision"] = _rank_df["Precision"].apply(lambda x: f"{x:.3f}")
            _rank_df[_thresh_col] = _rank_df[_thresh_col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else ("—" if x is None else str(x))
            )
            st.dataframe(_rank_df.set_index("Rank"), use_container_width=True)

            # ── Plain-language summary of best model ──────────────────────────
            _best = _top10[0]
            _best_thresh_raw = _best.get("Threshold")
            _best_recall     = _best["Recall"]
            _best_precision  = _best["Precision"]
            if task == "regression" and _cached_mcid is not None and _best_thresh_raw is not None:
                _gap = _best_thresh_raw - _cached_mcid
                if _best_thresh_raw < 0:
                    _thr_desc = (
                        f"only patients the model predicts will *lose* health (predicted gain ≤ "
                        f"**{_best_thresh_raw:.2f}**), which is {abs(_gap):.1f} pts below the MCID of {_cached_mcid:.0f}"
                    )
                elif abs(_gap) <= 1.5:
                    _thr_desc = (
                        f"patients with predicted health gain ≤ **{_best_thresh_raw:.2f}** — "
                        f"closely aligned with the clinical MCID of {_cached_mcid:.0f}"
                    )
                elif _gap < 0:
                    _thr_desc = (
                        f"patients with predicted health gain ≤ **{_best_thresh_raw:.2f}**, "
                        f"which is {abs(_gap):.1f} pts below the MCID of {_cached_mcid:.0f} "
                        f"(conservative — some 'No Benefit' patients near the boundary are missed)"
                    )
                else:
                    _thr_desc = (
                        f"patients with predicted health gain ≤ **{_best_thresh_raw:.2f}**, "
                        f"which is {_gap:.1f} pts above the MCID of {_cached_mcid:.0f} "
                        f"(some patients who would clinically benefit are also flagged)"
                    )
                st.info(
                    f"**Best model: {_best['Model']} [{_best['Dataset']}]**  \n"
                    f"At this threshold, the model correctly identifies "
                    f"**{_best_recall:.1%} of all '{_cached_pos}' patients** (recall = {_best_recall:.3f}) "
                    f"and **{_best_precision:.1%} of flagged patients are truly '{_cached_pos}'** "
                    f"(precision = {_best_precision:.3f}).  \n"
                    f"It flags {_thr_desc}."
                )
            else:
                st.info(
                    f"**Best model: {_best['Model']} [{_best['Dataset']}]**  \n"
                    f"Recall = **{_best_recall:.3f}**, Precision = **{_best_precision:.3f}** "
                    f"at this threshold."
                )

            if len(_achievers) > 10:
                st.caption(
                    f"{len(_achievers) - 10} additional model(s) also meet the precision target "
                    f"but are not shown. Lower the target precision to see more."
                )
        else:
            st.warning(
                f"No model + dataset combination can achieve precision ≥ {_cached_target:.2f} "
                f"for class **{_cached_pos}**. Try lowering the target precision."
            )

        if _non_achievers:
            with st.expander(
                f"{len(_non_achievers)} model(s) cannot achieve precision ≥ {_cached_target:.2f}"
            ):
                _na_df = pd.DataFrame(_non_achievers)
                _na_cols = [c for c in ["Model", "Dataset", "reason"] if c in _na_df.columns]
                st.dataframe(_na_df[_na_cols], hide_index=True, use_container_width=True)
else:
    st.info("No trained models found. Complete Stage 7 (Modelling) first.")

# ── Model detail — shared selector for confusion matrix + hyperparameters ──────
st.divider()
st.subheader("Model detail")

if _all_entries_full:
    _hp_labels = [
        f"{e['model_name']}  [{e['dataset']}]" for e in _all_entries_full
    ]
    _default_hp = (
        f"{_global_best['model_name']}  [{_global_best['dataset']}]"
        if _global_best and f"{_global_best['model_name']}  [{_global_best['dataset']}]" in _hp_labels
        else _hp_labels[0]
    )
    _sel_hp_label = st.selectbox(
        "Dataset / model:",
        _hp_labels,
        index=_hp_labels.index(_default_hp),
        key="hp_model_sel",
    )
    _sel_hp_entry = _all_entries_full[_hp_labels.index(_sel_hp_label)]
    _sel_hp_slug = next(
        (v["slug"] for v in _trained_variants if v["name"] == _sel_hp_entry["dataset"]),
        None,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    _tabs = st.tabs(["Outcome Threshold Explorer", "Hyperparameters"])

    # ── Tab: Outcome Threshold Explorer ──────────────────────────────────────
    with _tabs[0]:
        if _sel_hp_slug:
            _pipe_f = _mdir / f"{_sel_hp_slug}__{_sel_hp_entry['model_name']}.joblib"
            _sel_variant_meta = next(
                (v for v in _trained_variants if v["slug"] == _sel_hp_slug), None
            )
            if _pipe_f.exists() and _sel_variant_meta and _sel_variant_meta.get("test_path"):
                import warnings
                import numpy as np
                from sklearn.metrics import roc_auc_score, average_precision_score
                from shared.state import project_datasets_dir

                with st.spinner("Loading pipeline and test data…"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _th_pipe = load_joblib(_pipe_f)
                    _th_test = load_parquet(
                        project_datasets_dir() / _sel_variant_meta["test_path"]
                    )
                    _th_target = _sel_variant_meta["target_column"]
                    _th_X = _th_test.drop(columns=[_th_target], errors="ignore")
                    _th_y = np.asarray(_th_test[_th_target], dtype=float)

                def _synced_threshold_input(
                    label: str, t_min: float, t_max: float, t_default: float,
                    fmt: str, step: float, sk: str,
                ) -> float:
                    """Slider + number input that stay in sync via session state."""
                    _sv = f"{sk}_val"
                    _ss = f"{sk}_s"
                    _sn = f"{sk}_n"
                    if _sv not in st.session_state:
                        st.session_state[_sv] = t_default

                    def _from_slider():
                        st.session_state[_sv] = st.session_state[_ss]

                    def _from_num():
                        st.session_state[_sv] = st.session_state[_sn]

                    _c1, _c2 = st.columns([3, 1])
                    with _c1:
                        st.slider(
                            label, min_value=t_min, max_value=t_max,
                            value=float(st.session_state[_sv]),
                            step=step, format=fmt, key=_ss,
                            on_change=_from_slider,
                        )
                    with _c2:
                        st.number_input(
                            "Exact value", min_value=float(t_min),
                            max_value=float(t_max),
                            value=float(st.session_state[_sv]),
                            step=step * 5, format=fmt, key=_sn,
                            on_change=_from_num,
                        )
                    return float(st.session_state[_sv])

                def _interpretation(
                    TP: int, FN: int, FP: int, TN: int,
                    score: np.ndarray, y_bin: np.ndarray,
                    pos_label: str, neg_label: str,
                    threshold_label: str,
                    positive_below: bool = False,
                ) -> None:
                    """Render a plain-language summary using the user-defined class labels."""
                    total = TP + FN + FP + TN
                    n_pos = TP + FN
                    n_neg = FP + TN
                    recall      = TP / n_pos if n_pos > 0 else 0.0
                    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                    specificity = TN / n_neg if n_neg > 0 else 0.0
                    try:
                        _auc = float(roc_auc_score(y_bin, score))
                        _ap  = float(average_precision_score(y_bin, score))
                    except Exception:
                        _auc = _ap = None

                    st.divider()
                    st.markdown("**Plain-language interpretation**")
                    cols = st.columns(4)
                    cols[0].metric(
                        "Recall (sensitivity)", f"{recall:.1%}",
                        help=f"Of all patients actually labelled **{pos_label}**, how many did the model correctly flag?",
                    )
                    cols[1].metric(
                        "Precision (PPV)", f"{precision:.1%}",
                        help=f"Of everyone the model flagged as **{pos_label}**, how many actually were?",
                    )
                    cols[2].metric(
                        "Specificity", f"{specificity:.1%}",
                        help=f"Of all patients actually labelled **{neg_label}**, how many did the model correctly leave unflagged?",
                    )
                    cols[3].metric(
                        "ROC AUC", f"{_auc:.3f}" if _auc else "—",
                        help="How well the model ranks patients — 1.0 = perfect, 0.5 = random. Independent of the threshold.",
                    )

                    st.markdown(
                        f"At **{threshold_label}** across **{total:,}** patients:\n\n"
                        f"- **✅ {TP:,} True Positives** — predicted **{pos_label}** and actually **{pos_label}**. "
                        f"The model correctly identifies these patients.\n"
                        f"- **❌ {FN:,} False Negatives** — actually **{pos_label}** but model predicted **{neg_label}**. "
                        f"These patients are missed — the model under-predicts this outcome for them.\n"
                        f"- **⚠️ {FP:,} False Positives** — predicted **{pos_label}** but actually **{neg_label}**. "
                        f"The model over-predicts this outcome — these patients are incorrectly flagged.\n"
                        f"- **✅ {TN:,} True Negatives** — predicted **{neg_label}** and actually **{neg_label}**. "
                        f"The model correctly leaves these patients unflagged.\n"
                    )

                    # Direction-aware threshold advice
                    # "raise threshold" means moving it in the direction that reduces TP
                    # If positive_below: increasing t catches more positives (lowers threshold means tighter)
                    _raise = "raising" if not positive_below else "lowering"
                    _lower = "lowering" if not positive_below else "raising"

                    if recall < 0.5 and precision > 0.7:
                        advice = (
                            f"The model is **conservative** — it misses {FN:,} patients who are actually **{pos_label}** "
                            f"(low recall = {recall:.0%}). Consider **{_lower} the threshold** to flag more of them, "
                            "accepting some additional false positives."
                        )
                    elif recall > 0.8 and precision < 0.4:
                        advice = (
                            f"The model casts a **wide net** — {FP:,} flagged patients are not actually **{pos_label}** "
                            f"(low precision = {precision:.0%}). Consider **{_raise} the threshold** to reduce false positives, "
                            "accepting that some true positives will be missed."
                        )
                    else:
                        _f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        advice = (
                            f"The threshold gives a reasonable **balance** — recall {recall:.0%}, "
                            f"precision {precision:.0%}, F1 = {_f1:.2f}. "
                            f"Consider the clinical cost: is it worse to miss a **{pos_label}** patient (↓ recall) "
                            f"or to incorrectly flag a **{neg_label}** patient (↓ precision)?"
                        )
                    st.info(advice)

                    if _auc is not None:
                        st.caption(
                            f"**ROC AUC = {_auc:.3f}** — the model correctly ranks {_auc:.1%} of patient pairs "
                            f"(**{pos_label}** vs **{neg_label}**) by predicted score, regardless of the threshold chosen. "
                            f"Average Precision = {_ap:.3f}."
                        )

                # ── Regression branch ─────────────────────────────────────────
                if task == "regression":
                    _data_min  = float(np.nanmin(_th_y))
                    _data_max  = float(np.nanmax(_th_y))
                    _y_default = float(np.nanmedian(_th_y))

                    # ── User defines what "1" means for this project ──────────
                    st.markdown("**Define outcome classes**")
                    st.caption(
                        "Specify which side of the threshold is class 1 (positive) "
                        "and give each class a meaningful name for this project."
                    )
                    _dir_col, _pos_lbl_col, _neg_lbl_col = st.columns(3)
                    with _dir_col:
                        _pos_dir = st.radio(
                            "Class 1 (positive) = predicted value is",
                            options=["Below threshold  (< t)", "At or above threshold  (≥ t)"],
                            index=0,
                            key="conc_pos_dir",
                            help="e.g. NHS 'no benefit' = gain < 7 → choose 'Below threshold'",
                        )
                        _positive_below = _pos_dir.startswith("Below")
                    with _pos_lbl_col:
                        _pos_label = st.text_input(
                            "Name for class 1:", value="No benefit",
                            key="conc_pos_lbl",
                        )
                    with _neg_lbl_col:
                        _neg_label = st.text_input(
                            "Name for class 0:", value="Benefit",
                            key="conc_neg_lbl",
                        )

                    # ── Acceptable threshold range ────────────────────────────
                    st.markdown("**Acceptable threshold range**")
                    st.caption(
                        f"Data range: {_data_min:.3g} to {_data_max:.3g}. "
                        "Set the min/max to restrict the slider to clinically meaningful values."
                    )
                    _rng_col1, _rng_col2 = st.columns(2)
                    with _rng_col1:
                        _y_min = st.number_input(
                            "Slider minimum",
                            value=float(st.session_state.get("conc_range_min", _data_min)),
                            min_value=_data_min, max_value=_data_max,
                            step=0.01, format="%.3f", key="conc_range_min",
                        )
                    with _rng_col2:
                        _y_max = st.number_input(
                            "Slider maximum",
                            value=float(st.session_state.get("conc_range_max", _data_max)),
                            min_value=_data_min, max_value=_data_max,
                            step=0.01, format="%.3f", key="conc_range_max",
                        )
                    # Clamp default to the user-defined range
                    _y_default = float(np.clip(_y_default, _y_min, _y_max))

                    st.caption(
                        "The ROC / AUC use the raw continuous prediction as a ranking score "
                        "and remain valid at any threshold."
                    )
                    _thresh_val = _synced_threshold_input(
                        "Outcome threshold — drag or type exact value",
                        _y_min, _y_max, _y_default,
                        "%.3f", max((_y_max - _y_min) / 200, 0.001), "conc_thresh_reg",
                    )
                    try:
                        _th_pred = np.asarray(_th_pipe.predict(_th_X), dtype=float)
                        st.plotly_chart(
                            regression_outcome_plot(
                                y_true=_th_y, y_pred=_th_pred,
                                threshold=_thresh_val,
                                model_name=_sel_hp_entry["model_name"],
                                positive_below=_positive_below,
                                positive_label=_pos_label,
                                negative_label=_neg_label,
                            ),
                            use_container_width=True,
                        )
                        _y_bin  = (_th_y  < _thresh_val) if _positive_below else (_th_y  >= _thresh_val)
                        _yp_bin = (_th_pred < _thresh_val) if _positive_below else (_th_pred >= _thresh_val)
                        _TP = int(( _yp_bin  &  _y_bin).sum())
                        _FN = int((~_yp_bin  &  _y_bin).sum())
                        _FP = int(( _yp_bin  & ~_y_bin).sum())
                        _TN = int((~_yp_bin  & ~_y_bin).sum())
                        # ROC score must align with direction
                        _roc_score = -_th_pred if _positive_below else _th_pred
                        _interpretation(
                            _TP, _FN, _FP, _TN,
                            score=_roc_score, y_bin=_y_bin.astype(int),
                            pos_label=_pos_label, neg_label=_neg_label,
                            threshold_label=f"t = {_thresh_val:.3g}",
                            positive_below=_positive_below,
                        )
                    except Exception as _e:
                        st.warning(f"Could not compute predictions: {_e}")

                # ── Classification / ordinal branch ───────────────────────────
                else:
                    _clf_pos_col, _clf_neg_col = st.columns(2)
                    with _clf_pos_col:
                        _pos_label = st.text_input(
                            "Name for class 1:", value="Case",
                            key="conc_clf_pos_lbl",
                        )
                    with _clf_neg_col:
                        _neg_label = st.text_input(
                            "Name for class 0:", value="Non-case",
                            key="conc_clf_neg_lbl",
                        )
                    st.caption(
                        "Drag or type the probability threshold k. "
                        "ROC and Precision-Recall curves update to show where k sits."
                    )
                    _thresh_k = _synced_threshold_input(
                        "Classification threshold k — drag or type exact value",
                        0.01, 0.99, 0.50,
                        "%.2f", 0.01, "conc_thresh_k",
                    )
                    try:
                        _th_prob_all = _th_pipe.predict_proba(_th_X)
                        _th_prob = (
                            _th_prob_all[:, 1]
                            if _th_prob_all.shape[1] == 2
                            else _th_prob_all.max(axis=1)
                        )
                        st.plotly_chart(
                            threshold_explorer_plot(
                                y_true=_th_y, y_prob=_th_prob,
                                k=_thresh_k,
                                model_name=_sel_hp_entry["model_name"],
                            ),
                            use_container_width=True,
                        )
                        _y_true_int = _th_y.astype(int)
                        _y_pred_bin = (_th_prob >= _thresh_k).astype(int)
                        _pos_mask = _y_true_int == 1
                        _neg_mask = ~_pos_mask
                        _TP = int(((_y_pred_bin == 1) & _pos_mask).sum())
                        _FN = int(((_y_pred_bin == 0) & _pos_mask).sum())
                        _FP = int(((_y_pred_bin == 1) & _neg_mask).sum())
                        _TN = int(((_y_pred_bin == 0) & _neg_mask).sum())
                        _interpretation(
                            _TP, _FN, _FP, _TN,
                            score=_th_prob, y_bin=_y_true_int,
                            pos_label=_pos_label, neg_label=_neg_label,
                            threshold_label=f"k = {_thresh_k:.2f}",
                        )
                    except Exception as _e:
                        st.warning(f"Could not compute predictions: {_e}")
            else:
                st.info("Pipeline or test data not found for this model.")
        else:
            st.info("No variant found for the selected model.")

    # ── Tab: Hyperparameters ─────────────────────────────────────────────────
    _hp_tab = _tabs[1]
    with _hp_tab:
        _bp = _sel_hp_entry["best_params"]

        # Fallback: read params directly from the trained pipeline file.
        if not _bp and _sel_hp_slug:
            _pipe_file = _mdir / f"{_sel_hp_slug}__{_sel_hp_entry['model_name']}.joblib"
            if _pipe_file.exists():
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _hp_pipe = load_joblib(_pipe_file)
                    if hasattr(_hp_pipe, "steps"):
                        _estimator = _hp_pipe.steps[-1][1]
                        _bp = {
                            k: v for k, v in _estimator.get_params().items()
                            if not k.startswith("_") and v is not None
                        }
                except Exception:
                    pass

        if _bp:
            _hp_df = pd.DataFrame(
                [{"Parameter": k, "Value": str(v)} for k, v in sorted(_bp.items())]
            )
            st.dataframe(_hp_df, hide_index=True, use_container_width=True)
        else:
            st.info("No hyperparameters found for this model.")
else:
    st.info("No cached results found. Complete Stage 8 (Comparison) to populate this section.")

# ── Analyst notes ─────────────────────────────────────────────────────────────
st.subheader("Key findings & notes")
notes = st.text_area(
    "Document your key findings, limitations and recommendations",
    value=state.notes,
    height=200,
    placeholder=(
        "e.g. CatBoost achieved the lowest RMSE across all dataset variants. "
        "The main predictor of outcome was pre-op score. "
        "Class imbalance (85/15) required careful threshold tuning…"
    ),
)
if notes != state.notes:
    update_state({"notes": notes})

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export")
col_e1, col_e2, col_e3 = st.columns(3)

out_dir = project_reports_dir()

# Use cross-variant rows for export if available, else fall back to state
_export_rows = _all_metric_rows or state.comparison_table
_export_best = (_global_best["model_name"] if _global_best else state.best_model_name) or "—"

with col_e1:
    if st.button("📥 Export CSV comparison table"):
        if _export_rows:
            _rows_p = []
            for _r in _export_rows:
                try:
                    _rows_p.append(MetricRow.model_validate(_r))
                except Exception:
                    pass
            if _rows_p:
                csv_path = export_comparison_csv(
                    ComparisonResult(task=task, rows=_rows_p),
                    out_dir / "comparison.csv",
                )
                with open(csv_path, "rb") as f:
                    st.download_button("Download CSV", f, file_name="comparison.csv", mime="text/csv")

with col_e2:
    if st.button("📄 Export HTML report"):
        if _export_rows:
            _rows_p = []
            for _r in _export_rows:
                try:
                    _rows_p.append(MetricRow.model_validate(_r))
                except Exception:
                    pass
            if _rows_p:
                _best_pk_val = _global_best.get(_pk, 0.0) if _global_best else state.metrics_summary.get("best_value", 0.0)
                comparison = ComparisonResult(task=task, rows=_rows_p, best_model_name=_export_best)
                summary = PipelineRunSummary(
                    file_name=state.upload_cfg.get("file_name", "—"),
                    target_column=target,
                    task_type=task,
                    n_rows_raw=0,
                    n_rows_train=0,
                    n_rows_test=0,
                    n_features=0,
                    n_models_trained=len(_rows_p),
                    best_model=_export_best,
                    best_metric_name=_pk,
                    best_metric_value=float(_best_pk_val) if _best_pk_val else 0.0,
                    key_findings=notes,
                )
                html_path = export_html_report(summary, comparison, out_dir / "report.html")
                with open(html_path, "rb") as f:
                    st.download_button("Download HTML", f, file_name="report.html", mime="text/html")

with col_e3:
    if state.pipeline_config_path:
        cfg_path = project_models_dir() / state.pipeline_config_path
        if cfg_path.exists():
            with open(cfg_path, "rb") as f:
                st.download_button(
                    "📋 Download pipeline config (JSON)",
                    f,
                    file_name="pipeline_config.json",
                    mime="application/json",
                )

st.divider()
if not state.stage_complete.get("conclusions"):
    if st.button("Mark pipeline complete ✓", type="primary"):
        mark_stage_complete("conclusions")
        st.rerun()
else:
    st.success("🎉 Pipeline complete! All stages finished.")
