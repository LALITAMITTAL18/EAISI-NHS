"""Stage 9 — Clinical Insight: Azure ML Inference & Expert Advice.

Two complementary models are used together:
  1. Random Forest (tabular) — predicts post-operative health gain from patient
     demographics and clinical scores (pipeline: removedmissingextreamage).
  2. EfficientNet-B4 DL model — classifies KL grade (0–4) from a knee X-ray.

SHAP feature attribution from the RF model and the imaging KL grade are combined
via GPT-4o (Azure AI Foundry) to generate structured clinical decision support.
"""

from __future__ import annotations

import base64
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load MLPortal/.env BEFORE importing azure_ml so os.environ is populated
# when azure_ml.py reads DEFAULT_SUBSCRIPTION_ID at module-import time.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st

from shared.cache import load_stage_cache, save_stage_cache, update_stage_cache
from shared.io import load_joblib, load_parquet
from shared.nav import render_sidebar
from shared.state import (
    get_state,
    mark_stage_complete,
    project_datasets_dir,
    project_models_dir,
)
from stages.clinical_insight.azure_ml import (
    DEFAULT_RESOURCE_GROUP,
    DEFAULT_SUBSCRIPTION_ID,
    DEFAULT_TENANT_ID,
    DEFAULT_WORKSPACE_NAME,
    DL_BASE_IMAGE,
    DL_DEPLOYMENT_NAME,
    DL_ENDPOINT_NAME,
    DL_INSTANCE_TYPE,
    RF_BASE_IMAGE,
    RF_DEPLOYMENT_NAME,
    RF_ENDPOINT_NAME,
    RF_INSTANCE_TYPE,
    create_endpoint,
    deploy_model,
    get_endpoint_info,
    get_ml_client,
    invoke_xray_endpoint,
    list_workspace_models,
    register_model,
)
from stages.clinical_insight.llm_advisor import generate_medical_advice
from stages.clinical_insight.models import (
    AzureMLConfig,
    KL_DESCRIPTIONS,
    KL_SURGERY_GUIDANCE,
    ShapResult,
    TabularPredictionResult,
    XrayPredictionResult,
)
from stages.clinical_insight.plots import (
    class_probability_bar,
    health_gain_meter,
    kl_grade_gauge,
    shap_force_plot,
    shap_summary_bar,
    shap_waterfall,
)
from stages.clinical_insight.shap_explainer import (
    compute_shap_local,
    compute_shap_single_row,
)
from stages.clinical_insight.local_dl import (
    DEFAULT_CHECKPOINT,
    SAMPLE_IMAGES_DIR,
    list_sample_images,
    load_local_model,
    predict_xray,
)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Clinical Insight", page_icon="🏥", layout="wide")
render_sidebar()

st.title("Clinical Insight")
st.caption(
    "Azure ML-powered inference: **Random Forest** (tabular) + **Deep Learning** (X-ray) "
    "combined into structured advice for the clinical team."
)

state = get_state()
_MDIR = project_models_dir()
_DDIR = project_datasets_dir()

# Known model artifacts
RF_PIPELINE_SLUG = "removedmissingextreamage"
RF_MODEL_NAME = "RandomForestRegressor"
RF_JOBLIB = _MDIR / f"{RF_PIPELINE_SLUG}__{RF_MODEL_NAME}.joblib"
RF_TEST_PARQUET = _DDIR / f"{RF_PIPELINE_SLUG}_test.parquet"

# Azure ML scoring script directories (relative to this file's package root)
_PKG = Path(__file__).parent.parent
RF_SCORING_DIR = _PKG / "stages" / "clinical_insight" / "scoring" / "rf"
DL_SCORING_DIR = _PKG / "stages" / "clinical_insight" / "scoring" / "dl"


# ── Session-state helpers ─────────────────────────────────────────────────────

_STAGE = "clinical_insight"

# Keys to persist across restarts.  Map: session_state_key → cache_key (same here).
_CACHE_KEYS = [
    "rf_result",           # TabularPredictionResult dict
    "xray_result",         # XrayPredictionResult dict
    "shap_result",         # ShapResult dict (mean_abs_shap + feature_names only)
    "shap_feature_names",
    "shap_values_single",
    "shap_base_value",
    "shap_explanations",   # list[dict] from GPT-4o
    "llm_advice",          # dict from GPT-4o
    "patient_features",    # dict of feature inputs used for last prediction
    "xray_true_grade",     # int ground truth from folder name
    "dl_random_img",       # str path of last random X-ray
    "dl_random_grade",     # int grade of last random X-ray
    "aml_sub_id",          # Subscription ID (avoids re-typing)
    "shap_viz_choice",     # "Force Plot" or "Waterfall" — user's last selection
]


def _ss(key, default=None):
    return st.session_state.get(key, default)


def _set(key, value):
    st.session_state[key] = value


_PYDANTIC_CACHE_TYPES: dict[str, type] = {
    "rf_result": TabularPredictionResult,
    "xray_result": XrayPredictionResult,
    "shap_result": ShapResult,
}


def _restore_cache() -> None:
    """Load persisted Stage 9 results from disk into st.session_state (once per session).

    JSON cache stores Pydantic models as plain dicts (via model_dump()).
    This function reconstructs the original typed objects on restore so that
    attribute access (e.g. rf_result.prediction) works correctly everywhere.
    """
    if _ss("_stage9_cache_loaded"):
        return
    cache = load_stage_cache(_STAGE)
    for key in _CACHE_KEYS:
        if key in cache and cache[key] is not None and key not in st.session_state:
            val = cache[key]
            cls = _PYDANTIC_CACHE_TYPES.get(key)
            if cls is not None and isinstance(val, dict):
                try:
                    val = cls(**val)
                except Exception:
                    pass  # leave as dict rather than crashing — page will handle gracefully
            st.session_state[key] = val
    # Restore individual feature inputs
    for k, v in cache.items():
        if (k.startswith("rf_feat_") or k.startswith("local_feat_")) and k not in st.session_state:
            st.session_state[k] = v
    _set("_stage9_cache_loaded", True)


def _persist(key: str, value) -> None:
    """Set a session-state key AND write it to the disk cache immediately."""
    _set(key, value)
    _flush_cache()


def _flush_cache() -> None:
    """Write all cacheable session-state keys to disk."""
    data: dict = {}
    for key in _CACHE_KEYS:
        val = _ss(key)
        if val is not None:
            # Pydantic models → dict for JSON serialisation
            data[key] = val.model_dump() if hasattr(val, "model_dump") else val
    # Include feature inputs
    for k, v in st.session_state.items():
        if k.startswith("rf_feat_") or k.startswith("local_feat_"):
            data[k] = v
    update_stage_cache(_STAGE, data)


_SHAP_KEYS = [
    # Patient-specific — cleared on every new prediction.
    # shap_result (population mean |SHAP|) is model-wide, NOT listed here.
    "shap_feature_names", "shap_values_single", "shap_base_value",
    "shap_explanations", "llm_advice",
]


def _invalidate_shap() -> None:
    """Clear SHAP + explanation values from session state AND disk.

    Called before persisting a new rf_result so stale SHAP from a previous
    patient is never returned for the new one.  update_stage_cache merges, so
    we need explicit nulls on disk — save_stage_cache overwrites the file.
    """
    for k in _SHAP_KEYS:
        st.session_state.pop(k, None)
    existing = load_stage_cache(_STAGE)
    for k in _SHAP_KEYS:
        existing[k] = None
    save_stage_cache(_STAGE, existing)


@st.cache_resource(show_spinner=False)
def _load_rf_pipeline(joblib_path: str = "") -> object | None:
    """Load the RF pipeline, keyed on path so the cache is invalidated if the path changes."""
    from pathlib import Path as _Path
    p = _Path(joblib_path) if joblib_path else RF_JOBLIB
    if not p.exists():
        return None
    try:
        return load_joblib(p)
    except Exception:
        return None


# Restore on every page load (no-op after first call in a session)
_restore_cache()

# Lazy single-row SHAP — if rf_result came back from cache but SHAP wasn't computed yet,
# run it now using the local pipeline (TreeExplainer is fast, ~0.5 s).
if _ss("rf_result") is not None and _ss("shap_values_single") is None:
    _lazy_pipeline = _load_rf_pipeline(str(RF_JOBLIB))
    if _lazy_pipeline is not None:
        _cached_rf = _ss("rf_result")
        _fv = getattr(_cached_rf, "feature_values", {})
        if isinstance(_fv, dict) and _fv:
            try:
                _fn, _sv, _bv = compute_shap_single_row(_lazy_pipeline, _fv)
                _persist("shap_feature_names", _fn)
                _persist("shap_values_single", _sv)
                _persist("shap_base_value", _bv)
            except Exception as _lazy_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning("Lazy SHAP failed: %s", _lazy_exc)


# ── Azure ML connection section ───────────────────────────────────────────────

def _azure_connection_section() -> object | None:
    """Render workspace connection UI. Returns the MLClient if connected."""
    connected = _ss("aml_connected", False)
    with st.expander(
        "⚙️ Azure ML Workspace — " + ("✅ Connected" if connected else "🔴 Not connected"),
        expanded=not connected,
    ):
        col1, col2 = st.columns(2)
        with col1:
            # Read from env at render time so MLPortal/.env is always honoured
            _env_sub = os.environ.get("AZURE_SUBSCRIPTION_ID", DEFAULT_SUBSCRIPTION_ID)
            sub_id = st.text_input(
                "Subscription ID *",
                value=_ss("aml_sub_id", _env_sub),
                placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                help="Find in Azure Portal → Subscriptions",
            )
            rg = st.text_input("Resource Group", value=DEFAULT_RESOURCE_GROUP)
        with col2:
            ws = st.text_input("Workspace Name", value=DEFAULT_WORKSPACE_NAME)
            tenant = st.text_input("Tenant ID", value=DEFAULT_TENANT_ID)

        if not sub_id:
            st.warning(
                "Set your Azure Subscription ID above, or define the environment variable "
                "`AZURE_SUBSCRIPTION_ID` before launching the app."
            )

        col_btn, col_hint = st.columns([1, 3])
        with col_btn:
            connect_clicked = st.button("🔗 Connect", disabled=not sub_id)
        with col_hint:
            st.caption(
                "Uses `DefaultAzureCredential` — run `az login` in a terminal first "
                "if you are not logged in."
            )

        if connect_clicked and sub_id:
            cfg = AzureMLConfig(
                subscription_id=sub_id,
                resource_group=rg,
                workspace_name=ws,
                tenant_id=tenant,
            )
            with st.spinner("Connecting to Azure ML workspace…"):
                try:
                    client = get_ml_client(cfg)
                    # Validate connection by listing models
                    models = list_workspace_models(client)
                    _set("aml_client", client)
                    _set("aml_connected", True)
                    _persist("aml_sub_id", sub_id)
                    _set("aml_models", models)
                    st.success(
                        f"Connected to **{ws}** — found {len(models)} registered model(s)"
                    )
                except Exception as exc:
                    msg = str(exc)
                    if "AADSTS" in msg or "ClientAuthenticationError" in type(exc).__name__:
                        st.error(
                            "Azure authentication expired or missing. Run "
                            "`az login --scope https://management.azure.com/.default` "
                            "in a terminal, then click **Connect** again."
                        )
                        with st.expander("Full error details"):
                            st.code(msg)
                    else:
                        st.error(f"Connection failed: {exc}")
                    _set("aml_connected", False)

        if connected and _ss("aml_models"):
            with st.expander("Registered models in workspace", expanded=False):
                for m in _ss("aml_models", []):
                    st.markdown(f"- **{m.name}** v{m.version} — {m.description or '—'}")

    return _ss("aml_client") if connected else None


# ── Endpoint management helpers ───────────────────────────────────────────────

def _endpoint_status_badge(info) -> None:
    if info is None:
        st.warning("Endpoint does not exist yet.")
        return
    state_map = {
        "Succeeded": ("✅", "success"),
        "Creating": ("⏳", "info"),
        "Updating": ("⏳", "info"),
        "Failed": ("❌", "error"),
        "Deleting": ("🗑️", "warning"),
    }
    icon, kind = state_map.get(info.provisioning_state, ("❓", "warning"))
    msg = f"{icon} **{info.name}** — {info.provisioning_state}"
    if info.scoring_uri:
        msg += f"  \n`{info.scoring_uri}`"
    getattr(st, kind)(msg)


def _endpoint_panel(
    client,
    endpoint_name: str,
    deployment_name: str,
    model_ref: str | None,
    scoring_dir: Path,
    conda_file: str,
    instance_type: str,
    base_image: str,
    label: str,
) -> bool:
    """Render endpoint create/deploy UI. Returns True only when endpoint is
    Succeeded AND has at least one active deployment with traffic."""
    info = get_endpoint_info(client, endpoint_name)
    _endpoint_status_badge(info)

    # ── Endpoint does not exist yet ───────────────────────────────────────────
    if info is None:
        st.markdown(
            "No endpoint found. Click below to create the endpoint **and** deploy "
            "the selected model in one step (~10–15 minutes)."
        )
        if model_ref is None:
            st.error("Select a registered model above first.")
            return False
        if st.button(f"🚀 Create & Deploy {label} Endpoint", key=f"create_{endpoint_name}"):
            with st.spinner("Creating endpoint… (~2 min)"):
                try:
                    create_endpoint(client, endpoint_name, f"NHS Knee {label} endpoint")
                except Exception as exc:
                    st.error(f"Endpoint creation failed: {exc}")
                    return False
            with st.spinner("Deploying model… (~10–15 min, please wait)"):
                try:
                    deploy_model(
                        client, endpoint_name, deployment_name, model_ref,
                        scoring_dir, conda_file, instance_type, base_image=base_image,
                    )
                    st.success("Deployment complete!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Deployment failed: {exc}")
        return False

    # ── Endpoint is still provisioning ────────────────────────────────────────
    if info.provisioning_state in ("Creating", "Updating"):
        st.button("🔄 Refresh status", key=f"refresh_{endpoint_name}", on_click=st.rerun)
        return False

    # ── Endpoint failed ───────────────────────────────────────────────────────
    if info.provisioning_state == "Failed":
        st.error("Endpoint is in a failed state. Check the Azure ML portal for details.")
        return False

    # ── Endpoint exists but NO deployment yet (your current state) ────────────
    has_deployments = bool(info.deployment_names)
    has_traffic = bool(info.traffic)

    if info.provisioning_state == "Succeeded" and not has_deployments:
        st.warning(
            "⚠️ Endpoint created successfully but **no model is deployed yet**. "
            "This happens when the deployment step was interrupted or timed out. "
            "Click below to deploy the model to the existing endpoint."
        )
        if model_ref is None:
            st.error("Select a registered model above first.")
            return False
        if st.button(f"📦 Deploy Model to Existing Endpoint", key=f"deploy_only_{endpoint_name}",
                     type="primary"):
            with st.spinner("Deploying model to existing endpoint… (~10–15 min)"):
                try:
                    deploy_model(
                        client, endpoint_name, deployment_name, model_ref,
                        scoring_dir, conda_file, instance_type, base_image=base_image,
                    )
                    st.success("Deployment complete! Refresh to continue.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Deployment failed: {exc}")
                    st.caption(
                        "Tip: check Azure ML Studio → Endpoints → "
                        f"`{endpoint_name}` → Logs for details."
                    )
        return False

    # ── Endpoint succeeded with deployment but no traffic routed yet ──────────
    if has_deployments and not has_traffic:
        st.warning(
            f"Deployment `{info.deployment_names[0]}` exists but has **0% traffic**. "
            "Routing traffic now…"
        )
        try:
            from azure.ai.ml.entities import ManagedOnlineEndpoint as _MOE
            ep = client.online_endpoints.get(name=endpoint_name)
            ep.traffic = {info.deployment_names[0]: 100}
            client.online_endpoints.begin_create_or_update(ep).result()
            st.success("Traffic routed to 100%. Refreshing…")
            st.rerun()
        except Exception as exc:
            st.error(f"Traffic update failed: {exc}")
        return False

    # ── Fully ready ───────────────────────────────────────────────────────────
    if has_deployments:
        st.success(
            f"✅ Ready — deployment `{info.deployment_names[0]}` serving 100% traffic"
        )
    return info.provisioning_state == "Succeeded" and has_deployments


# ── RF model section ──────────────────────────────────────────────────────────

def _rf_section(client) -> None:
    st.subheader("🌲 Random Forest — Tabular Prediction")
    st.caption(
        "Pipeline: **removedmissingextreamage** (missing extreme-age groups removed) · "
        "Target: **health_gain** (OKS improvement post-surgery)"
    )

    # ── Model selector ────────────────────────────────────────────────────────
    workspace_models = _ss("aml_models", [])
    rf_suggestions = [
        m for m in workspace_models
        if any(kw in m.name.lower() for kw in ["randomforest", "rf", "knee", "nhs"])
    ]
    all_names = [m.name for m in workspace_models]
    suggestion_names = [m.name for m in rf_suggestions] or all_names

    if not all_names:
        st.info("No models found in workspace. Register your RF pipeline first.")
        if st.button("📤 Register local RF pipeline to workspace", key="reg_rf"):
            if RF_JOBLIB.exists():
                with st.spinner("Registering…"):
                    try:
                        ref = register_model(
                            client,
                            model_name="nhs-knee-rf-randomforest",
                            local_path=str(RF_JOBLIB),
                            description="NHS knee RF pipeline (removedmissingextreamage variant)",
                            tags={"pipeline": RF_PIPELINE_SLUG, "model": RF_MODEL_NAME},
                        )
                        _set("rf_model_ref", ref)
                        _set("aml_models", list_workspace_models(client))
                        st.success(f"Registered: {ref}")
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")
            else:
                st.error(f"Local pipeline not found: {RF_JOBLIB}")
        return

    selected_name = st.selectbox(
        "Select RF model from workspace",
        suggestion_names + [n for n in all_names if n not in suggestion_names],
        key="rf_model_select",
        help="Models whose names suggest a Random Forest are listed first.",
    )
    selected_version = next(
        (m.version for m in workspace_models if m.name == selected_name), "1"
    )
    rf_model_ref = f"azureml:{selected_name}:{selected_version}"
    _set("rf_model_ref", rf_model_ref)

    # ── Endpoint panel ────────────────────────────────────────────────────────
    with st.expander("Endpoint management", expanded=_ss("rf_endpoint_ready", False) is False):
        endpoint_ready = _endpoint_panel(
            client=client,
            endpoint_name=RF_ENDPOINT_NAME,
            deployment_name=RF_DEPLOYMENT_NAME,
            model_ref=rf_model_ref,
            scoring_dir=RF_SCORING_DIR,
            conda_file="conda.yaml",
            instance_type=RF_INSTANCE_TYPE,
            base_image=RF_BASE_IMAGE,
            label="RF",
        )
        _set("rf_endpoint_ready", endpoint_ready)

    if not _ss("rf_endpoint_ready", False):
        st.info("Complete endpoint setup above to enable inference.")
        return

    # ── Load local pipeline for SHAP + defaults ───────────────────────────────
    pipeline = _load_rf_pipeline(str(RF_JOBLIB))
    if pipeline is None:
        st.warning(
            f"Local pipeline file not found (`{RF_JOBLIB.name}`). "
            "SHAP and dummy-fill features will be unavailable. "
            "You can still run inference via the Azure ML endpoint."
        )

    feature_names, default_values = _get_rf_features_and_defaults(pipeline)

    # ── Patient feature input form ────────────────────────────────────────────
    st.markdown("#### Patient Data Entry")
    st.caption(
        "Fill in patient values manually, or click **🎲 Random Patient** to draw a "
        "random row from the test dataset — every click picks a different patient."
    )

    col_fill, _ = st.columns([1, 4])
    with col_fill:
        if st.button("🎲 Random Patient", key="fill_dummy_rf", type="secondary"):
            random_row = _get_random_row(feature_names)
            for fn in feature_names:
                new_val = str(random_row.get(fn, default_values.get(fn, "")))
                _set(f"rf_feat_{fn}", new_val)
                _set(f"rf_input_{fn}", new_val)  # update widget state so display refreshes

    if not feature_names:
        st.warning("Could not determine feature names from the local pipeline.")
        return

    row_input: dict = {}
    n_cols = min(4, len(feature_names))
    cols = st.columns(n_cols)

    for i, feat in enumerate(feature_names):
        with cols[i % n_cols]:
            default_str = str(default_values.get(feat, ""))
            current = _ss(f"rf_feat_{feat}", default_str)
            val = st.text_input(
                feat,
                value=current,
                key=f"rf_input_{feat}",
                label_visibility="visible",
            )
            try:
                row_input[feat] = float(val)
            except (ValueError, TypeError):
                row_input[feat] = val

    # ── Run prediction ────────────────────────────────────────────────────────
    st.markdown("---")
    col_run, col_local = st.columns([1, 1])
    with col_run:
        run_azure = st.button("▶ Predict via Azure ML", type="primary", key="run_rf_azure")
    with col_local:
        run_local = st.button("🖥 Predict Locally (offline)", key="run_rf_local")

    if run_azure or run_local:
        pred_val = None
        with st.spinner("Running prediction…"):
            if run_azure:
                try:
                    from stages.clinical_insight.azure_ml import _invoke_via_http
                    import json as _json
                    payload = _json.dumps({"features": row_input, "compute_shap": False})
                    result = _invoke_via_http(_ss("aml_client"), RF_ENDPOINT_NAME, payload)
                    pred_val = float(result["prediction"])
                except Exception as exc:
                    st.error(f"Azure ML inference failed: {exc}")
            elif run_local and pipeline is not None:
                try:
                    df = pd.DataFrame([row_input])
                    pred_val = float(pipeline.predict(df)[0])
                except Exception as exc:
                    st.error(f"Local inference failed: {exc}")
            elif run_local and pipeline is None:
                st.error("Local pipeline not available.")

        if pred_val is not None:
            _invalidate_shap()
            rf_result = TabularPredictionResult(
                prediction=pred_val,
                model_name=RF_MODEL_NAME,
                endpoint_used=RF_ENDPOINT_NAME if run_azure else "local",
                feature_values=row_input,
            )
            _persist("rf_result", rf_result)
            _persist("patient_features", row_input)

    # ── Results ───────────────────────────────────────────────────────────────
    rf_result: TabularPredictionResult | None = _ss("rf_result")
    if rf_result:
        st.markdown("---")
        st.markdown("#### Prediction Result")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(
                health_gain_meter(rf_result.prediction),
                use_container_width=True,
            )
        with c2:
            st.metric(
                "Predicted Health Gain",
                f"{rf_result.prediction:.2f} pts",
                help="Oxford Knee Score (OKS) improvement expected post-surgery",
            )
            mcid = 5.0
            if rf_result.prediction >= mcid:
                st.success(
                    f"Prediction ({rf_result.prediction:.1f}) meets the MCID threshold ({mcid} pts) — "
                    "clinically meaningful improvement expected."
                )
            else:
                st.warning(
                    f"Prediction ({rf_result.prediction:.1f}) is below the MCID threshold ({mcid} pts) — "
                    "consider whether surgery is the right pathway."
                )

        # SHAP analysis (requires local pipeline)
        if pipeline is not None:
            st.markdown("#### SHAP Feature Attribution")
            st.caption("Explains which patient factors are driving the prediction most.")
            with st.spinner("Computing SHAP values…"):
                try:
                    feat_names, row_shap, base_val = compute_shap_single_row(
                        pipeline, rf_result.feature_values
                    )
                    _persist("shap_feature_names", feat_names)
                    _persist("shap_values_single", row_shap)
                    _persist("shap_base_value", base_val)
                except Exception as exc:
                    st.warning(f"SHAP computation failed: {exc}")

            shap_res = _ss("shap_result")
            row_shap = _ss("shap_values_single")
            feat_names = _ss("shap_feature_names")
            base_val = _ss("shap_base_value", 0.0)

            if shap_res:
                tab_summary, tab_patient = st.tabs(
                    ["📊 Population summary (mean |SHAP|)", "🎯 This patient's prediction"]
                )
                with tab_summary:
                    st.caption(
                        "Mean absolute SHAP value across the test set — shows which features "
                        "matter most globally for the model."
                    )
                    st.plotly_chart(
                        shap_summary_bar(shap_res),
                        use_container_width=True,
                    )
                with tab_patient:
                    if row_shap and feat_names:
                        # Toggle cached between sessions
                        default_viz = _ss("shap_viz_choice", "Force Plot")
                        viz_choice = st.radio(
                            "Visualization style",
                            ["Force Plot", "Waterfall"],
                            index=0 if default_viz == "Force Plot" else 1,
                            horizontal=True,
                            key="shap_viz_radio",
                            help=(
                                "**Force Plot** — horizontal stacked bar showing cumulative "
                                "feature contributions from base to prediction.\n\n"
                                "**Waterfall** — vertical step chart, one bar per feature."
                            ),
                        )
                        if viz_choice != _ss("shap_viz_choice"):
                            _persist("shap_viz_choice", viz_choice)

                        if viz_choice == "Force Plot":
                            st.plotly_chart(
                                shap_force_plot(
                                    feat_names,
                                    row_shap,
                                    base_val,
                                    rf_result.prediction,
                                ),
                                use_container_width=True,
                            )
                            st.caption(
                                "🔴 Red = feature pushes prediction **up** (increases health gain).  "
                                "🔵 Blue = feature pushes prediction **down**.  "
                                f"Base value: **{base_val:.2f}** → Final prediction: **{rf_result.prediction:.2f}**"
                            )
                        else:
                            st.plotly_chart(
                                shap_waterfall(
                                    feat_names,
                                    row_shap,
                                    base_val,
                                    rf_result.prediction,
                                ),
                                use_container_width=True,
                            )
                            st.caption(
                                "🔴 Red bars increase prediction · 🔵 Blue bars decrease it · "
                                "🟢 Final bar = predicted health gain"
                            )
                    else:
                        st.info("Run a prediction above to see this patient's SHAP breakdown.")
        else:
            st.info(
                "Install the local pipeline (`removedmissingextreamage__RandomForestRegressor.joblib`) "
                "to enable SHAP analysis."
            )


# ── DL X-ray section ──────────────────────────────────────────────────────────

def _dl_section(client) -> None:
    st.subheader("🔬 Deep Learning — X-ray KL Grade Classification")
    st.caption(
        "Model: **EfficientNet-B4** · Task: ordinal classification of KL grade (0–4) "
        "· Trained with CLAHE preprocessing and Gaussian ordinal soft labels"
    )

    # ── Model selector ────────────────────────────────────────────────────────
    workspace_models = _ss("aml_models", [])
    dl_suggestions = [
        m for m in workspace_models
        if any(kw in m.name.lower() for kw in ["dl", "efficientnet", "xray", "kl", "knee_kl"])
    ]
    all_names = [m.name for m in workspace_models]
    suggestion_names = [m.name for m in dl_suggestions] or all_names

    if not all_names:
        st.info("No models in workspace. Register your DL checkpoint first.")
        return

    selected_name = st.selectbox(
        "Select DL model from workspace",
        suggestion_names + [n for n in all_names if n not in suggestion_names],
        key="dl_model_select",
    )
    selected_version = next(
        (m.version for m in workspace_models if m.name == selected_name), "1"
    )
    dl_model_ref = f"azureml:{selected_name}:{selected_version}"
    _set("dl_model_ref", dl_model_ref)

    # ── Endpoint panel ────────────────────────────────────────────────────────
    with st.expander("Endpoint management", expanded=_ss("dl_endpoint_ready", False) is False):
        endpoint_ready = _endpoint_panel(
            client=client,
            endpoint_name=DL_ENDPOINT_NAME,
            deployment_name=DL_DEPLOYMENT_NAME,
            model_ref=dl_model_ref,
            scoring_dir=DL_SCORING_DIR,
            conda_file="conda.yaml",
            instance_type=DL_INSTANCE_TYPE,
            base_image=DL_BASE_IMAGE,
            label="DL",
        )
        _set("dl_endpoint_ready", endpoint_ready)

    if not _ss("dl_endpoint_ready", False):
        st.info("Complete endpoint setup above to enable X-ray classification.")
        return

    # ── X-ray uploader ────────────────────────────────────────────────────────
    st.markdown("#### Upload Knee X-ray")
    st.caption(
        "Upload a frontal knee X-ray (AP view). Supported formats: JPG, PNG. "
        "The model applies CLAHE contrast enhancement automatically."
    )
    uploaded_file = st.file_uploader(
        "Choose X-ray image",
        type=["jpg", "jpeg", "png"],
        key="xray_upload",
        help="AP (anterior-posterior) view of the knee recommended.",
    )

    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        _set("xray_image_bytes", img_bytes)

        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(img_bytes, caption="Uploaded X-ray", use_container_width=True)
        with col_info:
            st.markdown("**Image ready for classification.**")
            st.caption(
                f"File: `{uploaded_file.name}` · "
                f"Size: {len(img_bytes) / 1024:.0f} KB"
            )
            if st.button("▶ Classify X-ray (KL Grade)", type="primary", key="run_dl"):
                with st.spinner("Running DL inference on Azure ML…"):
                    try:
                        result = invoke_xray_endpoint(
                            _ss("aml_client"),
                            DL_ENDPOINT_NAME,
                            img_bytes,
                        )
                        xray_result = XrayPredictionResult(
                            kl_grade=int(result["kl_grade"]),
                            confidence=float(result["confidence"]),
                            class_probabilities=[float(p) for p in result["class_probabilities"]],
                            class_names=result.get(
                                "class_names",
                                [f"Grade {i}" for i in range(5)],
                            ),
                            endpoint_used=DL_ENDPOINT_NAME,
                        )
                        _persist("xray_result", xray_result)
                    except Exception as exc:
                        st.error(f"DL inference failed: {exc}")

    # ── DL Results ────────────────────────────────────────────────────────────
    xray_result: XrayPredictionResult | None = _ss("xray_result")
    if xray_result:
        st.markdown("---")
        st.markdown("#### Classification Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(
                kl_grade_gauge(xray_result.kl_grade, xray_result.confidence),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                class_probability_bar(
                    xray_result.class_probabilities,
                    xray_result.class_names,
                ),
                use_container_width=True,
            )

        kl = xray_result.kl_grade
        col_desc, col_guide = st.columns(2)
        with col_desc:
            st.info(f"**Radiographic finding:** {KL_DESCRIPTIONS.get(kl, '—')}")
        with col_guide:
            guide = KL_SURGERY_GUIDANCE.get(kl, "—")
            if kl >= 3:
                st.warning(f"**Guidance:** {guide}")
            else:
                st.success(f"**Guidance:** {guide}")


# ── Combined clinical advice ──────────────────────────────────────────────────

def _clinical_advice_section() -> None:
    from stages.clinical_insight.llm_advisor import generate_shap_explanation

    rf_result: TabularPredictionResult | None = _ss("rf_result")
    xray_result: XrayPredictionResult | None = _ss("xray_result")
    shap_res = _ss("shap_result")

    if rf_result is None and xray_result is None:
        return

    st.divider()
    st.subheader("🏥 Clinical Decision Support")
    st.caption(
        "Powered by **GPT-4o (Azure AI Foundry)**. "
        "This is a decision support tool — always apply clinical judgement."
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if rf_result:
            above = rf_result.prediction >= 5
            st.metric(
                "Health Gain (predicted)",
                f"{rf_result.prediction:.1f} pts",
                delta="Above MCID ✓" if above else "Below MCID ✗",
                delta_color="normal" if above else "inverse",
                help="Oxford Knee Score improvement. MCID = 5 pts.",
            )
        else:
            st.metric("Health Gain (predicted)", "—")
    with col2:
        if xray_result:
            kl = xray_result.kl_grade
            st.metric(
                "KL Grade (X-ray)",
                f"Grade {kl}",
                delta=f"{xray_result.confidence:.0%} confidence",
                delta_color="off",
            )
        else:
            st.metric("KL Grade (X-ray)", "—")
    with col3:
        _s_names = _ss("shap_feature_names")
        _s_vals = _ss("shap_values_single") or []
        if shap_res and shap_res.feature_names:
            # Population-level summary available — use top feature by mean |SHAP|
            st.metric(
                "Top SHAP Driver",
                shap_res.feature_names[0][:18],
                delta=f"|SHAP| = {shap_res.mean_abs_shap[0]:.3f}",
                delta_color="off",
            )
        elif _s_names and _s_vals:
            # Fall back to this patient's single-row SHAP
            _top_i = max(range(len(_s_vals)), key=lambda i: abs(_s_vals[i]))
            st.metric(
                "Top SHAP Driver",
                _s_names[_top_i][:18],
                delta=f"|SHAP| = {abs(_s_vals[_top_i]):.3f}",
                delta_color="off",
            )
        else:
            st.metric("Top SHAP Driver", "—")
    with col4:
        if rf_result and xray_result:
            concordant = (xray_result.kl_grade >= 3 and rf_result.prediction >= 5) or \
                         (xray_result.kl_grade < 2 and rf_result.prediction < 5)
            st.metric(
                "Model Agreement",
                "Concordant" if concordant else "Discordant",
                delta="✓ Both point same way" if concordant else "⚠ Review carefully",
                delta_color="normal" if concordant else "inverse",
            )

    # ── Generate button ───────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_advice = st.button(
            "✨ Clinical Recommendation",
            type="primary",
            key="gen_advice",
            disabled=rf_result is None,
            help="Generates clinical recommendation AND SHAP feature explanations via GPT-4o",
        )

    if run_advice:
        # Step 1 — ensure single-row SHAP is available (fast, ~0.5 s).
        if not _ss("shap_feature_names"):
            _od_pipeline = _load_rf_pipeline(str(RF_JOBLIB))
            if _od_pipeline is not None:
                with st.spinner("Computing SHAP values for this patient…"):
                    try:
                        _fn, _sv, _bv = compute_shap_single_row(
                            _od_pipeline, rf_result.feature_values
                        )
                        _persist("shap_feature_names", _fn)
                        _persist("shap_values_single", _sv)
                        _persist("shap_base_value", _bv)
                    except Exception as exc:
                        st.warning(f"SHAP computation failed: {exc}")

        # Step 2 — GPT-4o clinical recommendation.
        with st.spinner("Generating clinical recommendation via GPT-4o…"):
            try:
                advice = generate_medical_advice(
                    rf_result=rf_result,
                    shap_result=shap_res,
                    xray_result=xray_result,
                    patient_features=_ss("patient_features"),
                )
                _persist("llm_advice", advice)
            except (ImportError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"GPT-4o call failed: {exc}")

        # Step 3 — GPT-4o SHAP feature explanations.
        _exp_names = _ss("shap_feature_names") or (shap_res.feature_names if shap_res else [])
        _exp_vals = _ss("shap_values_single") or (shap_res.mean_abs_shap if shap_res else [])
        if _exp_names:
            with st.spinner("Generating SHAP feature explanations via GPT-4o…"):
                try:
                    explanations = generate_shap_explanation(
                        feature_names=_exp_names,
                        shap_values=_exp_vals,
                        feature_values=rf_result.feature_values,
                        prediction=rf_result.prediction,
                    )
                    _persist("shap_explanations", explanations)
                except (ImportError, ValueError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"SHAP explanation failed: {exc}")

    # ── Results tabs ──────────────────────────────────────────────────────────
    advice: dict | None = _ss("llm_advice")
    shap_explanations: list | None = _ss("shap_explanations")

    if not advice and not shap_explanations:
        return

    st.markdown("")
    tab_rec, tab_shap, tab_export = st.tabs(
        ["📋 Clinical Recommendation", "🔬 SHAP — Why Each Feature Matters", "📄 Export"]
    )

    # ── Tab 1: Recommendation ─────────────────────────────────────────────────
    with tab_rec:
        if not advice:
            st.info("Click **✨ Clinical Recommendation** above to generate both tabs.")
        else:
            # Bottom line banner
            urgency_colours = {
                "Routine": "🟢", "Soon (3-6 months)": "🟡",
                "Urgent": "🔴", "Not indicated": "⚪",
            }
            urgency = advice.get("urgency", "")
            icon = urgency_colours.get(urgency, "⚪")
            st.markdown(
                f"<div style='background:#f0f4ff;border-left:4px solid #2563eb;"
                f"padding:12px 16px;border-radius:4px;margin-bottom:12px'>"
                f"<b>Bottom Line:</b> {advice.get('bottom_line', '—')}<br>"
                f"<small><b>Timing:</b> {icon} {urgency}</small></div>",
                unsafe_allow_html=True,
            )

            # Recommendation paragraph
            if advice.get("recommendation"):
                st.markdown("**Recommendation**")
                st.markdown(advice["recommendation"])

            col_l, col_r = st.columns(2)
            with col_l:
                if advice.get("key_considerations"):
                    st.markdown("**Key Considerations**")
                    for item in advice["key_considerations"]:
                        st.markdown(f"- {item}")
                if advice.get("caveats"):
                    st.markdown("**Caveats & Uncertainties**")
                    for item in advice["caveats"]:
                        st.markdown(f"- {item}")
            with col_r:
                if advice.get("next_steps"):
                    st.markdown("**Suggested Next Steps**")
                    for i, step in enumerate(advice["next_steps"], 1):
                        st.markdown(f"{i}. {step}")

    # ── Tab 2: SHAP Explanations ──────────────────────────────────────────────
    with tab_shap:
        if not shap_explanations:
            st.info("Click **✨ Clinical Recommendation** above to generate.")
        else:
            # Normalise: handle dict-of-dicts returned by old cache entries
            _items: list = (
                shap_explanations
                if isinstance(shap_explanations, list)
                else list(shap_explanations.values())
                if isinstance(shap_explanations, dict)
                else []
            )
            _items = [i for i in _items if isinstance(i, dict)]
            if not _items:
                st.warning("SHAP explanation data is in an unexpected format. Please click Clinical Recommendation again.")
            else:
                st.caption(
                    "Each row shows one patient factor, how much it influenced the predicted "
                    "health gain (SHAP value), and the clinical reason why."
                )
            for item in _items:
                fname = item.get("feature", "")
                fval = item.get("patient_value", "")
                shap_val = item.get("shap", 0.0)
                direction = item.get("direction", "positive")
                explanation = item.get("clinical_explanation", "")

                colour = "#dcfce7" if direction == "positive" else "#fee2e2"
                arrow = "▲" if direction == "positive" else "▼"
                border = "#16a34a" if direction == "positive" else "#dc2626"
                impact_word = "Increases benefit" if direction == "positive" else "Decreases benefit"

                st.markdown(
                    f"<div style='border-left:4px solid {border};background:{colour};"
                    f"padding:10px 14px;border-radius:4px;margin-bottom:8px'>"
                    f"<b>{fname}</b> &nbsp;·&nbsp; Patient value: <code>{fval}</code> "
                    f"&nbsp;·&nbsp; {arrow} SHAP: <b>{float(shap_val):+.3f}</b> "
                    f"&nbsp;<span style='color:{border}'>{impact_word}</span><br>"
                    f"<small>{explanation}</small></div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Export ─────────────────────────────────────────────────────────
    with tab_export:
        import json as _json

        export_parts = []
        if rf_result:
            export_parts.append(f"PREDICTED HEALTH GAIN: {rf_result.prediction:.2f} pts")
        if xray_result:
            export_parts.append(
                f"KL GRADE: {xray_result.kl_grade} (confidence {xray_result.confidence:.0%})"
            )
        if advice:
            export_parts += [
                "",
                f"BOTTOM LINE: {advice.get('bottom_line', '')}",
                f"URGENCY: {advice.get('urgency', '')}",
                "",
                "RECOMMENDATION:",
                advice.get("recommendation", ""),
                "",
                "KEY CONSIDERATIONS:",
                *[f"- {c}" for c in advice.get("key_considerations", [])],
                "",
                "CAVEATS:",
                *[f"- {c}" for c in advice.get("caveats", [])],
                "",
                "NEXT STEPS:",
                *[f"{i+1}. {s}" for i, s in enumerate(advice.get("next_steps", []))],
            ]
        if shap_explanations:
            export_parts += ["", "SHAP FEATURE EXPLANATIONS:"]
            for item in shap_explanations:
                if isinstance(item, dict):
                    export_parts.append(
                        f"  {item.get('feature','')} (value={item.get('patient_value','')}, "
                        f"SHAP={float(item.get('shap',0)):+.3f}): "
                        f"{item.get('clinical_explanation','')}"
                    )

        export_text = "\n".join(export_parts)
        col_txt, col_json, _ = st.columns([1, 1, 2])
        with col_txt:
            st.download_button(
                "📄 Download Report (TXT)",
                data=export_text,
                file_name="clinical_insight_report.txt",
                mime="text/plain",
            )
        with col_json:
            export_json = {
                "rf_prediction": rf_result.prediction if rf_result else None,
                "kl_grade": xray_result.kl_grade if xray_result else None,
                "recommendation": advice,
                "shap_explanations": shap_explanations,
            }
            st.download_button(
                "📊 Download Data (JSON)",
                data=_json.dumps(export_json, indent=2),
                file_name="clinical_insight_data.json",
                mime="application/json",
            )


# ── Local pipeline / DL utilities ────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_dl_model_cached():
    """Load the local EfficientNet-B4 checkpoint once and cache it."""
    try:
        return load_local_model(DEFAULT_CHECKPOINT)
    except FileNotFoundError:
        return None, None
    except ImportError:
        return None, None
    except Exception:
        return None, None


def _get_rf_features_and_defaults(pipeline) -> tuple[list[str], dict]:
    """Return (feature_names, median_defaults_dict) from the pipeline + test set."""
    feature_names: list[str] = []
    defaults: dict = {}

    if pipeline is not None:
        try:
            feature_names = list(pipeline.feature_names_in_)
        except AttributeError:
            try:
                preprocessor = pipeline.named_steps.get("preprocessor")
                if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
                    feature_names = list(preprocessor.feature_names_in_)
            except Exception:
                pass

    if RF_TEST_PARQUET.exists():
        try:
            df = load_parquet(RF_TEST_PARQUET).drop(columns=["health_gain"], errors="ignore")
            if not feature_names:
                feature_names = df.columns.tolist()
            for col in feature_names:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        defaults[col] = round(float(df[col].median()), 3)
                    else:
                        mode = df[col].mode()
                        defaults[col] = mode.iloc[0] if not mode.empty else ""
        except Exception:
            pass

    return feature_names, defaults


def _get_random_row(feature_names: list[str]) -> dict:
    """Pick a random patient and randomise the top SHAP features across their full
    distribution so each click produces a meaningfully different prediction.

    Low-impact features come from a single randomly chosen patient row (realistic
    co-occurrence).  Top-SHAP features are re-sampled independently from a random
    percentile so the prediction score changes noticeably each time.
    """
    if not RF_TEST_PARQUET.exists():
        return {}
    try:
        df = load_parquet(RF_TEST_PARQUET).drop(columns=["health_gain"], errors="ignore")

        # Base row — realistic values for low-impact features
        base_row = df.sample(1, random_state=random.randint(0, 999_999)).iloc[0]
        result: dict = {}
        for col in feature_names:
            if col in df.columns:
                val = base_row[col]
                result[col] = round(float(val), 3) if pd.api.types.is_numeric_dtype(df[col]) else str(val)

        # Re-sample top SHAP features from a random quantile for maximum variation.
        # This ensures clicking "Random Patient" always changes the prediction visibly.
        top_shap = (_ss("shap_feature_names") or [])[:8]
        rng = random.Random()  # fresh Random instance — not affected by any global seed
        for feat in top_shap:
            if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                q = rng.random()  # 0.0–1.0
                result[feat] = round(float(df[feat].quantile(q)), 3)

        return result
    except Exception:
        return {}


# ── Page layout ───────────────────────────────────────────────────────────────

# ── Cache status banner ───────────────────────────────────────────────────────
_has_cached = any(
    _ss(k) is not None
    for k in ["rf_result", "xray_result", "llm_advice", "shap_result"]
)
if _has_cached:
    _c1, _c2 = st.columns([4, 1])
    with _c1:
        _saved = load_stage_cache(_STAGE).get("_saved_at", "")
        st.success(
            f"💾 Results loaded from local cache"
            + (f" — last saved {_saved}" if _saved else "")
            + ". Scroll down to see previous results."
        )
    with _c2:
        if st.button("🗑 Clear Cache", key="clear_stage9_cache"):
            from shared.cache import clear_stage_cache
            clear_stage_cache(_STAGE)
            for k in _CACHE_KEYS:
                st.session_state.pop(k, None)
            _set("_stage9_cache_loaded", False)
            st.rerun()

# Section 1: Azure ML connection (always shown; ml_client is None when not connected)
ml_client = _azure_connection_section()

if ml_client is None:
    st.info(
        "Connect to the Azure ML workspace above to manage endpoints. "
        "The **Random Forest tab** works offline using the local pipeline file. "
        "The **Deep Learning tab** requires an active Azure ML endpoint."
    )

# Section 2: Both tabs always visible regardless of Azure connection
tab_rf, tab_dl = st.tabs(
    ["🌲 Random Forest (Tabular)", "🔬 Deep Learning (X-ray)"]
)

with tab_rf:
    if ml_client is not None:
        # Full Azure mode: endpoint management + Azure or local inference
        _rf_section(ml_client)
    else:
        # Offline mode: local pipeline only
        st.subheader("🌲 Random Forest — Local Prediction (offline)")
        st.caption(
            "Pipeline: **removedmissingextreamage** · Target: **health_gain** (OKS improvement) "
            "· Azure ML not connected — running locally"
        )
        pipeline = _load_rf_pipeline(str(RF_JOBLIB))
        if pipeline is None:
            st.warning(
                f"Local pipeline file not found: `{RF_JOBLIB.name}`. "
                "Connect to Azure ML above to use the endpoint instead."
            )
        else:
            feature_names, default_values = _get_rf_features_and_defaults(pipeline)
            if feature_names:
                col_fill, _ = st.columns([1, 4])
                with col_fill:
                    if st.button("🎲 Random Patient", key="fill_dummy_local", type="secondary"):
                        random_row = _get_random_row(feature_names)
                        for fn in feature_names:
                            new_val = str(random_row.get(fn, default_values.get(fn, "")))
                            _set(f"local_feat_{fn}", new_val)
                            _set(f"local_input_{fn}", new_val)  # update widget state so display refreshes

                row_input: dict = {}
                n_cols = min(4, len(feature_names))
                form_cols = st.columns(n_cols)
                for i, feat in enumerate(feature_names):
                    with form_cols[i % n_cols]:
                        default_str = str(default_values.get(feat, ""))
                        val = st.text_input(
                            feat,
                            value=_ss(f"local_feat_{feat}", default_str),
                            key=f"local_input_{feat}",
                        )
                        try:
                            row_input[feat] = float(val)
                        except (ValueError, TypeError):
                            row_input[feat] = val

                if st.button("▶ Predict Locally", type="primary", key="local_predict"):
                    with st.spinner("Predicting…"):
                        try:
                            pred = float(pipeline.predict(pd.DataFrame([row_input]))[0])
                            _invalidate_shap()
                            _persist("rf_result", TabularPredictionResult(
                                prediction=pred,
                                model_name=RF_MODEL_NAME,
                                endpoint_used="local",
                                feature_values=row_input,
                            ))
                            _persist("patient_features", row_input)
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")

                    # Compute single-row SHAP immediately after prediction (local pipeline available).
                    # Population SHAP (compute_shap_local) is skipped here — it takes ~30s and
                    # is not needed for the GPT-4o explanation button.
                    if _ss("rf_result") is not None:
                        with st.spinner("Computing SHAP values…"):
                            try:
                                feat_names, row_shap, base_val = compute_shap_single_row(
                                    pipeline, row_input
                                )
                                _persist("shap_feature_names", feat_names)
                                _persist("shap_values_single", row_shap)
                                _persist("shap_base_value", base_val)
                            except Exception as exc:
                                st.warning(f"SHAP computation failed: {exc}")

                rf_result = _ss("rf_result")
                if rf_result:
                    st.markdown("---")
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.plotly_chart(
                            health_gain_meter(rf_result.prediction),
                            use_container_width=True,
                        )
                    with c2:
                        st.metric("Predicted Health Gain", f"{rf_result.prediction:.2f} pts")
                        mcid = 5.0
                        if rf_result.prediction >= mcid:
                            st.success(f"Meets MCID ({mcid} pts) — meaningful improvement expected.")
                        else:
                            st.warning(f"Below MCID ({mcid} pts) — review surgical pathway.")

                    # ── SHAP charts (offline mode) ────────────────────────────
                    shap_res_local = _ss("shap_result")
                    row_shap_local = _ss("shap_values_single")
                    feat_names_local = _ss("shap_feature_names")
                    base_val_local = _ss("shap_base_value", 0.0)

                    if shap_res_local:
                        st.markdown("#### SHAP Feature Attribution")
                        st.caption("Explains which patient factors drive this prediction most.")
                        tab_sum, tab_pat = st.tabs(
                            ["📊 Population summary (mean |SHAP|)", "🎯 This patient's prediction"]
                        )
                        with tab_sum:
                            st.plotly_chart(
                                shap_summary_bar(shap_res_local),
                                use_container_width=True,
                            )
                        with tab_pat:
                            if row_shap_local and feat_names_local:
                                default_viz = _ss("shap_viz_choice", "Force Plot")
                                viz_choice = st.radio(
                                    "Visualization style",
                                    ["Force Plot", "Waterfall"],
                                    index=0 if default_viz == "Force Plot" else 1,
                                    horizontal=True,
                                    key="shap_viz_radio_local",
                                )
                                if viz_choice != _ss("shap_viz_choice"):
                                    _persist("shap_viz_choice", viz_choice)
                                if viz_choice == "Force Plot":
                                    st.plotly_chart(
                                        shap_force_plot(
                                            feat_names_local, row_shap_local,
                                            base_val_local, rf_result.prediction,
                                        ),
                                        use_container_width=True,
                                    )
                                    st.caption(
                                        "🔴 Red = pushes prediction **up** · "
                                        "🔵 Blue = pushes prediction **down** · "
                                        f"Base: **{base_val_local:.2f}** → "
                                        f"Prediction: **{rf_result.prediction:.2f}**"
                                    )
                                else:
                                    st.plotly_chart(
                                        shap_waterfall(
                                            feat_names_local, row_shap_local,
                                            base_val_local, rf_result.prediction,
                                        ),
                                        use_container_width=True,
                                    )
                                    st.caption(
                                        "🔴 Red bars increase prediction · "
                                        "🔵 Blue bars decrease it · "
                                        "🟢 Final bar = predicted health gain"
                                    )
                            else:
                                st.info("Run a prediction to see the per-patient SHAP breakdown.")

with tab_dl:
    if ml_client is not None:
        _dl_section(ml_client)
    else:
        # ── Offline DL inference using local checkpoint ───────────────────────
        st.subheader("🔬 Deep Learning — X-ray KL Grade Classification")
        st.caption(
            "Model: **EfficientNet-B4** · KL grades 0–4 · CLAHE preprocessing · "
            "Checkpoint: `notebooks/DeepLearning/model/best_efficientnet_b4.pth`"
        )

        dl_model, dl_cfg = _load_dl_model_cached()

        if dl_model is None:
            if not DEFAULT_CHECKPOINT.exists():
                st.error(
                    f"Local checkpoint not found: `{DEFAULT_CHECKPOINT.name}`  \n"
                    f"Expected at: `{DEFAULT_CHECKPOINT}`"
                )
            else:
                st.warning(
                    "PyTorch is required for local DL inference.  \n"
                    "Install: `pip install torch torchvision opencv-python`  \n"
                    "Then restart the Streamlit app."
                )
        else:
            st.success(
                f"Local model loaded — EfficientNet-B4 · "
                f"{dl_cfg.get('num_classes', 5)} classes · "
                f"Input size {dl_cfg.get('img_size', 380)}×{dl_cfg.get('img_size', 380)}"
            )

            GRADE_LABELS = dl_cfg.get(
                "grade_labels",
                {0: "Healthy", 1: "Doubtful", 2: "Minimal", 3: "Moderate", 4: "Severe"},
            )

            # ── Image source ──────────────────────────────────────────────────
            img_source = st.radio(
                "Image source",
                ["🎲 Random from dataset", "📂 Upload my own X-ray"],
                horizontal=True,
                key="dl_img_source",
            )

            image_bytes: bytes | None = None
            image_label: str = ""
            true_grade: int | None = None   # ground truth from folder name

            if img_source == "🎲 Random from dataset":
                if not SAMPLE_IMAGES_DIR.exists():
                    st.warning(f"Sample image directory not found: `{SAMPLE_IMAGES_DIR}`")
                else:
                    # Pick random grade + random image on button click
                    if st.button("🎲 Pick Random X-ray", type="secondary", key="dl_random_pick"):
                        all_grades = [g for g in range(5) if list_sample_images(g)]
                        chosen_grade = random.choice(all_grades)
                        samples = list_sample_images(chosen_grade)
                        chosen_img = random.choice(samples)
                        _persist("dl_random_grade", chosen_grade)
                        _persist("dl_random_img", str(chosen_img))

                    # Show currently selected random image (persists across reruns)
                    stored_grade = _ss("dl_random_grade")
                    stored_img = _ss("dl_random_img")
                    if stored_img and Path(stored_img).exists():
                        true_grade = stored_grade
                        image_bytes = Path(stored_img).read_bytes()
                        image_label = Path(stored_img).name
                    else:
                        st.info("Click **🎲 Pick Random X-ray** to load a random knee X-ray.")

            else:
                uploaded = st.file_uploader(
                    "Choose knee X-ray (JPG / PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="dl_upload_local",
                    help="AP (anterior-posterior) view recommended. Ground truth unknown for uploads.",
                )
                if uploaded:
                    image_bytes = uploaded.read()
                    image_label = uploaded.name

            # ── Preview ───────────────────────────────────────────────────────
            if image_bytes:
                col_prev, col_run = st.columns([1, 2])
                with col_prev:
                    caption = image_label
                    if true_grade is not None:
                        caption += f"  \n📁 Dataset folder: {true_grade} → Actual KL Grade {true_grade}"
                    st.image(image_bytes, caption=caption, use_container_width=True)
                with col_run:
                    if true_grade is not None:
                        st.markdown(
                            f"**Ground Truth:** Grade {true_grade} — "
                            f"*{GRADE_LABELS.get(true_grade, '')}*"
                        )
                    st.caption("CLAHE contrast enhancement applied automatically before classification.")
                    if st.button("▶ Classify X-ray", type="primary", key="dl_classify_local"):
                        with st.spinner("Running EfficientNet-B4 inference locally…"):
                            try:
                                result = predict_xray(dl_model, dl_cfg, image_bytes)
                                _persist("xray_result", XrayPredictionResult(
                                    kl_grade=result["kl_grade"],
                                    confidence=result["confidence"],
                                    class_probabilities=result["class_probabilities"],
                                    class_names=result["class_names"],
                                    endpoint_used="local",
                                ))
                                _persist("xray_true_grade", true_grade)
                            except Exception as exc:
                                st.error(f"Local DL inference failed: {exc}")

            # ── DL results ────────────────────────────────────────────────────
            xray_result: XrayPredictionResult | None = _ss("xray_result")
            if xray_result:
                st.markdown("---")
                st.markdown("#### Classification Result")

                # Predicted vs Actual banner
                pred_kl = xray_result.kl_grade
                stored_true = _ss("xray_true_grade")
                if stored_true is not None:
                    correct = pred_kl == stored_true
                    match_icon = "✅ Correct" if correct else "❌ Incorrect"
                    colour = "#dcfce7" if correct else "#fee2e2"
                    border = "#16a34a" if correct else "#dc2626"
                    st.markdown(
                        f"<div style='background:{colour};border-left:4px solid {border};"
                        f"padding:10px 16px;border-radius:4px;margin-bottom:10px'>"
                        f"<b>Predicted:</b> Grade {pred_kl} ({GRADE_LABELS.get(pred_kl,'')})"
                        f" &nbsp;|&nbsp; "
                        f"<b>Actual (Ground Truth):</b> Grade {stored_true} "
                        f"({GRADE_LABELS.get(stored_true,'')})"
                        f" &nbsp;→&nbsp; {match_icon}</div>",
                        unsafe_allow_html=True,
                    )

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.plotly_chart(
                        kl_grade_gauge(pred_kl, xray_result.confidence),
                        use_container_width=True,
                    )
                with c2:
                    st.plotly_chart(
                        class_probability_bar(
                            xray_result.class_probabilities,
                            xray_result.class_names,
                        ),
                        use_container_width=True,
                    )
                kl = pred_kl
                col_desc, col_guide = st.columns(2)
                with col_desc:
                    st.info(f"**Radiographic finding:** {KL_DESCRIPTIONS.get(kl, '—')}")
                with col_guide:
                    guide = KL_SURGERY_GUIDANCE.get(kl, "—")
                    if kl >= 3:
                        st.warning(f"**Guidance:** {guide}")
                    else:
                        st.success(f"**Guidance:** {guide}")

# Section 3: Combined clinical advice (always shown when any result exists)
_clinical_advice_section()

# ── Navigation ────────────────────────────────────────────────────────────────

st.divider()
if st.button("Continue to Conclusions →", type="primary"):
    mark_stage_complete("explanation")
    st.switch_page("pages/10_conclusions.py")
