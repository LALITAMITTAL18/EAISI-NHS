"""ML Pipeline — Streamlit application entry point.

Run with:
    cd MLPortal
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from shared.nav import render_sidebar
from shared.state import (
    STAGES,
    create_project,
    delete_project,
    get_active_project,
    get_state,
    list_projects,
    set_active_project,
)

st.set_page_config(
    page_title="ML Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project gate — must select or create a project before entering pipeline ───
active = get_active_project()

if active is None:
    # ── Project landing ───────────────────────────────────────────────────────
    st.title("🔬 ML Pipeline")
    st.subheader("Select or create a project")
    st.caption(
        "Each project stores its data, models and session state independently. "
        "You can return to any project at any time and resume where you left off."
    )

    projects = list_projects()

    tab_open, tab_new = st.tabs(["📂 Open existing project", "➕ Create new project"])

    with tab_new:
        st.markdown("#### New project")
        col_n, col_d = st.columns([1, 2])
        with col_n:
            new_name = st.text_input("Project name", placeholder="e.g. NHS Knee Replacement")
        with col_d:
            new_desc = st.text_input("Description (optional)", placeholder="Brief description of the analysis")
        if st.button("Create project", type="primary", disabled=not new_name.strip()):
            try:
                proj = create_project(new_name.strip(), new_desc.strip())
                set_active_project(proj)
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    with tab_open:
        if not projects:
            st.info("No projects yet. Create one using the tab above.")
        else:
            st.markdown("#### Your projects")
            for proj in projects:
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.markdown(f"**{proj['name']}**")
                    if proj.get("description"):
                        st.caption(proj["description"])
                    st.caption(f"Created: {proj['created_at']}")
                with col_b:
                    if st.button("Open", key=f"open_{proj['slug']}", type="primary", use_container_width=True):
                        set_active_project(proj)
                        st.rerun()
                with col_c:
                    if st.button("Delete", key=f"del_{proj['slug']}", type="secondary", use_container_width=True):
                        st.session_state[f"_confirm_del_{proj['slug']}"] = True
                if st.session_state.get(f"_confirm_del_{proj['slug']}"):
                    st.warning(
                        f"⚠️ Delete **{proj['name']}** and all its data? This cannot be undone."
                    )
                    col_y, col_n2 = st.columns(2)
                    with col_y:
                        if st.button("Yes, delete", key=f"yes_{proj['slug']}", type="primary"):
                            delete_project(proj["slug"])
                            st.session_state.pop(f"_confirm_del_{proj['slug']}", None)
                            st.rerun()
                    with col_n2:
                        if st.button("Cancel", key=f"no_{proj['slug']}"):
                            st.session_state.pop(f"_confirm_del_{proj['slug']}", None)
                            st.rerun()
                st.divider()
    st.stop()

# ── Project is active — show the pipeline home page ───────────────────────────
render_sidebar()

get_state()  # bootstrap project session state from disk

state = get_state()
completed = sum(state.stage_complete.get(s, False) for s in STAGES)

st.title(f"🔬 {active['name']}")
if active.get("description"):
    st.caption(active["description"])

st.progress(completed / len(STAGES), text=f"{completed}/{len(STAGES)} stages complete")

st.markdown(
    """
---

### Pipeline stages

| Stage | Name | Description |
|---|---|---|
| 1 | **Upload Data** | Import CSV, Excel, Parquet or JSON; define or derive outcome variable; configure sentinels |
| 2 | **Explore** | Distributions, correlations, Q-Q plots, subgroup analysis |
| 3 | **Outlier Detection** | IQR + Z-score dual flagging; per-column action selection |
| 4 | **Missing Data** | Missingness analysis; configure imputation per column |
| 5 | **Feature Engineering** | Derived features, polynomial interactions, transforms, encoding |
| 6 | **Preparation & Split** | Train/test split (pipeline or bypass); Outcome Threshold |
| 7 | **Modelling** | Select any model from the registry; Optuna HPO; fine-tuning |
| 8 | **Model Comparison** | Metrics, Bland-Altman, calibration, equity, threshold explorer |
| 9 | **Explanation** | SHAP, permutation importance, PDP, live prediction explorer |
| 10 | **Conclusions** | Model card, performance matrix, export HTML/CSV report |

---
"""
)

if state.upload_cfg:
    st.info(
        f"📂 Dataset: **{state.upload_cfg.get('file_name', '—')}**  |  "
        f"🎯 Target: **{state.upload_cfg.get('target_column', '—')}**  |  "
        f"⚙️ Task: **{state.upload_cfg.get('task_type', '—').replace('_', ' ').title()}**"
    )
    st.page_link("pages/1_upload.py", label="→ Go to Stage 1 — Upload Data", icon="📤")
else:
    st.info("👈 Use the sidebar or the link below to begin.")
    st.page_link("pages/1_upload.py", label="→ Start: Stage 1 — Upload Data", icon="📤")


# ── Landing page ──────────────────────────────────────────────────────────────
st.title("🔬 General-Purpose ML Pipeline")
st.markdown(
    """
Welcome! This tool guides you through a complete, reproducible machine learning pipeline
in **10 sequential stages** — from raw data upload to model conclusions.

---

### How it works

| Stage | Name | Description |
|---|---|---|
| 1 | **Upload Data** | Import CSV, Excel, Parquet or JSON; configure sentinels and target |
| 2 | **Explore** | Distributions, correlations, Q-Q plots, subgroup analysis |
| 3 | **Outlier Detection** | IQR + Z-score dual flagging; per-column action selection |
| 4 | **Missing Data** | Missingness analysis; configure imputation per column |
| 5 | **Feature Engineering** | Derived features, polynomial interactions, transforms, encoding |
| 6 | **Preparation & Split** | Train/test split; apply all transforms; Outcome Threshold |
| 7 | **Modelling** | Select any model from the registry; Optuna HPO; fine-tuning |
| 8 | **Model Comparison** | Metrics, Bland-Altman, calibration, equity analysis |
| 9 | **Explanation** | SHAP, permutation importance, PDP, live prediction explorer |
| 10 | **Conclusions** | Model card, performance matrix, export HTML/CSV report |

---

### Getting started

Use the **sidebar** to navigate between stages. Completed stages are marked ✅.  
Your progress is **automatically saved** — you can close and reopen the app at any time.

---

### Adding custom models

Edit `website/data/custom_models.json` to register any scikit-learn compatible model
without touching the code. The schema is documented in `website/data/models_schema.json`.

```json
[
  {
    "name": "MyCustomForest",
    "display_name": "My Custom Random Forest",
    "task": ["regression"],
    "constructor_path": "sklearn.ensemble.ExtraTreesRegressor",
    "default_params": {"n_estimators": 300, "random_state": 42},
    "optuna_space": {
      "n_estimators": {"type": "int", "low": 100, "high": 700},
      "max_depth":    {"type": "int", "low": 3,   "high": 15}
    },
    "finetune_window": 0.2,
    "requires_scaling": false,
    "supports_warmstart": false,
    "tags": ["bagging", "tree"]
  }
]
```
    """
)

state = get_state()
completed = sum(state.stage_complete.get(s, False) for s in STAGES)
st.progress(completed / len(STAGES), text=f"{completed}/{len(STAGES)} stages complete")

if state.upload_cfg:
    st.info(
        f"📂 Current dataset: **{state.upload_cfg.get('file_name', '—')}**  |  "
        f"🎯 Target: **{state.upload_cfg.get('target_column', '—')}**  |  "
        f"⚙️ Task: **{state.upload_cfg.get('task_type', '—').replace('_', ' ').title()}**"
    )
else:
    st.info("👈 Start by navigating to **Stage 1 — Upload Data** in the sidebar.")
