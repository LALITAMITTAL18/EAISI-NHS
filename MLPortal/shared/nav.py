"""Sidebar navigation rendered on every page."""

from __future__ import annotations

import streamlit as st

from shared.state import STAGES, get_active_project, get_state, reset_state

_STAGE_LABELS: dict[str, str] = {
    "upload":      "1 · Upload Data",
    "explore":     "2 · Explore",
    "outliers":    "3 · Outlier Detection",
    "missing":     "4 · Missing Data",
    "features":    "5 · Feature Engineering",
    "preparation": "6 · Preparation & Split",
    "modelling":   "7 · Modelling",
    "comparison":  "8 · Model Comparison",
    "explanation": "9 · Explanation",
    "conclusions": "10 · Conclusions",
}

_PAGE_PATHS: dict[str, str] = {
    "upload":      "pages/1_upload.py",
    "explore":     "pages/2_explore.py",
    "outliers":    "pages/3_outliers.py",
    "missing":     "pages/4_missing.py",
    "features":    "pages/5_features.py",
    "preparation": "pages/6_preparation.py",
    "modelling":   "pages/7_modelling.py",
    "comparison":  "pages/8_comparison.py",
    "explanation": "pages/9_clinical_insight.py",
    "conclusions": "pages/10_conclusions.py",
}


def render_sidebar() -> None:
    """Draw the persistent sidebar with stage progress indicators.
    
    Also enforces the project gate — if no project is active, redirects to app.py.
    """
    active = get_active_project()
    if active is None:
        st.warning("No project selected.")
        st.page_link("app.py", label="← Select or create a project")
        st.stop()

    state = get_state()

    with st.sidebar:
        st.markdown("## 🔬 ML Pipeline")

        # ── Active project ────────────────────────────────────────────────────
        if active:
            st.caption(f"📁 Project: **{active['name']}**")
            if st.button("← Switch project", use_container_width=True, key="_nav_switch"):
                st.session_state.pop("_active_project", None)
                st.session_state.pop("_app_state", None)
                st.switch_page("app.py")
        st.divider()

        # ── Stage progress ────────────────────────────────────────────────────
        st.markdown("**Pipeline stages**")
        for stage in STAGES:
            complete = state.stage_complete.get(stage, False)
            icon = "✅" if complete else "○"
            label = _STAGE_LABELS[stage]
            st.page_link(_PAGE_PATHS[stage], label=f"{icon} {label}")

        st.divider()

        # ── Dataset summary ───────────────────────────────────────────────────
        if state.upload_cfg:
            fname = state.upload_cfg.get("file_name", "—")
            task = state.upload_cfg.get("task_type", "—").replace("_", " ").title()
            target = state.upload_cfg.get("target_column", "—")
            st.caption(f"📂 **File:** {fname}")
            st.caption(f"🎯 **Target:** {target}")
            st.caption(f"⚙️ **Task:** {task}")
            st.divider()

        # ── Session options ───────────────────────────────────────────────────
        with st.expander("⚙️ Session options"):
            if st.button("🔄 Reset all progress", type="secondary", use_container_width=True):
                if st.session_state.get("_confirm_reset"):
                    reset_state()
                    st.session_state.pop("_confirm_reset", None)
                    st.rerun()
                else:
                    st.session_state["_confirm_reset"] = True
                    st.warning("Click again to confirm reset. This deletes all saved progress for this project.")

