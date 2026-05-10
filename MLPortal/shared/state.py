"""Session state management — persistence across Streamlit restarts.

SessionState is a Pydantic model that is:
- Serialized to outputs/<project>/session/session_state.json after every meaningful action
- Restored from disk when the app starts (if the file exists)
- Accessed via get_state() and mutated via update_state()

Projects are stored in outputs/<project_slug>/ so multiple ML experiments can
coexist on the same machine without interfering with each other.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import BaseModel, ConfigDict, Field

# ── Constants ─────────────────────────────────────────────────────────────────

STAGES = [
    "upload",
    "explore",
    "outliers",
    "missing",
    "features",
    "preparation",
    "modelling",
    "comparison",
    "explanation",
    "conclusions",
]

_BASE_DIR = Path(__file__).parent.parent  # MLPortal/ root
PROJECTS_DIR = _BASE_DIR / "outputs"
PROJECTS_INDEX = PROJECTS_DIR / "projects.json"

# ── Project helpers ───────────────────────────────────────────────────────────


def _slug(name: str) -> str:
    """Convert a project name to a safe directory slug."""
    return re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())[:60]


def list_projects() -> list[dict]:
    """Return list of {name, slug, description, created_at} dicts."""
    if PROJECTS_INDEX.exists():
        try:
            return json.loads(PROJECTS_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def create_project(name: str, description: str = "") -> dict:
    """Create a new project directory and register it. Returns the project dict."""
    import datetime
    slug = _slug(name)
    if not slug:
        raise ValueError("Project name must contain at least one letter or digit.")
    project_dir = PROJECTS_DIR / slug
    project_dir.mkdir(parents=True, exist_ok=True)

    projects = list_projects()
    existing_slugs = [p["slug"] for p in projects]
    if slug in existing_slugs:
        raise ValueError(f"A project named '{name}' already exists.")

    entry = {
        "name": name,
        "slug": slug,
        "description": description,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    projects.append(entry)
    PROJECTS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    PROJECTS_INDEX.write_text(json.dumps(projects, indent=2), encoding="utf-8")
    return entry


def delete_project(slug: str) -> None:
    """Delete a project directory and remove from index."""
    project_dir = PROJECTS_DIR / slug
    if project_dir.exists():
        shutil.rmtree(project_dir)
    projects = [p for p in list_projects() if p["slug"] != slug]
    PROJECTS_INDEX.write_text(json.dumps(projects, indent=2), encoding="utf-8")


def get_project_state_path(slug: str) -> Path:
    return PROJECTS_DIR / slug / "session" / "session_state.json"


def get_project_datasets_dir(slug: str) -> Path:
    return PROJECTS_DIR / slug / "datasets"


def get_project_models_dir(slug: str) -> Path:
    return PROJECTS_DIR / slug / "models"


def get_project_reports_dir(slug: str) -> Path:
    return PROJECTS_DIR / slug / "reports"


# ── Active project helpers ────────────────────────────────────────────────────


def get_active_project() -> dict | None:
    """Return the active project dict stored in st.session_state, or None."""
    return st.session_state.get("_active_project")


def set_active_project(project: dict) -> None:
    """Set the active project and clear any loaded state so it reloads from disk."""
    st.session_state["_active_project"] = project
    st.session_state.pop("_app_state", None)  # force reload from project's state file


# ── Pydantic model ────────────────────────────────────────────────────────────


class SessionState(BaseModel):
    """Fully JSON-serializable snapshot of all pipeline decisions and artifact paths."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    # Stage completion flags
    stage_complete: dict[str, bool] = Field(
        default_factory=lambda: {s: False for s in STAGES}
    )
    last_stage: str = "upload"
    notes: str = ""

    # Artifact paths (relative to project outputs dir, kept as str for JSON compat)
    raw_data_path: str | None = None
    outlier_data_path: str | None = None
    imputed_data_path: str | None = None
    features_data_path: str | None = None
    train_data_path: str | None = None
    test_data_path: str | None = None
    results_cache_path: str | None = None
    best_model_path: str | None = None
    pipeline_config_path: str | None = None

    # Per-stage configs stored as plain dicts (JSON-serializable)
    upload_cfg: dict[str, Any] = Field(default_factory=dict)
    outlier_cfg: dict[str, Any] = Field(default_factory=dict)
    missing_cfg: dict[str, Any] = Field(default_factory=dict)
    feature_cfg: dict[str, Any] = Field(default_factory=dict)
    prep_cfg: dict[str, Any] = Field(default_factory=dict)
    train_cfg: dict[str, Any] = Field(default_factory=dict)

    # Dataset variants — multiple prepared train/test pairs within one project
    # Each entry: {name, slug, description, train_path, test_path, n_train, n_test,
    #              n_features, created_at, notes, pipeline_method}
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    active_dataset: str | None = None  # slug of the currently selected variant

    # Lightweight results summary
    best_model_name: str | None = None
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)
    run_summary: dict[str, Any] = Field(default_factory=dict)


# ── Persistence ───────────────────────────────────────────────────────────────


def _active_state_path() -> Path:
    """Return the state JSON path for the current active project."""
    project = get_active_project()
    if project:
        return get_project_state_path(project["slug"])
    return PROJECTS_DIR / "_default" / "session" / "session_state.json"


def _active_base_dir() -> Path:
    """Return the output base dir for the active project."""
    project = get_active_project()
    if project:
        return PROJECTS_DIR / project["slug"]
    return PROJECTS_DIR / "_default"


def save_state(state: SessionState, path: Path | None = None) -> None:
    """Serialize *state* to disk as JSON."""
    p = path or _active_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def load_state(path: Path | None = None) -> SessionState:
    """Restore state from disk; return a blank SessionState if file does not exist."""
    p = path or _active_state_path()
    if p.exists():
        try:
            return SessionState.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            return SessionState()
    return SessionState()


def reset_state(path: Path | None = None) -> SessionState:
    """Delete the persisted state file and return a blank SessionState."""
    p = path or _active_state_path()
    if p.exists():
        p.unlink()
    fresh = SessionState()
    st.session_state["_app_state"] = fresh
    return fresh


# ── Streamlit accessors ───────────────────────────────────────────────────────


def get_state() -> SessionState:
    """Return the current SessionState (loads from project disk on first call)."""
    if "_app_state" not in st.session_state:
        st.session_state["_app_state"] = load_state()
    return st.session_state["_app_state"]  # type: ignore[return-value]


def update_state(updates: dict[str, Any], auto_save: bool = True) -> SessionState:
    """Apply *updates* to the current state, persist, and return the new state."""
    current = get_state()
    updated = current.model_copy(update=updates)
    st.session_state["_app_state"] = updated
    if auto_save:
        save_state(updated)
    return updated


def mark_stage_complete(stage: str, auto_save: bool = True) -> SessionState:
    """Mark a pipeline stage as complete and advance last_stage."""
    state = get_state()
    new_complete = {**state.stage_complete, stage: True}
    next_stage = _next_stage(stage)
    return update_state(
        {"stage_complete": new_complete, "last_stage": next_stage or stage},
        auto_save=auto_save,
    )


def _next_stage(current: str) -> str | None:
    try:
        idx = STAGES.index(current)
        return STAGES[idx + 1] if idx + 1 < len(STAGES) else None
    except ValueError:
        return None


def project_artifact_path(relative: str) -> Path:
    """Resolve a relative artifact path against the active project's output dir."""
    return _active_base_dir() / relative


def project_datasets_dir() -> Path:
    return _active_base_dir() / "datasets"


def project_models_dir() -> Path:
    return _active_base_dir() / "models"


def project_reports_dir() -> Path:
    return _active_base_dir() / "reports"


# ── Dataset variant helpers ───────────────────────────────────────────────────


def _dataset_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())[:40]


def list_project_datasets() -> list[dict]:
    """Return all registered dataset variants for the active project."""
    state = get_state()
    return state.datasets


def list_trained_variants() -> list[dict]:
    """Return metadata for every variant that has trained models on disk.

    A variant is considered trained if it has either:
    - a ``{slug}_results_cache.joblib`` file, OR
    - at least one ``{slug}__{model_name}.joblib`` individual model file.

    Each returned dict contains:
      slug, name, cache_path (filename or ""), task_type, target_column, test_path
    Values fall back to project-level upload_cfg when the variant has no overrides.
    """
    models_dir = project_models_dir()
    if not models_dir.exists():
        return []
    state = get_state()
    variant_map = {v["slug"]: v for v in state.datasets}
    upload_cfg = state.upload_cfg
    _cache_suffix = "_results_cache.joblib"

    # Collect slugs from cache files
    slugs_seen: dict[str, str] = {}  # slug -> cache_path filename (or "")
    for cache_file in models_dir.glob(f"*{_cache_suffix}"):
        slug = cache_file.name[: -len(_cache_suffix)]
        slugs_seen[slug] = cache_file.name

    # Also collect slugs from individual model files ({slug}__{model}.joblib)
    for f in models_dir.glob("*__*.joblib"):
        slug = f.name.split("__")[0]
        if slug not in slugs_seen:
            slugs_seen[slug] = ""  # no cache file, but individual models exist

    trained: list[dict] = []
    for slug in sorted(slugs_seen):
        meta = variant_map.get(slug, {})
        trained.append({
            "slug": slug,
            "name": meta.get("name", slug),
            "cache_path": slugs_seen[slug],
            "task_type": meta.get("task_type") or upload_cfg.get("task_type", "regression"),
            "target_column": meta.get("target_column") or upload_cfg.get("target_column", ""),
            "test_path": meta.get("test_path") or state.test_data_path,
        })
    return trained


def load_variant_results(slug: str, models_dir: Path, variant_task: str = "regression") -> list:
    """Load a variant's results cache and supplement with individual model files on disk.

    The results cache only contains models from a single training run.  If the
    user has trained different model subsets in separate runs, some models exist
    only as individual ``{slug}__{model_name}.joblib`` files.  This function:

    1. Loads the results cache (if it exists) to get ``TrainResult`` objects with
       full HPO history and CV scores.
    2. For any ``TrainResult`` whose ``pipeline`` is ``None`` (field is excluded
       from Pydantic serialisation and may not survive pickling), loads the
       pipeline from the matching individual file.
    3. Adds a minimal ``TrainResult`` for every individual file whose model name
       is NOT already in the cache.

    Returns a list of ``TrainResult`` objects, all with ``pipeline`` set.
    """
    # Late import to avoid circular dependency at module load time.
    from stages.modelling.models import TrainResult
    from shared.io import load_joblib

    cache_file = models_dir / f"{slug}_results_cache.joblib"
    cached_raw = []
    if cache_file.exists():
        try:
            cached_raw = load_joblib(cache_file)
        except Exception:
            # Corrupt or truncated cache (e.g. interrupted write) — fall back to
            # individual model files below.  Remove the bad file so a fresh run
            # can write a clean one.
            try:
                cache_file.unlink()
            except OSError:
                pass
    # Support both old format (list[TrainResult] pickled objects) and new format (list[dict]).
    # New format avoids Pydantic class-identity pickling errors on Streamlit reruns.
    cached: list[TrainResult] = []
    for item in cached_raw:
        if isinstance(item, dict):
            item.pop("pipeline", None)  # never stored, but guard against stale keys
            try:
                cached.append(TrainResult(**item))
            except Exception:
                pass
        elif hasattr(item, "model_name"):
            cached.append(item)  # old format: already a TrainResult
    cached_map: dict[str, TrainResult] = {r.model_name: r for r in cached}

    prefix = f"{slug}__"
    for f in sorted(models_dir.glob(f"{prefix}*.joblib")):
        model_name = f.name[len(prefix) : -len(".joblib")]
        if model_name in cached_map:
            # Restore pipeline if it was lost (exclude=True field)
            if cached_map[model_name].pipeline is None:
                cached_map[model_name].pipeline = load_joblib(f)
        else:
            # Model trained in a separate run — create a minimal TrainResult
            pipeline = load_joblib(f)
            cached_map[model_name] = TrainResult(
                model_name=model_name,
                task=variant_task,
                best_params={},
                optuna_history=[],
                pipeline=pipeline,
            )

    # Return in consistent order: cached entries first, then extras alphabetically
    cached_names = [r.model_name for r in cached]
    extras = sorted(k for k in cached_map if k not in cached_names)
    return [cached_map[n] for n in cached_names + extras]


def register_dataset(
    name: str,
    train_path: str,
    test_path: str,
    *,
    description: str = "",
    n_train: int = 0,
    n_test: int = 0,
    n_features: int = 0,
    notes: str = "",
    pipeline_method: str = "",
    set_active: bool = True,
) -> dict:
    """Register a new dataset variant in the project state and optionally activate it.

    Args:
        name: Human-readable name (e.g. "MICE Imputation").
        train_path: Filename of the train parquet relative to project datasets dir.
        test_path: Filename of the test parquet relative to project datasets dir.

    Returns the registered variant dict.
    """
    import datetime

    slug = _dataset_slug(name)
    state = get_state()

    # Overwrite if same slug already registered
    existing = [d for d in state.datasets if d["slug"] != slug]
    entry: dict[str, Any] = {
        "name": name,
        "slug": slug,
        "description": description,
        "train_path": train_path,
        "test_path": test_path,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "pipeline_method": pipeline_method,
    }
    existing.append(entry)

    updates: dict[str, Any] = {"datasets": existing}
    if set_active or state.active_dataset is None:
        updates["active_dataset"] = slug
        updates["train_data_path"] = train_path
        updates["test_data_path"] = test_path

    update_state(updates)
    return entry


def set_active_dataset(slug: str) -> None:
    """Switch the active dataset variant. Updates train/test paths in state."""
    state = get_state()
    variant = next((d for d in state.datasets if d["slug"] == slug), None)
    if variant is None:
        raise ValueError(f"No dataset variant with slug '{slug}'")
    update_state({
        "active_dataset": slug,
        "train_data_path": variant["train_path"],
        "test_data_path": variant["test_path"],
    })


# ── Streamlit accessors ───────────────────────────────────────────────────────


def get_state() -> SessionState:
    """Return the current SessionState from st.session_state (load from disk on first call)."""
    if "_app_state" not in st.session_state:
        st.session_state["_app_state"] = load_state()
    return st.session_state["_app_state"]  # type: ignore[return-value]


def update_state(updates: dict[str, Any], auto_save: bool = True) -> SessionState:
    """Apply *updates* to the current state, persist to disk, and return the new state."""
    current = get_state()
    updated = current.model_copy(update=updates)
    st.session_state["_app_state"] = updated
    if auto_save:
        save_state(updated)
    return updated


def mark_stage_complete(stage: str, auto_save: bool = True) -> SessionState:
    """Mark a pipeline stage as complete and update last_stage."""
    state = get_state()
    new_complete = {**state.stage_complete, stage: True}
    # Advance last_stage to next incomplete
    next_stage = _next_stage(stage)
    return update_state(
        {"stage_complete": new_complete, "last_stage": next_stage or stage},
        auto_save=auto_save,
    )


def _next_stage(current: str) -> str | None:
    """Return the name of the stage after *current*, or None if it is the last."""
    try:
        idx = STAGES.index(current)
        return STAGES[idx + 1] if idx + 1 < len(STAGES) else None
    except ValueError:
        return None
