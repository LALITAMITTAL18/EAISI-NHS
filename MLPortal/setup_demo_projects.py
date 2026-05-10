"""Bootstrap NHS Hip and Knee Replacement demo projects in MLPortal.

Copies pre-processed data from notebooks/Experiment/{Hip,Knee}/data/interim/
and notebooks/Experiment/Knee/prepared_datasets/ into the MLPortal outputs
directory structure and writes fully-formed session_state.json files so the
projects appear ready-to-use on first launch.

Dataset variants registered per project
-----------------------------------------
NHS Knee Replacement:
  1. Manual Imputation         — 2.1-train/test (hand-crafted, 48 features incl. health_gain)
  2. Median / Mode + Flags     — pipeline2_median_mode (50 features, missingness indicator cols)
  3. MICE                      — pipeline3_mice (50 features, MICE imputation)
  4. Reduced Features          — pipeline4_reduced_features (30 features, fewer comorbidities)

NHS Hip Replacement:
  1. Manual Imputation         — 2.1-train/test (50 features)

Run once from the MLPortal/ directory:
    python setup_demo_projects.py

Safe to re-run: skips projects and files that already exist.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

MLPORTAL = Path(__file__).parent  # MLPortal/
REPO_ROOT = MLPORTAL.parent  # EAISI-NHS/
OUTPUTS = MLPORTAL / "outputs"
PROJECTS_INDEX = OUTPUTS / "projects.json"

KNEE_INTERIM = REPO_ROOT / "notebooks" / "Experiment" / "Knee" / "data" / "interim"
KNEE_PREPARED = REPO_ROOT / "notebooks" / "Experiment" / "Knee" / "prepared_datasets"
HIP_INTERIM = REPO_ROOT / "notebooks" / "Experiment" / "Hip" / "data" / "interim"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())[:60]


def _dslug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())[:40]


def _load_index() -> list[dict]:
    if PROJECTS_INDEX.exists():
        return json.loads(PROJECTS_INDEX.read_text(encoding="utf-8"))
    return []


def _save_index(projects: list[dict]) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PROJECTS_INDEX.write_text(json.dumps(projects, indent=2), encoding="utf-8")


def _register_project(name: str, description: str, slug: str) -> dict:
    projects = _load_index()
    existing = [p for p in projects if p["slug"] == slug]
    if existing:
        print(f"  [skip] Project '{name}' already registered.")
        return existing[0]
    entry = {
        "name": name,
        "slug": slug,
        "description": description,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    projects.append(entry)
    _save_index(projects)
    print(f"  [ok]   Registered project '{name}' (slug: {slug})")
    return entry


def _copy_parquet(src: Path, dst: Path) -> tuple[int, int]:
    """Copy a parquet file, return (n_rows, n_cols)."""
    df = pd.read_parquet(src)
    if dst.exists():
        print(f"  [skip] {dst.name} already exists.")
        return df.shape[0], df.shape[1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False, compression="gzip")
    size_mb = dst.stat().st_size / 1_048_576
    print(f"  [ok]   {dst.name}  ({df.shape[0]:,} rows × {df.shape[1]} cols, {size_mb:.1f} MB)")
    return df.shape[0], df.shape[1]


def _make_dataset_entry(
    name: str,
    train_path: str,
    test_path: str,
    n_train: int,
    n_test: int,
    n_features: int,
    *,
    description: str = "",
    pipeline_method: str = "",
    notes: str = "",
) -> dict:
    slug = _dslug(name)
    return {
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


def _write_session_state(
    slug: str,
    *,
    file_name: str,
    target_column: str,
    task_type: str,
    n_rows: int,
    n_cols: int,
    derived_formula: str | None = None,
    derived_column: str | None = None,
    sentinel_values: list[str] | None = None,
    datasets: list[dict] | None = None,
    active_dataset: str | None = None,
) -> None:
    state_path = OUTPUTS / slug / "session" / "session_state.json"
    if state_path.exists():
        print(f"  [skip] session_state.json already exists.")
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)

    upload_cfg: dict = {
        "file_name": file_name,
        "target_column": target_column,
        "task_type": task_type,
        "sentinel_values": sentinel_values or ["*", " ", "9", "999"],
        "csv_separator": ",",
        "n_rows": n_rows,
        "n_cols": n_cols,
    }
    if derived_formula:
        upload_cfg["derived_outcome_formula"] = derived_formula
        upload_cfg["derived_outcome_column"] = derived_column

    stage_complete = {
        "upload": True,
        "explore": True,
        "outliers": True,
        "missing": True,
        "features": True,
        "preparation": True,
        "modelling": False,
        "comparison": False,
        "explanation": False,
        "conclusions": False,
    }

    # Derive train/test paths from the first (default) dataset variant
    first = (datasets or [{}])[0]
    train_data_path = first.get("train_path")
    test_data_path = first.get("test_path")
    if active_dataset:
        active = next((d for d in (datasets or []) if d["slug"] == active_dataset), first)
        train_data_path = active.get("train_path")
        test_data_path = active.get("test_path")

    state = {
        "stage_complete": stage_complete,
        "last_stage": "modelling",
        "notes": (
            f"Demo project bootstrapped from NHS PROMs experimental notebooks. "
            f"Data sourced from England NHS {file_name.split('.')[0]} dataset "
            f"(2016-17 to 2018-19). Target: '{target_column}' ({task_type})."
        ),
        "raw_data_path": "1_raw.parquet",
        "outlier_data_path": None,
        "imputed_data_path": None,
        "features_data_path": None,
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        "results_cache_path": None,
        "best_model_path": None,
        "pipeline_config_path": None,
        "upload_cfg": upload_cfg,
        "outlier_cfg": {
            "method": "iqr_zscore",
            "note": "Applied during notebook pre-processing (1.2-Outlier-Detection.ipynb)",
        },
        "missing_cfg": {
            "column_specs": [],
            "global_indicator_threshold": 0.1,
            "note": "Imputation applied during notebook pre-processing",
        },
        "feature_cfg": {
            "derived_features": [
                {"name": "comorbidity_burden", "formula": "sum of comorbidity flags"},
                {"name": "depression_pain_interaction", "formula": "Depression * Pre-Op Q Discomfort"},
                {"name": "depression_anxiety_interaction", "formula": "Depression * Pre-Op Q Anxiety"},
            ],
        },
        "prep_cfg": {
            "test_size": 0.3,
            "random_state": 42,
            "note": "Split from notebook 2.1-data-preparation-Manual.ipynb",
        },
        "train_cfg": {},
        "datasets": datasets or [],
        "active_dataset": active_dataset or (first.get("slug") if first else None),
        "best_model_name": None,
        "metrics_summary": {},
        "comparison_table": [],
        "run_summary": {},
    }

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"  [ok]   session_state.json written ({len(datasets or [])} dataset variant(s)).")


# ── Project: NHS Knee Replacement ─────────────────────────────────────────────

def setup_knee() -> None:
    print("\n=== NHS Knee Replacement ===")
    slug = "nhs_knee_replacement"
    datasets_dir = OUTPUTS / slug / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    raw_src = KNEE_INTERIM / "1.1-Reduced.parquet"
    if not raw_src.exists():
        print(f"  [ERROR] {raw_src} not found. Skipping.")
        return

    _copy_parquet(raw_src, datasets_dir / "1_raw.parquet")
    raw_df = pd.read_parquet(raw_src)

    dataset_variants: list[dict] = []

    # Variant 1 — Manual Imputation (2.1)
    v1_train_src = KNEE_INTERIM / "2.1-train.parquet"
    v1_test_src = KNEE_INTERIM / "2.1-test.parquet"
    if v1_train_src.exists():
        n_rows, n_cols = _copy_parquet(v1_train_src, datasets_dir / "manual_imputation_train.parquet")
        nt_rows, _ = _copy_parquet(v1_test_src, datasets_dir / "manual_imputation_test.parquet")
        dataset_variants.append(_make_dataset_entry(
            name="Manual Imputation",
            train_path="manual_imputation_train.parquet",
            test_path="manual_imputation_test.parquet",
            n_train=n_rows,
            n_test=nt_rows,
            n_features=n_cols - 1,
            description="Hand-crafted imputation from notebook 2.1",
            pipeline_method="Manual imputation | 48 features",
            notes=(
                "Source: notebooks/Experiment/Knee/data/interim/2.1-train/test.parquet\n"
                "Strategy: manual, domain-guided imputation of missing values. "
                "Includes engineered features: comorbidity_burden, depression interactions."
            ),
        ))

    # Variant 2 — Median / Mode + Missingness Flags
    v2_train_src = KNEE_PREPARED / "pipeline2_median_mode_train.parquet"
    v2_test_src = KNEE_PREPARED / "pipeline2_median_mode_test.parquet"
    if v2_train_src.exists():
        n_rows, n_cols = _copy_parquet(v2_train_src, datasets_dir / "median_mode_train.parquet")
        nt_rows, _ = _copy_parquet(v2_test_src, datasets_dir / "median_mode_test.parquet")
        dataset_variants.append(_make_dataset_entry(
            name="Median / Mode + Flags",
            train_path="median_mode_train.parquet",
            test_path="median_mode_test.parquet",
            n_train=n_rows,
            n_test=nt_rows,
            n_features=n_cols - 1,
            description="Median/mode imputation with missingness indicator columns",
            pipeline_method="Median/mode imputation | 50 features | missingness flags",
            notes=(
                "Source: notebooks/Experiment/Knee/prepared_datasets/pipeline2_median_mode_*.parquet\n"
                "Strategy: numeric → median, categorical → mode. "
                "Adds *_missing flag columns for Age Band, Gender and Pre-Op questionnaire items."
            ),
        ))

    # Variant 3 — MICE
    v3_train_src = KNEE_PREPARED / "pipeline3_mice_train.parquet"
    v3_test_src = KNEE_PREPARED / "pipeline3_mice_test.parquet"
    if v3_train_src.exists():
        n_rows, n_cols = _copy_parquet(v3_train_src, datasets_dir / "mice_train.parquet")
        nt_rows, _ = _copy_parquet(v3_test_src, datasets_dir / "mice_test.parquet")
        dataset_variants.append(_make_dataset_entry(
            name="MICE Imputation",
            train_path="mice_train.parquet",
            test_path="mice_test.parquet",
            n_train=n_rows,
            n_test=nt_rows,
            n_features=n_cols - 1,
            description="Multiple Imputation by Chained Equations (MICE / IterativeImputer)",
            pipeline_method="MICE imputation | 50 features | missingness flags",
            notes=(
                "Source: notebooks/Experiment/Knee/prepared_datasets/pipeline3_mice_*.parquet\n"
                "Strategy: sklearn IterativeImputer (BayesianRidge estimator, 10 iterations). "
                "Same missingness flags as pipeline2 but imputed values differ."
            ),
        ))

    # Variant 4 — Reduced Features
    v4_train_src = KNEE_PREPARED / "pipeline4_reduced_features_train.parquet"
    v4_test_src = KNEE_PREPARED / "pipeline4_reduced_features_test.parquet"
    if v4_train_src.exists():
        n_rows, n_cols = _copy_parquet(v4_train_src, datasets_dir / "reduced_features_train.parquet")
        nt_rows, _ = _copy_parquet(v4_test_src, datasets_dir / "reduced_features_test.parquet")
        dataset_variants.append(_make_dataset_entry(
            name="Reduced Features",
            train_path="reduced_features_train.parquet",
            test_path="reduced_features_test.parquet",
            n_train=n_rows,
            n_test=nt_rows,
            n_features=n_cols - 1,
            description="Smaller feature set — rare comorbidities and redundant columns removed",
            pipeline_method="Manual imputation | 30 features | reduced comorbidities",
            notes=(
                "Source: notebooks/Experiment/Knee/prepared_datasets/pipeline4_reduced_features_*.parquet\n"
                "Dropped: Cancer, Circulation, Depression (separate), Heart Disease, Kidney Disease, "
                "Liver Disease, Lung Disease, Nervous System, Revision Flag, "
                "Pre-Op Q Living Arrangements_3/4/null. "
                "Useful for checking whether rare features genuinely improve the model."
            ),
        ))

    if not dataset_variants:
        print("  [ERROR] No variant files found.")
        return

    # Use manual imputation as default active variant
    active_ds = _dslug("Manual Imputation") if dataset_variants else None

    _register_project(
        name="NHS Knee Replacement",
        slug=slug,
        description=(
            "NHS PROMs knee replacement outcomes (England, 2016-19). "
            "Predicts health_gain (post-op OKS minus pre-op OKS). "
            f"{len(dataset_variants)} dataset variants pre-loaded."
        ),
    )

    _write_session_state(
        slug=slug,
        file_name="1.1-Reduced.parquet",
        target_column="health_gain",
        task_type="regression",
        n_rows=raw_df.shape[0],
        n_cols=raw_df.shape[1],
        derived_formula="Knee Replacement Post-Op Q Score - Knee Replacement Pre-Op Q Score",
        derived_column="health_gain",
        sentinel_values=["*", " ", "9", "999"],
        datasets=dataset_variants,
        active_dataset=active_ds,
    )
    print(f"  Variants: {[v['name'] for v in dataset_variants]}")


# ── Project: NHS Hip Replacement ──────────────────────────────────────────────

def setup_hip() -> None:
    print("\n=== NHS Hip Replacement ===")
    slug = "nhs_hip_replacement"
    datasets_dir = OUTPUTS / slug / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    raw_src = HIP_INTERIM / "1.1-Reduced.parquet"
    if not raw_src.exists():
        print(f"  [ERROR] {raw_src} not found. Skipping.")
        return

    _copy_parquet(raw_src, datasets_dir / "1_raw.parquet")
    raw_df = pd.read_parquet(raw_src)

    dataset_variants: list[dict] = []

    v1_train_src = HIP_INTERIM / "2.1-train.parquet"
    v1_test_src = HIP_INTERIM / "2.1-test.parquet"
    if v1_train_src.exists():
        n_rows, n_cols = _copy_parquet(v1_train_src, datasets_dir / "manual_imputation_train.parquet")
        nt_rows, _ = _copy_parquet(v1_test_src, datasets_dir / "manual_imputation_test.parquet")
        dataset_variants.append(_make_dataset_entry(
            name="Manual Imputation",
            train_path="manual_imputation_train.parquet",
            test_path="manual_imputation_test.parquet",
            n_train=n_rows,
            n_test=nt_rows,
            n_features=n_cols - 1,
            description="Hand-crafted imputation from notebook 2.1",
            pipeline_method="Manual imputation | 50 features",
            notes=(
                "Source: notebooks/Experiment/Hip/data/interim/2.1-train/test.parquet\n"
                "Strategy: manual, domain-guided imputation. Includes comorbidity_burden, "
                "depression interactions, disability_missing flag."
            ),
        ))

    if not dataset_variants:
        print("  [ERROR] No variant files found.")
        return

    _register_project(
        name="NHS Hip Replacement",
        slug=slug,
        description=(
            "NHS PROMs hip replacement outcomes (England, 2016-19). "
            "Predicts health_gain (post-op OHS minus pre-op OHS). "
            f"{len(dataset_variants)} dataset variant(s) pre-loaded."
        ),
    )

    _write_session_state(
        slug=slug,
        file_name="1.1-Reduced.parquet",
        target_column="health_gain",
        task_type="regression",
        n_rows=raw_df.shape[0],
        n_cols=raw_df.shape[1],
        derived_formula="Hip Replacement Post-Op Q Score - Hip Replacement Pre-Op Q Score",
        derived_column="health_gain",
        sentinel_values=["*", " ", "9", "999"],
        datasets=dataset_variants,
        active_dataset=_dslug("Manual Imputation"),
    )
    print(f"  Variants: {[v['name'] for v in dataset_variants]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MLPortal — bootstrapping demo projects")
    print(f"  MLPortal root: {MLPORTAL}")
    print(f"  Outputs dir:   {OUTPUTS}")

    # Wipe existing session states so they get rewritten with dataset variant info
    for slug in ["nhs_knee_replacement", "nhs_hip_replacement"]:
        sp = OUTPUTS / slug / "session" / "session_state.json"
        if sp.exists():
            sp.unlink()
            print(f"  [reset] {sp} removed (will be rewritten)")
    # Wipe projects.json so slugs get re-registered cleanly
    if PROJECTS_INDEX.exists():
        PROJECTS_INDEX.unlink()
        print(f"  [reset] projects.json removed (will be rewritten)")

    setup_knee()
    setup_hip()

    projects = _load_index()
    print(f"\nDone. {len(projects)} project(s) registered.")
    for p in projects:
        state_path = OUTPUTS / p["slug"] / "session" / "session_state.json"
        state = json.loads(state_path.read_text())
        nv = len(state.get("datasets", []))
        print(f"  - {p['name']} ({p['slug']}) -- {nv} dataset variant(s)")
    print("\nStart the app with:  streamlit run app.py")
