"""Pure read/write helpers for parquet, joblib and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# ── Parquet ───────────────────────────────────────────────────────────────────


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write *df* to a gzip-compressed parquet file; create parent dirs if needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, compression="gzip", index=False)
    return dest


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file and return a DataFrame."""
    return pd.read_parquet(Path(path))


# ── Joblib ────────────────────────────────────────────────────────────────────


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist any object with joblib; creates parent dirs if needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, dest)
    return dest


def load_joblib(path: str | Path) -> Any:
    """Load an object previously saved with joblib."""
    return joblib.load(Path(path))


# ── JSON ──────────────────────────────────────────────────────────────────────


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write *data* as pretty-printed JSON; creates parent dirs if needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
    return dest


def load_json(path: str | Path) -> Any:
    """Load JSON from *path*."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_exists(path: str | None) -> bool:
    """Return True if *path* is not None and the file exists on disk."""
    return path is not None and Path(path).exists()
