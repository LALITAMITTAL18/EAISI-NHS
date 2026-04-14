"""File I/O utilities.

Thin wrappers around polars/pandas parquet I/O and joblib serialisation
so that all raw file-system access is centralised here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import polars as pl

from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


# ── Parquet helpers ───────────────────────────────────────────────────────────


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a gzip-compressed Parquet file into a Polars DataFrame.

    Args:
        path: Absolute or relative path to the ``.parquet`` file.

    Returns:
        Loaded Polars DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pl.read_parquet(str(path))
    logger.debug("Read parquet: %s  shape=%s", path, df.shape)
    return df


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a Polars DataFrame to a gzip-compressed Parquet file.

    Creates parent directories if they do not exist.

    Args:
        df:   DataFrame to write.
        path: Destination file path (should end with ``.parquet``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path), compression="gzip")
    logger.info("Written parquet: %s  shape=%s", path, df.shape)


# ── Joblib helpers ────────────────────────────────────────────────────────────


def load_joblib(path: Path) -> Any:
    """Deserialise a joblib-compressed object.

    Args:
        path: Path to a ``.joblib`` file.

    Returns:
        Deserialised Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Joblib file not found: {path}")
    obj = joblib.load(str(path))
    logger.debug("Loaded joblib: %s", path)
    return obj


def dump_joblib(obj: Any, path: Path) -> None:
    """Serialise *obj* to a joblib-compressed file.

    Creates parent directories if they do not exist.

    Args:
        obj:  Python object to serialise.
        path: Destination file path (should end with ``.joblib``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, str(path))
    logger.info("Saved joblib: %s", path)
