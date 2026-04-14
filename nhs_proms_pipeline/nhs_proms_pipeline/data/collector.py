"""Data collection — Step 1.1.

Loads the raw NHS PROMs CSV, reduces its memory footprint, and writes
the result to an interim Parquet file.

Corresponds to notebook: 1.1-Collect-Data.ipynb
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nhs_proms_pipeline.config import PipelineSettings
from nhs_proms_pipeline.utils.io import write_parquet
from nhs_proms_pipeline.utils.logging import get_logger
from nhs_proms_pipeline.utils.memory import optimise_pandas_memory

logger = get_logger(__name__)

# Parquet file produced by this step
_OUTPUT_FILENAME = "1.1-Reduced.parquet"


def collect_data(settings: PipelineSettings) -> Path:
    """Load the raw CSV, optimise memory, and persist as Parquet.

    This is the first step in the pipeline.  The raw CSV is read once,
    memory is optimised with dtype downcasting, and the result is stored
    as a compressed Parquet file for all downstream steps.

    Args:
        settings: Resolved pipeline settings (paths, procedure type, etc.).

    Returns:
        Path to the written ``.parquet`` file.

    Raises:
        FileNotFoundError: If the raw CSV does not exist at the configured path.
    """
    csv_path = settings.raw_csv_path()
    logger.info("Loading raw CSV: %s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {csv_path}. "
            "Verify that RAW_DATA_DIR is set correctly in your .env file."
        )

    df: pd.DataFrame = pd.read_csv(str(csv_path), sep=",", low_memory=False)
    logger.info("Loaded raw data — shape: %s", df.shape)

    df = optimise_pandas_memory(df)

    output_path = settings.interim_path(_OUTPUT_FILENAME)
    write_parquet(
        # polars read from pandas
        _pandas_to_polars_and_write(df, output_path),
        output_path,
    )
    return output_path


def _pandas_to_polars_and_write(df: pd.DataFrame, path: Path) -> "pl.DataFrame":  # noqa: F821
    """Convert the optimised pandas DataFrame to polars and return it.

    The conversion is necessary because Polars is used in all downstream
    steps.  Writing is handled by the caller so this stays a pure
    transformation function.
    """
    import polars as pl  # local import to avoid top-level cost when not needed

    # Retain only columns that didn't become all-null after optimisation
    df = df.dropna(axis=1, how="all")

    # Convert category columns with boolean-like values back to int before polars
    for col in df.select_dtypes(include="category").columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("float32")
        except (ValueError, TypeError):
            pass

    pl_df = pl.from_pandas(df)
    logger.debug("Converted to Polars DataFrame — shape: %s", pl_df.shape)
    return pl_df
