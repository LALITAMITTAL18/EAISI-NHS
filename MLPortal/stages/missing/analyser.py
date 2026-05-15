"""Pure functions for missing data analysis."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def replace_sentinels(
    df: pd.DataFrame,
    sentinel_values: Sequence[int | float | str] | None,
) -> pd.DataFrame:
    """Return a copy of *df* with sentinel values replaced by NaN.

    Only numeric columns are checked for numeric sentinels; string/object
    columns are checked for string sentinels.  This prevents accidental
    replacement in free-text columns.
    """
    if not sentinel_values:
        return df
    df = df.copy()
    num_sentinels = [v for v in sentinel_values if isinstance(v, (int, float))]
    str_sentinels = [v for v in sentinel_values if isinstance(v, str)]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and num_sentinels:
            df[col] = df[col].replace(num_sentinels, pd.NA)
        elif pd.api.types.is_object_dtype(df[col]) and str_sentinels:
            df[col] = df[col].replace(str_sentinels, pd.NA)
    return df


def missingness_summary(
    df: pd.DataFrame,
    sentinel_values: Sequence[int | float | str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with per-column missingness counts and percentages.

    If *sentinel_values* is provided, those values are treated as missing
    before computing counts (e.g. ``9`` in NHS PROMs Likert columns).
    """
    n = len(df)
    df_check = replace_sentinels(df, sentinel_values)
    counts = df_check.isna().sum()
    return pd.DataFrame(
        {
            "column": counts.index,
            "n_missing": counts.values,
            "pct_missing": (counts.values / max(n, 1) * 100).round(2),
        }
    ).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def co_missing_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a symmetric matrix of co-missingness rates between all column pairs."""
    miss = df.isna()
    cols = miss.columns.tolist()
    n = len(df)
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            both = (miss[c1] & miss[c2]).sum()
            matrix.loc[c1, c2] = round(both / max(n, 1) * 100, 2)
    return matrix


def add_missing_indicators(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.02,
) -> tuple[pd.DataFrame, list[str]]:
    """Add binary indicator columns for columns with pct missing > threshold.

    Returns (augmented_df, list_of_added_column_names).
    """
    df = df.copy()
    added: list[str] = []
    n = len(df)
    for col in columns:
        if col in df.columns and df[col].isna().sum() / max(n, 1) > threshold:
            indicator_col = f"{col}_missing"
            df[indicator_col] = df[col].isna().astype("uint8")
            added.append(indicator_col)
    return df, added


def missingness_umap_data(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute 2D UMAP embedding of the binary missingness indicator matrix.

    Returns a DataFrame with columns ['umap_1', 'umap_2'].
    Raises ImportError with instructions if umap-learn is not installed.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "Install umap-learn to enable missingness pattern visualisation: "
            "pip install umap-learn"
        ) from exc

    miss_matrix = df.isna().astype("uint8").to_numpy()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        init="random",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(miss_matrix)
    return pd.DataFrame({"umap_1": embedding[:, 0], "umap_2": embedding[:, 1]})
