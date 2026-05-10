"""Pure functions for missing data analysis."""

from __future__ import annotations

import pandas as pd


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-column missingness counts and percentages."""
    n = len(df)
    counts = df.isna().sum()
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
