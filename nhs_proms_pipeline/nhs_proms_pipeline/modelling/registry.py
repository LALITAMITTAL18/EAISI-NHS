"""Model registry — canonical list of all candidate estimators.

Adding a new model is a single-line change in ``MODELS``.
All models are clones (via ``sklearn.base.clone``) before training so
that repeated calls to the trainer do not share state.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

# ── Optional dependency: XGBoost ───────────────────────────────────────────────
try:
    from xgboost import XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed — 'XGBoost' model will be unavailable.")


# ── Model registry ─────────────────────────────────────────────────────────────
# Hyperparameters are chosen as sensible starting points.
# Tune further via grid/random/Bayesian search as needed.

_BASE_MODELS: dict[str, BaseEstimator] = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=42),
    "Lasso": Lasso(alpha=0.01, random_state=42, max_iter=10_000),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=1
    ),
    # HistGradientBoosting is ~10× faster than GradientBoostingRegressor
    "GradientBoosting": HistGradientBoostingRegressor(max_iter=300, random_state=42),
}

if _XGBOOST_AVAILABLE:
    _BASE_MODELS["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def get_model(name: str) -> BaseEstimator:
    """Return a fresh clone of the named model.

    Using ``clone`` ensures each call returns an unfitted estimator with
    the same hyperparameters, preventing state sharing between runs.

    Args:
        name: Key from the model registry (e.g. ``'RandomForest'``).

    Returns:
        A cloned, unfitted sklearn estimator.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _BASE_MODELS:
        available = list(_BASE_MODELS.keys())
        raise KeyError(
            f"Model '{name}' not found. Available: {available}"
        )
    return clone(_BASE_MODELS[name])


def available_models() -> list[str]:
    """Return the list of registered model names."""
    return list(_BASE_MODELS.keys())


def get_models(names: list[str] | None = None) -> dict[str, BaseEstimator]:
    """Return a dict of ``{name: cloned_estimator}`` for the requested names.

    Args:
        names: Subset of model names to return. ``None`` returns all models.

    Returns:
        Dict mapping model names to fresh cloned estimators.
    """
    target_names: list[str] = names if names is not None else available_models()
    unknown = [n for n in target_names if n not in _BASE_MODELS]
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}. Available: {available_models()}")
    return {name: get_model(name) for name in target_names}
