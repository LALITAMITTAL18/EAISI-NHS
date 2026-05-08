"""Pure functions for model explanation using SHAP and permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.explanation.models import (
    ExplanationConfig,
    FeatureImportanceRow,
    PdpResult,
    ShapResult,
)


def compute_shap(
    pipeline,
    X: pd.DataFrame,
    config: ExplanationConfig,
) -> ShapResult:
    """Compute SHAP values for the model inside *pipeline*.

    Uses TreeExplainer for tree models, LinearExplainer for linear models,
    and falls back to KernelExplainer (slow, sampling applied).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap: pip install shap") from exc

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    sample = X.sample(
        min(config.max_shap_samples, len(X)), random_state=42
    )
    X_transformed = preprocessor.transform(sample)
    feature_names = _get_feature_names(preprocessor, X.columns.tolist())

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_transformed)
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, X_transformed)
            shap_vals = explainer.shap_values(X_transformed)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_transformed, 50))
            shap_vals = explainer.shap_values(X_transformed, nsamples=100)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

    mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
    sorted_idx = np.argsort(mean_abs)[::-1][: config.top_n_features]

    return ShapResult(
        model_name=config.model_name,
        feature_names=[feature_names[i] for i in sorted_idx],
        mean_abs_shap=[round(mean_abs[i], 6) for i in sorted_idx],
        shap_values=shap_vals[:, sorted_idx].tolist(),
        sample_features=sample.iloc[:, list(sorted_idx)].to_dict("records")
        if hasattr(sample, "iloc")
        else [],
    )


def compute_permutation_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> list[FeatureImportanceRow]:
    """Compute permutation feature importance (model-agnostic)."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    rows: list[FeatureImportanceRow] = []
    for i in np.argsort(result.importances_mean)[::-1]:
        rows.append(
            FeatureImportanceRow(
                feature=X.columns[i],
                importance=round(float(result.importances_mean[i]), 6),
                method="permutation",
            )
        )
    return rows


def compute_native_importance(
    pipeline,
    X: pd.DataFrame,
) -> list[FeatureImportanceRow]:
    """Extract native feature importances from the model (tree-based models)."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = _get_feature_names(preprocessor, X.columns.tolist())

    importances: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    if importances is None:
        return []

    rows: list[FeatureImportanceRow] = []
    for i in np.argsort(importances)[::-1]:
        if i < len(feature_names):
            rows.append(
                FeatureImportanceRow(
                    feature=feature_names[i],
                    importance=round(float(importances[i]), 6),
                    method="native",
                )
            )
    return rows


def compute_pdp(
    pipeline,
    X: pd.DataFrame,
    feature: str,
    n_grid: int = 50,
    n_samples: int = 200,
) -> PdpResult:
    """Compute partial dependence for a single feature."""
    sample = X.sample(min(n_samples, len(X)), random_state=42)
    col = sample[feature].dropna()
    grid = np.linspace(col.min(), col.max(), n_grid)
    avg_preds: list[float] = []
    for val in grid:
        tmp = sample.copy()
        tmp[feature] = val
        avg_preds.append(float(pipeline.predict(tmp).mean()))
    return PdpResult(
        feature=feature,
        grid_values=grid.tolist(),
        avg_predictions=avg_preds,
    )


def predict_single(pipeline, row: dict) -> float | list:
    """Predict for a single data point provided as a dict."""
    df = pd.DataFrame([row])
    return pipeline.predict(df)[0]


def _get_feature_names(preprocessor, original_cols: list[str]) -> list[str]:
    """Try to extract feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return original_cols
