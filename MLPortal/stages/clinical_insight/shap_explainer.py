"""Local SHAP computation for the RF pipeline using the joblib artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stages.clinical_insight.models import ShapResult


def compute_shap_local(
    pipeline,
    X: pd.DataFrame,
    model_name: str = "RandomForestRegressor",
    top_n: int = 15,
    max_samples: int = 300,
) -> ShapResult:
    """Compute SHAP values using the local sklearn pipeline.

    Works with Tree models (TreeExplainer), linear models (LinearExplainer),
    and anything else (KernelExplainer — slow).
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap: pip install shap") from exc

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    sample = X.sample(min(max_samples, len(X)), random_state=42)
    X_transformed = preprocessor.transform(sample)
    feature_names = _extract_feature_names(preprocessor, X.columns.tolist())

    explainer, shap_vals = _fit_explainer(model, X_transformed)
    _ev = np.asarray(explainer.expected_value).ravel() if hasattr(explainer, "expected_value") else np.array([0.0])
    base_value = float(_ev[0]) if _ev.size > 0 else 0.0

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

    mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]

    return ShapResult(
        model_name=model_name,
        feature_names=[feature_names[i] for i in sorted_idx],
        mean_abs_shap=[round(mean_abs[i], 6) for i in sorted_idx],
        shap_values=shap_vals[:, sorted_idx].tolist(),
        sample_features=sample.iloc[:, list(sorted_idx)].to_dict("records"),
        base_value=base_value,
    )


def compute_shap_single_row(
    pipeline,
    row: dict,
    feature_names_in: list[str] | None = None,
) -> tuple[list[str], list[float], float]:
    """Return (feature_names, shap_values, base_value) for one data point.

    Used to build the waterfall chart for the patient's specific prediction.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap: pip install shap") from exc

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    df = pd.DataFrame([row])
    X_transformed = preprocessor.transform(df)
    feature_names = _extract_feature_names(
        preprocessor,
        feature_names_in or list(df.columns),
    )

    explainer, shap_vals = _fit_explainer(model, X_transformed)
    _ev = np.asarray(explainer.expected_value).ravel() if hasattr(explainer, "expected_value") else np.array([0.0])
    base_value = float(_ev[0]) if _ev.size > 0 else 0.0

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]

    row_shap = shap_vals[0].tolist()
    return feature_names, row_shap, base_value


def _fit_explainer(model, X_transformed):
    """Try TreeExplainer → LinearExplainer → KernelExplainer."""
    import shap

    def _to_numpy(sv):
        """Normalise shap_values output — handles Explanation objects (shap >= 0.45)."""
        if hasattr(sv, "values"):
            return sv.values
        return sv

    try:
        explainer = shap.TreeExplainer(model)
        # check_additivity=False avoids the "0-dimensional array" bug with sklearn >= 1.6
        sv = explainer.shap_values(X_transformed, check_additivity=False)
        return explainer, _to_numpy(sv)
    except Exception:
        pass
    try:
        explainer = shap.LinearExplainer(model, X_transformed)
        sv = explainer.shap_values(X_transformed)
        return explainer, _to_numpy(sv)
    except Exception:
        pass
    background = shap.sample(X_transformed, min(50, len(X_transformed)))
    explainer = shap.KernelExplainer(model.predict, background)
    sv = explainer.shap_values(X_transformed, nsamples=100)
    return explainer, _to_numpy(sv)


def _extract_feature_names(preprocessor, original_cols: list[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return original_cols
