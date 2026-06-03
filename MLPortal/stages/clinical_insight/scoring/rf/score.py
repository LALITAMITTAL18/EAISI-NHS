"""Azure ML scoring script for the NHS Knee RF pipeline.

The model is a full sklearn Pipeline: preprocessor + RandomForestRegressor.
Input JSON schema:
  { "features": {"age_at_operation": 70, "pre_op_oks": 18, ...},
    "compute_shap": true }

Output JSON schema:
  { "prediction": 12.4,
    "shap_values": {"age_at_operation": 0.82, ...},   # null if not requested
    "feature_names": ["age_at_operation", ...] }
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
pipeline = None


def init():
    global pipeline
    model_dir = os.environ.get("AZUREML_MODEL_DIR", ".")
    candidates = [
        os.path.join(model_dir, "model.joblib"),
        os.path.join(model_dir, "pipeline.joblib"),
    ]
    # Walk the dir in case the file is in a subdirectory
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".joblib"):
                candidates.insert(0, os.path.join(root, f))
                break

    for path in candidates:
        if os.path.exists(path):
            pipeline = joblib.load(path)
            logger.info("Pipeline loaded from %s", path)
            return

    raise FileNotFoundError(f"No .joblib file found under {model_dir}")


def run(raw_data: str) -> str:
    data = json.loads(raw_data)
    features: dict = data.get("features", {})
    compute_shap: bool = data.get("compute_shap", False)

    df = pd.DataFrame([features])

    prediction = float(pipeline.predict(df)[0])

    shap_values = None
    feature_names = list(df.columns)

    if compute_shap:
        try:
            import shap

            preprocessor = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]
            X_transformed = preprocessor.transform(df)

            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                pass

            try:
                explainer = shap.TreeExplainer(model)
                vals = explainer.shap_values(X_transformed)
            except Exception:
                explainer = shap.KernelExplainer(
                    model.predict, shap.sample(X_transformed, min(50, len(X_transformed)))
                )
                vals = explainer.shap_values(X_transformed, nsamples=100)

            if isinstance(vals, list):
                vals = vals[0]
            shap_values = dict(zip(feature_names, vals[0].tolist()))
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)

    return json.dumps(
        {
            "prediction": prediction,
            "shap_values": shap_values,
            "feature_names": feature_names,
        }
    )
