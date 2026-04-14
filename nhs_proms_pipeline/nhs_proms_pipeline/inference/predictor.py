"""Inference — load best model and predict for a single patient.

A doctor provides a :class:`~nhs_proms_pipeline.schemas.patient.PatientRecord`
and receives a :class:`~nhs_proms_pipeline.schemas.results.PredictionResult`
containing:
- The predicted health gain (regression score)
- Whether the predicted gain ≥ MCID (benefit classification)
- A plain-English confidence note based on Bland-Altman limits of agreement

The predictor is stateless after loading.  Cache or reuse a single
:class:`Predictor` instance in production to avoid repeated disk reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nhs_proms_pipeline.config import (
    COMORBIDITY_INDICATOR_COLS,
    EQ5D_PRE_OP_COLS,
    PipelineSettings,
)
from nhs_proms_pipeline.features.eq5d import calculate_eq5d_index
from nhs_proms_pipeline.schemas.patient import PatientRecord
from nhs_proms_pipeline.schemas.results import BestModelInfo, PredictionResult
from nhs_proms_pipeline.utils.io import load_joblib
from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


class Predictor:
    """Loads the best trained pipeline from disk and generates patient predictions.

    Args:
        settings: Resolved pipeline settings.

    Raises:
        FileNotFoundError: If the model artefact or metadata file is missing.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._proc_cfg = settings.get_procedure_config()

        meta_path = settings.model_path("best_model_meta.joblib")
        pipeline_path = settings.model_path("best_pipeline.joblib")

        self._meta: BestModelInfo = load_joblib(meta_path)
        self._pipeline = load_joblib(pipeline_path)

        logger.info(
            "Predictor loaded: %s × %s  (RMSE=%.4f, R²=%.4f)",
            self._meta.dataset_label,
            self._meta.model_name,
            self._meta.test_rmse,
            self._meta.test_r2,
        )

    def predict(self, patient: PatientRecord) -> PredictionResult:
        """Generate a health-gain prediction for a single patient.

        Args:
            patient: Validated :class:`~nhs_proms_pipeline.schemas.patient.PatientRecord`.

        Returns:
            :class:`~nhs_proms_pipeline.schemas.results.PredictionResult` with the
            predicted health gain, benefit classification, and a confidence note.
        """
        feature_row = self._build_feature_row(patient)
        feature_df = pd.DataFrame([feature_row])
        predicted_gain: float = float(self._pipeline.predict(feature_df)[0])

        mcid = self._proc_cfg.mcid_threshold
        predicted_benefit = predicted_gain >= mcid

        confidence_note = self._build_confidence_note(predicted_gain, mcid)

        logger.info(
            "Prediction: health_gain=%.2f, benefit=%s, model=%s",
            predicted_gain,
            predicted_benefit,
            self._meta.model_name,
        )

        return PredictionResult(
            predicted_health_gain=round(predicted_gain, 2),
            predicted_benefit=predicted_benefit,
            mcid_threshold=mcid,
            confidence_note=confidence_note,
            model_name=self._meta.model_name,
            dataset_label=self._meta.dataset_label,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_feature_row(self, patient: PatientRecord) -> dict[str, Any]:
        """Map a PatientRecord to the flat feature dictionary expected by sklearn.

        The mapping must exactly mirror the columns produced by the
        data preparation pipeline so that the fitted ColumnTransformer
        receives the same feature space it was trained on.
        """
        proc_cfg = self._proc_cfg
        pre_op_dims = [
            patient.pre_op_q_1,
            patient.pre_op_q_2,
            patient.pre_op_q_3,
            patient.pre_op_q_4,
            patient.pre_op_q_5,
            patient.pre_op_q_6,
            patient.pre_op_q_7,
            patient.pre_op_q_8,
            patient.pre_op_q_9,
            patient.pre_op_q_10,
            patient.pre_op_q_11,
            patient.pre_op_q_12,
        ]

        # Derive dimension column names from prefix (e.g. "Knee Replacement Pre-Op Q 1")
        dim_col_names = [
            f"{proc_cfg.pre_op_q_prefix} {i}" for i in range(1, 13)
        ]

        eq5d_profile = patient.eq5d.profile
        eq5d_index = calculate_eq5d_index(eq5d_profile)

        row: dict[str, Any] = {
            "Age Band": patient.age_band,
            "Gender": patient.gender,
            proc_cfg.pre_op_score_col: patient.pre_op_score,
            "Pre-Op Q Symptom Period": patient.symptom_period,
            "Pre-Op Q Previous Surgery": patient.previous_surgery,
            "Pre-Op Q Assisted": patient.assisted if patient.assisted is not None else np.nan,
            "Pre-Op Q Living Arrangements": patient.living_arrangements,
            "Pre-Op Q Disability": patient.disability,
            # EQ-5D
            "Pre-Op Q EQ5D Index Profile": eq5d_profile,
            "Pre-Op Q EQ5D Index": eq5d_index,
            **dict(zip(EQ5D_PRE_OP_COLS, [
                patient.eq5d.mobility,
                patient.eq5d.self_care,
                patient.eq5d.activity,
                patient.eq5d.discomfort,
                patient.eq5d.anxiety,
            ])),
        }

        # Oxford score dimensions
        for col_name, val in zip(dim_col_names, pre_op_dims):
            row[col_name] = val

        # Comorbidities (mapped to NHS column names)
        comorbidity_map = {
            "Arthritis": patient.comorbidities.arthritis,
            "Cancer": patient.comorbidities.cancer,
            "Circulation": patient.comorbidities.circulation,
            "Depression": patient.comorbidities.depression,
            "Diabetes": patient.comorbidities.diabetes,
            "Heart Disease": patient.comorbidities.heart_disease,
            "High Bp": patient.comorbidities.high_bp,
            "Kidney Disease": patient.comorbidities.kidney_disease,
            "Liver Disease": patient.comorbidities.liver_disease,
            "Lung Disease": patient.comorbidities.lung_disease,
            "Nervous System": patient.comorbidities.nervous_system,
            "Stroke": patient.comorbidities.stroke,
        }
        row.update(comorbidity_map)

        return row

    def _build_confidence_note(self, predicted_gain: float, mcid: float) -> str:
        """Build a plain-English note about prediction certainty.

        Uses the Bland-Altman limits of agreement stored in the model
        metadata to contextualise the individual prediction.
        """
        if self._meta.mcid_f2 is not None:
            f2_note = f"Model F₂ (clinical recall) = {self._meta.mcid_f2:.3f}. "
        else:
            f2_note = ""

        boundary_note = (
            "The predicted gain is close to the MCID threshold; "
            "individual predictions carry inherent uncertainty. "
            if abs(predicted_gain - mcid) < 2
            else ""
        )

        return (
            f"{f2_note}"
            f"Predicted health gain: {predicted_gain:.1f} pts "
            f"(MCID threshold for this procedure: {mcid:.0f} pts). "
            f"{boundary_note}"
            "This prediction is generated by a regression model trained on "
            "NHS PROMs data and should be used alongside clinical judgement."
        )
