"""End-to-end pipeline orchestrator.

:class:`TrainingPipeline` runs every step from raw CSV to a persisted
best model.  :class:`InferencePipeline` wraps the :class:`Predictor`
lifecycle for easy reuse in web services or scripts.

Design:
- Each pipeline step is independently importable (separation of concerns).
- The orchestrator only calls steps and handles errors/logging.
- Steps write intermediate artefacts so any step can be re-run in isolation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from nhs_proms_pipeline.config import PipelineSettings
from nhs_proms_pipeline.data.collector import collect_data
from nhs_proms_pipeline.data.preparator import SplitDataset, prepare_data
from nhs_proms_pipeline.data.preprocessor import preprocess_data
from nhs_proms_pipeline.inference.predictor import Predictor
from nhs_proms_pipeline.modelling.evaluator import (
    bland_altman_stats,
    build_comparison_table,
    calibration_by_decile,
    mcid_classification_metrics,
    plot_bland_altman,
    plot_calibration,
)
from nhs_proms_pipeline.modelling.registry import get_models
from nhs_proms_pipeline.modelling.trainer import TrainingOutput, build_regression_pipeline
from nhs_proms_pipeline.schemas.patient import PatientRecord
from nhs_proms_pipeline.schemas.results import (
    BestModelInfo,
    PredictionResult,
)
from nhs_proms_pipeline.utils.io import dump_joblib, load_joblib, read_parquet
from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_RESULTS_CACHE_FILENAME = "all_results_cache.joblib"
_BEST_META_FILENAME = "best_model_meta.joblib"
_BEST_PIPELINE_FILENAME = "best_pipeline.joblib"


class TrainingPipeline:
    """Orchestrates all training steps from raw CSV to persisted best model.

    Args:
        settings: Resolved :class:`~nhs_proms_pipeline.config.PipelineSettings`.

    Example::

        from nhs_proms_pipeline.config import PipelineSettings
        from nhs_proms_pipeline.pipeline import TrainingPipeline

        settings = PipelineSettings()
        pipeline = TrainingPipeline(settings)
        best = pipeline.run()
        print(best.model_name, best.test_rmse)
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        self._proc_cfg = settings.get_procedure_config()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        skip_collection: bool = False,
        skip_preprocessing: bool = False,
        skip_preparation: bool = False,
    ) -> BestModelInfo:
        """Run the full training pipeline and return the best model info.

        Args:
            skip_collection:    Skip Step 1 (assumes parquet already exists).
            skip_preprocessing: Skip Step 2.0 (assumes parquet already exists).
            skip_preparation:   Skip Step 2.1 (assumes train/test parquet exist).

        Returns:
            :class:`~nhs_proms_pipeline.schemas.results.BestModelInfo`.
        """
        start = time.time()
        logger.info(
            "=== Training pipeline START (procedure=%s) ===",
            self._settings.procedure_type.value,
        )

        if not skip_collection:
            self._run_collection()

        if not skip_preprocessing:
            self._run_preprocessing()

        if not skip_preparation:
            self._run_preparation()

        split = self._load_split()
        all_results = self._run_model_training(split)
        best_info = self._select_and_persist_best(all_results, split)

        elapsed = time.time() - start
        logger.info(
            "=== Training pipeline COMPLETE in %.1f s — best: %s × %s (RMSE=%.4f) ===",
            elapsed,
            best_info.dataset_label,
            best_info.model_name,
            best_info.test_rmse,
        )
        return best_info

    # ── Private step runners ──────────────────────────────────────────────────

    def _run_collection(self) -> None:
        logger.info("Step 1/4 — Data collection")
        collect_data(self._settings)

    def _run_preprocessing(self) -> None:
        logger.info("Step 2/4 — Data pre-processing")
        preprocess_data(self._settings)

    def _run_preparation(self) -> None:
        logger.info("Step 3/4 — Data preparation (split + imputation + outcome)")
        prepare_data(self._settings)

    def _load_split(self) -> SplitDataset:
        """Load the prepared train/test parquet files from disk."""
        train_path = self._settings.interim_path("2.1-train.parquet")
        test_path = self._settings.interim_path("2.1-test.parquet")
        return SplitDataset(
            train=read_parquet(train_path),
            test=read_parquet(test_path),
            train_path=train_path,
            test_path=test_path,
            imputation_values={},
        )

    def _run_model_training(
        self, split: SplitDataset
    ) -> dict[str, TrainingOutput]:
        """Train all enabled models and return their :class:`TrainingOutput` objects.

        A results cache is written after each model so a failed run can be
        resumed (already-cached models are skipped).
        """
        logger.info("Step 4/4 — Model training (%d models)", len(self._settings.enabled_models))

        cache_path = self._settings.model_path(_RESULTS_CACHE_FILENAME)
        # Store serialisable metric dicts for the cache
        cached_metrics: dict[str, Any] = {}
        if cache_path.exists():
            cached_metrics = load_joblib(cache_path)
            logger.info("Found cache with %d entries at %s", len(cached_metrics), cache_path)

        models = get_models(list(self._settings.enabled_models))
        drop_cols = ["OHS_Success"] if "OHS_Success" in split.train.columns else []
        all_outputs: dict[str, TrainingOutput] = {}

        for model_name, estimator in models.items():
            logger.info("  Training: %s", model_name)
            output: TrainingOutput = build_regression_pipeline(
                train_df=split.train,
                test_df=split.test,
                target="health_gain",
                model=estimator,
                model_name=model_name,
                dataset_label="2.1-Manual",
                drop_cols=drop_cols,
                n_splits=self._settings.cv_splits,
                n_repeats=self._settings.cv_repeats,
                random_state=self._settings.random_seed,
            )
            all_outputs[model_name] = output
            cached_metrics[model_name] = {
                "cv_metrics": output.result.cv_metrics,
                "test_metrics": output.result.test_metrics,
                "feature_names": output.result.feature_names,
            }
            dump_joblib(cached_metrics, cache_path)

        return all_outputs

    def _select_and_persist_best(
        self,
        all_outputs: dict[str, TrainingOutput],
        split: SplitDataset,
    ) -> BestModelInfo:
        """Select the model with the lowest test RMSE and save its artefacts."""
        comparison_input = {
            "2.1-Manual": {
                name: {
                    "cv_metrics": out.result.cv_metrics,
                    "test_metrics": out.result.test_metrics,
                }
                for name, out in all_outputs.items()
            }
        }
        comparison = build_comparison_table(comparison_input)
        best_row = comparison.iloc[0]
        best_model_name = str(best_row["Model"])
        best_output = all_outputs[best_model_name]

        mcid = self._proc_cfg.mcid_threshold
        y_test = best_output.y_test
        y_pred = best_output.y_pred

        mcid_m = mcid_classification_metrics(y_test, y_pred, mcid)
        bland_altman_stats(y_test, y_pred, mcid)

        plot_bland_altman(
            y_test, y_pred, mcid,
            title=f"Bland-Altman — 2.1-Manual × {best_model_name}",
            save_path=self._settings.reports_dir / "bland_altman_best.png",
        )
        calibration = calibration_by_decile(y_test, y_pred)
        plot_calibration(
            calibration,
            title=f"Calibration by Decile — 2.1-Manual × {best_model_name}",
            save_path=self._settings.reports_dir / "calibration_best.png",
        )

        pipeline_path = self._settings.model_path(_BEST_PIPELINE_FILENAME)
        dump_joblib(best_output.fitted_pipeline, pipeline_path)

        best_info = BestModelInfo(
            dataset_label="2.1-Manual",
            model_name=best_model_name,
            test_rmse=float(best_row["Test RMSE"]),
            test_r2=float(best_row["Test R²"]),
            mcid_f2=mcid_m.f2,
            pipeline_path=pipeline_path,
        )
        dump_joblib(best_info, self._settings.model_path(_BEST_META_FILENAME))

        logger.info(
            "Best model saved: %s (RMSE=%.4f, R²=%.4f, F2=%.4f)",
            best_model_name,
            best_info.test_rmse,
            best_info.test_r2,
            best_info.mcid_f2 or 0,
        )
        return best_info


class InferencePipeline:
    """Wraps the :class:`Predictor` for easy use in scripts or web services.

    Args:
        settings: Resolved pipeline settings.

    Example::

        from nhs_proms_pipeline.config import PipelineSettings
        from nhs_proms_pipeline.pipeline import InferencePipeline
        from nhs_proms_pipeline.schemas.patient import PatientRecord, PreOpEQ5D

        settings = PipelineSettings()
        pipe = InferencePipeline(settings)
        record = PatientRecord(
            age_band="65 to 69",
            gender=2.0,
            pre_op_q_1=3, pre_op_q_2=2, pre_op_q_3=3, pre_op_q_4=2,
            pre_op_q_5=3, pre_op_q_6=2, pre_op_q_7=3, pre_op_q_8=2,
            pre_op_q_9=3, pre_op_q_10=2, pre_op_q_11=3, pre_op_q_12=2,
            eq5d=PreOpEQ5D(mobility=2, self_care=1, activity=2, discomfort=3, anxiety=1),
            symptom_period=2,
        )
        result = pipe.predict(record)
        print(result.predicted_health_gain, result.predicted_benefit)
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._predictor = Predictor(settings)

    def predict(self, patient: PatientRecord) -> PredictionResult:
        """Predict health gain for a single patient.

        Args:
            patient: Validated patient record.

        Returns:
            :class:`~nhs_proms_pipeline.schemas.results.PredictionResult`.
        """
        return self._predictor.predict(patient)
