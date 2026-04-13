"""Modelling package."""

from nhs_proms_pipeline.modelling.evaluator import (
    bland_altman_stats,
    build_comparison_table,
    calibration_by_decile,
    find_optimal_threshold,
    mcid_classification_metrics,
    plot_bland_altman,
    plot_calibration,
)
from nhs_proms_pipeline.modelling.registry import available_models, get_model, get_models
from nhs_proms_pipeline.modelling.trainer import build_regression_pipeline

__all__ = [
    "available_models",
    "bland_altman_stats",
    "build_comparison_table",
    "build_regression_pipeline",
    "calibration_by_decile",
    "find_optimal_threshold",
    "get_model",
    "get_models",
    "mcid_classification_metrics",
    "plot_bland_altman",
    "plot_calibration",
]
