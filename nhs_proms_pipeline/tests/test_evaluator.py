"""Tests for model evaluator functions."""

import numpy as np
import pytest

from nhs_proms_pipeline.modelling.evaluator import (
    bland_altman_stats,
    calibration_by_decile,
    find_optimal_threshold,
    mcid_classification_metrics,
)


def _make_arrays(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=8, scale=5, size=n)
    y_pred = y_true + rng.normal(loc=0, scale=2, size=n)
    return y_true, y_pred


class TestBlandAltman:
    def test_zero_bias_for_perfect_predictions(self) -> None:
        y = np.linspace(-10, 20, 100)
        ba = bland_altman_stats(y, y.copy(), mcid=5.0)
        assert ba.bias == pytest.approx(0.0, abs=1e-8)
        assert ba.loa_width == pytest.approx(0.0, abs=1e-8)

    def test_positive_bias(self) -> None:
        y_true = np.zeros(100)
        y_pred = np.ones(100) * 3
        ba = bland_altman_stats(y_true, y_pred, mcid=5.0)
        assert ba.bias == pytest.approx(3.0, abs=1e-8)

    def test_within_mcid_flag(self) -> None:
        y = np.linspace(0, 10, 500)
        ba = bland_altman_stats(y, y.copy(), mcid=5.0)
        assert ba.within_mcid is True


class TestMCIDMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0.0, 10.0, 0.0, 10.0])
        y_pred = np.array([0.0, 10.0, 0.0, 10.0])
        m = mcid_classification_metrics(y_true, y_pred, mcid=5.0)
        assert m.recall == pytest.approx(1.0, abs=1e-4)
        assert m.precision == pytest.approx(1.0, abs=1e-4)
        assert m.f2 == pytest.approx(1.0, abs=1e-4)

    def test_no_positive_predictions(self) -> None:
        y_true = np.array([10.0, 10.0, 10.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        m = mcid_classification_metrics(y_true, y_pred, mcid=5.0)
        assert m.tp == 0
        assert m.fn == 3
        assert m.recall == pytest.approx(0.0, abs=1e-4)

    def test_mcid_fields(self) -> None:
        y_true, y_pred = _make_arrays()
        m = mcid_classification_metrics(y_true, y_pred, mcid=5.0)
        assert m.threshold == 5.0
        assert 0.0 <= m.auroc <= 1.0
        assert 0.0 <= m.f2 <= 1.0


class TestOptimalThreshold:
    def test_optimal_threshold_reduces_fn(self) -> None:
        y_true, y_pred = _make_arrays()
        result = find_optimal_threshold(y_true, y_pred, mcid=5.0)
        assert "optimal_threshold" in result
        assert "fn_saved" in result
        assert result["fn_saved"] >= 0

    def test_optimal_f2_ge_mcid_f2(self) -> None:
        y_true, y_pred = _make_arrays()
        result = find_optimal_threshold(y_true, y_pred, mcid=5.0)
        mcid_m = mcid_classification_metrics(y_true, y_pred, mcid=5.0)
        assert result["optimal_f2"] >= mcid_m.f2 - 1e-6  # allow tiny float error


class TestCalibration:
    def test_returns_correct_columns(self) -> None:
        y_true, y_pred = _make_arrays()
        cal = calibration_by_decile(y_true, y_pred)
        assert "mean_predicted" in cal.columns
        assert "mean_actual" in cal.columns
        assert "bias" in cal.columns

    def test_n_bins(self) -> None:
        y_true, y_pred = _make_arrays(n=300)
        cal = calibration_by_decile(y_true, y_pred, n_bins=5)
        assert len(cal) <= 5
