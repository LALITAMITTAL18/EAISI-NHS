"""Tests for Pydantic patient schema validation."""

import pytest
from pydantic import ValidationError

from nhs_proms_pipeline.schemas.patient import (
    ComorbidityProfile,
    PatientRecord,
    PreOpEQ5D,
)


def _make_patient(**overrides) -> PatientRecord:
    defaults = dict(
        age_band="65 to 69",
        gender=2.0,
        pre_op_q_1=3, pre_op_q_2=3, pre_op_q_3=3, pre_op_q_4=3,
        pre_op_q_5=3, pre_op_q_6=3, pre_op_q_7=3, pre_op_q_8=3,
        pre_op_q_9=3, pre_op_q_10=3, pre_op_q_11=3, pre_op_q_12=3,
        eq5d=PreOpEQ5D(mobility=2, self_care=1, activity=2, discomfort=3, anxiety=1),
        symptom_period=2,
    )
    defaults.update(overrides)
    return PatientRecord(**defaults)


class TestPatientRecord:
    def test_valid_patient(self) -> None:
        patient = _make_patient()
        assert patient.age_band == "65 to 69"
        assert patient.gender == 2.0

    def test_pre_op_score_property(self) -> None:
        patient = _make_patient(
            pre_op_q_1=1, pre_op_q_2=1, pre_op_q_3=1, pre_op_q_4=1,
            pre_op_q_5=1, pre_op_q_6=1, pre_op_q_7=1, pre_op_q_8=1,
            pre_op_q_9=1, pre_op_q_10=1, pre_op_q_11=1, pre_op_q_12=1,
        )
        assert patient.pre_op_score == 12

    def test_invalid_age_band(self) -> None:
        with pytest.raises(ValidationError):
            _make_patient(age_band="50 to 55")  # wrong format

    def test_invalid_gender(self) -> None:
        with pytest.raises(ValidationError):
            _make_patient(gender=3.0)  # must be 1 or 2

    def test_invalid_dimension_value(self) -> None:
        with pytest.raises(ValidationError):
            _make_patient(pre_op_q_1=6)  # must be 1–5

    def test_invalid_eq5d_dimension(self) -> None:
        with pytest.raises(ValidationError):
            PreOpEQ5D(mobility=4, self_care=1, activity=2, discomfort=1, anxiety=1)

    def test_eq5d_profile_property(self) -> None:
        eq5d = PreOpEQ5D(mobility=2, self_care=1, activity=2, discomfort=3, anxiety=1)
        assert eq5d.profile == "21231"

    def test_comorbidity_defaults(self) -> None:
        patient = _make_patient()
        assert patient.comorbidities.diabetes == 0
        assert patient.comorbidities.heart_disease == 0

    def test_invalid_comorbidity_value(self) -> None:
        with pytest.raises(ValidationError):
            ComorbidityProfile(diabetes=2)  # must be 0 or 1

    def test_invalid_symptom_period(self) -> None:
        with pytest.raises(ValidationError):
            _make_patient(symptom_period=5)  # must be 1–4
