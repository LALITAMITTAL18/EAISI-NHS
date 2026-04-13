"""Pydantic schemas for patient input data.

These models validate and document the fields a doctor must supply
when requesting a prediction for a patient.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ComorbidityProfile(BaseModel):
    """Binary comorbidity indicators (0 = No, 1 = Yes)."""

    arthritis: int = Field(default=0, ge=0, le=1)
    cancer: int = Field(default=0, ge=0, le=1)
    circulation: int = Field(default=0, ge=0, le=1)
    depression: int = Field(default=0, ge=0, le=1)
    diabetes: int = Field(default=0, ge=0, le=1)
    heart_disease: int = Field(default=0, ge=0, le=1)
    high_bp: int = Field(default=0, ge=0, le=1)
    kidney_disease: int = Field(default=0, ge=0, le=1)
    liver_disease: int = Field(default=0, ge=0, le=1)
    lung_disease: int = Field(default=0, ge=0, le=1)
    nervous_system: int = Field(default=0, ge=0, le=1)
    stroke: int = Field(default=0, ge=0, le=1)


class PreOpEQ5D(BaseModel):
    """Pre-operative EQ-5D-3L dimension scores (1=No problems, 2=Moderate, 3=Extreme)."""

    mobility: int = Field(..., ge=1, le=3, description="Mobility score 1-3")
    self_care: int = Field(..., ge=1, le=3, description="Self-care score 1-3")
    activity: int = Field(..., ge=1, le=3, description="Usual activities score 1-3")
    discomfort: int = Field(..., ge=1, le=3, description="Pain/discomfort score 1-3")
    anxiety: int = Field(..., ge=1, le=3, description="Anxiety/depression score 1-3")

    @property
    def profile(self) -> str:
        """Return the 5-digit EQ-5D profile string e.g. '11121'."""
        return f"{self.mobility}{self.self_care}{self.activity}{self.discomfort}{self.anxiety}"


class PatientRecord(BaseModel):
    """Full pre-operative patient record used for health-gain prediction.

    All fields that the model was trained on must be present.
    Optional fields are imputed to their training-set defaults if absent.
    """

    # ── Demographics ──────────────────────────────────────────────────────────
    age_band: str = Field(
        ...,
        description=(
            "Patient age band, e.g. '65 to 69'. "
            "Valid values follow the NHS standard age band coding."
        ),
    )
    gender: float = Field(
        ...,
        ge=1,
        le=2,
        description="Patient gender (1 = Male, 2 = Female).",
    )

    # ── Oxford Knee/Hip Score — 12 pre-op dimensions ─────────────────────────
    # Scoring: 1 = no difficulty / best, 5 = unable to do / worst (original NHS coding).
    # Total pre-op score = sum of all 12 questions (range 12–60; lower = better function).
    pre_op_q_1: int = Field(
        ..., ge=1, le=5,
        description="Pain level (1=None, 2=Very mild, 3=Mild, 4=Moderate, 5=Severe).",
    )
    pre_op_q_2: int = Field(
        ..., ge=1, le=5,
        description=(
            "Trouble washing and drying yourself (all over) "
            "(1=No trouble … 5=Impossible to do)."
        ),
    )
    pre_op_q_3: int = Field(
        ..., ge=1, le=5,
        description=(
            "Trouble getting in/out of a car or using public transport "
            "(1=No trouble … 5=Impossible to do)."
        ),
    )
    pre_op_q_4: int = Field(
        ..., ge=1, le=5,
        description=(
            "Walking distance before severe pain begins "
            "(1=>30 min or no pain, 2=16–30 min, 3=5–15 min, 4=House only, 5=Not at all)."
        ),
    )
    pre_op_q_5: int = Field(
        ..., ge=1, le=5,
        description=(
            "Pain standing up from a chair after a meal "
            "(1=Not at all painful … 5=Unbearable)."
        ),
    )
    pre_op_q_6: int = Field(
        ..., ge=1, le=5,
        description=(
            "Limping when walking "
            "(1=Rarely/never, 2=Sometimes, 3=Often, 4=Most of the time, 5=All of the time)."
        ),
    )
    pre_op_q_7: int = Field(
        ..., ge=1, le=5,
        description=(
            "OKS: Kneeling down and getting up again. "
            "OHS: Going up and down stairs. "
            "(1=Yes easily … 5=No/impossible)."
        ),
    )
    pre_op_q_8: int = Field(
        ..., ge=1, le=5,
        description=(
            "Pain in bed at night "
            "(1=No nights, 2=1–2 nights, 3=Some nights, 4=Most nights, 5=Every night)."
        ),
    )
    pre_op_q_9: int = Field(
        ..., ge=1, le=5,
        description=(
            "Interference of pain with usual work (including housework) "
            "(1=Not at all, 2=A little, 3=Moderately, 4=Greatly, 5=Totally)."
        ),
    )
    pre_op_q_10: int = Field(
        ..., ge=1, le=5,
        description=(
            "Feeling that the joint might suddenly give way "
            "(1=Rarely/never, 2=Sometimes, 3=Often, 4=Most of the time, 5=All of the time)."
        ),
    )
    pre_op_q_11: int = Field(
        ..., ge=1, le=5,
        description=(
            "Ability to do household shopping independently "
            "(1=Yes easily … 5=No/impossible)."
        ),
    )
    pre_op_q_12: int = Field(
        ..., ge=1, le=5,
        description=(
            "Ability to walk down a flight of stairs "
            "(1=Yes easily … 5=No/impossible)."
        ),
    )

    # ── EQ-5D dimensions ──────────────────────────────────────────────────────
    eq5d: PreOpEQ5D = Field(..., description="Pre-operative EQ-5D-3L profile.")

    # ── Clinical history ─────────────────────────────────────────────────────
    symptom_period: int = Field(
        ...,
        ge=1,
        le=4,
        description=(
            "Duration of symptoms before surgery. "
            "1=<1yr, 2=1-5yrs, 3=6-10yrs, 4=>10yrs."
        ),
    )
    previous_surgery: int = Field(
        default=2,
        ge=1,
        le=2,
        description="Previous surgery on the joint (1=Yes, 2=No).",
    )

    # ── Administrative / optional ─────────────────────────────────────────────
    assisted: Optional[int] = Field(
        default=None,
        ge=1,
        le=2,
        description="Questionnaire completed with assistance (1=Yes, 2=No). Imputed if absent.",
    )
    living_arrangements: int = Field(
        default=9,
        description="Living arrangements code (9=Unknown/Not disclosed).",
    )
    disability: int = Field(
        default=9,
        description="Disability code (9=Not Disclosed).",
    )

    # ── Comorbidities ─────────────────────────────────────────────────────────
    comorbidities: ComorbidityProfile = Field(
        default_factory=ComorbidityProfile,
        description="Comorbidity flags. Defaults to all 0 (none reported).",
    )

    @field_validator("age_band")
    @classmethod
    def _validate_age_band(cls, v: str) -> str:
        valid_bands = {
            "Under 45",
            "45 to 49",
            "50 to 54",
            "55 to 59",
            "60 to 64",
            "65 to 69",
            "70 to 74",
            "75 to 79",
            "80 to 84",
            "85 to 89",
            "90 and over",
        }
        if v not in valid_bands:
            raise ValueError(
                f"Invalid age_band '{v}'. Must be one of: {sorted(valid_bands)}"
            )
        return v

    @property
    def pre_op_score(self) -> int:
        """Oxford Knee/Hip Score = sum of the 12 pre-op dimension questions."""
        return sum(
            [
                self.pre_op_q_1,
                self.pre_op_q_2,
                self.pre_op_q_3,
                self.pre_op_q_4,
                self.pre_op_q_5,
                self.pre_op_q_6,
                self.pre_op_q_7,
                self.pre_op_q_8,
                self.pre_op_q_9,
                self.pre_op_q_10,
                self.pre_op_q_11,
                self.pre_op_q_12,
            ]
        )
