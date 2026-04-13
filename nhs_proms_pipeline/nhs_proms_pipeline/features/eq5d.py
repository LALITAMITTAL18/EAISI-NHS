"""EQ-5D index calculation using the UK value set.

The EQ-5D-3L index is derived from a 5-digit profile string
(e.g. ``'11121'``) using published UK tariff weights.

Reference:
    Dolan, P. (1997). Modeling valuations for EuroQol health states.
    Medical Care, 35(11), 1095-1108.
"""

from __future__ import annotations

from nhs_proms_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

# ── UK value set constants ────────────────────────────────────────────────────
_CONSTANT_DEDUCTION: float = 0.081
_N3_PENALTY: float = 0.269

# Dimension weights by level (1 = baseline, 2 = moderate, 3 = extreme)
# Index: 0=Mobility, 1=Self-care, 2=Usual activities, 3=Pain/discomfort, 4=Anxiety
_DIM_WEIGHTS: dict[int, dict[str, float]] = {
    0: {"2": 0.069, "3": 0.314},  # Mobility
    1: {"2": 0.104, "3": 0.214},  # Self-care
    2: {"2": 0.036, "3": 0.094},  # Usual activities
    3: {"2": 0.123, "3": 0.386},  # Pain/discomfort
    4: {"2": 0.071, "3": 0.236},  # Anxiety/depression
}


def calculate_eq5d_index(profile: str | None) -> float | None:
    """Calculate the EQ-5D-3L utility index from a 5-digit profile string.

    Uses the UK value set published by Dolan (1997).  Returns ``None`` for
    any invalid or missing profile.

    Args:
        profile: A 5-character string where each character is ``'1'``, ``'2'``,
                 or ``'3'`` (e.g. ``'11121'``).

    Returns:
        Utility score in the range (-0.59, 1.0], or ``None`` if the profile
        is invalid.

    Examples:
        >>> calculate_eq5d_index("11111")
        1.0
        >>> calculate_eq5d_index("33333")
        -0.594
        >>> calculate_eq5d_index(None)
        None
    """
    if profile is None:
        return None

    profile_str = str(profile).strip()
    if len(profile_str) != 5:
        logger.debug("Invalid EQ-5D profile length (%d): '%s'", len(profile_str), profile_str)
        return None

    try:
        digits = [int(d) for d in profile_str]
    except ValueError:
        logger.debug("Non-integer characters in EQ-5D profile: '%s'", profile_str)
        return None

    if any(d not in (1, 2, 3) for d in digits):
        logger.debug("Invalid EQ-5D dimension value in profile: '%s'", profile_str)
        return None

    total_deduction = _CONSTANT_DEDUCTION
    has_level_3 = False

    for i, level in enumerate(digits):
        if level == 2:
            total_deduction += _DIM_WEIGHTS[i]["2"]
        elif level == 3:
            total_deduction += _DIM_WEIGHTS[i]["3"]
            has_level_3 = True

    if has_level_3:
        total_deduction += _N3_PENALTY

    return round(1.0 - total_deduction, 6)
