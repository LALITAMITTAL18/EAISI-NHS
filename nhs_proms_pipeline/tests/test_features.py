"""Tests for the EQ-5D index calculation."""

import pytest

from nhs_proms_pipeline.features.eq5d import calculate_eq5d_index


class TestCalculateEQ5DIndex:
    def test_perfect_health(self) -> None:
        result = calculate_eq5d_index("11111")
        assert result == pytest.approx(1.0 - 0.081, abs=1e-4)

    def test_worst_health(self) -> None:
        result = calculate_eq5d_index("33333")
        # 1.0 - 0.081 - (0.314+0.214+0.094+0.386+0.236) - 0.269
        expected = round(
            1.0 - 0.081 - (0.314 + 0.214 + 0.094 + 0.386 + 0.236) - 0.269, 6
        )
        assert result == pytest.approx(expected, abs=1e-4)

    def test_mixed_profile(self) -> None:
        result = calculate_eq5d_index("11211")
        # deduction = 0.081 (constant) + 0.036 (activity level 2)
        expected = 1.0 - 0.081 - 0.036
        assert result == pytest.approx(expected, abs=1e-4)

    def test_none_profile(self) -> None:
        assert calculate_eq5d_index(None) is None

    def test_invalid_length(self) -> None:
        assert calculate_eq5d_index("1111") is None
        assert calculate_eq5d_index("111111") is None

    def test_invalid_digit(self) -> None:
        assert calculate_eq5d_index("11411") is None
        assert calculate_eq5d_index("abcde") is None

    def test_zero_digit(self) -> None:
        assert calculate_eq5d_index("10111") is None

    def test_n3_penalty_applied(self) -> None:
        """If any dimension is 3, the N3 penalty (0.269) must be added."""
        result_with_3 = calculate_eq5d_index("11113")
        result_without_3 = calculate_eq5d_index("11111")
        # Difference = anxiety level 3 weight + N3 penalty
        expected_diff = 0.236 + 0.269
        # without_3 > with_3 (worse health index when any dim = 3)
        assert result_without_3 is not None and result_with_3 is not None
        assert result_without_3 - result_with_3 == pytest.approx(expected_diff, abs=1e-4)
