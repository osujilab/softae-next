"""The rig's replicate reproducibility, and the rule it encodes.

The constant is a zero-day prior and these tests pin its *contract*, not its precision —
a per-material measurement is expected to supersede the value, and doing so must not
require rewriting the semantics around it.
"""

from __future__ import annotations

import math

import pytest

from softae.analysis import reproducibility as R


class TestTheConstantAndItsStanding:
    def test_the_shipped_spread_is_a_rounded_factor_of_two_not_the_raw_measurement(self):
        # 1.83x was measured; 2.0 ships. Rounding is deliberate anti-false-precision and
        # is the conservative direction for a floor -- a test so the rounding cannot be
        # "corrected" back to the raw figure without someone reading why.
        assert R.REPLICATE_SIGMA_SPREAD == 2.0
        assert R.REPLICATE_SIGMA_SPREAD > 1.83

    def test_the_provenance_travels_with_the_constant(self):
        """The number's value is mostly its caveats; they must not be strippable."""
        doc = R.__doc__ or ""
        assert "zero-day prior" in doc
        assert "supersedes" in doc


class TestResolvable:
    def test_a_difference_under_the_floor_is_not_resolvable(self):
        assert R.resolvable(1e-5, 1.5e-5) is False

    def test_a_difference_over_the_floor_is_resolvable(self):
        assert R.resolvable(1e-5, 3.0e-5) is True

    def test_exactly_at_the_floor_is_not_resolvable(self):
        """Strict inequality: a difference equal to the noise is not above it."""
        assert R.resolvable(1e-5, 2e-5) is False

    def test_the_comparison_is_symmetric_in_its_arguments(self):
        assert R.resolvable(3e-5, 1e-5) == R.resolvable(1e-5, 3e-5)

    def test_it_compares_a_ratio_not_a_difference(self):
        """Conductivity spans decades; an absolute tolerance would be meaningless.

        The same 2x ratio must give the same verdict five decades apart -- which an
        absolute-difference implementation could not do.
        """
        assert R.resolvable(1e-9, 3e-9) is R.resolvable(1e-4, 3e-4)

    def test_a_non_positive_or_nan_input_is_not_resolvable_rather_than_raising(self):
        # A caller comparing an unavailable sigma must get "cannot tell", not an
        # exception and not an accidental True.
        assert R.resolvable(float("nan"), 1e-5) is False
        assert R.resolvable(0.0, 1e-5) is False
        assert R.resolvable(-1e-5, 1e-5) is False

    def test_an_explicit_spread_overrides_the_shipped_prior(self):
        """A per-material measurement must be usable without editing the module."""
        assert R.resolvable(1e-5, 1.5e-5, spread=1.2) is True


class TestRangeToStandardDeviation:
    def test_no_spread_means_no_noise(self):
        assert R.replicate_log_sd(1.0) == pytest.approx(0.0)

    def test_the_shipped_prior_is_about_forty_percent_relative(self):
        assert R.replicate_log_sd() == pytest.approx(0.409, abs=0.005)

    def test_the_conversion_round_trips_through_the_stated_assumption(self):
        """exp(sd x factor) recovers the spread -- the assumption is not hidden."""
        sd = R.replicate_log_sd(2.5)
        assert math.exp(sd * R.RANGE_TO_SD_N3) == pytest.approx(2.5)

    def test_it_refuses_an_uncalibrated_sample_count_rather_than_reusing_the_n3_factor(self):
        # The range-to-sd factor depends on n. Silently reusing the n=3 value for n=5
        # would understate the sd by ~30% while looking like a supported call.
        with pytest.raises(ValueError, match="only n=3 is calibrated"):
            R.replicate_log_sd(2.0, n=5)

    def test_a_degenerate_spread_yields_nan_rather_than_a_number(self):
        assert math.isnan(R.replicate_log_sd(0.0))
        assert math.isnan(R.replicate_log_sd(float("nan")))
