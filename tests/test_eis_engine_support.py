"""``_resolve_reported_resistance``: which resistance the evidence licenses reporting.

The subject here is the **two-sided** degeneracy test that landed 2026-09-03. Until then
the engine used a signed comparison, ``ρ <= rho_degenerate``, while
:func:`~softae.analysis.eis.gates.gate_degeneracy` had already moved to the magnitude in
``3e51ac0`` — a divergence that was deliberate and is now retired by operator direction.

**The positive branch is reached by real data, not only by fixtures.** The 10 MΩ metal-film
reference of ``20260903T213729Z_commission_reference_r`` (channel 25, the first spectrum on
this rig with a known answer) fits at ``ρ(R_series, R_bulk) = +1.000000``, and roughly a
third of the covariance-bearing fits of ``20260825T154521Z_arrhenius_sweep`` sit on the same
side. That matters because a gate can discriminate perfectly on a branch the population
never visits; this one is visited.

**What the two-sided rule does and does not buy.** It stops the row claiming a partition the
likelihood surface never constrained, and it replaces a fictitiously tight ``se`` with an
honest one. It is *not* an accuracy fix: on that 10 MΩ spectrum ``R_series`` is 289 Ω beside
a 9.02 MΩ ``R_bulk``, so sum and split agree to 3e-5 and the −9.8 % error against nominal is
untouched. The tests below therefore assert on the **basis and the standard error**, which
are what actually move, rather than on a resistance that mostly does not.

Nothing here reads config, touches the DataStore, or fits a spectrum: every case is a
hand-built covariance, so the correlation under test is exact rather than approximately
whatever the optimiser happened to produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.eis.engine_support import _resolve_reported_resistance
from softae.analysis.eis.fitter import FitCovariance

RHO_DEGENERATE = -0.95
R_SERIES, R_BULK = 50.0, 2000.0


def _fit_with_rho(rho: float, *, singular: bool = False, pcov=None):
    """A minimal fit whose ``R0``/``R1`` correlation is exactly *rho*.

    Unit diagonal, so ``pcov`` *is* the correlation matrix: ``se(R1)`` is 1.0 and
    ``sum_se`` is ``sqrt(2 + 2ρ)`` — 0 at ρ = −1 and 2 at ρ = +1, which is the whole
    asymmetry the two-sided rule has to live with, readable by eye.
    """
    if pcov is None:
        pcov = np.array([[1.0, float(rho)], [float(rho), 1.0]])

    class _Fit:
        model_name = "blocking_coplanar"
        R1 = R_BULK
        covariance = FitCovariance(
            names=("R0", "R1"), values=np.array([R_SERIES, R_BULK]),
            pcov=np.asarray(pcov, dtype=float), singular=singular,
        )

    return _Fit()


def _resolve(fit, rho_degenerate: float = RHO_DEGENERATE):
    return _resolve_reported_resistance(fit, rho_degenerate=rho_degenerate)


class TestNegativeDegeneracyIsUnchanged:
    """The change adds a side; it must not swap one."""

    @pytest.mark.parametrize("rho", [-0.95, -0.977, -1.0])
    def test_a_negative_correlation_past_the_threshold_still_reports_the_sum(self, rho):
        R, se, basis, reported_rho = _resolve(_fit_with_rho(rho))
        assert basis == "sum"
        assert R == pytest.approx(R_SERIES + R_BULK)
        assert reported_rho == pytest.approx(rho)

    def test_a_negative_correlation_inside_the_threshold_still_reports_the_split(self):
        R, se, basis, _ = _resolve(_fit_with_rho(-0.9))
        assert basis == "split_bulk"
        assert R == pytest.approx(R_BULK)

    def test_the_negative_side_still_reports_the_sum_with_the_collapsed_variance(self):
        # sqrt(2 + 2ρ) → 0 as ρ → −1: the sum is determined exactly where the split is
        # meaningless. This is the variance argument the threshold was built on, and it
        # is the half that does NOT generalise to positive ρ.
        _, se, _, _ = _resolve(_fit_with_rho(-1.0))
        assert se == pytest.approx(0.0, abs=1e-9)
        assert se < _fit_with_rho(-1.0).covariance.se("R1")


class TestPositiveDegeneracyIsNowDetected:
    """The new behaviour: the same rank deficiency, ridge running the other way."""

    @pytest.mark.parametrize("rho", [0.95, 0.977, 1.0])
    def test_a_positive_correlation_past_the_threshold_now_reports_the_sum(self, rho):
        R, se, basis, reported_rho = _resolve(_fit_with_rho(rho))
        assert basis == "sum"
        assert R == pytest.approx(R_SERIES + R_BULK)
        assert reported_rho == pytest.approx(rho)

    def test_the_old_one_sided_rule_would_have_reported_the_split_here(self):
        # The pin on the defect itself. `ρ <= -0.95` is False at ρ = +1, so the shipped
        # engine returned `R_bulk` alone on the 10 MΩ reference resistor while the gate
        # beside it recorded "the split is invented".
        rho = 1.0
        assert not (rho <= RHO_DEGENERATE), "the old rule passed this cleanly"
        assert _resolve(_fit_with_rho(rho))[2] == "sum"

    def test_a_positive_correlation_inside_the_threshold_still_reports_the_split(self):
        R, _, basis, _ = _resolve(_fit_with_rho(0.9))
        assert basis == "split_bulk"
        assert R == pytest.approx(R_BULK)

    def test_the_positive_side_reports_the_sum_with_an_inflated_not_collapsed_error(self):
        # sqrt(2 + 2ρ) → 2 as ρ → +1, larger than either individual SE of 1.0. The sum
        # is the LESS well determined quantity here, and reporting it honestly says so
        # rather than borrowing the negative side's collapse.
        fit = _fit_with_rho(1.0)
        _, se, _, _ = _resolve(fit)
        assert se == pytest.approx(2.0)
        assert se > fit.covariance.se("R1")


class TestTheBoundaryAndTheSignOfTheThreshold:
    def test_the_magnitude_exactly_at_the_threshold_is_degenerate_on_both_sides(self):
        # `>=`, matching `gate_degeneracy`'s `abs(rho) < threshold` for `passed`. The
        # two must agree at the boundary or one spectrum in the corpus reads both ways.
        assert _resolve(_fit_with_rho(-0.95))[2] == "sum"
        assert _resolve(_fit_with_rho(0.95))[2] == "sum"

    def test_just_inside_the_threshold_is_not_degenerate_on_either_side(self):
        assert _resolve(_fit_with_rho(-0.9499))[2] == "split_bulk"
        assert _resolve(_fit_with_rho(0.9499))[2] == "split_bulk"

    def test_a_positively_written_threshold_gives_the_identical_decision(self):
        # `rho_degenerate` is a config key and an operator may write it either way.
        # abs() is applied to the configured value, so the sign cannot silently invert
        # the test — the same guard `gate_degeneracy` applies.
        for rho in (-1.0, -0.5, 0.5, 1.0):
            assert (_resolve(_fit_with_rho(rho), rho_degenerate=0.95)[2]
                    == _resolve(_fit_with_rho(rho), rho_degenerate=-0.95)[2])


class TestTheFailClosedAndUnknownPaths:
    def test_a_singular_covariance_still_reports_the_sum_whatever_rho_reads(self):
        # Fail-closed and untouched by the two-sided change: `rho()` returns NaN when
        # singular, so this branch cannot be reached through the comparison at all.
        R, se, basis, rho = _resolve(_fit_with_rho(0.0, singular=True))
        assert basis == "sum"
        assert R == pytest.approx(R_SERIES + R_BULK)
        assert np.isnan(rho) and np.isnan(se)

    def test_a_nan_correlation_on_a_non_singular_fit_is_not_treated_as_degenerate(self):
        # "Unknown" must not be spelled with the same token as "checked and degenerate".
        # A zero variance makes `rho()` return NaN without setting `singular`, and
        # `abs(nan) >= t` is False anyway — the `rho == rho` guard states the intent
        # rather than relying on that.
        fit = _fit_with_rho(0.0, pcov=np.array([[0.0, 0.0], [0.0, 1.0]]))
        R, _, basis, rho = _resolve(fit)
        assert np.isnan(rho)
        assert basis == "split_bulk"
        assert R == pytest.approx(R_BULK)

    def test_a_fit_without_covariance_falls_back_to_the_split_as_the_legacy_path_does(self):
        class _Fit:
            model_name = "simpleSalt"
            R1 = 1234.0
            covariance = None

        R, se, basis, rho = _resolve(_Fit())
        assert (R, basis) == (1234.0, "split_bulk")
        assert np.isnan(se) and np.isnan(rho)


class TestTheRecordNamesWhichSideItWas:
    """The two sides share a basis but are not the same finding."""

    def _msg(self, fit, monkeypatch):
        seen = {}
        import softae.analysis.eis.engine_support as mod

        monkeypatch.setattr(mod.logger, "info",
                            lambda event, **kw: seen.update(kw, event=event))
        _resolve(fit)
        return seen

    def test_the_negative_side_is_logged_as_the_relaxation_corner_leaving_the_band(
            self, monkeypatch):
        seen = self._msg(_fit_with_rho(-1.0), monkeypatch)
        assert seen["event"] == "eis_split_degenerate"
        assert "relaxation corner out of band" in seen["msg"]

    def test_the_positive_side_is_logged_as_the_data_pinning_the_difference(
            self, monkeypatch):
        seen = self._msg(_fit_with_rho(1.0), monkeypatch)
        assert "positive degeneracy" in seen["msg"]
        assert "relaxation corner" not in seen["msg"]

    def test_a_singular_covariance_is_logged_as_singular_not_as_a_sign(
            self, monkeypatch):
        seen = self._msg(_fit_with_rho(0.0, singular=True), monkeypatch)
        assert "singular" in seen["msg"]
