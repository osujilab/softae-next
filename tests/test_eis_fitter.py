"""Covariance-preserving fitter (E1) — identifiability, scaling, and sum reporting.

Three defects converge here, and the tests are written so each is pinned separately:

* ``impedance.py`` discards ``pcov``, so ``ρ(R_series, R_bulk)`` — the quantity that
  decides whether the split may be reported at all — was unavailable;
* ``curve_fit`` is called unscaled, so on parameters spanning fourteen orders of
  magnitude it terminates at iteration zero and returns the initial guess; and
* the fit is unweighted while the grader weights by modulus.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("impedance")

from softae.analysis.eis.engine import _resolve_reported_resistance  # noqa: E402
from softae.analysis.eis.fitter import (  # noqa: E402
    FitCovariance,
    fit_spectrum,
    fit_with_covariance,
)
from tests.eis_synthetic import (  # noqa: E402
    DEFAULT_C_PAR,
    DEFAULT_CPE_N,
    DEFAULT_CPE_Q,
    DEFAULT_R_BULK,
    DEFAULT_R_SERIES,
    as_eis_result,
    reference_spectrum,
)

CIRCUIT = "R0-CPE0-p(R1,C0)"
#: Deliberately offset from truth, mimicking what ``extract_features`` produces.
GUESS = [194.0, 1e-7, 0.8, 51259.0, 3.0e-10]


class TestParameterRecovery:
    def test_a_synthetic_spectrum_recovers_its_generating_parameters(self):
        f, Z = reference_spectrum()
        cov = fit_with_covariance(f, Z, CIRCUIT, GUESS)
        assert cov is not None
        assert cov.value("R0") == pytest.approx(DEFAULT_R_SERIES, rel=1e-3)
        assert cov.value("R1") == pytest.approx(DEFAULT_R_BULK, rel=1e-3)
        assert cov.value("CPE0_0") == pytest.approx(DEFAULT_CPE_Q, rel=1e-2)
        assert cov.value("CPE0_1") == pytest.approx(DEFAULT_CPE_N, rel=1e-3)
        assert cov.value("C0") == pytest.approx(DEFAULT_C_PAR, rel=1e-2)

    def test_parameters_are_named_so_roles_never_resolve_positionally(self):
        f, Z = reference_spectrum()
        cov = fit_with_covariance(f, Z, CIRCUIT, GUESS)
        assert cov.names == ("R0", "CPE0_0", "CPE0_1", "R1", "C0")

    def test_a_noisy_spectrum_recovers_its_parameters_within_the_reported_errors(self):
        f, Z = reference_spectrum(noise_pct=1.0, seed=3)
        cov = fit_with_covariance(f, Z, CIRCUIT, GUESS)
        assert abs(cov.value("R1") - DEFAULT_R_BULK) < 5.0 * cov.se("R1")


class TestOptimiserScaling:
    def test_an_unscaled_fit_returns_the_initial_guess_which_is_why_x_scale_is_set(self):
        # The whole reason fit_with_covariance exists rather than a thin wrapper.
        # curve_fit's default x_scale=1.0 makes a unit step simultaneously absurd for
        # C_par (~3e-10) and invisible to R_bulk (~5e4), so it stops immediately —
        # and no amount of tightening ftol helps.
        from impedance.models.circuits.fitting import set_default_bounds, wrapCircuit
        from scipy.optimize import curve_fit

        f, Z = reference_spectrum()
        popt, _ = curve_fit(
            wrapCircuit(CIRCUIT, {}), f, np.hstack([Z.real, Z.imag]),
            p0=GUESS, bounds=set_default_bounds(CIRCUIT, constants={}),
            ftol=1e-13, max_nfev=20_000,
        )
        assert popt == pytest.approx(GUESS, rel=1e-9), (
            "if scipy ever fixes this, the x_scale rationale needs revisiting")

        scaled = fit_with_covariance(f, Z, CIRCUIT, GUESS)
        assert scaled.value("R0") == pytest.approx(DEFAULT_R_SERIES, rel=1e-3)

    def test_modulus_weighting_is_on_by_default_because_the_grader_already_grades_that_way(self):
        f, Z = reference_spectrum()
        assert fit_with_covariance(f, Z, CIRCUIT, GUESS).weight_by_modulus is True


class TestFailureModes:
    def test_a_fit_that_cannot_converge_returns_none_rather_than_raising(self):
        f = np.array([1.0, 2.0])
        Z = np.array([np.nan + 1j, np.nan + 1j])
        assert fit_with_covariance(f, Z, CIRCUIT, GUESS) is None

    def test_too_few_points_return_none_rather_than_a_meaningless_fit(self):
        f, Z = reference_spectrum()
        assert fit_with_covariance(f[:1], Z[:1], CIRCUIT, GUESS) is None

    def test_a_singular_covariance_reports_unidentifiable_rather_than_a_fabricated_rho(self):
        # One bad spectrum must not end a 32-channel batch, and a rho invented from a
        # singular Jacobian would be worse than none at all.
        cov = FitCovariance(
            names=("R0", "R1"), values=np.array([50.0, 5e4]),
            pcov=np.full((2, 2), np.nan), singular=True,
        )
        assert np.isnan(cov.rho("R0", "R1"))
        assert np.isnan(cov.se("R0"))
        assert np.isnan(cov.sum_se("R0", "R1"))

    def test_an_unknown_parameter_name_yields_nan_rather_than_an_index_error(self):
        f, Z = reference_spectrum()
        cov = fit_with_covariance(f, Z, CIRCUIT, GUESS)
        assert cov.index("Rnope") is None
        assert np.isnan(cov.value("Rnope"))
        assert np.isnan(cov.rho("R0", "Rnope"))


class TestIdentifiability:
    def test_an_in_band_relaxation_corner_leaves_the_split_identifiable(self):
        f, Z = reference_spectrum(R_bulk=5e4, noise_pct=1.0, seed=3)
        cov = fit_with_covariance(f, Z, CIRCUIT, GUESS)
        assert cov.rho("R0", "R1") > -0.95

    def test_a_corner_above_f_max_drives_rho_toward_minus_one(self):
        # R_bulk = 200 puts f_c at 2.3 MHz, an order of magnitude above the band.
        f, Z = reference_spectrum(R_bulk=200.0, noise_pct=1.0, seed=3)
        cov = fit_with_covariance(f, Z, CIRCUIT,
                                  [50.0, 1e-7, 0.8, 200.0, 3.5e-10])
        assert cov.rho("R0", "R1") < -0.95

    def test_the_sum_variance_collapses_exactly_where_the_split_is_meaningless(self):
        f, Z = reference_spectrum(R_bulk=200.0, noise_pct=1.0, seed=3)
        cov = fit_with_covariance(f, Z, CIRCUIT,
                                  [50.0, 1e-7, 0.8, 200.0, 3.5e-10])
        var_sum = cov.sum_se("R0", "R1") ** 2
        var_indep = cov.se("R0") ** 2 + cov.se("R1") ** 2
        assert var_sum < var_indep / 10.0

        # ...and the sum is the better-determined observable, not merely the safer one.
        sum_rel = cov.sum_se("R0", "R1") / cov.sum_value("R0", "R1")
        assert sum_rel < cov.rel_se("R0")

    def test_a_parameter_resting_on_a_bound_is_reported_as_pegged(self):
        cov = FitCovariance(
            names=("R0", "R1"), values=np.array([0.0, 5e4]),
            pcov=np.eye(2),
            bounds=(np.array([0.0, 0.0]), np.array([np.inf, np.inf])),
        )
        assert cov.pegged() == ("R0",)

    def test_no_bounds_means_nothing_can_be_pegged(self):
        cov = FitCovariance(names=("R0",), values=np.array([0.0]), pcov=np.eye(1))
        assert cov.pegged() == ()


class TestPeggedAgainstTheBoundsProductionActuallyBuilds:
    """``pegged`` is tested against ``set_default_bounds``' output, not a fixture.

    The test above it passes ``values = [0.0, ...]``, which a relative rule catches
    for a reason that has nothing to do with the rule: ``|v - 0| <= tol*max(|v|, 1e-30)``
    at ``v = 0`` is ``0 <= 1e-33``. That is the whole of what the zero-bound case used
    to be — an **effective cut of ``tol × 1e-30``**, three decades below the floor the
    legacy path applies and, worse, a function of the caller's ``[gates] bound_tol``.
    Not a chosen threshold at all, but a by-product of a formula written for finite
    bounds, so the fixture passed while the rule underneath it was arithmetic
    (``SUBAGENT_RULES`` §3.2). The bounds below are read from the library rather than
    written down, so this fixture cannot drift away from the production path the way
    the legacy-registry bounds in ``test_eis_railed_parameters`` have.
    """

    @staticmethod
    def _bounds():
        from impedance.models.circuits.fitting import set_default_bounds

        lo, hi = set_default_bounds(CIRCUIT)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def _cov(self, values):
        return FitCovariance(
            names=("R0", "CPE0_0", "CPE0_1", "R1", "C0"),
            values=np.asarray(values, dtype=float),
            pcov=np.eye(5) * 1e-4,
            bounds=self._bounds(),
        )

    def test_the_production_bounds_are_zero_below_and_mostly_infinite_above(self):
        """The premise, pinned: every lower bound is exactly 0, and the only finite
        upper bound belongs to the CPE exponent. A relative rule therefore has
        *nothing* to bite on for either resistance."""
        lo, hi = self._bounds()
        assert list(lo) == [0.0] * 5
        assert list(hi) == [np.inf, np.inf, 1.0, np.inf, np.inf]

    @pytest.mark.parametrize("value", [0.0, 5e-324, 4.6e-62, 1e-31, 1e-30])
    def test_a_resistance_collapsed_onto_a_zero_bound_is_pegged(self, value):
        """``R0 = 4.6e-62`` Ω is the stored shape: 449 of 3 618 corpus fits carry an
        ``R0`` below 1e-30 Ω. The last two are the ones the old cut of ``tol × 1e-30``
        let through, and they are not a rounding difference — they are the band the
        legacy path has demoted since ``443c948`` while the gated path did not."""
        assert "R0" in self._cov([value, 1e-7, 0.7, 5.0e4, 1e-10]).pegged()

    @pytest.mark.parametrize("value", [1e-29, 1e-12, 1.0, 50.0])
    def test_a_resistance_above_the_floor_is_set_by_the_data(self, value):
        assert "R0" not in self._cov([value, 1e-7, 0.7, 5.0e4, 1e-10]).pegged()

    def test_a_healthy_fit_pegs_nothing_at_all(self):
        assert self._cov([50.0, 1e-7, 0.7, 5.0e4, 1e-10]).pegged() == ()

    def test_the_finite_upper_bound_still_pegs_relatively(self):
        """``CPE0_1`` is the one parameter with a finite bound on the gated path, and
        the relative rule is right for it — a regression pin, not a new catch."""
        assert self._cov([50.0, 1e-7, 0.9997, 5.0e4, 1e-10]).pegged() == ("CPE0_1",)

    def test_an_infinite_bound_is_never_a_constraint(self):
        """A huge ``R1`` is a bad fit, not a pegged one: ``+inf`` is not a bound the
        optimiser can stop against."""
        assert "R1" not in self._cov([50.0, 1e-7, 0.7, 1e300, 1e-10]).pegged()

    def test_a_nan_parameter_is_not_reported_as_pegged(self):
        assert self._cov([np.nan, 1e-7, 0.7, 5.0e4, 1e-10]).pegged() == ()

    def test_the_zero_bound_cut_does_not_move_with_tol(self):
        """The two geometries are not two tolerances on one geometry, and the old
        code made them one: at ``tol = 1e-3`` the cut was 1e-33 and at ``tol = 0.5``
        it was 5e-31, so ``[gates] bound_tol`` silently redefined what "collapsed to
        zero ohms" meant. ``tol`` governs the window around a *finite* bound and
        nothing else."""
        collapsed = self._cov([1e-31, 1e-7, 0.7, 5.0e4, 1e-10])
        healthy = self._cov([1e-20, 1e-7, 0.7, 5.0e4, 1e-10])
        for tol in (1e-3, 0.5):
            assert "R0" in collapsed.pegged(tol)
            assert "R0" not in healthy.pegged(tol)


class TestSumVersusSplitIsBehavioural:
    """R2: the pipeline selects, the analyst is never handed the choice."""

    def _fit(self, R_bulk: float, guess):
        f, Z = reference_spectrum(R_bulk=R_bulk, noise_pct=1.0, seed=3)
        return fit_spectrum(as_eis_result(f, Z), "blocking_coplanar")

    def test_an_identifiable_split_reports_the_bulk_resistance_alone(self):
        fit = self._fit(5e4, GUESS)
        R, se, basis, rho = _resolve_reported_resistance(fit, rho_degenerate=-0.95)
        assert basis == "split_bulk"
        assert R == pytest.approx(fit.R1)

    def test_a_degenerate_split_reports_the_sum_instead_of_either_term(self):
        fit = self._fit(200.0, [50.0, 1e-7, 0.8, 200.0, 3.5e-10])
        R, se, basis, rho = _resolve_reported_resistance(fit, rho_degenerate=-0.95)
        assert basis == "sum"
        assert rho < -0.95
        assert R == pytest.approx(fit.R0 + fit.R1, rel=1e-6)

    def test_a_singular_fit_reports_the_sum_because_the_split_is_unknowable(self):
        class _Fit:
            model_name = "blocking_coplanar"
            R1 = 5e4
            covariance = FitCovariance(
                names=("R0", "R1"), values=np.array([50.0, 5e4]),
                pcov=np.full((2, 2), np.nan), singular=True,
            )

        _, _, basis, _ = _resolve_reported_resistance(_Fit(), rho_degenerate=-0.95)
        assert basis == "sum"

    def test_a_fit_without_covariance_falls_back_to_the_split_as_the_legacy_path_does(self):
        class _Fit:
            model_name = "simpleSalt"
            R1 = 1234.0
            covariance = None

        R, se, basis, rho = _resolve_reported_resistance(_Fit(), rho_degenerate=-0.95)
        assert (R, basis) == (1234.0, "split_bulk")
        assert np.isnan(se) and np.isnan(rho)


class TestFitSpectrumShape:
    def test_it_returns_the_same_fitresult_type_so_every_existing_consumer_keeps_working(self):
        from softae.analysis.circuit_fitting import FitResult

        fit = fit_spectrum(as_eis_result(*reference_spectrum()), "blocking_coplanar")
        assert isinstance(fit, FitResult)
        assert fit.success and fit.covariance is not None
        assert fit.quality, "goodness-of-fit metrics must still be computed"

    def test_resistances_resolve_by_element_name_not_by_position(self):
        fit = fit_spectrum(as_eis_result(*reference_spectrum()), "blocking_coplanar")
        assert fit.R0 == pytest.approx(DEFAULT_R_SERIES, rel=1e-3)
        assert fit.R1 == pytest.approx(DEFAULT_R_BULK, rel=1e-3)

    def test_an_unknown_model_name_raises_rather_than_fitting_the_wrong_topology(self):
        with pytest.raises(ValueError, match="Unknown circuit model"):
            fit_spectrum(as_eis_result(*reference_spectrum()), "nope")

    def test_a_legacy_model_name_still_resolves_so_stored_records_stay_readable(self):
        fit = fit_spectrum(as_eis_result(*reference_spectrum()), "simpleSalt")
        assert fit.success


class TestConvergenceTolerance:
    """Why the tolerance is 1e-10, asserted rather than left to a comment.

    Loosening it looks like free speed and is not. The trap is the *metric*: judged on
    ``R_series + R_bulk`` every tolerance looks equivalent, because an error in the
    50 Ω series term is invisible against a 50 kΩ sum. Judged on ``R_series`` alone —
    the term this whole module exists to recover (§3.1) — 1e-8 is an order worse.
    """

    def _fit(self, tol):
        from softae.analysis.eis.fitter import fit_with_covariance

        from .eis_synthetic import reference_spectrum

        f, Z = reference_spectrum(noise_pct=0.0)
        return fit_with_covariance(
            f, Z, "R_0-p(R_1,C_1)-CPE_1",
            initial_guess=[100.0, 4.5e4, 3e-10, 1e-7, 0.8], tol=tol)

    def test_the_shipped_default_recovers_the_small_series_term(self):
        from softae.analysis.eis.fitter import DEFAULT_FIT_TOL

        cov = self._fit(DEFAULT_FIT_TOL)
        assert cov is not None
        assert abs(cov.value("R_0") - 50.0) / 50.0 < 0.01

    def test_loosening_it_degrades_the_series_term_but_not_the_sum(self):
        """The measurement that decided this, kept as a guard.

        If a future change makes the loose fit as good as the tight one, this fails and
        the tolerance can be revisited on evidence — which is the only basis on which it
        should be.
        """
        tight, loose = self._fit(1e-10), self._fit(1e-8)
        assert tight is not None and loose is not None

        err = lambda c, n, t: abs(c.value(n) - t) / t          # noqa: E731
        assert err(loose, "R_0", 50.0) > 5 * err(tight, "R_0", 50.0)

        # ...while the sum is indistinguishable, which is exactly what makes the
        # loosening look safe if you measure the wrong thing.
        sum_err = lambda c: abs(c.sum_value("R_0", "R_1") - 5.005e4) / 5.005e4  # noqa: E731
        assert abs(sum_err(loose) - sum_err(tight)) < 1e-3

    def test_on_noisy_data_the_tolerance_stops_mattering(self):
        """Real spectra are noise-limited, not tolerance-limited — which is why the
        several-second fit is a synthetic-only cost and not an operational one."""
        from softae.analysis.eis.fitter import fit_with_covariance

        from .eis_synthetic import reference_spectrum

        f, Z = reference_spectrum(noise_pct=1.0, seed=11)
        fits = [fit_with_covariance(
            f, Z, "R_0-p(R_1,C_1)-CPE_1",
            initial_guess=[100.0, 4.5e4, 3e-10, 1e-7, 0.8], tol=t)
            for t in (1e-6, 1e-10)]
        assert all(c is not None for c in fits)
        loose, tight = fits
        assert loose.value("R_1") == pytest.approx(tight.value("R_1"), rel=1e-3)


class TestPerQuestionIdentifiability:
    """Three consumers ask three questions of one covariance and get three answers.

    ``singular`` is one bool about the whole matrix, and every older accessor
    short-circuits on it. On the measured corpus that is right about exactly one of
    the three questions: the R0/R1 split is genuinely dead, while the sum and R1 alone
    are determined to parts in 10^4. These pin the separation.

    ``rho = -1`` with equal variances is the shape [p91] measured on 30 of 30, so it
    is the fixture rather than an invented edge case.
    """

    @staticmethod
    def _cov(*, saa=1e4, sbb=1e4, rho=-1.0, r0=3.0, r1=1.0e6, singular=False):
        sab = rho * np.sqrt(saa * sbb)
        return FitCovariance(
            names=("R0", "R1"),
            values=np.array([r0, r1], dtype=float),
            pcov=np.array([[saa, sab], [sab, sbb]], dtype=float),
            singular=singular,
            n_points=30,
        )

    def test_the_split_is_refused_at_rho_minus_one(self):
        # The one question the global flag gets right, and it stays right.
        assert self._cov(rho=-1.0).split_identifiable("R0", "R1") is False

    def test_the_sum_survives_the_dead_split(self):
        # Complement, not a weaker version: the sum is determined BECAUSE one
        # parameter carries the whole uncertainty.
        assert self._cov(rho=-1.0).sum_determined("R0", "R1") is True

    def test_the_sum_also_survives_rho_PLUS_one(self):
        # [p91] §2's correction, pinned. rho = +1 makes Var(a+b) MAXIMAL, and the sum
        # is still determined -- so the sign of rho is not the mechanism, and a test
        # written against the rho=-1 cancellation story would pass for a wrong reason.
        assert self._cov(rho=+1.0).sum_determined("R0", "R1") is True

    def test_one_parameter_is_supported_while_the_other_is_not(self):
        # The statement no pair-level bool can make. Same matrix, both answers.
        cov = self._cov(rho=-1.0)
        assert cov.supports("R1") is True
        assert cov.supports("R0") is False

    def test_a_finite_variance_is_not_enough_to_support_a_collapsed_parameter(self):
        # R0 collapsed onto a zero lower bound has a perfectly finite variance and no
        # information in it. Judging on variance alone would call that supported.
        cov = self._cov(saa=1e-20, r0=1e-33)
        assert np.isfinite(cov.pcov[0, 0])
        assert cov.supports("R0") is False

    def test_none_of_the_three_read_the_global_singular_flag(self):
        # THE POINT. A whole-matrix flag must not blank a per-parameter question --
        # that conflation is what these exist to remove. Same numbers, flag flipped.
        on = self._cov(rho=-1.0, singular=True)
        off = self._cov(rho=-1.0, singular=False)
        assert on.supports("R1") == off.supports("R1") is True
        assert on.sum_determined("R0", "R1") == off.sum_determined("R0", "R1") is True
        assert on.split_identifiable("R0", "R1") == off.split_identifiable("R0", "R1")
        # And the older accessors still DO gate on it, which is the contrast.
        assert np.isnan(on.se("R1")) and np.isfinite(off.se("R1"))

    def test_an_all_nan_covariance_refuses_every_question_on_its_own_terms(self):
        cov = FitCovariance(names=("R0", "R1"), values=np.array([3.0, 1e6]),
                            pcov=np.full((2, 2), np.nan), singular=True, n_points=30)
        assert cov.supports("R1") is False
        assert cov.sum_determined("R0", "R1") is False
        assert cov.split_identifiable("R0", "R1") is False

    def test_a_well_conditioned_pair_passes_all_three(self):
        # Anti-vacuity: predicates that always refuse would satisfy every test above.
        cov = self._cov(saa=1e-4, sbb=1e-4, rho=0.1, r0=1.0e3, r1=1.0e6)
        assert cov.split_identifiable("R0", "R1") is True
        assert cov.sum_determined("R0", "R1") is True
        assert cov.supports("R0") is True and cov.supports("R1") is True

    def test_an_unknown_parameter_name_refuses_rather_than_raising(self):
        cov = self._cov()
        assert cov.supports("nope") is False
        assert cov.sum_determined("R0", "nope") is False
        assert cov.split_identifiable("nope", "R1") is False
