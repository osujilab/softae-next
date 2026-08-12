"""Geometry series as a self-validating calibration route (E5, §5.6, R13).

The route's whole claim is that σ falls out of the *slope* while every nuisance sits in
the intercept, so no blank subtraction is needed. These tests pin the three things that
make the claim safe rather than merely convenient:

1. the slope really does recover σ with an arbitrary fixture conductance present,
2. a slope that drifts with frequency is **refused**, not caveated, and
3. the design checks (levels, span, replicates, F12) are separate from fit quality —
   a perfect regression over a confounded design still cannot support the claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from softae.analysis.eis.geometry_series import (
    MIN_LEVELS,
    GeometrySeriesFit,
    SeriesMember,
    fit_geometry_series,
)

L_GAP = 0.2
L_STRIPE = 0.2
FREQ = np.geomspace(1e5, 1.0, 40)


#: Geometric factor: ``G_bulk = σ · GEOM · t``, matching ``K = L_gap/(t·L_stripe)``.
GEOM = L_STRIPE / L_GAP

#: Coplanar geometric capacitance per cm of film thickness. Like the conductance, it
#: scales with how much film is there — which is what puts it in the *slope* and leaves
#: only ``C_stray`` in the intercept.
C_CELL_PER_CM = 1e-9 / 0.015


def _member(t_um, sigma, *, channel=-1, G_fixture=0.0, C_stray=0.0,
            dielectric=0.0, noise=0.0, seed=0):
    """A synthetic spectrum whose parallel core is exactly what §5.6 assumes::

        Y = (σ + ε''·ω)·GEOM·t + G_fixture  +  jω(C_cell·t + C_stray)

    Two details are load-bearing and were both wrong on the first attempt:

    * **The sample capacitance scales with thickness.** A constant ``C_cell`` would sit
      in the intercept alongside ``C_stray`` and the two would be inseparable — which
      is a real confound, just not the one this route claims to solve.
    * **The dielectric loss scales with thickness too.** A lossy film's loss grows with
      how much film there is, exactly as its conduction does, so the contaminant lands
      in the **slope**. A frequency-dependent term added to the *intercept* would be a
      fixture artifact and the slope would stay clean — which is why this test has to
      put it in the right place to mean anything.
    """
    t_cm = t_um * 1e-4
    omega = 2.0 * np.pi * FREQ
    G = (sigma + dielectric * omega) * GEOM * t_cm + G_fixture
    Y = G + 1j * omega * (C_CELL_PER_CM * t_cm + C_stray)
    if noise:
        rng = np.random.default_rng(seed)
        Y = Y * (1.0 + noise * rng.standard_normal(Y.shape))
    return SeriesMember(thickness_cm=t_cm, frequency=FREQ, Z=1.0 / Y, channel=channel)


def _clean_series(sigma=1e-5, *, crossed=True, **kw):
    """Four levels, two replicates each, channel index crossed against level."""
    levels = [100.0, 150.0, 200.0, 250.0]
    order = ([100, 200, 150, 250, 200, 100, 250, 150] if crossed
             else [100, 100, 150, 150, 200, 200, 250, 250])
    assert sorted(order) == sorted(levels * 2)
    return [_member(t, sigma, channel=i + 1, seed=i, **kw)
            for i, t in enumerate(order)]


class TestSlopeRecoversSigma:
    def test_the_slope_recovers_sigma_through_an_arbitrary_fixture_conductance(self):
        # The point of the route (§5.6.1): the fixture is an additive intercept, so it
        # drops out of the derivative. A blank subtraction that got this wrong -- which
        # is what F6 recorded on this fixture -- cannot affect the answer here.
        fit = fit_geometry_series(_clean_series(1e-5, G_fixture=3e-6),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.usable
        assert fit.sigma_S_per_cm == pytest.approx(1e-5, rel=0.02)

    def test_a_larger_fixture_conductance_does_not_move_the_answer(self):
        a = fit_geometry_series(_clean_series(1e-5, G_fixture=0.0),
                                L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        b = fit_geometry_series(_clean_series(1e-5, G_fixture=1e-4),
                                L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert b.sigma_S_per_cm == pytest.approx(a.sigma_S_per_cm, rel=0.02)
        assert b.intercept_S > a.intercept_S      # it went into the intercept instead

    def test_the_intercept_is_a_free_measurement_of_the_stray_capacitance(self):
        # §5.6.2. Worth having independently of sigma: it cross-checks a blank, and on
        # this fixture the blank is the artifact that is hardest to trust.
        fit = fit_geometry_series(_clean_series(1e-5, C_stray=12e-12),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.C_stray_F == pytest.approx(12e-12, rel=0.1)

    def test_it_survives_realistic_noise(self):
        fit = fit_geometry_series(_clean_series(1e-5, noise=0.01),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.usable
        assert fit.sigma_S_per_cm == pytest.approx(1e-5, rel=0.15)


class TestFrequencyIndependenceIsTheValidation:
    """§5.6.3 — the one test that separates conductivity from dielectric loss."""

    def test_a_dielectric_contaminated_slope_is_refused_not_caveated(self):
        # The failure this exists for: a loss term scaling with omega fits a straight
        # line in thickness at EVERY frequency, so per-frequency R^2 stays near 1 and
        # nothing within a single frequency can tell it from conduction.
        members = _clean_series(1e-5, dielectric=2e-11)
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert not fit.slope_frequency_independent
        assert math.isnan(fit.sigma_S_per_cm)
        assert "not σ" in fit.describe() or "REFUSED" in fit.describe()

    def test_per_frequency_r_squared_stays_excellent_even_when_refused(self):
        # The evidence for refusing rather than reporting-with-a-flag: goodness of fit
        # is no defence at all here, so a caveat attached to a number would be read as
        # a good number with a note.
        fit = fit_geometry_series(_clean_series(1e-5, dielectric=2e-11),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        r2 = [s.r_squared for s in fit.slopes if s.r_squared == s.r_squared]
        assert r2 and min(r2) > 0.99
        assert not fit.usable

    def test_the_raw_median_is_still_reported_so_the_refusal_is_inspectable(self):
        fit = fit_geometry_series(_clean_series(1e-5, dielectric=2e-11),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert math.isnan(fit.sigma_S_per_cm)
        assert fit.sigma_median_raw == fit.sigma_median_raw

    def test_a_clean_series_has_a_flat_power_law_exponent(self):
        fit = fit_geometry_series(_clean_series(1e-5),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert abs(fit.slope_exponent) < 0.05

    def test_contamination_drives_the_exponent_toward_one(self):
        # Residual dielectric loss scales ~omega, so a fully contaminated slope tends
        # to exponent 1. This is what makes the exponent the sharper of the two tests.
        fit = fit_geometry_series(_clean_series(1e-9, dielectric=2e-11),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.slope_exponent > 0.5


class TestDesignAdequacyIsSeparateFromFitQuality:
    def test_three_levels_are_inadequate_however_well_they_fit(self):
        members = [_member(t, 1e-5, channel=i + 1)
                   for i, t in enumerate([100, 150, 200, 100, 150, 200])]
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.n_levels == 3
        assert not fit.adequate
        assert not fit.usable
        assert any(str(MIN_LEVELS) in i for i in fit.issues)

    def test_a_narrow_span_is_rejected(self):
        members = [_member(t, 1e-5, channel=i + 1)
                   for i, t in enumerate([100, 110, 120, 130, 100, 110, 120, 130])]
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.span_ratio < 2.0
        assert not fit.adequate

    def test_unreplicated_levels_are_rejected(self):
        members = [_member(t, 1e-5, channel=i + 1)
                   for i, t in enumerate([100, 150, 200, 250])]
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.min_replicates == 1
        assert not fit.adequate

    def test_f12_is_rechecked_on_the_data_as_it_arrived(self):
        # The planner prevents this before casting; this catches a sound plan followed
        # inattentively, which produces exactly the dataset the plan existed to prevent.
        # CH1/2 = 100, CH3/4 = 150, CH5/6 = 200, CH7/8 = 250 -- the real F12 pattern.
        fit = fit_geometry_series(_clean_series(1e-5, crossed=False),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.confound_verdict == "confounded"
        assert not fit.adequate
        assert not fit.usable
        assert any("F12" in i for i in fit.issues)

    def test_a_crossed_design_passes_the_confound_check(self):
        fit = fit_geometry_series(_clean_series(1e-5),
                                  L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.confound_verdict == "ok"
        assert fit.adequate

    def test_missing_channels_say_the_check_did_not_run(self):
        # Silence would read as a pass. F12 is invisible in the spectra, so an unrun
        # check has to be reported as unrun.
        members = [_member(t, 1e-5) for t in [100, 150, 200, 250] * 2]
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.confound_verdict == "indeterminate"
        assert any("did not run" in i for i in fit.issues)


class TestDeadHeightIsNotIdentifiableHere:
    """The finding that outranks the overhaul's own ±20 µm caveat.

    One line, one intercept, two unknowns: ``b = G_fixture − m·h``. More levels do not
    help, because the deficiency is rank rather than noise — so ``h`` waits on an
    independent ``G_fixture``, i.e. the open blank, i.e. the RE→CE jumper.
    """

    def test_dead_height_is_nan_without_an_independent_fixture_conductance(self):
        fit = fit_geometry_series(_clean_series(1e-5), L_gap_cm=L_GAP,
                                  L_stripe_cm=L_STRIPE)
        assert math.isnan(fit.dead_height_cm(float("nan")))
        assert math.isnan(fit.dead_height_cm(None))  # type: ignore[arg-type]

    def test_a_supplied_fixture_conductance_recovers_the_dead_height(self):
        # Cast the series as if h = 40 um: the film conducts only over (t - h), so the
        # x-intercept sits at h rather than at zero.
        h_um, G_fix, sigma = 40.0, 5e-6, 1e-5
        members = []
        for i, t_um in enumerate([100, 200, 150, 250, 200, 100, 250, 150]):
            eff = (t_um - h_um) * 1e-4
            G = sigma * L_STRIPE * eff / L_GAP + G_fix
            omega = 2.0 * np.pi * FREQ
            Y = G + 1j * omega * 1e-9
            members.append(SeriesMember(thickness_cm=t_um * 1e-4, frequency=FREQ,
                                        Z=1.0 / Y, channel=i + 1))
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.usable
        assert fit.dead_height_cm(G_fix) == pytest.approx(h_um * 1e-4, rel=0.05)

    def test_the_wrong_fixture_conductance_gives_the_wrong_dead_height(self):
        # Why it is advisory even when supplied: h inherits every error in G_fixture,
        # and the open blank it comes from is itself unvalidated on this fixture.
        h_um, G_fix, sigma = 40.0, 5e-6, 1e-5
        members = []
        for i, t_um in enumerate([100, 200, 150, 250, 200, 100, 250, 150]):
            eff = (t_um - h_um) * 1e-4
            G = sigma * L_STRIPE * eff / L_GAP + G_fix
            omega = 2.0 * np.pi * FREQ
            members.append(SeriesMember(thickness_cm=t_um * 1e-4, frequency=FREQ,
                                        Z=1.0 / (G + 1j * omega * 1e-9),
                                        channel=i + 1))
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        truth = fit.dead_height_cm(G_fix)
        skewed = fit.dead_height_cm(G_fix * 1.5)
        assert not math.isclose(truth, skewed, rel_tol=0.2)

    def test_the_intercept_is_not_named_g_fixture_because_it_is_not_one(self):
        fit = GeometrySeriesFit()
        assert not hasattr(fit, "G_fixture_S")
        assert hasattr(fit, "intercept_S")


class TestNeverRaises:
    """A series is analysed long after the samples are gone. A traceback loses it."""

    @pytest.mark.parametrize("members", [
        [],
        [SeriesMember(thickness_cm=0.015, frequency=np.asarray([]),
                      Z=np.asarray([], dtype=complex))],
        [SeriesMember(thickness_cm=float("nan"), frequency=FREQ,
                      Z=np.ones_like(FREQ, dtype=complex))],
        [SeriesMember(thickness_cm=-1.0, frequency=FREQ,
                      Z=np.ones_like(FREQ, dtype=complex))],
    ])
    def test_degenerate_input_returns_a_report_naming_the_problem(self, members):
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert not fit.usable
        assert fit.issues
        assert isinstance(fit.describe(), str)

    def test_members_on_disjoint_frequency_grids_report_rather_than_interpolate(self):
        # Interpolating onto a synthetic grid would let one member's extrapolated tail
        # set a slope for everyone.
        a = SeriesMember(thickness_cm=0.010, frequency=np.geomspace(1e5, 1e3, 10),
                         Z=np.full(10, 1e4 + 0j), channel=1)
        b = SeriesMember(thickness_cm=0.020, frequency=np.geomspace(1e2, 1.0, 10),
                         Z=np.full(10, 5e3 + 0j), channel=2)
        fit = fit_geometry_series([a, b], L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert not fit.usable
        assert any("every member" in i for i in fit.issues)


class TestThresholdsComeFromConfig:
    def test_the_tolerances_default_to_eis_gates(self):
        from softae.analysis.eis.settings import eis_settings

        g = eis_settings().gates
        assert g.geom_sigma_spread_tol > 0
        assert g.geom_slope_exponent_tol > 0

    def test_an_explicit_tolerance_overrides_the_config(self):
        # Loosening is how a refusal gets overridden after inspection -- the raw median
        # is reported precisely so that judgement can be made on evidence.
        members = _clean_series(1e-5, dielectric=2e-11)
        strict = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        loose = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE,
                                    sigma_spread_tol=1e6, slope_exponent_tol=1e6)
        assert not strict.slope_frequency_independent
        assert loose.slope_frequency_independent
        assert loose.sigma_S_per_cm == pytest.approx(strict.sigma_median_raw)


class TestDeadHeightAcceptsAFrequencyDependentFixture:
    """The fixture's real part is dielectric loss, not an ohmic leak.

    Seven tied open blanks give ``d ln G / d ln f`` between +0.87 and +1.04 — i.e.
    ``G = ωC·tan δ`` with tan δ ≈ 0.05. An ohmic leak would give 0. So there is no
    single ``G_fixture``, and a scalar describes the fixture at exactly one frequency.
    """

    def _series_with_dead_height(self, h_um=40.0, G0=5e-9, f0=1e3, sigma=1e-5):
        """A series whose fixture conductance rises linearly with frequency."""
        members = []
        omega = 2.0 * np.pi * FREQ
        G_fix = G0 * (FREQ / f0)                      # G proportional to omega
        for i, t_um in enumerate([100, 200, 150, 250, 200, 100, 250, 150]):
            eff = (t_um - h_um) * 1e-4
            G = sigma * GEOM * eff + G_fix
            members.append(SeriesMember(thickness_cm=t_um * 1e-4, frequency=FREQ,
                                        Z=1.0 / (G + 1j * omega * 1e-9),
                                        channel=i + 1))
        return members, dict(zip(FREQ, G_fix))

    def test_a_per_frequency_mapping_recovers_the_dead_height(self):
        members, G_of_f = self._series_with_dead_height()
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert fit.usable
        assert fit.dead_height_cm(G_of_f) == pytest.approx(40e-4, rel=0.05)

    def test_a_scalar_is_only_right_at_the_frequency_it_was_measured_at(self):
        # The reason the mapping form exists -- and the size of the error depends on
        # WHICH frequency the scalar came from, which is precisely what a scalar does
        # not record. A value taken at 1 kHz happens to land near the log-centre of
        # this band and is ~8% out; the same measurement taken at 100 kHz is an order
        # of magnitude out. Neither is a property the caller can see.
        members, G_of_f = self._series_with_dead_height()
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        good = fit.dead_height_cm(G_of_f)
        assert good == pytest.approx(40e-4, rel=0.05)

        near_band_centre = fit.dead_height_cm(5e-9)          # G at 1 kHz
        at_top_of_band = fit.dead_height_cm(5e-9 * 100)      # G at 100 kHz
        assert abs(near_band_centre - good) / good < 0.2
        assert abs(at_top_of_band - good) / good > 5.0

    def test_the_profile_exposes_drift_that_a_median_would_hide(self):
        # h is geometric and cannot depend on frequency. If it does, either the slope
        # or G_fixture is wrong -- and the median is exactly what would conceal it.
        members, _ = self._series_with_dead_height()
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        profile = fit.dead_height_profile(5e-9)     # deliberately the wrong form
        hs = [h for _, h in profile]
        assert len(hs) > 3
        assert max(hs) - min(hs) > 1e-4             # visibly inconsistent

    def test_a_flat_profile_confirms_a_consistent_answer(self):
        members, G_of_f = self._series_with_dead_height()
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        hs = [h for _, h in fit.dead_height_profile(G_of_f)]
        assert max(hs) - min(hs) < 1e-5

    def test_a_sequence_aligned_to_the_slopes_also_works(self):
        members, G_of_f = self._series_with_dead_height()
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        seq = [G_of_f[min(G_of_f, key=lambda k: abs(k - s.frequency_hz))]
               for s in fit.slopes]
        assert fit.dead_height_cm(seq) == pytest.approx(
            fit.dead_height_cm(G_of_f), rel=0.02)

    def test_an_unusable_conductance_still_returns_nan_rather_than_raising(self):
        members, _ = self._series_with_dead_height()
        fit = fit_geometry_series(members, L_gap_cm=L_GAP, L_stripe_cm=L_STRIPE)
        assert math.isnan(fit.dead_height_cm({}))
        assert math.isnan(fit.dead_height_cm("nonsense"))
        assert math.isnan(fit.dead_height_cm(None))
