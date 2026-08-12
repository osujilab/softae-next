"""The P.22 σ(t) statistics: τ, t_tol, and the measured noise floor.

The fitter's job is as much to **refuse** as to fit. A τ that comes back from a
window shorter than itself, or from a series that is only noise, becomes the
campaign's conditioning hold duration and then a night of off-equilibrium
spectra. Every refusal below is a case where a plausible number was available and
is deliberately not emitted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from softae.analysis.conditions import TEMPERATURE_MIXED, TEMPERATURE_UNAVAILABLE
from softae.analysis.equilibration import (
    DEFAULT_N_SETTLE,
    EXCLUDED_ABSENT,
    EXCLUDED_RAILED,
    EXCLUDED_SIGMA_NULL,
    R1_AGREEMENT_TOL_REL,
    REFUSAL_NO_MODEL,
    REFUSAL_NOISE_DOMINATED,
    REFUSAL_NON_MONOTONIC,
    REFUSAL_R1_UNAVAILABLE,
    REFUSAL_SIGMA_UNAVAILABLE,
    REFUSAL_TOO_FEW_POINTS,
    REFUSAL_WINDOW_SHORTER_THAN_TAU,
    RoundFit,
    SettleTracker,
    add_r1_diagnostic,
    endorse_tolerance,
    fit_equilibration,
    fit_run,
    is_railed,
    load_round_fits,
    load_sigma_series,
    make_equilibration_fitter,
    noise_floor,
    r1_conductance,
    r1_lower_bound_ohms,
    series_temperature,
    session_drift,
    settle_check,
    settling_time,
    window_noise_floor,
)
from softae.errors import AnalysisError


def _relaxation(tau_s: float, *, span_s: float, step_s: float,
                sigma_inf: float = 1.0e-4, sigma_0: float = 2.0e-4):
    """A noiseless σ(t) = σ_∞ + (σ₀ − σ_∞)·exp(−t/τ)."""
    times = np.arange(0.0, span_s + step_s / 2, step_s)
    sigmas = sigma_inf + (sigma_0 - sigma_inf) * np.exp(-times / tau_s)
    return list(times), list(sigmas)


# ── The fit, and the four digits ─────────────────────────────────────────────

class TestExponentialFit:
    def test_the_exponential_fitter_recovers_a_known_tau_from_a_synthetic_series_to_four_digits(self):
        # curve_fit's default step is unit-scaled and tau lives in the 1e2-1e4 s
        # range, so an unscaled fit wanders and then reports convergence. The fit
        # is done in units of a seed taken from the model-free t_tol.
        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        result = fit_equilibration("exponential", times, sigmas, channel=3)

        assert result.fit_success, result.refusal
        assert result.tau_s == pytest.approx(600.0, rel=1e-4)
        assert result.sigma_inf == pytest.approx(1.0e-4, rel=1e-4)
        assert result.r_squared > 0.9999

    def test_a_tau_far_from_unity_is_still_recovered_which_is_the_scaling_bug_pinned(self):
        # 8000 s is four orders off curve_fit's unit default; without the rescale
        # this is where the fit silently stops moving.
        times, sigmas = _relaxation(8000.0, span_s=40000.0, step_s=1000.0)
        result = fit_equilibration("exponential", times, sigmas)
        assert result.fit_success, result.refusal
        assert result.tau_s == pytest.approx(8000.0, rel=1e-4)


# ── The refusals ─────────────────────────────────────────────────────────────

class TestRefusals:
    def test_the_fitter_refuses_a_series_shorter_than_two_time_constants_rather_than_extrapolating(self):
        times, sigmas = _relaxation(3000.0, span_s=1500.0, step_s=100.0)
        result = fit_equilibration("exponential", times, sigmas)

        assert not result.fit_success
        assert result.refusal == REFUSAL_WINDOW_SHORTER_THAN_TAU
        assert math.isnan(result.tau_s), "a refused fit must emit no number"

    def test_the_fitter_refuses_a_noise_dominated_series_rather_than_reporting_tau_zero(self):
        rng = np.random.default_rng(20260810)
        times = list(np.arange(0.0, 1500.0, 100.0))
        sigmas = list(1.0e-4 + rng.normal(0.0, 1.0e-6, size=len(times)))
        result = fit_equilibration("exponential", times, sigmas)

        assert not result.fit_success
        assert result.refusal == REFUSAL_NOISE_DOMINATED
        assert math.isnan(result.tau_s)

    def test_the_fitter_refuses_when_any_joined_sigma_is_null_rather_than_fitting_a_nan(self):
        # P.20 Stage B stores NULL when the geometry is absent; the loader passes
        # that through as NaN precisely so it can be refused here.
        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        sigmas[7] = float("nan")
        result = fit_equilibration("exponential", times, sigmas)

        assert not result.fit_success
        assert result.refusal == REFUSAL_SIGMA_UNAVAILABLE

    def test_the_fitter_refuses_four_points_because_three_parameters_fit_any_tau(self):
        times, sigmas = _relaxation(600.0, span_s=1800.0, step_s=600.0)
        assert len(times) == 4
        result = fit_equilibration("exponential", times, sigmas)
        assert result.refusal == REFUSAL_TOO_FEW_POINTS

    def test_min_points_for_tau_is_the_exact_boundary_the_acquisition_side_imports(self):
        # `settle_floor_rounds` and `EquilibrationConfig.validate` both fold this
        # constant into how long a setpoint runs, by importing it rather than
        # restating 5. So the number has to be the REAL boundary of the refusal
        # here: one point below it refuses, and at it the refusal is something
        # else or nothing at all.
        from softae.analysis.equilibration import MIN_POINTS_FOR_TAU

        step = 600.0
        short = _relaxation(600.0, span_s=step * (MIN_POINTS_FOR_TAU - 2),
                            step_s=step)
        exact = _relaxation(600.0, span_s=step * (MIN_POINTS_FOR_TAU - 1),
                            step_s=step)
        assert (len(short[0]), len(exact[0])) == (MIN_POINTS_FOR_TAU - 1,
                                                  MIN_POINTS_FOR_TAU)

        assert fit_equilibration("exponential", *short).refusal \
            == REFUSAL_TOO_FEW_POINTS
        assert fit_equilibration("exponential", *exact).refusal \
            != REFUSAL_TOO_FEW_POINTS

    def test_the_fitter_refuses_two_relaxations_because_the_model_describes_one(self):
        # Down, up, down: two reversals in the smoothed first difference. The
        # one-relaxation model cannot describe this and says so.
        times = list(np.arange(0.0, 4000.0, 100.0))
        sigmas = [1.0e-4 + 1.0e-4 * math.exp(-t / 400.0)
                  + 0.6e-4 * (1 - math.exp(-max(0.0, t - 1200.0) / 300.0))
                  - 0.6e-4 * (1 - math.exp(-max(0.0, t - 2600.0) / 300.0))
                  for t in times]
        result = fit_equilibration("exponential", times, sigmas)
        assert result.refusal == REFUSAL_NON_MONOTONIC

    def test_a_single_reversal_is_inside_the_threshold_the_spec_set(self):
        # The rule is "> 1 sign change", which leaves one unit of slack. With the
        # sub-noise filter already removing spurious crossings, that slack lets a
        # genuine down-then-up series past the monotonicity gate. Pinned as the
        # specified behaviour rather than silently tightened.
        times = list(np.arange(0.0, 3000.0, 100.0))
        sigmas = [1.0e-4 + 1.0e-4 * math.exp(-t / 400.0)
                  + 0.6e-4 * (1 - math.exp(-max(0.0, t - 1200.0) / 400.0))
                  for t in times]
        result = fit_equilibration("exponential", times, sigmas)
        assert result.refusal != REFUSAL_NON_MONOTONIC

    def test_a_refusal_never_raises_because_an_unfittable_series_is_an_outcome(self):
        result = fit_equilibration("exponential", [0.0, 1.0], [1.0, 1.0])
        assert not result.fit_success and result.refusal

    def test_an_unknown_model_raises_because_that_is_a_caller_bug_not_an_outcome(self):
        with pytest.raises(AnalysisError):
            make_equilibration_fitter("stretched_exponential")


# ── tau from R1: the cell constant has to cancel ─────────────────────────────

class TestR1Diagnostic:
    def test_tau_from_r1_equals_tau_from_sigma_because_the_cell_constant_cancels(self):
        # sigma = K/R1 with K constant during a hold, so 1/R1 is sigma times a
        # constant and an exponential's tau is invariant under that scaling. The
        # identity is EXACT, which is what makes a disagreement diagnostic rather
        # than expected.
        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        cell_constant = 57.3                       # K, /cm -- any value will do
        r1 = [cell_constant / s for s in sigmas]

        result = fit_equilibration("exponential", times, sigmas, channel=3)
        add_r1_diagnostic(result, times, r1)

        assert result.fit_success and result.r1_fit_success
        assert result.tau_r1_s == pytest.approx(result.tau_s, rel=1e-6)
        assert result.tau_agreement_rel == pytest.approx(0.0, abs=1e-6)
        assert result.r1_diagnostic_ok is True

    def test_the_diagnostic_is_independent_of_the_cell_constants_value(self):
        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        taus = []
        for k in (1.0, 57.3, 1.0e5):
            result = fit_equilibration("exponential", times, sigmas)
            add_r1_diagnostic(result, times, [k / s for s in sigmas])
            taus.append(result.tau_r1_s)
        assert taus[0] == pytest.approx(taus[1], rel=1e-6)
        assert taus[1] == pytest.approx(taus[2], rel=1e-6)

    def test_a_cell_constant_that_drifts_across_the_series_is_caught(self):
        # THE failure the diagnostic exists for: K is not constant, so sigma and
        # R1 no longer describe the same relaxation. Nothing else in the pipeline
        # would notice.
        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        r1 = [57.3 / (s * math.exp(-t / 5000.0)) for t, s in zip(times, sigmas)]

        result = fit_equilibration("exponential", times, sigmas)
        add_r1_diagnostic(result, times, r1)

        assert result.fit_success
        assert result.r1_fit_success
        assert result.tau_agreement_rel > R1_AGREEMENT_TOL_REL
        assert result.r1_diagnostic_ok is False
        assert "SUSPECT" in result.describe_r1()

    def test_the_diagnostic_refuses_when_r1_is_null_and_never_stands_in_for_sigma(self):
        # R1 goes NULL exactly when a circuit fit failed. A diagnostic that
        # quietly fitted the surviving points would report agreement about a
        # series it did not see.
        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        r1 = [57.3 / s for s in sigmas]
        r1[4] = None

        result = fit_equilibration("exponential", times, sigmas)
        add_r1_diagnostic(result, times, r1)

        assert result.fit_success, "the sigma fit must be untouched"
        assert not result.r1_fit_success
        assert result.r1_refusal == REFUSAL_R1_UNAVAILABLE
        assert math.isnan(result.tau_r1_s)
        assert result.tau_agreement_rel is None
        assert result.r1_diagnostic_ok is None, \
            "'not checked' must not read as 'checked and fine'"

    def test_a_refused_sigma_fit_is_not_rescued_by_a_successful_r1_one(self):
        times, sigmas = _relaxation(3000.0, span_s=1500.0, step_s=100.0)
        result = fit_equilibration("exponential", times, sigmas)
        add_r1_diagnostic(result, times, [57.3 / s for s in sigmas])

        assert not result.fit_success
        assert result.refusal == REFUSAL_WINDOW_SHORTER_THAN_TAU
        assert math.isnan(result.tau_s)
        assert result.tau_agreement_rel is None

    def test_a_nonpositive_resistance_is_not_a_conductance(self):
        assert all(math.isnan(g) for g in r1_conductance([0.0, -1.0, None, "x"]))
        assert r1_conductance([2.0])[0] == pytest.approx(0.5)


# ── The model-free half ──────────────────────────────────────────────────────

class TestModelFree:
    def test_the_time_only_model_returns_a_tolerance_time_without_attempting_a_fit(self, monkeypatch):
        import scipy.optimize

        def _boom(*_a, **_kw):
            raise AssertionError("the 'none' model must not reach a fitter")

        monkeypatch.setattr(scipy.optimize, "curve_fit", _boom)

        times, sigmas = _relaxation(600.0, span_s=3000.0, step_s=120.0)
        result = fit_equilibration("none", times, sigmas)

        assert result.t_tol_s is not None and result.t_tol_s > 0
        assert result.sigma_settled == pytest.approx(1.0e-4, rel=2e-2)
        assert math.isnan(result.tau_s)
        assert result.refusal == REFUSAL_NO_MODEL

    def test_a_series_that_never_settles_reports_no_tolerance_time_rather_than_the_last_one(self):
        times = list(np.arange(0.0, 1500.0, 100.0))
        sigmas = [1.0e-4 * (1.0 + 0.05 * i) for i in range(len(times))]
        assert settling_time(times, sigmas, tol_rel=0.02) is None

    def test_the_noise_floor_is_computed_from_the_settled_tail_and_labelled_an_upper_bound(self):
        tail = [1.0, 1.02, 0.98, 1.01, 0.99]
        sigmas = [5.0, 3.0, 2.0] + tail          # the drift before it settled
        floor = noise_floor(sigmas, n_settle=5)

        assert floor == pytest.approx(float(np.std(tail, ddof=1)) / 1.0)
        # It is measurement noise PLUS residual drift; the two cannot be separated
        # without a repeat taken with zero time between them.
        result = fit_equilibration("none", list(range(8)), sigmas, n_settle=5)
        assert result.noise_floor_rel == pytest.approx(floor)
        assert result.noise_floor_is_upper_bound is True

    def test_a_tail_of_one_point_has_no_measurable_noise_floor(self):
        assert noise_floor([1.0], n_settle=DEFAULT_N_SETTLE) is None


class TestSessionDrift:
    """The idea the retired ``Longest`` anchors were conflated with.

    A settled block at the start of the up leg and one at the end of the down leg
    sit at the same nominal condition, so their disagreement is session drift —
    and it doubles as the retrace evidence at the reference point. It now costs no
    instrument time: those blocks are ordinary series tails.
    """

    def _result(self, leg, sp_idx, sigma, *, floor=0.005, channel=1):
        from softae.analysis.equilibration import EquilibrationResult

        return EquilibrationResult(channel=channel, leg=leg, setpoint_index=sp_idx,
                                   sigma_settled=sigma, noise_floor_rel=floor)

    def test_a_drift_larger_than_the_measured_noise_floor_is_flagged(self):
        rows = session_drift([
            self._result("up", 0, 1.0e-4),
            self._result("down", 3, 1.2e-4),
        ])
        assert len(rows) == 1
        assert rows[0]["drift_rel"] == pytest.approx(0.2 / 1.1, rel=1e-6)
        assert rows[0]["significant"] is True
        assert rows[0]["end"]["setpoint_index"] == 3

    def test_a_drift_inside_the_noise_floor_is_not_evidence_of_anything(self):
        rows = session_drift([
            self._result("up", 0, 1.000e-4, floor=0.05),
            self._result("down", 3, 1.002e-4, floor=0.05),
        ])
        assert rows[0]["significant"] is False

    def test_without_a_measured_noise_floor_the_verdict_is_unknown_not_stable(self):
        rows = session_drift([
            self._result("up", 0, 1.0e-4, floor=None),
            self._result("down", 3, 1.0e-4, floor=None),
        ])
        assert rows[0]["significant"] is None

    def test_a_run_with_only_one_leg_has_no_drift_evidence_and_says_so(self):
        assert session_drift([self._result("up", 0, 1.0e-4)]) == []

    def test_each_channel_is_paired_with_itself(self):
        rows = session_drift([
            self._result("up", 0, 1.0e-4, channel=1),
            self._result("down", 3, 1.0e-4, channel=1),
            self._result("up", 0, 5.0e-4, channel=2),
            self._result("down", 3, 9.0e-4, channel=2),
        ])
        assert [r["channel"] for r in rows] == [1, 2]
        assert rows[0]["significant"] is False
        assert rows[1]["significant"] is True


class TestToleranceEndorsement:
    def test_report_refuses_to_endorse_a_conditioning_tolerance_below_the_measured_noise_floor(self):
        ok, why = endorse_tolerance(0.005, 0.016)
        assert not ok
        assert "below the measured noise floor" in why.lower()

    def test_a_tolerance_above_the_floor_is_endorsed(self):
        ok, _ = endorse_tolerance(0.05, 0.016)
        assert ok

    def test_an_unmeasured_noise_floor_is_not_an_endorsement(self):
        ok, why = endorse_tolerance(0.05, None)
        assert not ok and "no noise floor" in why


# ── The join, against a real store ───────────────────────────────────────────

def _store_with_series(tmp_path, *, sigmas, stems, r1s=None,
                       stage_temp_pv_C=45.4, stage_temp_sp_C=45.0, chamber_air_C=29.1):
    """A store whose ``conditions`` rows carry BOTH thermometers.

    The defaults are the real run's 45 °C setpoint: the stage controller reads
    45.4 °C while the humidity sensor's air probe reads 29.1 °C at the same moment.
    They are deliberately far apart so a test cannot pass by reading either one.
    """
    from softae.core.data_store import DataStore

    store = DataStore(str(tmp_path / "proj"), db_filename="test.db")
    run_id = store.start_run("equilibration_characterization", mode="characterization")
    resistances = list(r1s) if r1s is not None else [None] * len(stems)
    for i, (stem, sigma) in enumerate(zip(stems, sigmas)):
        cur = store._conn.execute(
            "INSERT INTO measurements (run_id, channel, timestamp, eis_file_path) "
            "VALUES (?, ?, ?, ?)",
            (run_id, 1, f"2026-08-10T00:{i:02d}:00", f"eis/{run_id}/{stem}.txt"),
        )
        mid = cur.lastrowid
        if sigma is not None:
            store._conn.execute(
                "INSERT INTO fit_results (measurement_id, run_id, model_name, "
                "sigma_S_per_cm, R1, fitted_at) "
                "VALUES (?, ?, 'simpleSalt', ?, ?, 'now')",
                (mid, run_id, sigma, resistances[i]),
            )
        store._conn.execute(
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp, "
            "stage_temp_sp_C, stage_temp_pv_C, chamber_air_C, rh_sp_pct, rh_pv_pct) "
            "VALUES (?, ?, 'measurement', ?, ?, ?, ?, 15.0, 15.4)",
            (mid, run_id, f"2026-08-10T00:{i:02d}:00",
             stage_temp_sp_C, stage_temp_pv_C, chamber_air_C),
        )
    store._conn.commit()
    return store, run_id


class TestSeriesReconstruction:
    def test_the_sigma_series_is_reconstructed_by_joining_measurements_fits_and_conditions(
            self, tmp_path):
        # No schema change: every column here is already written by the router on
        # an ordinary EIS shot. Only the COORDINATE comes from the sidecar, because
        # router.handle reads a fixed list of tags and drops the rest.
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(4)]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, 1.5e-4, 1.2e-4, 1.1e-4], stems=stems)
        sidecar = {"points": [
            {"step_name": f"eq_ch1_Lup_S0_R{i}", "channel": 1, "leg": "up",
             "setpoint_index": 0, "round_index": i, "kind": "series",
             "t_since_hold_s": 120.0 * i}
            for i in range(4)
        ]}

        series = load_sigma_series(store, run_id, sidecar)
        try:
            key = (1, "up", 0, "series")
            assert list(series) == [key]
            points = series[key]
            assert [p["t_since_hold_s"] for p in points] == [0.0, 120.0, 240.0, 360.0]
            assert [p["sigma"] for p in points] == [2.0e-4, 1.5e-4, 1.2e-4, 1.1e-4]
            assert points[0]["rh_pv_pct"] == pytest.approx(15.4)
            # The humidity sensor's own air temperature survives the join under a
            # name that says what it is — the RH physics still needs it.
            assert points[0]["chamber_air_C"] == pytest.approx(29.1)
        finally:
            store.close()

    def test_the_join_also_carries_r1_so_the_cross_check_costs_no_second_query(
            self, tmp_path):
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(3)]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, 1.5e-4, 1.2e-4], stems=stems,
            r1s=[1000.0, 1333.0, None])
        sidecar = {"points": [
            {"step_name": f"eq_ch1_Lup_S0_R{i}", "channel": 1, "leg": "up",
             "setpoint_index": 0, "round_index": i, "kind": "series",
             "t_since_hold_s": 120.0 * i}
            for i in range(3)
        ]}
        series = load_sigma_series(store, run_id, sidecar)
        try:
            assert [p["R1"] for p in series[(1, "up", 0, "series")]] == [
                1000.0, 1333.0, None]
        finally:
            store.close()

    def test_fit_run_attaches_the_cross_check_without_being_asked(self, tmp_path):
        # A consistency check nobody remembers to request is one that never runs.
        times, sigmas = _relaxation(400.0, span_s=1680.0, step_s=120.0)
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(len(times))]
        store, run_id = _store_with_series(
            tmp_path, sigmas=sigmas, stems=stems, r1s=[57.3 / s for s in sigmas])
        sidecar = {"points": [
            {"step_name": f"eq_ch1_Lup_S0_R{i}", "channel": 1, "leg": "up",
             "setpoint_index": 0, "round_index": i, "kind": "series",
             "t_since_hold_s": t}
            for i, t in enumerate(times)
        ]}
        try:
            results = fit_run(load_sigma_series(store, run_id, sidecar),
                              run_id=run_id)
            assert len(results) == 1
            assert results[0].fit_success and results[0].r1_fit_success
            assert results[0].tau_r1_s == pytest.approx(results[0].tau_s, rel=1e-6)
            assert results[0].r1_diagnostic_ok is True
        finally:
            store.close()

    def test_a_null_sigma_survives_the_join_so_the_fitter_can_refuse_it(self, tmp_path):
        # Dropping it here would hide a geometry gap behind a shorter but
        # apparently clean series.
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(3)]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, None, 1.2e-4], stems=stems)
        sidecar = {"points": [
            {"step_name": f"eq_ch1_Lup_S0_R{i}", "channel": 1, "leg": "up",
             "setpoint_index": 0, "round_index": i, "kind": "series",
             "t_since_hold_s": 120.0 * i}
            for i in range(3)
        ]}
        series = load_sigma_series(store, run_id, sidecar)
        try:
            assert [p["sigma"] for p in series[(1, "up", 0, "series")]] == [
                2.0e-4, None, 1.2e-4]
        finally:
            store.close()

    def test_the_up_leg_and_down_leg_points_at_the_same_temperature_are_separately_addressable(
            self, tmp_path):
        stems = ["eq_ch1_Lup_S1_R0_ch1", "eq_ch1_Ldown_S2_R0_ch1"]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, 1.9e-4], stems=stems)
        sidecar = {"points": [
            {"step_name": "eq_ch1_Lup_S1_R0", "channel": 1, "leg": "up",
             "setpoint_index": 1, "round_index": 0, "kind": "series",
             "t_since_hold_s": 0.0},
            {"step_name": "eq_ch1_Ldown_S2_R0", "channel": 1, "leg": "down",
             "setpoint_index": 2, "round_index": 0, "kind": "series",
             "t_since_hold_s": 0.0},
        ]}
        series = load_sigma_series(store, run_id, sidecar)
        try:
            assert set(series) == {(1, "up", 1, "series"), (1, "down", 2, "series")}
        finally:
            store.close()


# ── Which thermometer the series is labelled with ────────────────────────────
#
# `_SERIES_SQL` selected the humidity sensor's air probe — the column then named
# `c.temp_pv_C`, now `c.chamber_air_C` — until 2026-08-11, which on the production
# run was up to 42 C below the stage. These tests pin that the join now reads BOTH
# and says which one it used.

def _sidecar_rounds(n, *, leg="up", setpoint_index=0, channel=1, step_s=120.0):
    return {"points": [
        {"step_name": f"eq_ch{channel}_L{leg}_S{setpoint_index}_R{i}",
         "channel": channel, "leg": leg, "setpoint_index": setpoint_index,
         "round_index": i, "kind": "series", "t_since_hold_s": step_s * i}
        for i in range(n)
    ]}


class TestSeriesTemperatureSource:
    def test_load_sigma_series_every_point_carries_the_stage_pv_and_says_so(
            self, tmp_path):
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(3)]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, 1.5e-4, 1.2e-4], stems=stems,
            stage_temp_pv_C=85.0, stage_temp_sp_C=85.0, chamber_air_C=42.9)
        try:
            points = load_sigma_series(
                store, run_id, _sidecar_rounds(3))[(1, "up", 0, "series")]
            assert [p["temperature_source"] for p in points] == ["stage_pv"] * 3
            assert [p["temperature_C"] for p in points] == [85.0] * 3
            # And the air probe is still there, still 42.1 C wrong, still labelled.
            assert [p["chamber_air_C"] for p in points] == [42.9] * 3
        finally:
            store.close()

    def test_fit_run_result_carries_the_hold_temperature_with_its_source(
            self, tmp_path):
        times, sigmas = _relaxation(400.0, span_s=1680.0, step_s=120.0)
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(len(times))]
        store, run_id = _store_with_series(
            tmp_path, sigmas=sigmas, stems=stems,
            stage_temp_pv_C=65.0, stage_temp_sp_C=65.0, chamber_air_C=36.2)
        try:
            results = fit_run(
                load_sigma_series(store, run_id,
                                  _sidecar_rounds(len(times))), run_id=run_id)
            assert results[0].temperature_C == pytest.approx(65.0)
            assert results[0].temperature_source == "stage_pv"
            # An operator reading the line can tell which thermometer it was.
            assert "65.0C[stage_pv]" in results[0].describe()
        finally:
            store.close()

    def test_load_sigma_series_no_stage_reading_still_resolves_and_names_the_fallback(
            self, tmp_path):
        # Regression: a legacy run written before `stage_temp_pv_C` existed must
        # still produce a temperature — labelled as the lesser source, not dropped.
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(3)]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, 1.5e-4, 1.2e-4], stems=stems,
            stage_temp_pv_C=None, stage_temp_sp_C=45.0, chamber_air_C=29.1)
        try:
            points = load_sigma_series(
                store, run_id, _sidecar_rounds(3))[(1, "up", 0, "series")]
            assert [p["temperature_source"] for p in points] == ["stage_sp"] * 3
            assert [p["temperature_C"] for p in points] == [45.0] * 3
        finally:
            store.close()

    def test_load_sigma_series_no_stage_and_no_setpoint_falls_back_to_chamber_air(
            self, tmp_path):
        stems = [f"eq_ch1_Lup_S0_R{i}_ch1" for i in range(3)]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4, 1.5e-4, 1.2e-4], stems=stems,
            stage_temp_pv_C=None, stage_temp_sp_C=None, chamber_air_C=29.1)
        try:
            points = load_sigma_series(
                store, run_id, _sidecar_rounds(3))[(1, "up", 0, "series")]
            assert [p["temperature_source"] for p in points] == ["chamber_air"] * 3
        finally:
            store.close()


class TestSeriesTemperatureAggregate:
    @staticmethod
    def _points(pairs):
        return [{"temperature_C": t, "temperature_source": s} for t, s in pairs]

    def test_series_temperature_takes_the_median_so_one_approach_round_cannot_drag_it(
            self):
        # The first round is taken while the stage is still arriving at 85 C.
        value, source = series_temperature(
            self._points([(78.0, "stage_pv")] + [(85.0, "stage_pv")] * 8))
        assert value == pytest.approx(85.0)
        assert source == "stage_pv"

    def test_series_temperature_rounds_from_two_thermometers_are_labelled_mixed(self):
        value, source = series_temperature(
            self._points([(85.0, "stage_pv"), (85.0, "stage_pv"),
                          (42.9, "chamber_air")]))
        assert source == TEMPERATURE_MIXED
        assert np.isfinite(value)

    def test_series_temperature_nothing_resolvable_is_unavailable_not_zero(self):
        value, source = series_temperature(self._points(
            [(float("nan"), TEMPERATURE_UNAVAILABLE)] * 3))
        assert math.isnan(value)
        assert source == TEMPERATURE_UNAVAILABLE

    def test_series_temperature_a_lost_round_does_not_make_the_series_mixed(self):
        value, source = series_temperature(self._points([
            (85.0, "stage_pv"), (float("nan"), TEMPERATURE_UNAVAILABLE),
            (85.0, "stage_pv")]))
        assert value == pytest.approx(85.0)
        assert source == "stage_pv"


# ── The adaptive settle criterion ────────────────────────────────────────────
#
# Pure, so the whole rule is exercised here rather than through a simulated run:
# a window of fits in, a verdict out. The one that matters most is the railed
# case -- a railed fit returns the same number every round, and a constant is
# trivially "settled".

def _window(*rounds: dict) -> list[list[RoundFit]]:
    """``{channel: (sigma, R1)}`` per round -> the window the criterion reads."""
    return [[RoundFit(channel=ch, sigma=sigma, r1_ohms=r1)
             for ch, (sigma, r1) in sorted(spec.items())] for spec in rounds]


def _flat(channels, sigma=2.0e-4, r1=5.0e3, n=3, jitter=0.0):
    return _window(*[{ch: (sigma * (1.0 + jitter * i), r1) for ch in channels}
                     for i in range(n)])


class TestRailedFitDetection:
    def test_the_r1_lower_bound_is_read_off_the_circuit_model_not_written_down(self):
        # simpleSalt fits R0-CPE0-p(R1,C0) with R1 bounded below at 1e2 ohm, and
        # z_indices already names which parameter R1 is -- so the bound is derived
        # rather than restated, and cannot drift from the fitter's own.
        assert r1_lower_bound_ohms("simpleSalt") == pytest.approx(100.0)

    def test_an_unbounded_model_reports_no_bound_rather_than_a_plausible_one(self):
        # flexSalt declares bounds=None. "No railing check was possible" must not
        # be spelled the same as "nothing was railed".
        assert r1_lower_bound_ohms("flexSalt") is None
        assert r1_lower_bound_ohms("no such model") is None

    def test_a_fit_resting_on_the_bound_is_railed_and_one_above_it_is_not(self):
        assert is_railed(100.0, 100.0) is True
        assert is_railed(100.02, 100.0) is True      # trf stops near, not on
        assert is_railed(5.0e3, 100.0) is False

    def test_railing_cannot_be_asserted_without_a_bound_to_assert_it_against(self):
        assert is_railed(100.0, None) is False
        assert is_railed(None, 100.0) is False
        assert is_railed(float("nan"), 100.0) is False


class TestSettleParticipation:
    def test_settle_check_a_flat_window_of_good_channels_is_settled(self):
        check = settle_check(_flat([1, 2, 3]), tol_rel=0.10, min_channels=3,
                             r1_bound_ohms=100.0)
        assert check.evaluable and check.settled
        assert check.participating == [1, 2, 3]
        assert check.max_deviation_rel == pytest.approx(0.0)

    def test_settle_check_a_drifting_window_is_evaluable_but_not_settled(self):
        check = settle_check(_flat([1, 2, 3], jitter=0.30), tol_rel=0.10,
                             min_channels=3, r1_bound_ohms=100.0)
        assert check.evaluable and not check.settled
        assert check.max_deviation_rel > 0.10

    def test_settle_check_railed_channels_are_excluded_so_a_dead_board_never_settles(
            self):
        # THE critical case. Four channels railed at the simpleSalt R1 bound
        # return sigma = 0.5 S/cm on every round with success = 1. Perfectly
        # constant, therefore perfectly "settled" -- and the whole run would be
        # under-conditioned on the strength of four dead channels.
        railed = _flat([9, 10, 15, 16], sigma=0.5, r1=100.0)
        check = settle_check(railed, tol_rel=0.10, min_channels=3,
                             r1_bound_ohms=100.0)

        assert check.settled is False
        assert check.evaluable is False, "no evidence is not evidence of settling"
        assert check.participating == []
        assert set(check.excluded.values()) == {EXCLUDED_RAILED}

    def test_settle_check_a_null_sigma_channel_is_excluded_from_participation(self):
        window = _window(
            {1: (2.0e-4, 5.0e3), 2: (None, 5.0e3), 3: (2.0e-4, 5.0e3)},
            {1: (2.0e-4, 5.0e3), 2: (None, 5.0e3), 3: (2.0e-4, 5.0e3)},
            {1: (2.0e-4, 5.0e3), 2: (None, 5.0e3), 3: (2.0e-4, 5.0e3)},
        )
        check = settle_check(window, tol_rel=0.10, min_channels=2,
                             r1_bound_ohms=100.0)
        assert check.participating == [1, 3]
        assert check.excluded[2] == EXCLUDED_SIGMA_NULL

    def test_settle_check_a_channel_that_misses_one_round_cannot_carry_the_window(self):
        window = _window(
            {1: (2.0e-4, 5.0e3), 2: (2.0e-4, 5.0e3)},
            {1: (2.0e-4, 5.0e3)},
            {1: (2.0e-4, 5.0e3), 2: (2.0e-4, 5.0e3)},
        )
        check = settle_check(window, tol_rel=0.10, min_channels=1,
                             r1_bound_ohms=100.0)
        assert check.participating == [1]
        assert check.excluded[2] == EXCLUDED_ABSENT

    def test_settle_check_too_few_participants_is_not_evaluable_rather_than_unsettled(
            self):
        check = settle_check(_flat([1, 2]), tol_rel=0.10, min_channels=3,
                             r1_bound_ohms=100.0)
        assert check.evaluable is False and check.settled is False
        assert "cannot be evaluated" in check.reason

    def test_settle_check_an_empty_window_is_never_settled(self):
        check = settle_check([], tol_rel=0.10, min_channels=1)
        assert not check.evaluable and not check.settled


class TestSettleTracker:
    def test_tracker_says_nothing_until_it_has_a_full_window(self):
        tracker = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0)
        assert tracker.observe(_flat([1], n=1)[0]) is None
        assert tracker.observe(_flat([1], n=1)[0]) is None
        assert tracker.observe(_flat([1], n=1)[0]) is not None
        assert tracker.settled is True

    def test_tracker_a_disabled_criterion_never_settles_and_never_evaluates(self):
        tracker = SettleTracker(enabled=False, n_rounds=2, min_channels=1)
        for _ in range(5):
            tracker.observe(_flat([1], n=1)[0])
        assert tracker.settled is False
        assert tracker.outcome(stopped_early=False) == "disabled"

    def test_tracker_outcome_distinguishes_a_ceiling_from_an_unevaluable_setpoint(self):
        moving = SettleTracker(n_rounds=3, min_channels=1, r1_bound_ohms=100.0)
        for i in range(4):
            moving.observe([RoundFit(1, 1.0e-4 * (1 + i), 5.0e3)])
        assert moving.outcome(stopped_early=False) == "ceiling"

        blind = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0)
        for _ in range(4):
            blind.observe([RoundFit(9, 0.5, 100.0)])
        assert blind.outcome(stopped_early=False) == "not_evaluable"

    def test_tracker_endorsement_refuses_a_tolerance_below_the_measured_noise_floor(
            self):
        # `endorse_tolerance` is reused rather than a second rule being written:
        # a tolerance under the floor can never be met by any number of rounds.
        tracker = SettleTracker(tol_rel=0.01, n_rounds=3, min_channels=1,
                                r1_bound_ohms=100.0)
        for i in range(3):
            tracker.observe([RoundFit(1, 2.0e-4 * (1 + 0.05 * (i % 2)), 5.0e3)])
        ok, why, floor = tracker.endorsement()

        assert ok is False
        assert "BELOW the measured noise floor" in why
        assert floor is not None and floor > 0.01

    def test_tracker_endorsement_is_none_rather_than_true_when_nothing_participated(
            self):
        tracker = SettleTracker(n_rounds=3, min_channels=3, r1_bound_ohms=100.0)
        for _ in range(3):
            tracker.observe([RoundFit(9, 0.5, 100.0)])
        ok, why, floor = tracker.endorsement()
        assert ok is None and floor is None and "no participating" in why

    def test_window_noise_floor_is_the_median_scatter_of_the_participants(self):
        window = _flat([1, 2], jitter=0.10)
        floor = window_noise_floor(window, [1, 2])
        assert floor is not None and floor > 0.0


class TestRoundFitsFromTheStore:
    def test_load_round_fits_returns_one_entry_per_requested_channel(self, tmp_path):
        # A channel with no row must come back as an all-None fit, not be dropped:
        # a shorter list reads to settle_check as a smaller board rather than as
        # missing evidence, and a smaller board can satisfy min_channels.
        stems = ["eq_ch1_Lup_S0_R0_ch1"]
        store, run_id = _store_with_series(
            tmp_path, sigmas=[2.0e-4], stems=stems, r1s=[5.0e3])
        try:
            fits = load_round_fits(store, run_id, {
                1: "eq_ch1_Lup_S0_R0", 2: "eq_ch2_Lup_S0_R0"})
            assert [f.channel for f in fits] == [1, 2]
            assert fits[0].sigma == pytest.approx(2.0e-4)
            assert fits[0].r1_ohms == pytest.approx(5.0e3)
            assert fits[1].sigma is None and fits[1].r1_ohms is None
        finally:
            store.close()

    def test_load_round_fits_carries_a_railed_r1_through_rather_than_hiding_it(
            self, tmp_path):
        store, run_id = _store_with_series(
            tmp_path, sigmas=[0.5], stems=["eq_ch1_Lup_S0_R0_ch1"], r1s=[100.0])
        try:
            fits = load_round_fits(store, run_id, {1: "eq_ch1_Lup_S0_R0"})
            assert is_railed(fits[0].r1_ohms, r1_lower_bound_ohms("simpleSalt"))
        finally:
            store.close()
