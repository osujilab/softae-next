"""Measurement quality gate (P4).

An unattended campaign turns every measurement into a decision. The failures
that matter are the ones that still produce plausible numbers: a dead channel
averages to a real float, and a non-converged fit still reports an R1 that
`σ = L/(R·w·t)` happily consumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.quality import (
    Verdict,
    compute_fit_quality,
    gate_raw_measurement,
    grade_fit,
    quality_config,
    validate_eis_trace,
    validate_raw_eis,
)


class _Trace:
    def __init__(self, freq, z_real, z_imag_neg):
        self.frequency = np.asarray(freq, dtype=float)
        self.z_real = np.asarray(z_real, dtype=float)
        self.z_imag_neg = np.asarray(z_imag_neg, dtype=float)


def _good_trace(n=20):
    return _Trace(np.logspace(5, 1, n), np.linspace(100, 200, n),
                  np.linspace(10, 50, n))


def _raw(n=20, *, z_real=None, z_imag=None):
    return np.column_stack([
        np.logspace(5, 1, n),
        np.linspace(100, 200, n) if z_real is None else z_real,
        np.linspace(10, 50, n) if z_imag is None else z_imag,
    ])


# ── Trace validation ─────────────────────────────────────────────────────────

class TestTraceValidation:
    def test_a_healthy_trace_is_accepted(self):
        assert validate_eis_trace(_good_trace()).verdict is Verdict.ACCEPT

    def test_empty_trace_is_rejected(self):
        assert validate_eis_trace(_Trace([], [], [])).verdict is Verdict.REJECT

    def test_all_non_finite_is_rejected(self):
        t = _good_trace()
        t.z_real = np.full_like(t.z_real, np.nan)
        assert validate_eis_trace(t).verdict is Verdict.REJECT

    def test_a_few_non_finite_points_are_flagged_not_rejected(self):
        """Losing two points of twenty degrades a fit; it does not invalidate it."""
        t = _good_trace()
        t.z_real[3] = np.nan
        t.z_real[7] = np.inf
        report = validate_eis_trace(t)
        assert report.verdict is Verdict.SUSPECT
        assert report.ok

    def test_too_few_points_to_fit_is_rejected(self):
        assert validate_eis_trace(_good_trace(n=4)).verdict is Verdict.REJECT

    def test_shorted_channel_is_rejected(self):
        t = _Trace(np.logspace(5, 1, 20), np.full(20, 1e-9), np.full(20, 1e-9))
        report = validate_eis_trace(t)
        assert report.verdict is Verdict.REJECT
        assert "short" in report.summary().lower() or "dead" in report.summary().lower()

    def test_open_circuit_is_rejected(self):
        t = _Trace(np.logspace(5, 1, 20), np.full(20, 1e15), np.full(20, 1e15))
        assert validate_eis_trace(t).verdict is Verdict.REJECT

    def test_a_stuck_instrument_is_rejected(self):
        """Identical |Z| at every frequency is not a spectrum."""
        t = _Trace(np.logspace(5, 1, 20), np.full(20, 150.0), np.full(20, 20.0))
        report = validate_eis_trace(t)
        assert report.verdict is Verdict.REJECT
        assert "stuck" in report.summary()

    def test_non_monotonic_frequency_is_flagged(self):
        """Interleaved sweeps would silently mix two measurements."""
        t = _good_trace()
        t.frequency = np.concatenate([t.frequency[:10][::-1], t.frequency[10:]])
        assert "monotonic" in validate_eis_trace(t).summary()

    def test_unusual_but_finite_physics_is_not_rejected(self):
        """A gate that rejects unfamiliar spectra would bias the campaign."""
        t = _Trace(np.logspace(5, 1, 20), np.linspace(1e5, 5e5, 20),
                   np.linspace(-100, 900, 20))
        assert validate_eis_trace(t).ok


# ── Raw-array adapter (the autonomous path) ──────────────────────────────────

class TestRawAdapter:
    def test_none_is_rejected(self):
        assert validate_raw_eis(None).verdict is Verdict.REJECT

    def test_a_healthy_raw_array_is_accepted(self):
        assert validate_raw_eis(_raw()).verdict is Verdict.ACCEPT

    def test_a_list_wrapped_array_is_unwrapped(self):
        """Matches the objective extractor's own convention."""
        assert validate_raw_eis([_raw()]).verdict is Verdict.ACCEPT

    def test_a_1d_result_is_flagged_not_rejected(self):
        """No Z' / Z'' structure to check — defer, do not invent a verdict."""
        report = validate_raw_eis(np.arange(20, dtype=float))
        assert report.verdict is Verdict.SUSPECT
        assert report.ok

    def test_garbage_is_rejected_without_raising(self):
        assert validate_raw_eis(object()).verdict is Verdict.REJECT


# ── Fit grading ──────────────────────────────────────────────────────────────

class TestFitGrading:
    def test_a_non_converged_fit_is_rejected(self):
        assert grade_fit({"r_squared": 0.99}, success=False).verdict is Verdict.REJECT

    def test_a_converged_but_wrong_fit_is_rejected(self):
        """Its R1 is a fitting artefact; the conductivity from it is not data."""
        report = grade_fit({"r_squared": 0.4, "residual_rms_pct": 3.0})
        assert report.verdict is Verdict.REJECT
        assert "R" in report.summary()

    def test_a_loose_fit_is_rejected_on_residuals(self):
        assert grade_fit(
            {"r_squared": 0.99, "residual_rms_pct": 40.0}).verdict is Verdict.REJECT

    def test_a_good_fit_is_accepted(self):
        assert grade_fit(
            {"r_squared": 0.995, "residual_rms_pct": 2.0}).verdict is Verdict.ACCEPT

    def test_missing_metrics_are_unknown_not_bad(self):
        """Discarding good data because quality is unrecorded would be worse."""
        report = grade_fit({})
        assert report.verdict is Verdict.SUSPECT
        assert report.ok


# ── Metrics ──────────────────────────────────────────────────────────────────

class TestFitMetrics:
    def test_a_perfect_fit_scores_perfectly(self):
        t = _good_trace()
        z_fit = t.z_real - 1j * t.z_imag_neg
        m = compute_fit_quality(t, z_fit, n_params=5)
        assert m["r_squared"] == pytest.approx(1.0)
        assert m["residual_rms_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_a_poor_fit_scores_poorly(self):
        t = _good_trace()
        z_fit = (t.z_real * 2.0) - 1j * (t.z_imag_neg * 2.0)
        m = compute_fit_quality(t, z_fit, n_params=5)
        assert m["r_squared"] < 0.95
        assert m["residual_rms_pct"] > 15.0

    def test_missing_z_fit_yields_no_metrics(self):
        assert compute_fit_quality(_good_trace(), None) == {}

    def test_metrics_survive_non_finite_points(self):
        t = _good_trace()
        z_fit = t.z_real - 1j * t.z_imag_neg
        z_fit[2] = np.nan
        assert compute_fit_quality(t, z_fit, n_params=5)["r_squared"] > 0.9


# ── Gate policy + config ─────────────────────────────────────────────────────

class TestGatePolicy:
    def test_disabled_gate_observes_without_discarding(self):
        """Watch it against real runs before giving it authority over data."""
        short = _raw(z_real=np.full(20, 1e-9), z_imag=np.full(20, 1e-9))
        report = gate_raw_measurement(short, config={"enabled": False})
        assert report.verdict is Verdict.SUSPECT
        assert report.ok
        assert "gate disabled" in report.summary()

    def test_enabled_gate_rejects(self):
        short = _raw(z_real=np.full(20, 1e-9), z_imag=np.full(20, 1e-9))
        assert gate_raw_measurement(
            short, config={"enabled": True}).verdict is Verdict.REJECT

    def test_a_healthy_measurement_passes_either_way(self):
        for enabled in (True, False):
            assert gate_raw_measurement(_raw(), config={"enabled": enabled}).ok

    def test_thresholds_come_from_config(self):
        cfg = quality_config({"min_r_squared": 0.5, "max_residual_pct": 99.0})
        assert cfg["min_r_squared"] == 0.5
        assert cfg["max_residual_pct"] == 99.0

    def test_bad_config_values_fall_back_to_defaults(self):
        assert quality_config({"min_r_squared": "not a number"})["min_r_squared"] > 0


# ── Integration with the unmeasured path (P0.1) ──────────────────────────────

class TestObjectiveIntegration:
    def test_a_rejected_measurement_becomes_unmeasured_not_zero(self):
        """It must reuse the existing unmeasured route, never report 0.0."""
        from softae.core.autonomous_wiring import _scalar_from_eis_raw

        short = _raw(z_real=np.full(20, 1e-9), z_imag=np.full(20, 1e-9))
        with _gate_enabled():
            assert _scalar_from_eis_raw(short) is None

    def test_a_healthy_measurement_still_produces_a_value(self):
        from softae.core.autonomous_wiring import _scalar_from_eis_raw

        with _gate_enabled():
            value = _scalar_from_eis_raw(_raw())
        assert value is not None and value > 0

    def test_a_broken_gate_does_not_discard_data(self):
        """The gate is a safeguard, not a dependency."""
        import softae.analysis.quality as q
        from softae.core.autonomous_wiring import _scalar_from_eis_raw

        original = q.gate_raw_measurement
        q.gate_raw_measurement = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("checker broken"))
        try:
            assert _scalar_from_eis_raw(_raw()) is not None
        finally:
            q.gate_raw_measurement = original


class _gate_enabled:
    """Force ``[quality] enabled = true`` for the duration of a block."""

    def __enter__(self):
        import softae.analysis.quality as q

        self._q = q
        self._orig = q.gate_raw_measurement
        q.gate_raw_measurement = lambda raw, **kw: self._orig(
            raw, config={"enabled": True})
        return self

    def __exit__(self, *exc):
        self._q.gate_raw_measurement = self._orig
        return False


class TestOpenCircuitScreenIsDerivedNotChosen:
    """The open-circuit threshold comes from the stray capacitance, not a constant.

    ``max_abs_z = 1e12`` is an absolute backstop that on this rig cannot fire — the
    whole stored corpus has a median |Z| below 5.4e7. A threshold nothing can reach is
    not a threshold, and the fix is not a smaller magic number: an open circuit is the
    fixture's stray capacitance with nothing across it, so the value is computable.
    """

    @staticmethod
    def _flat_trace(z_ohm, f_lo=3.91, f_hi=2.0e5, n=35):
        """A trace whose |Z| MEDIAN is exactly ``z_ohm``, over a realistic band.

        |Z| deliberately VARIES by +/-10%: a constant-|Z| trace trips the
        stuck-instrument check and is rejected before this screen is ever consulted,
        so a flat fixture would test the wrong branch. ``n`` is odd and the factors
        are log-symmetric about 1, so the median is exactly ``z_ohm``.
        """
        freq = np.geomspace(f_lo, f_hi, n)
        mag = float(z_ohm) * np.geomspace(1 / 1.1, 1.1, n)
        return _Trace(freq, mag, np.zeros(n))

    def test_the_threshold_tracks_the_stray_capacitance(self):
        from softae.analysis.quality import open_circuit_z_ohm

        freq = np.geomspace(3.91, 2.0e5, 34)
        # Z_open = 1/(2 pi f C): double the capacitance, halve the impedance.
        assert open_circuit_z_ohm(freq, 18.5e-12) == pytest.approx(
            2.0 * open_circuit_z_ohm(freq, 37.0e-12), rel=1e-9)

    def test_the_threshold_tracks_the_swept_band(self):
        # THE REASON THIS IS A FUNCTION AND NOT A CONSTANT. The same fixture presents a
        # different |Z| over a different sweep, so a scalar encodes one run's geometry
        # as though it were a property of the hardware.
        from softae.analysis.quality import open_circuit_z_ohm

        wide = open_circuit_z_ohm(np.geomspace(3.91, 2.0e5, 34))
        low = open_circuit_z_ohm(np.geomspace(0.016, 1.0e3, 34))
        assert low > wide * 10.0

    def test_an_unbridged_cell_is_flagged(self):
        from softae.analysis.quality import open_circuit_z_ohm

        freq = np.geomspace(3.91, 2.0e5, 34)
        z_open = open_circuit_z_ohm(freq)
        report = validate_eis_trace(self._flat_trace(z_open * 1.5))
        assert any("open-circuit" in i for i in report.issues)

    def test_it_screens_rather_than_rejects(self):
        # The population it flags is five-week-old dried films, and a film too
        # resistive to measure is an upper BOUND on sigma — a result, not a failure.
        # Rejecting would discard it. This is the whole posture of the change.
        from softae.analysis.quality import open_circuit_z_ohm

        freq = np.geomspace(3.91, 2.0e5, 34)
        report = validate_eis_trace(self._flat_trace(open_circuit_z_ohm(freq) * 1.5))
        assert report.verdict is Verdict.SUSPECT
        assert report.verdict is not Verdict.REJECT
        assert report.ok           # still usable, which is the point

    def test_an_ordinary_film_is_not_flagged(self):
        # Anti-vacuity: a screen that fires on everything is not a screen. The corpus
        # median is ~7.6e5, three decades under the open-circuit reading.
        report = validate_eis_trace(self._flat_trace(7.6e5))
        assert not any("open-circuit" in i for i in report.issues)
        assert report.verdict is Verdict.ACCEPT

    def test_the_threshold_is_reported_so_a_reader_can_check_it(self):
        report = validate_eis_trace(self._flat_trace(7.6e5))
        assert report.metrics["z_open_circuit"] == pytest.approx(9.73e6, rel=0.02)

    def test_an_unusable_band_yields_no_opinion_rather_than_a_passing_threshold(self):
        # nan must mean "no opinion". Were it read as a threshold, `z_med >= nan` is
        # False and the screen would silently never fire — the failure this whole
        # exercise is about, reintroduced one level down.
        from softae.analysis.quality import open_circuit_z_ohm

        assert np.isnan(open_circuit_z_ohm([1.0]))
        assert np.isnan(open_circuit_z_ohm([3.91, 2e5], stray_c_f=0.0))
        # A degenerate band is NOT a no-opinion case: every point at one frequency
        # still has a well-defined open-circuit |Z| at that frequency.
        assert np.isfinite(open_circuit_z_ohm([1.0] * 12))
        # No usable frequency at all is. The trace is refused for other reasons; what
        # matters is that no threshold is invented for it.
        n = 35
        mag = 7.6e5 * np.geomspace(1 / 1.1, 1.1, n)
        report = validate_eis_trace(_Trace(np.full(n, -1.0), mag, np.zeros(n)))
        assert "z_open_circuit" not in report.metrics

    def test_the_stray_comes_from_the_instrument_section_not_a_quality_copy(self):
        # One physical quantity, one home. A `[quality]` copy would be free to drift
        # from the blanks that produced it.
        cfg = quality_config({})
        assert cfg["stray_c_f"] == pytest.approx(18.5e-12, rel=1e-6)
