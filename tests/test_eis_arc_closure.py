"""Whether the impedance semicircle closed inside the swept window.

The synthetic cases pin the rule; the four rig cases pin it against the
instrument. Both are needed: a rule that only ever meets constructed spectra
tests its own author, and archived spectra alone cannot say what *should* have
happened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.arc import (
    CLOSED,
    NOISE_TOL_REL,
    OPEN,
    TRAILING_POINTS,
    UNKNOWN,
    ArcClosure,
    _peak_prominence,
    annotate_arc_closure,
    arc_closure,
)
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant
from tests.eis_synthetic import as_eis_result, reference_spectrum

CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")

#: The characterisation run, read only. Skipped where it is not present, because
#: the rest of the suite must stay runnable on a machine that never saw the rig.
RUN_DIR = Path("C:/Users/Osuji/softae_data/runs/"
               "20260811T023757Z_equilibration_characterization/data/eis")


def semicircle(f_peak: float = 1.0e3, R: float = 1.0e5,
               f_lo: float = 20.0, f_hi: float = 2.0e5, npts: int = 25):
    """A bare Debye arc: ``−Z″`` peaks at *f_peak* and falls away either side."""
    f = np.logspace(np.log10(f_hi), np.log10(f_lo), npts)
    x = f / f_peak
    return f, R * x / (1.0 + x**2)


class TestArcClosureReadsTheShapeOfTheSweep:
    def test_arc_closure_full_semicircle_reads_closed(self):
        f, y = semicircle()
        arc = arc_closure(f, y)
        assert arc.state == CLOSED
        assert arc.f_peak_hz > arc.f_low_hz

    def test_arc_closure_truncated_semicircle_reads_open(self):
        # Same arc, sweep stopped an order of magnitude above the peak: −Z″ is
        # still climbing when the instrument runs out of frequency.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        arc = arc_closure(f, y)
        assert arc.state == OPEN
        assert arc.f_peak_hz == pytest.approx(arc.f_low_hz)

    def test_arc_closure_open_arc_carries_phase_at_the_sweep_floor_as_severity(self):
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        phase = np.full(f.size, -80.0)
        assert arc_closure(f, y, phase).phase_low_deg == pytest.approx(-80.0)

    def test_arc_closure_without_phase_still_decides_the_state(self):
        # Severity is unavailable, the verdict is not: the peak decides it.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        arc = arc_closure(f, y)
        assert arc.state == OPEN
        assert arc.phase_low_deg != arc.phase_low_deg          # NaN, not 0°

    def test_arc_closure_sweep_order_does_not_change_the_verdict(self):
        f, y = semicircle()
        assert arc_closure(f[::-1], y[::-1]).state == arc_closure(f, y).state


class TestArcClosureIsRobustToOneNoisyPoint:
    """The sweep floor is the noisiest point in the spectrum, and it is also the
    one ``argmax`` is most likely to pick. A guard that did not exist here would
    make every low-frequency outlier read as a physical finding."""

    def test_arc_closure_single_spike_at_the_sweep_floor_stays_closed(self):
        f, y = semicircle()
        y[-1] = y.max() * 3.0            # f ascends last in this descending sweep
        arc = arc_closure(f, y)
        assert arc.state == CLOSED
        assert "excursion" in arc.reason
        # The reported peak is the arc's, not the spike's.
        assert arc.f_peak_hz > arc.f_low_hz

    def test_arc_closure_genuine_rise_at_the_sweep_floor_is_not_suppressed(self):
        # The guard must not swallow the real thing it is guarding against.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        assert arc_closure(f, y).state == OPEN

    def test_arc_closure_noise_level_wobble_below_the_peak_still_reads_open(self):
        # A 1 % wobble two points up-sweep is scatter, not a peak.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        y[-2] *= 0.99
        assert arc_closure(f, y).state == OPEN


class TestArcClosureJudgesTheUsablePointsRatherThanRefusingWhole:
    """``ch22_003``: 52 usable points of 53, ``fit_ok=True``, ``arc_closed=False``,
    ``reason='non-finite point in the sweep'``.

    That pair of answers is what a hardware fault looks like — an unclosed reference
    electrode — and ch22 was read as one for two days. ``fit_circuit`` was cured by
    :func:`~softae.analysis.circuit_fitting.usable_points`; this is the other half,
    and these tests are what stops the refusal coming back."""

    def test_arc_closure_one_unusable_point_among_many_still_returns_a_verdict(self):
        f, y = semicircle(npts=53)
        y[3] = np.nan
        arc = arc_closure(f, y)
        assert arc.state == CLOSED
        assert arc.n_dropped == 1
        assert arc.f_peak_hz > arc.f_low_hz

    def test_arc_closure_one_unusable_point_does_not_move_the_verdict(self):
        # The excluded point is not near the peak, so the arc it is read off is the
        # same arc. A verdict that flipped here would mean the mask, not the rule,
        # was deciding.
        f, y = semicircle(npts=53)
        clean = arc_closure(f, y)
        y[3] = np.nan
        assert arc_closure(f, y).state == clean.state
        assert arc_closure(f, y).f_peak_hz == pytest.approx(clean.f_peak_hz)

    def test_arc_closure_open_verdict_survives_an_unusable_point_too(self):
        # Both directions, so the mask cannot be quietly biasing toward CLOSED.
        f, y = semicircle(f_peak=2.0, f_lo=20.0, npts=53)
        y[3] = np.nan
        arc = arc_closure(f, y)
        assert arc.state == OPEN
        assert arc.n_dropped == 1

    def test_arc_closure_non_positive_frequency_is_excluded_like_a_non_finite_one(self):
        # The shared predicate is `isfinite(f) & isfinite(Z) & (f > 0)`, not just
        # finiteness: `band_below_apex_decades` is `log10(f_apex / f_low)`, and a
        # zero floor makes that infinite rather than wrong-looking.
        f, y = semicircle(npts=53)
        f = f.copy()
        f[-1] = 0.0
        arc = arc_closure(f, y)
        assert arc.n_dropped == 1
        assert arc.f_low_hz > 0.0

    def test_arc_closure_phase_is_read_at_the_lowest_surviving_frequency(self):
        # Severity must describe a point the verdict actually saw. Masking `f` and
        # `y` but not `phase` would misalign the arrays and report a neighbour's.
        f, y = semicircle(f_peak=2.0, f_lo=20.0, npts=53)
        phase = np.linspace(-10.0, -80.0, f.size)      # f descends, so phase[-1] is the floor
        f = f.copy()
        f[-1] = np.nan
        arc = arc_closure(f, y, phase)
        assert arc.phase_low_deg == pytest.approx(phase[-2])

    def test_arc_closure_phase_of_a_kept_point_is_not_itself_a_reason_to_drop(self):
        # Phase is severity only and is documented as NaN-able. Admitting it to the
        # predicate would drop points for a field the state never consults.
        f, y = semicircle(npts=53)
        phase = np.full(f.size, -45.0)
        phase[7] = np.nan
        arc = arc_closure(f, y, phase)
        assert arc.n_dropped == 0
        assert arc.state == CLOSED

    def test_arc_closure_masked_verdict_is_still_order_invariant(self):
        f, y = semicircle(npts=53)
        y[3] = np.nan
        ascending, descending = arc_closure(f[::-1], y[::-1]), arc_closure(f, y)
        assert ascending.state == descending.state
        assert ascending.f_peak_hz == pytest.approx(descending.f_peak_hz)
        assert ascending.n_dropped == descending.n_dropped

    def test_arc_closure_reports_the_exclusion_in_the_operator_facing_detail(self):
        f, y = semicircle(npts=53)
        y[3] = np.nan
        assert "1 unusable point(s) excluded" in arc_closure(f, y).detail

    def test_arc_closure_dropped_count_reaches_the_gate_log_record(self):
        # `as_record`'s readers sum `n_dropped` across entries; it was a hardcoded
        # zero, which is the only reason no key had to be added for this.
        f, y = semicircle(npts=53)
        y[3] = np.nan
        assert arc_closure(f, y).as_record()["n_dropped"] == 1

    def test_engine_legacy_path_fits_and_judges_the_same_gappy_spectrum(self):
        # End to end on the shipped engine — the ch22_003 pair of answers, which
        # must now agree with each other instead of contradicting.
        f, Z = reference_spectrum()
        Z = Z.astype(complex).copy()
        Z[3] = complex(np.nan, np.nan)
        report = analyze_spectrum(as_eis_result(f, Z), cell=CELL, engine="legacy")
        assert report.fit.success is True
        assert report.fit.n_points_dropped == 1
        assert report.fit.arc_closure.state != UNKNOWN
        assert report.fit.arc_closure.n_dropped == 1


class TestArcClosureRefusesRatherThanGuesses:
    def test_arc_closure_too_few_points_reads_unknown(self):
        f, y = semicircle(npts=4)
        arc = arc_closure(f, y)
        assert arc.state == UNKNOWN
        assert arc.state != CLOSED
        assert "4 points" in arc.reason

    def test_arc_closure_remnant_below_min_points_reads_unknown(self):
        # Refusing must stay possible. What changed is *what* is refused: a sweep
        # whose survivors cannot support a verdict, not a sweep with one bad sample.
        f, y = semicircle()
        y[4:] = np.nan                       # 4 usable of 25, below min_points=5
        arc = arc_closure(f, y)
        assert arc.state == UNKNOWN
        assert "4 of 25 points are usable" in arc.reason
        assert arc.n_dropped == 21

    def test_arc_closure_wholly_non_finite_sweep_reads_unknown(self):
        f, y = semicircle()
        arc = arc_closure(f, np.full(y.size, np.nan))
        assert arc.state == UNKNOWN
        assert "0 of 25" in arc.reason

    def test_arc_closure_short_sweep_is_refused_as_short_not_as_unusable(self):
        # The two refusals are not the same finding and must not read alike: an
        # operator who chose a 4-point preset has a configuration problem, not a
        # data-quality one.
        f, y = semicircle(npts=4)
        arc = arc_closure(f, y)
        assert arc.reason == "4 points, need 5"
        assert arc.n_dropped == 0

    def test_arc_closure_flat_sweep_reads_unknown(self):
        f, _ = semicircle()
        assert arc_closure(f, np.ones(f.size)).state == UNKNOWN

    def test_arc_closure_mismatched_lengths_reads_unknown(self):
        f, y = semicircle()
        assert arc_closure(f, y[:-1]).state == UNKNOWN


class TestArcClosureAgainstTheRig:
    """Four spectra from run ``20260811T023757Z_equilibration_characterization``,
    round 7 of their block. ch1 is a low-conductivity channel whose arc never
    closes; ch14 is a high-conductivity one whose arc closes at both ends of the
    sweep. Read from the netCDF payloads, never written."""

    CASES = [
        ("eq_ch1_Lup_S0_R7_ch1", OPEN, 20.0, -45.1),
        ("eq_ch1_Ldown_S3_R7_ch1", OPEN, 20.0, -93.8),
        ("eq_ch14_Lup_S0_R7_ch14", CLOSED, 2000.0, -14.7),
        ("eq_ch14_Ldown_S3_R7_ch14", CLOSED, 43.09, -34.4),
    ]

    @pytest.mark.parametrize("stem,state,f_peak,phase", CASES)
    def test_arc_closure_stored_spectrum_matches_the_measured_verdict(
        self, stem, state, f_peak, phase
    ):
        xr = pytest.importorskip("xarray")
        path = RUN_DIR / f"{stem}.nc"
        if not path.exists():
            pytest.skip("characterisation run not present on this machine")
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            arc = arc_closure(ds["frequency_hz"].values, ds["z_imag_neg"].values,
                              ds["phase"].values)
        assert arc.state == state
        assert arc.f_peak_hz == pytest.approx(f_peak, rel=1e-3)
        assert arc.phase_low_deg == pytest.approx(phase, abs=0.05)
        assert arc.f_low_hz == pytest.approx(20.0, rel=1e-3)


class TestTheEngineAnnotatesWithoutDemoting:
    def test_engine_open_arc_does_not_fail_the_fit(self):
        # The regression pin against over-refusing. An extrapolated R₁ is a weaker
        # claim, not a failed measurement, and a third of every run is in that
        # state — demoting them would throw the cold end of the sweep away.
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, engine="legacy")
        assert report.fit.arc_closure.state == OPEN
        assert report.fit.success is True
        assert report.fit.error_msg == ""
        assert report.fit.R1 == report.fit.R1                  # not NaN'd
        assert report.sigma.mode == "value"

    def test_engine_open_arc_is_named_in_the_quality_issues(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, engine="legacy")
        assert any("did not close" in issue for issue in report.quality.issues)

    def test_engine_gated_path_annotates_the_fit_too(self):
        from softae.analysis.eis.settings import EISSettings, GateSettings

        report = analyze_spectrum(
            as_eis_result(*reference_spectrum()), cell=CELL,
            settings=EISSettings(engine="gated", gates=GateSettings(enabled=False)))
        assert report.fit.arc_closure.state in (OPEN, CLOSED)


class TestTheAnnotationIsPersisted:
    def test_record_fit_legacy_path_stores_the_arc_record(self, tmp_path):
        # The path that ships: `[eis] engine` is legacy, and `analysis/eis/router.py`
        # is what writes a campaign's fit rows. The record lands in the four columns,
        # read straight off the fit with no `report=` in the call.
        import json

        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "project")
        try:
            run_id = store.start_run("arc")
            eis = as_eis_result(*reference_spectrum())
            measurement_id = store.record_measurement(run_id, eis)
            fit = analyze_spectrum(eis, cell=CELL, engine="legacy").fit
            fit_id = store.record_fit(measurement_id, fit, L_cm=0.2, t_cm=0.015,
                                      w_cm=0.2)
            row = dict(store._conn.execute(
                "SELECT gate_log_json, parameters_json, success, gate_verdict, engine, "
                "arc_state, arc_f_peak_hz, arc_f_low_hz, arc_phase_low_deg "
                "FROM fit_results WHERE fit_id = ?", (fit_id,)).fetchone())
        finally:
            store.close()

        assert row["arc_state"] == OPEN
        assert row["arc_f_peak_hz"] == pytest.approx(row["arc_f_low_hz"])
        assert row["arc_f_low_hz"] == pytest.approx(20.0)
        assert row["arc_phase_low_deg"] < 0.0
        # The columns agree with the annotation the engine put on the fit — one
        # verdict, not a copy that can drift.
        assert row["arc_state"] == fit.arc_closure.state
        assert row["arc_f_peak_hz"] == pytest.approx(fit.arc_closure.f_peak_hz)
        # Everything else on the row is what it was before the annotation existed,
        # `gate_log_json` included: no report is passed, so it keeps its literal.
        assert row["gate_log_json"] == "[]"
        assert row["success"] == 1
        assert row["gate_verdict"] is None
        # `engine` is the one column here that is a claim rather than an absence, and
        # no report was passed, so nothing declared one. The `engine="legacy"` above
        # is an argument to `analyze_spectrum`, which `record_fit` never sees.
        assert row["engine"] == "unknown"
        assert json.loads(row["parameters_json"]) == pytest.approx(
            fit.parameters.tolist())


def arc_with_blocking_tail(
    f_apex: float = 1.0e3, apex_ohm: float = 5.3e4, tail_ohm_at_floor: float = 8.1e5,
    f_lo: float = 6.475, f_hi: float = 2.0e5, npts: int = 41,
):
    """An arc **under** a blocking tail — the case the interior fields exist for.

    Measured shape, and the numbers are the measured ones: an 8.1×10⁵ Ω tail at the
    sweep floor over a 5.3×10⁴ Ω arc apex, 15× larger. ``argmax`` of ``−Z″`` is the
    tail, so ``f_peak_hz`` is honestly the floor — while the arc worth putting 24
    dense points on sits two decades up-sweep.

    Local rather than in ``tests/eis_synthetic.py`` because that module is
    parallel-session's and this claim is read-only on it; ``semicircle`` above set
    the precedent for an arc generator living beside the tests that use it.
    """
    f = np.logspace(np.log10(f_hi), np.log10(f_lo), npts)
    x = f / f_apex
    arc = 2.0 * apex_ohm * x / (1.0 + x**2)          # peaks at apex_ohm, at f_apex
    return f, arc + tail_ohm_at_floor * f_lo / f     # −Z″ = 1/(ωC), a blocking tail


def arc_verdict_before_the_extension(freq, z_imag_neg, *, min_points: int = 5):
    """``(state, f_peak_hz)`` exactly as ``arc.py`` computed them before this task.

    A frozen copy, not an import: the point of a characterization test is that the
    baseline cannot move when the implementation does. If this ever has to change to
    keep :func:`arc_closure` agreeing with it, that *is* the finding — stored
    ``arc_state`` values would be moving, which is T7.9's operator-authorized
    territory and deliberately outside this change.
    """
    f = np.asarray(freq, dtype=float).ravel()
    y = np.asarray(z_imag_neg, dtype=float).ravel()
    if f.size != y.size or f.size < min_points:
        return UNKNOWN, float("nan")
    if not (np.isfinite(f).all() and np.isfinite(y).all()):
        return UNKNOWN, float("nan")
    order = np.argsort(f)
    f_s, y_s = f[order], y[order]
    if f_s[0] == f_s[-1] or y_s.max() == y_s.min():
        return UNKNOWN, float("nan")
    peak = int(np.argmax(y_s))
    if peak != 0:
        return CLOSED, float(f_s[peak])
    trailing = y_s[:TRAILING_POINTS]
    if np.any(np.diff(trailing) > NOISE_TOL_REL * np.abs(trailing[1:])):
        return CLOSED, float(f_s[1 + int(np.argmax(y_s[1:]))])
    return OPEN, float(f_s[0])


#: The epoch-6 boundary, as ``SCHEMA_EPOCHS`` version 6 (2026-08-26, data-epoch)
#: states it. Rows fitted **before** it were produced by the all-or-nothing
#: finiteness rule; rows fitted **on or after** it were produced by the masking one.
#:
#: Compared as a plain string against ``fit_results.fitted_at``, and that is sound
#: rather than lucky: measured over the shipped database, all 3619 rows are exactly
#: 32 characters of ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``, none NULL and not one
#: carrying any other UTC offset. Fixed width, zero-padded, single offset — so
#: lexicographic order *is* chronological order and no parsing is needed. A column
#: that mixed offsets would make this comparison quietly wrong and would have to be
#: parsed instead, which is why the format is asserted here in words.
EPOCH6_BOUNDARY = "2026-08-26"


def _peak_agrees(computed: float, stored: float | None) -> bool:
    """``arc_f_peak_hz`` round-trips a NaN peak as SQL NULL, so absent equals absent.

    ``_f_or_none`` is what writes the column and every UNKNOWN verdict carries a NaN
    peak, so a bare ``==`` would read a correctly stored refusal as a mismatch.
    :meth:`~TestTheExtensionMovedNothingThatWasAlreadyStored._replay` already makes
    the same both-absent allowance against the frozen baseline. On the shipped
    database this changes nothing today — all 91 fully-usable rows store a real
    number — it is what keeps the post-boundary branch below correct for the first
    genuinely unjudgeable spectrum the new rule refuses.
    """
    if stored is None:
        return computed != computed
    return computed == stored


def check_stored_arc_row(*, stored_state, stored_peak, fitted_at, arc, where) -> str:
    """Assert one stored ``fit_results`` row against a freshly computed *arc*.

    The partition is the claim, not a convenience. ``fitted_at`` says which rule
    wrote the row, and an epoch sanctions a difference between two rules — so it can
    only ever excuse a row that the *old* rule wrote. Returns ``'exact'``,
    ``'epoch6_judged'`` or ``'epoch6_refused'`` so callers can pin the population.
    """
    if fitted_at >= EPOCH6_BOUNDARY:
        # Written BY the masking rule. It must reproduce exactly — unusable points
        # and all. Without this branch a gappy row written next week would be waved
        # through the epoch relaxation below on the strength of a difference that
        # cannot exist, because only one rule ever ran on it.
        assert arc.state == stored_state, where
        assert _peak_agrees(arc.f_peak_hz, stored_peak), where
        return "exact"
    if arc.n_dropped == 0:
        # Pre-boundary and fully usable: the two rules read the identical array, so
        # this is the original assertion at its original strength.
        assert arc.state == stored_state, where
        assert _peak_agrees(arc.f_peak_hz, stored_peak), where
        return "exact"
    # Pre-boundary and gappy — the only population epoch 6 speaks about, and the
    # divergence is one-directional: the old rule refused every spectrum carrying an
    # unusable point, so the stored value can only be `unknown`. Anything else means
    # the mask is *moving* a judged row, which epoch 6 does not sanction.
    assert stored_state == UNKNOWN, where
    if arc.state == UNKNOWN:
        # Still refused, but now for the informative reason: the remnant rather
        # than a single bad sample.
        assert "usable" in arc.reason, where
        return "epoch6_refused"
    return "epoch6_judged"


class TestTheExtensionMovedNothingThatWasAlreadyStored:
    """A1/A11 — the additive-only constraint, which is what keeps this change out of
    T7.9's epoch-grade scope. ``arc_state`` and ``arc_f_peak_hz`` are populated
    columns; if the new fields could move them, the extension would need the
    operator authorisation T7.9 carries. They cannot, and these tests are the proof
    rather than the claim."""

    #: The characterisation run's spectra, read only.
    CORPUS = sorted(RUN_DIR.glob("*.nc"))
    #: The shipped DataStore, whose ``fit_results`` rows carry stored verdicts.
    DB = Path("C:/Users/Osuji/softae_data/db/softae.db")

    def _replay(self, paths):
        """Agreement with the frozen baseline, plus a *measured* account of the one
        class where this tree deliberately stopped agreeing.

        The finiteness change moves stored ``arc_state`` on exactly one class of
        spectrum: the ones carrying an unusable point, which the baseline refuses
        whole and this now judges on the remnant. That is the finding
        :func:`arc_verdict_before_the_extension` was written to surface, and it is
        surfaced here rather than papered over — the divergence is asserted to be
        one-directional (baseline UNKNOWN → a verdict) and confined to spectra with
        ``n_dropped > 0``. Everything with a fully usable sweep is pinned as before.

        Returns ``(diverged, judged)`` so the callers can pin the blast radius.
        """
        xr = pytest.importorskip("xarray")
        diverged = judged = 0
        for path in paths:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                f = ds["frequency_hz"].values
                y = ds["z_imag_neg"].values
            arc = arc_closure(f, y)
            expected_state, expected_peak = arc_verdict_before_the_extension(f, y)
            if arc.n_dropped:
                diverged += 1
                # One-directional: the baseline can only have refused. If it had a
                # verdict here, the mask would be *changing* a judged spectrum,
                # which is T7.9's territory and not this change's.
                assert expected_state == UNKNOWN, path.name
                if arc.state == UNKNOWN:
                    # Still refused, but for the informative reason.
                    assert "usable" in arc.reason, path.name
                else:
                    judged += 1
                continue
            assert arc.state == expected_state, path.name
            assert (arc.f_peak_hz == expected_peak
                    or (arc.f_peak_hz != arc.f_peak_hz
                        and expected_peak != expected_peak)), path.name
        return diverged, judged

    def test_arc_state_and_f_peak_unchanged_over_a_sample_of_the_corpus(self):
        # Every twelfth spectrum of the 1440, deterministically — the same property
        # the full sweep below asserts, at a twelfth of the I/O, so an ordinary
        # tier-1 run still meets real data.
        if not self.CORPUS:
            pytest.skip("characterisation run not present on this machine")
        diverged, judged = self._replay(self.CORPUS[::12])
        # Measured: 3 of the 120 sampled. Pinned rather than tolerated, because a
        # silent widening of this class is the thing that would be hard to notice.
        assert (diverged, judged) == (3, 3)

    @pytest.mark.slow
    def test_arc_state_and_f_peak_unchanged_over_the_stored_corpus(self):
        # All 1440. Slow because it is 1440 file opens (~37 s), which is I/O and not
        # a defect in the code under test; the sampled version above is the one that
        # runs by default.
        if not self.CORPUS:
            pytest.skip("characterisation run not present on this machine")
        diverged, judged = self._replay(self.CORPUS)
        # 33 of 1440 carry an unusable point; the smallest remnant is 9 of 25, so
        # every one of them clears `min_points=5` and is judged rather than refused.
        # This is the non-fallback path proved on real data, not on constructions.
        assert (diverged, judged) == (33, 33)

    def test_stored_arc_columns_are_reproduced_exactly_by_the_extended_function(self):
        """The other direction: not "agrees with a frozen copy" but "agrees with what
        is actually written down". These rows were computed by the un-extended
        function, so any drift would be visible here as a changed column.

        One class of drift is **sanctioned**, and only that one: ``SCHEMA_EPOCHS``
        version 6 (2026-08-26, data-epoch) records that ``arc_state`` may now hold a
        verdict where the old all-or-nothing finiteness rule stored ``unknown``.

        ``fitted_at`` splits the table into the two halves that claim needs, and the
        split is what keeps this test stable while the database grows. The
        **pre-boundary** half was written by the old rule, is frozen, and is the only
        half an epoch can excuse, so its populations are pinned exactly. The
        **post-boundary** half was written by the new rule, so it must reproduce
        exactly — gaps and all — and is guarded by equality rather than by a count,
        which holds at any size. Counting the growing half would be a number that goes
        red for a legitimate reason, and the predictable repair is to loosen it.
        """
        import sqlite3
        from collections import Counter

        from softae.analysis.eis_data import EISResult

        if not self.DB.exists():
            pytest.skip("no shipped DataStore on this machine")
        conn = sqlite3.connect(f"file:{self.DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT f.arc_state, f.arc_f_peak_hz, f.fitted_at, m.eis_file_path "
                "FROM fit_results f JOIN measurements m "
                "  ON m.measurement_id = f.measurement_id "
                "WHERE f.arc_state IS NOT NULL AND m.eis_file_path IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

        before: Counter[str] = Counter()
        checked = after = 0
        for row in rows:
            path = self.DB.parent.parent / row["eis_file_path"]
            if not path.exists():
                continue
            checked += 1
            result = EISResult.load(path)
            outcome = check_stored_arc_row(
                stored_state=row["arc_state"], stored_peak=row["arc_f_peak_hz"],
                fitted_at=row["fitted_at"], where=path.name,
                arc=arc_closure(result.frequency, result.z_imag_neg,
                                getattr(result, "phase", None)))
            if row["fitted_at"] >= EPOCH6_BOUNDARY:
                # Asserted exact inside, and deliberately NOT pinned: this half gains
                # a row every time the rig fits a spectrum.
                after += 1
            else:
                before[outcome] += 1
        if checked == 0:
            pytest.skip("no stored arc verdicts with a readable spectrum")
        # Measured against the shipped DataStore, not guessed: 101 joinable rows with
        # a readable spectrum, every one of them fitted between 2026-08-17 and
        # 2026-08-25 and so pre-boundary. 91 have a fully usable sweep; 10 carry
        # unusable points, and 9 of those now get a verdict where the old rule stored
        # `unknown`. The tenth keeps 2 of 34 points, below `min_points`, and is still
        # refused — the non-fallback refusal proved on real data rather than on a
        # construction. Pinned exactly, in the style of the `_replay` tests above,
        # because a silent widening of this class is what would otherwise go
        # unnoticed. These three numbers cannot move on their own: nothing writes to
        # a row after it is fitted, so a change here is a change in the rule.
        assert (before["exact"], before["epoch6_judged"],
                before["epoch6_refused"]) == (91, 9, 1), (
            f"{checked} rows checked, {after} of them post-boundary")

    def test_a_row_fitted_after_the_boundary_reproduces_exactly_unusable_points_and_all(
        self, tmp_path
    ):
        """The post-boundary half of the partition, which the shipped database cannot
        exercise: all 101 of its joinable arc rows were fitted between 2026-08-17 and
        2026-08-25, so the branch that *refuses* to relax has no real row to fire on
        and would be code proved by nothing.

        So one is constructed, in the one shape that matters — a spectrum carrying an
        unusable point, fitted now. ``record_fit`` stamps ``fitted_at`` off the clock,
        which is on or after the boundary by construction. Without the partition this
        row would be waved through the epoch branch on the strength of a difference
        between two rules, when only one rule ever touched it.
        """
        from softae.core.data_store import DataStore

        f, Z = reference_spectrum()
        Z = Z.astype(complex).copy()
        Z[3] = complex(np.nan, np.nan)
        eis = as_eis_result(f, Z)

        store = DataStore(tmp_path / "post_epoch6")
        try:
            run_id = store.start_run("arc-post-epoch6")
            measurement_id = store.record_measurement(run_id, eis)
            fit = analyze_spectrum(eis, cell=CELL, engine="legacy").fit
            fit_id = store.record_fit(measurement_id, fit, L_cm=0.2, t_cm=0.015,
                                      w_cm=0.2)
            row = dict(store._conn.execute(
                "SELECT arc_state, arc_f_peak_hz, fitted_at FROM fit_results "
                "WHERE fit_id = ?", (fit_id,)).fetchone())
        finally:
            store.close()

        arc = arc_closure(eis.frequency, eis.z_imag_neg, getattr(eis, "phase", None))
        # The two preconditions that make this the case the live table has none of.
        assert row["fitted_at"] >= EPOCH6_BOUNDARY, row["fitted_at"]
        assert arc.n_dropped > 0
        # And the stored verdict is a real one, so a relaxation that fired here would
        # be comparing the new rule against itself.
        assert row["arc_state"] != UNKNOWN
        assert check_stored_arc_row(
            stored_state=row["arc_state"], stored_peak=row["arc_f_peak_hz"],
            fitted_at=row["fitted_at"], arc=arc,
            where="constructed post-boundary row") == "exact"

    def test_as_record_is_unchanged(self):
        # The record travels in `fit_results.gate_log_json`, whose readers sum
        # `n_dropped` over every entry. The planner reads the live object, so the
        # new fields buy nothing here and adding keys would move stored JSON.
        f, y = arc_with_blocking_tail()
        record = arc_closure(f, y, np.full(f.size, -70.0)).as_record()
        assert set(record) == {"gate", "severity", "passed", "n_dropped", "detail",
                               "state", "f_peak_hz", "f_low_hz", "phase_low_deg"}

    def test_existing_positional_construction_still_binds_reason(self):
        # The two constructions inside `arc_closure` are positional; a field
        # inserted before `reason` would rebind it silently rather than fail.
        arc = ArcClosure(CLOSED, 1.0e3, 20.0, -45.0, "lone excursion at the sweep floor")
        assert arc.reason == "lone excursion at the sweep floor"
        assert arc.f_apex_interior_hz != arc.f_apex_interior_hz     # NaN by default


class TestTheInteriorApexIsForPlanningNotForTheVerdict:
    def test_interior_apex_found_above_a_dominant_tail(self):
        f, y = arc_with_blocking_tail()
        arc = arc_closure(f, y)
        # The verdict is unchanged and still honest: the maximum IS the floor.
        assert arc.f_peak_hz == pytest.approx(arc.f_low_hz)
        # And the arc a planner should aim at is two decades up-sweep of it.
        assert arc.f_apex_interior_hz == pytest.approx(1.0e3, rel=0.15)
        assert arc.f_apex_interior_hz > 100.0 * arc.f_low_hz

    def test_global_scaling_would_have_missed_that_arc(self):
        # Why the prominence is relative to the candidate's own height. Against the
        # spectrum's global maximum — which the tail owns — the real arc scores an
        # order of magnitude below the 0.10 cut and would be discarded.
        f, y = arc_with_blocking_tail()
        arc = arc_closure(f, y)
        k = int(np.argmin(np.abs(np.sort(f) - arc.f_apex_interior_hz)))
        prominence = _peak_prominence(np.asarray(y)[np.argsort(f)], k)
        assert arc.apex_prominence_rel > 0.10          # survives relative scaling
        assert prominence / np.max(y) < 0.10           # would not have survived global

    def test_endpoints_are_never_interior_apex_candidates(self):
        # `gate_valley_feature`'s precedent, on the sibling feature. A window edge
        # admitted as an extremum is how F15 happened, twice.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)       # still climbing at the floor
        arc = arc_closure(f, y)
        assert arc.state == OPEN
        assert arc.f_apex_interior_hz != arc.f_low_hz

    def test_new_fields_are_nan_when_no_interior_maximum(self):
        f, y = semicircle(f_peak=2.0, f_lo=20.0)       # monotonic across the window
        arc = arc_closure(f, y)
        for value in (arc.f_apex_interior_hz, arc.apex_prominence_rel,
                      arc.band_below_apex_decades):
            assert value != value

    @pytest.mark.parametrize("case", ["mismatched", "too_few", "remnant", "flat"])
    def test_new_fields_are_nan_on_every_unknown_path(self, case):
        f, y = arc_with_blocking_tail()
        if case == "mismatched":
            arc = arc_closure(f, y[:-1])
        elif case == "too_few":
            arc = arc_closure(f[:4], y[:4])
        elif case == "remnant":
            # Was `non_finite` with a single NaN, which no longer reaches UNKNOWN —
            # that spectrum is now judged, and `TestArcClosureJudgesTheUsablePoints…`
            # is where it went. What still refuses is a remnant too small to judge.
            y = y.copy()
            y[4:] = np.nan
            arc = arc_closure(f, y)
        else:
            arc = arc_closure(f, np.ones(f.size))
        assert arc.state == UNKNOWN
        assert arc.f_apex_interior_hz != arc.f_apex_interior_hz
        assert arc.apex_prominence_rel != arc.apex_prominence_rel
        assert arc.band_below_apex_decades != arc.band_below_apex_decades

    def test_new_fields_are_order_invariant(self):
        # Cheap, and it stops a later edit from computing the new fields before the
        # sort at arc.py:145 — which is the whole reason no separate ordering
        # regression test is needed for them.
        f, y = arc_with_blocking_tail()
        ascending = arc_closure(f[::-1], y[::-1])
        descending = arc_closure(f, y)
        assert ascending.f_apex_interior_hz == descending.f_apex_interior_hz
        assert ascending.apex_prominence_rel == descending.apex_prominence_rel
        assert ascending.band_below_apex_decades == descending.band_below_apex_decades

    def test_band_below_apex_decades_is_log10_apex_over_floor(self):
        f, y = arc_with_blocking_tail()
        arc = arc_closure(f, y)
        assert arc.band_below_apex_decades == pytest.approx(
            np.log10(arc.f_apex_interior_hz / arc.f_low_hz))
        assert arc.band_below_apex_decades > 0.0


class TestProminenceIsTheTopographicOne:
    #: Three peaks, hand-checkable: a saddle at 4 between peaks of 10 and 8, and a
    #: shoulder at 6 hanging off the tallest.
    Y = np.array([0.0, 10.0, 4.0, 8.0, 6.0, 7.0, 1.0])

    def test_prominence_matches_the_topographic_definition(self):
        # y[1] = 10 is the tallest: nothing higher either side, so both bases are
        # the array's own minima -> 10 - max(0.0, 1.0) = 9.0.
        assert _peak_prominence(self.Y, 1) == pytest.approx(9.0)
        # y[3] = 8 is walled in by the 10 on its left; left base is the 4 saddle,
        # right base is the 1 at the end -> 8 - max(4.0, 1.0) = 4.0.
        assert _peak_prominence(self.Y, 3) == pytest.approx(4.0)
        # y[5] = 7 is a shoulder: walled by the 8, its left base is the 6.
        assert _peak_prominence(self.Y, 5) == pytest.approx(1.0)

    def test_prominence_agrees_with_scipy_where_scipy_is_available(self):
        # Cross-check only — `arc.py` gains no scipy dependency from this, and the
        # test skips rather than requires it.
        signal = pytest.importorskip("scipy.signal")
        peaks = [1, 3, 5]
        expected = signal.peak_prominences(self.Y, peaks)[0]
        assert [_peak_prominence(self.Y, k) for k in peaks] == pytest.approx(expected)

    def test_the_tallest_interior_maximum_wins_when_several_qualify(self):
        f = np.logspace(5, 1, self.Y.size)             # descending, as the rig sweeps
        arc = arc_closure(f, self.Y)
        assert arc.f_apex_interior_hz == pytest.approx(float(f[1]))


class TestAnnotateArcClosureLeavesTheFitAlone:
    def test_annotate_arc_closure_touches_nothing_but_the_annotation(self):
        from softae.analysis.circuit_fitting import fit_circuit

        eis = as_eis_result(*reference_spectrum())
        fit = fit_circuit(eis, "simpleSalt")
        before = (fit.success, fit.R1, fit.error_msg, fit.parameters.copy())
        annotate_arc_closure(fit, eis)
        assert (fit.success, fit.R1, fit.error_msg) == before[:3]
        assert np.array_equal(fit.parameters, before[3])
