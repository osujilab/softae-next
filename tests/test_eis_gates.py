"""Admission gates (E0) — does this spectrum contain the physics we extract from it?

The expensive EIS failure is not a fit that fails; it is a fit that *succeeds* on a
spectrum containing no parallel conduction and hands the resulting number to a
campaign. These tests pin the discriminators that stop that, and — per framework
§8.3 — that each pathology is caught by the **intended** gate and no other.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.eis_synthetic import (
    dispersive_dielectric,
    hf_phase_artifact,
    log_frequencies,
    negative_real_part,
    over_range,
    pure_series_rc,
    reference_spectrum,
    stuck_instrument,
)
from softae.analysis.eis.gates import (
    BLOCK_POINT,
    BLOCK_SPECTRUM,
    FLAG,
    TOPOLOGY_TRIAD,
    GateResult,
    gate_cap_flatness,
    gate_finiteness,
    gate_hf_inductive,
    gate_kk_truncation,
    gate_magnitude,
    gate_min_points,
    gate_quadrant,
    gate_series_rc,
    gate_stuck_instrument,
    gate_tand_slope,
    run_gates,
)
from softae.analysis.eis.admittance import (
    MIN_FALLING_SEGMENT_POINTS,
    log_slope,
    parallel_branch_window,
)
from softae.analysis.eis.policy import build_context
from softae.analysis.eis.settings import GateSettings
from tests.test_eis_kk import RIG_FREQ


def _ctx(**overrides):
    gates = GateSettings(**overrides) if overrides else GateSettings()
    return build_context(gates=gates)


def _failed(results) -> set[str]:
    return {r.name for r in results if not r.passed}


class TestReferenceSpectrumPasses:
    def test_a_well_formed_blocking_cell_spectrum_passes_every_gate(self):
        f, Z = reference_spectrum()
        mask, results, _ = run_gates(f, Z, _ctx())
        assert _failed(results) == set()
        assert mask.all(), "no point should be dropped from a clean spectrum"

    def test_a_noisy_but_well_formed_spectrum_still_passes(self):
        f, Z = reference_spectrum(noise_pct=0.5)
        _, results, _ = run_gates(f, Z, _ctx())
        assert _failed(results) == set()


class TestTopologyAdmission:
    def test_a_pure_series_rc_is_rejected_because_no_fitting_recovers_a_conductance_that_is_absent(self):
        f, Z = pure_series_rc()
        _, results, _ = run_gates(f, Z, _ctx())
        assert "tand_slope" in _failed(results)

    def test_the_triad_runs_as_a_group_so_two_independent_formulations_confirm_each_other(self):
        # The printed runner breaks on the first failed block_spectrum, which would
        # mean series_rc_topology never evaluates once tand_slope has failed — and
        # §3.5.3 calls their agreement "the confirmation".
        f, Z = pure_series_rc()
        _, results, _ = run_gates(f, Z, _ctx())
        failed = _failed(results)
        assert {"tand_slope", "series_rc_topology"} <= failed
        assert len([r for r in results if r.name == "series_rc_topology"]) == 1

    def test_a_parallel_rc_passes_because_falling_tand_is_the_signature_of_the_measurand(self):
        f, Z = reference_spectrum()
        r = gate_tand_slope(f, Z, _ctx())
        assert r.passed
        assert r.metrics["tand_slope"] < GateSettings().tand_slope_max

    def test_the_slope_is_fitted_above_the_tand_peak_because_a_global_fit_rejects_good_data(self):
        # A blocking electrode makes tanδ non-monotonic: it rises through the
        # CPE-dominated low-frequency region before falling. Fitted globally this
        # spectrum reads about -0.24 and the spec's own -0.3 threshold rejects it.
        f, Z = reference_spectrum()
        tand = Z.real / np.abs(Z.imag)
        ok = np.isfinite(tand) & (tand > 0)
        global_slope = np.polyfit(np.log10(f[ok]), np.log10(tand[ok]), 1)[0]
        windowed = gate_tand_slope(f, Z, _ctx()).metrics["tand_slope"]

        assert global_slope > GateSettings().tand_slope_max, (
            "if this stops being true the deviation is no longer needed")
        assert windowed < global_slope
        assert windowed <= GateSettings().tand_slope_max

    def test_a_dispersive_dielectric_is_flagged_by_capacitance_flatness_and_nothing_else(self):
        f, Z = dispersive_dielectric()
        _, results, _ = run_gates(f, Z, _ctx())
        assert _failed(results) == {"cap_flatness"}

    def test_capacitance_flatness_is_advisory_because_escalating_it_needs_substrate_loss_quantified(self):
        f, Z = dispersive_dielectric()
        r = gate_cap_flatness(f, Z, _ctx())
        assert r.severity == FLAG
        assert not r.passed

    def test_an_ideal_capacitance_reads_flat_over_the_top_decade(self):
        f, Z = reference_spectrum()
        r = gate_cap_flatness(f, Z, _ctx())
        assert r.passed
        assert abs(r.metrics["cap_slope"]) <= GateSettings().cap_flatness_max

    def test_the_series_rc_test_agrees_with_the_loss_tangent_on_a_clean_spectrum(self):
        f, Z = reference_spectrum()
        assert gate_series_rc(f, Z, _ctx()).passed


class TestTandWindowShapes:
    """§3.5.1 — the slope window is the *falling segment*, not "above the argmax".

    ``tan δ`` takes three shapes on this fixture, and an ``argmax``-only anchor is
    correct for one of them. The other two are the discriminator's whole job:

    * **unimodal, peak in band** — the CPE limb rises, peaks, then falls to the top of
      the sweep. Anchor at the peak; today's window, and this class pins it unchanged.
    * **U-shaped, peak below band** — a larger ``R_bulk`` pushes the relaxation corner
      below 1.351 Hz, so what is left in band is the falling limb plus the
      high-frequency rise (``Z → R_series``, ``Im Z → 0``, ``tan δ → ∞``). ``argmax``
      then lands on the low-frequency *endpoint* and ``f >= f_peak`` is the entire
      band — returned by the **primary** path, since that mask clears ``min_points``,
      so nothing distinguishes it from a legitimate selection. This is the rig's
      normal condition at large ``R_bulk``, not an edge case.
    * **monotone rising** — a series parasitic. No falling segment exists, and the
      full-band fallback is load-bearing: the global fit then correctly returns ``+1``
      and the gate rejects. Narrowing this shape would break the discriminator.
    """

    @staticmethod
    def _tand(Z):
        """``tan δ`` computed independently of the module under test."""
        return np.asarray(Z).real / np.abs(np.asarray(Z).imag)

    @staticmethod
    def _falling_segment_of(*, points: int, rise_above: float):
        """``(f, Z)`` whose ``tan δ`` falling segment is exactly ``points`` long.

        Built in ``tan δ`` rather than from a circuit, because segment *length* is the
        quantity under test and no ``R_bulk`` sets it directly — reaching a chosen
        length through :func:`reference_spectrum` would mean solving for a corner
        frequency and would tie the fixture to the sweep grid. ``Z = tan δ − 1j``
        makes ``loss_tangent(Z)`` exactly ``tan δ``: the module note in
        ``admittance.py`` gives ``tan δ = Z′/(−Z″)``, and here ``−Z″ = 1``.

        The shape is the rig's own three-limb one, per this class's docstring — the
        CPE limb rising to a peak, the falling parallel-conduction limb, then the
        high-frequency rise where ``Z → R_series`` and ``tan δ → ∞``. ``rise_above``
        stays shallow enough that the tail never overtakes the peak, or ``argmax``
        would move to the top endpoint and the fixture would be the monotone shape.
        """
        peak, npts = 4, 12
        trough = peak + points - 1
        f_asc = log_frequencies(npts=npts, descending=False)
        lf = np.log10(f_asc)
        lt = np.empty(npts)
        lt[:peak + 1] = 1.0 - 1.3 * (lf[peak] - lf[:peak + 1])          # CPE limb, up
        lt[peak:trough + 1] = 1.0 - 0.70 * (lf[peak:trough + 1] - lf[peak])
        lt[trough:] = lt[trough] + rise_above * (lf[trough:] - lf[trough])
        return f_asc[::-1], (10.0 ** lt)[::-1] - 1j                     # descending

    def test_tand_window_with_an_interior_peak_runs_from_the_peak_to_the_top_of_band(self):
        # Regression pin. The default synthetic has its tanδ minimum at the top of the
        # sweep, so "peak → first minimum above it" *is* "at or above the peak" — the
        # numbers the module docstring quotes (global -0.24, windowed -0.83) must not
        # move, and neither must the mask.
        f, Z = reference_spectrum()
        tand = self._tand(Z)
        f_peak = float(f[int(np.argmax(tand))])

        window = parallel_branch_window(f, Z)
        assert np.array_equal(window, f >= f_peak)
        assert window[int(np.argmax(f))], "the top of the band belongs to this window"
        assert 0 < int(window.sum()) < f.size, "a real selection, not the whole band"
        assert log_slope(f[window], tand[window]) == pytest.approx(-0.82, abs=0.02)

    def test_tand_window_with_the_peak_below_the_sweep_stops_at_the_tand_minimum(self):
        # The rig's own grid at rig R_bulk. Fitting through the high-frequency rise
        # reads -0.49 against a -0.3 threshold; the falling segment reads -0.85.
        f, Z = reference_spectrum(RIG_FREQ, R_bulk=5.0e7)
        tand = self._tand(Z)
        ascending = np.argsort(f)

        assert int(np.argmax(tand[ascending])) == 0, (
            "fixture must be the U shape — the peak has to sit at the low endpoint")
        assert tand[ascending][-1] > tand[ascending].min(), "and it must rise again"

        window = parallel_branch_window(f, Z)
        assert not window.all(), "the full band is the pre-fix degenerate answer"
        assert not window[int(np.argmax(f))], "the rising HF tail must be excluded"
        f_trough = float(f[ascending][int(np.argmin(tand[ascending]))])
        assert float(f[window].max()) == f_trough
        assert float(f[window].min()) == float(f.min()), "the falling limb starts low"

        assert log_slope(f[window], tand[window]) == pytest.approx(-0.85, abs=0.02)
        assert log_slope(f, tand) == pytest.approx(-0.49, abs=0.02), (
            "the full-band fit is what the old anchor returned here")

    def test_tand_window_on_a_monotone_rising_series_parasitic_falls_back_to_full_band(self):
        # Load-bearing: a series parasitic's maximum *is* the top of the band, so the
        # falling segment is empty. The full-band fit is what makes the slope read +1
        # and the gate reject; a narrowed window would report NaN instead.
        f, Z = pure_series_rc(RIG_FREQ)
        tand = self._tand(Z)
        ascending = np.argsort(f)
        assert int(np.argmax(tand[ascending])) == tand.size - 1, "monotone rising"

        window = parallel_branch_window(f, Z)
        assert window.all(), "no falling segment exists; the whole band is the answer"

        r = gate_tand_slope(f, Z, _ctx())
        assert r.metrics["tand_slope"] == pytest.approx(1.0, abs=0.01)
        assert not r.passed and r.severity == BLOCK_SPECTRUM

    def test_tand_window_with_too_few_valid_points_returns_the_full_band(self):
        # Three usable points spread across a 41-point sweep: too few to locate a peak,
        # but enough that a peak-to-trough span would still clear min_points in the
        # *original* array — so the guard has to be what returns the full band here.
        f, Z = reference_spectrum()
        Z = Z.copy()
        drop = np.ones(f.size, dtype=bool)
        drop[[0, 20, 40]] = False
        Z[drop] = -np.abs(Z[drop].real) + 1j * Z[drop].imag   # tanδ <= 0 ⇒ unusable

        tand = self._tand(Z)
        assert int((np.isfinite(tand) & (tand > 0)).sum()) == 3, "fewer than min_points"

        window = parallel_branch_window(f, Z)
        assert window.all() and window.size == f.size

    def test_tand_window_with_a_four_point_falling_segment_is_fitted_not_discarded(self):
        """The false-reject population, reproduced: four points is a window, not a stub.

        196 of 203 rejections on the ``20260811T023757Z_equilibration_characterization``
        corpus were this shape and every one was false — the segment held exactly four
        points against a threshold of five, so the full-band fallback fired and fitted
        the CPE limb the window exists to exclude. The two slopes here (−0.70 segment,
        +0.31 full band) sit on the measured medians (−0.73 and +0.28), and they land on
        opposite sides of the −0.3 threshold, which is the whole defect in one fixture.
        """
        f, Z = self._falling_segment_of(points=4, rise_above=0.50)
        tand = self._tand(Z)
        ascending = np.argsort(f)

        assert int(np.argmax(tand[ascending])) == 4, "interior peak, not an endpoint"
        assert int((np.isfinite(tand) & (tand > 0)).sum()) == f.size, (
            "every point usable — so the segment-length guard is the one under test, "
            "not the data-sufficiency guard that precedes it")

        window = parallel_branch_window(f, Z)
        assert int(window.sum()) == MIN_FALLING_SEGMENT_POINTS == 4
        assert not window.all(), "a real selection — the fallback must not have fired"
        assert log_slope(f[window], tand[window],
                         min_points=MIN_FALLING_SEGMENT_POINTS) == pytest.approx(-0.70, abs=0.02)

        # The pre-fix answer, reached through the parameter rather than a patched module:
        # five discards this segment and returns the whole sweep, whose fit is positive.
        old = parallel_branch_window(f, Z, min_points=5)
        assert old.all(), "the threshold of five is what produced the false rejects"
        assert log_slope(f[old], tand[old]) == pytest.approx(+0.31, abs=0.02)
        assert log_slope(f[old], tand[old]) > GateSettings().tand_slope_max

        r = gate_tand_slope(f, Z, _ctx())
        assert r.passed and r.checked
        assert r.metrics["tand_window_pts"] == 4.0
        assert r.metrics["tand_slope"] == pytest.approx(-0.70, abs=0.02)

    def test_tand_window_with_a_three_point_falling_segment_still_returns_the_full_band(self):
        """One below the new line, and the corpus says nothing about it.

        Distinct from the too-few-valid-points test above: there the sweep never had
        enough usable points to locate a peak at all, so the *first* guard returned the
        full band. Here all twelve points are usable, the peak and trough are both
        found, and it is the resulting *segment* that is too short — the second guard.
        The measured false-reject population's p10 is 4, so lowering the threshold past
        what was measured would be loosening this fix never bought evidence for.
        """
        f, Z = self._falling_segment_of(points=3, rise_above=0.20)
        tand = self._tand(Z)
        ascending = np.argsort(f)
        tand_asc = tand[ascending]

        assert int((np.isfinite(tand) & (tand > 0)).sum()) == f.size, (
            "the data-sufficiency guard must be satisfied, or this tests that instead")
        i_peak = int(np.argmax(tand_asc))
        i_trough = i_peak + int(np.argmin(tand_asc[i_peak:]))
        assert i_trough - i_peak + 1 == 3 < MIN_FALLING_SEGMENT_POINTS

        window = parallel_branch_window(f, Z)
        assert window.all(), "three is still short enough to fall back to the full band"
        assert gate_tand_slope(f, Z, _ctx()).metrics["tand_window_pts"] == float(f.size)

    def test_log_slope_still_needs_five_points_so_only_the_tand_gate_relaxed(self):
        """``log_slope``'s own default did not move, and must not be merged into the new one.

        It has four call sites in ``gates.py``: ``gate_tand_slope`` — windowed by
        ``parallel_branch_window`` and the only one this fix touches — ``gate_cap_flatness``,
        windowed by ``top_decade_window``, and two unwindowed calls in ``gate_series_rc``.
        Collapsing the two constants into one shared default would silently relax the
        other three, on a corpus that measured nothing about them.
        """
        x = np.array([1.0, 2.0, 4.0, 8.0])
        y = np.array([8.0, 4.0, 2.0, 1.0])

        assert log_slope(x, y) != log_slope(x, y), "four points is still NaN by default"
        assert log_slope(x, y, min_points=MIN_FALLING_SEGMENT_POINTS) == pytest.approx(-1.0)
        assert log_slope(np.append(x, 16.0), np.append(y, 0.5)) == pytest.approx(-1.0), (
            "five is what the untouched default admits")


class TestPointGates:
    def test_negative_re_z_points_are_dropped_because_they_are_artefact_not_measurement(self):
        f, Z = negative_real_part(n_points=3)
        r = gate_quadrant(f, Z, _ctx())
        assert r.severity == BLOCK_POINT
        assert r.n_dropped == 3

    def test_quadrant_after_finiteness_drops_exactly_the_nonpositive_real_points(self):
        """Pin `quadrant.n_dropped` as a *negative-real count*, for its consumer.

        A downstream session reads this number out of ``gate_log_json`` and treats
        it as "how many points had ``Re Z <= 0``". That reading is only sound
        because ``gate_finiteness`` is ``FRONT1_PRE_CORRECTION[0]`` and
        ``gate_quadrant`` is ``[2]``: the phase clause at the top of
        ``gate_quadrant`` is redundant with ``Re > 0`` for every *finite* point
        (``np.angle`` is ``atan2 ∈ (−180°, 180°]``), and the one class it rejects
        that ``Re > 0`` admits — finite positive ``Re`` with non-finite ``Im`` — has
        already been masked away by the time quadrant runs.

        Reorder the ladder, or relax ``gate_finiteness``, and the consumer starts
        over-counting with nothing else going red. Hence this test rather than a
        comment. Note ``<=``, not ``<``: ``Re == 0`` fails the ``Re > 0`` clause
        while *passing* the phase clause at exactly 90°, so it is dropped too.
        """
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[:3] = -np.abs(Z[:3].real) + 1j * Z[:3].imag   # unphysical: Re Z < 0
        Z[5] = 0.0 + 1j * Z[5].imag                     # the Re == 0 boundary
        Z[7] = 1000.0 + 1j * np.nan                     # Re > 0, Im NaN: the split class

        # Mirror `run_gates`: finiteness first, then quadrant on the survivors.
        fin = gate_finiteness(f, Z, _ctx())
        assert fin.n_dropped == 1, "only the Im-NaN point is non-finite here"
        survivors = np.where(fin.mask)[0]
        Z_surv = Z[survivors]

        r = gate_quadrant(f[survivors], Z_surv, _ctx())
        expected = int((Z_surv.real <= 0).sum())
        assert expected > 0, "an all-valid spectrum would make this test vacuous"
        assert r.n_dropped == expected

    def test_hf_inductive_truncation_removes_only_the_contiguous_run_at_the_top(self):
        f, Z = hf_phase_artifact(n_points=4)
        r = gate_hf_inductive(f, Z, _ctx())
        assert r.n_dropped == 4
        assert not r.mask[np.argsort(f)[-1]], "the highest frequency must be dropped"

    def test_hf_inductive_truncation_is_skipped_on_a_non_blocking_cell(self):
        # Without a blocking electrode, Im Z > 0 may be genuine lead inductance and
        # removing it would be discarding a real measurement.
        f, Z = hf_phase_artifact()
        ctx = build_context(blocking=False)
        r = gate_hf_inductive(f, Z, ctx)
        assert r.passed and r.n_dropped == 0

    @pytest.mark.parametrize("blocking", [True, False])
    def test_cell_blocking_reaches_the_kk_ladder_from_a_real_built_context(
        self, monkeypatch, blocking
    ):
        # `blocking` decides `add_cap` on the K–K ladder (kk.py), so a non-blocking
        # cell fitted with the blocking basis is a wrong answer that reports as a
        # clean one. The context must be built by `build_context`, not by hand: it
        # files the flag under `cell`, and a hand-built `{"blocking": ...}` would
        # satisfy a top-level `ctx.get("blocking")` that production never sees.
        import softae.analysis.eis.kk as kk_module

        assert "blocking" not in build_context(blocking=blocking), \
            "build_context files the flag under 'cell'; a top-level read cannot work"

        seen: dict[str, object] = {}

        def _capture(f, Z, *, blocking=True, **kw):
            seen["blocking"] = blocking
            return kk_module.LinKKResult(error="stubbed — kwarg capture only")

        monkeypatch.setattr(kk_module, "lin_kk", _capture)

        f, Z = reference_spectrum()
        gate_kk_truncation(f, Z, build_context(blocking=blocking))
        assert seen["blocking"] is blocking

    def test_points_outside_the_magnitude_window_are_dropped_pointwise_not_by_median(self):
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[:2] *= 1e9                       # two points over range, median untouched
        r = gate_magnitude(f, Z, _ctx())
        assert r.n_dropped == 2

    def test_non_positive_frequencies_are_dropped_so_the_triad_cannot_divide_by_zero(self):
        f, Z = reference_spectrum()
        f = f.copy()
        f[0] = 0.0
        r = gate_finiteness(f, Z, _ctx())
        assert not r.mask[0]

    def test_duplicate_frequencies_are_dropped_because_they_break_the_fit_jacobian(self):
        f, Z = reference_spectrum()
        f = f.copy()
        f[5] = f[4]
        r = gate_finiteness(f, Z, _ctx())
        assert r.n_dropped == 1 and not r.mask[5]


class TestInstrumentHealth:
    def test_a_stuck_instrument_is_rejected_although_the_framework_has_no_such_gate(self):
        f, Z = stuck_instrument()
        r = gate_stuck_instrument(f, Z, _ctx())
        assert r.severity == BLOCK_SPECTRUM
        assert not r.passed

    def test_an_over_range_spectrum_is_caught_by_the_magnitude_window_and_then_min_points(self):
        f, Z = over_range()
        _, results, _ = run_gates(f, Z, _ctx())
        failed = _failed(results)
        assert "magnitude_window" in failed
        assert "min_points" in failed

    def test_a_spectrum_far_from_the_phase_calibration_is_flagged_not_removed(self):
        # Replaces a test of the withdrawn "phase-reliable ceiling". Phase noise was
        # characterised at 9.9 kΩ on a resistive load; a film three decades higher is
        # outside that band, and the loss-tangent floor there is extrapolated.
        f, Z = reference_spectrum()
        _, results, _ = run_gates(f, Z * 1e3, _ctx())
        flagged = next(r for r in results if r.name == "phase_noise_extrapolated")
        assert flagged.severity == FLAG
        assert not flagged.passed
        assert flagged.n_dropped == 0
        assert flagged.metrics["phase_noise_valid"] == 0.0

    def test_a_spectrum_inside_the_phase_calibration_band_passes(self):
        f, Z = reference_spectrum(R_bulk=1.0e4, noise_pct=0.0)
        _, results, _ = run_gates(f, Z, _ctx())
        flagged = next(r for r in results if r.name == "phase_noise_extrapolated")
        assert flagged.passed


class TestReferenceElectrodeAttribution:
    """F13/R19 — a quadrant violation is a wiring fault until proven otherwise.

    Attributing this signature to the instrument is exactly how the withdrawn
    ``Z_φ ≈ 5×10⁷ Ω`` ceiling came to be believed.
    """

    def test_a_widespread_violation_with_unverified_re_names_the_reference_electrode(self):
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[: int(0.4 * Z.size)] = -np.abs(Z[: int(0.4 * Z.size)].real) + 1j * Z[
            : int(0.4 * Z.size)].imag
        r = gate_quadrant(f, Z, build_context(re_connection="unverified"))
        assert "RE integrity UNVERIFIED" in r.detail
        assert "tie RE to CE" in r.detail

    def test_a_film_bridging_the_stripes_closes_the_loop_so_the_fault_is_the_instrument(self):
        # The RE stripe sits between CE and WE and reaches the cell only through cast
        # material — so a film closes the loop itself. That is why historical sample
        # spectra are sound while two-terminal reference components across CE/WE,
        # which touch no RE stripe, produced the pathology.
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[: int(0.4 * Z.size)] = -np.abs(Z[: int(0.4 * Z.size)].real) + 1j * Z[
            : int(0.4 * Z.size)].imag
        r = gate_quadrant(f, Z, build_context(re_connection="bridged_by_sample"))
        assert "RE integrity UNVERIFIED" not in r.detail
        assert "instrument-side" in r.detail

    def test_a_bare_board_violation_is_reported_as_structural_not_as_a_fault(self):
        # Air between the stripes cannot close the loop. There is nothing to repair
        # and nothing to blame on the instrument.
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[: int(0.4 * Z.size)] = -np.abs(Z[: int(0.4 * Z.size)].real) + 1j * Z[
            : int(0.4 * Z.size)].imag
        r = gate_quadrant(f, Z, build_context(re_connection="open_by_geometry"))
        assert "expected" in r.detail
        assert "UNVERIFIED" not in r.detail
        assert "instrument-side" not in r.detail

    def test_a_few_scattered_violations_do_not_accuse_the_reference_electrode(self):
        # Below the suspicion threshold this is ordinary noise, not a floating loop.
        f, Z = negative_real_part(n_points=1)
        r = gate_quadrant(f, Z, build_context(re_connection="unverified"))
        assert "RE integrity" not in r.detail

    def test_points_are_dropped_either_way_because_both_causes_are_artefact(self):
        f, Z = negative_real_part(n_points=3)
        for state in ("unverified", "connected"):
            assert gate_quadrant(f, Z, build_context(re_connection=state)).n_dropped == 3


class TestRunner:
    def test_every_dropped_point_carries_a_gate_name_and_a_reason(self):
        f, Z = negative_real_part()
        _, _, log = run_gates(f, Z, _ctx())
        dropping = [e for e in log if e["n_dropped"]]
        assert dropping
        for entry in dropping:
            assert entry["gate"] and entry["detail"]

    def test_a_non_triad_blocking_failure_stops_the_chain_so_no_later_gate_sees_rejected_data(self):
        f, Z = stuck_instrument()
        _, results, _ = run_gates(f, Z, _ctx())
        names = [r.name for r in results]
        assert "stuck_instrument" in names
        assert not any(g.__name__.replace("gate_", "") in names for g in TOPOLOGY_TRIAD)

    def test_the_mask_indexes_the_original_arrays_so_survivors_stay_identifiable(self):
        f, Z = negative_real_part(n_points=3)
        mask, _, _ = run_gates(f, Z, _ctx())
        assert mask.size == f.size
        assert not mask[:3].any()

    def test_a_gate_that_raises_is_skipped_rather_than_discarding_the_measurement(self):
        def exploding(f, Z, ctx):
            raise RuntimeError("boom")

        f, Z = reference_spectrum()
        mask, results, log = run_gates(f, Z, _ctx(), gates=(exploding,))
        assert mask.all()
        assert results[0].passed
        assert "boom" in log[0]["detail"]

    def test_min_points_reports_how_many_survived_against_what_was_needed(self):
        f, Z = reference_spectrum(freq=log_frequencies(npts=5))
        r = gate_min_points(f, Z, _ctx())
        assert not r.passed
        assert r.metrics["n_surviving"] == 5.0


class TestFrontTwo:
    """Post-fit gates (§4.1–4.5). They grade the answer; only one can reject."""

    def _ctx_with_fit(self, fit, **overrides):
        ctx = _ctx(**overrides)
        ctx["fit"] = fit
        return ctx

    def test_only_the_residual_norm_can_reject_because_the_data_is_already_admitted(self):
        from softae.analysis.eis.gates import FRONT2_GATES

        blocking = [g for g in FRONT2_GATES
                    if g(*reference_spectrum(), self._ctx_with_fit(None)).severity
                    == BLOCK_SPECTRUM]
        assert [g.__name__ for g in blocking] == ["gate_residual_norm"]

    def test_a_catastrophic_residual_is_rejected_even_though_the_fit_converged(self):
        from softae.analysis.eis.gates import gate_residual_norm

        class _Fit:
            quality = {"residual_rms_pct": 850.0}

        f, Z = reference_spectrum()
        r = gate_residual_norm(f, Z, self._ctx_with_fit(_Fit()))
        assert not r.passed and r.severity == BLOCK_SPECTRUM

    def test_an_ordinary_poor_fit_is_left_to_the_grader_not_rejected_here(self):
        # 30 % is bad by grade_fit's 15 % standard but nowhere near catastrophic;
        # duplicating that judgement here would reject twice for one problem.
        from softae.analysis.eis.gates import gate_residual_norm

        class _Fit:
            quality = {"residual_rms_pct": 30.0}

        f, Z = reference_spectrum()
        assert gate_residual_norm(f, Z, self._ctx_with_fit(_Fit())).passed

    def test_residuals_all_of_one_sign_are_flagged_as_an_offset_model(self):
        from softae.analysis.eis.gates import gate_residual_structure

        f, Z = reference_spectrum()

        class _Fit:
            z_fit = Z * 0.9      # uniformly low ⇒ every residual the same sign

        r = gate_residual_structure(f, Z, self._ctx_with_fit(_Fit()))
        assert not r.passed
        assert "same sign" in r.detail

    def test_random_residuals_pass_the_runs_test(self):
        from softae.analysis.eis.gates import gate_residual_structure

        f, Z = reference_spectrum()
        rng = np.random.default_rng(0)

        class _Fit:
            z_fit = Z * (1.0 + 0.01 * rng.standard_normal(Z.size))

        assert gate_residual_structure(f, Z, self._ctx_with_fit(_Fit())).passed

    def test_a_pegged_parameter_is_named(self):
        from softae.analysis.eis.fitter import FitCovariance
        from softae.analysis.eis.gates import gate_pegged_parameters

        class _Fit:
            covariance = FitCovariance(
                names=("R0", "R1"), values=np.array([0.0, 5e4]), pcov=np.eye(2),
                bounds=(np.array([0.0, 0.0]), np.array([np.inf, np.inf])),
            )

        f, Z = reference_spectrum()
        r = gate_pegged_parameters(f, Z, self._ctx_with_fit(_Fit()))
        assert not r.passed and "R0" in r.detail

    def test_degeneracy_reports_rho_and_says_the_sum_was_used(self):
        from softae.analysis.eis.fitter import FitCovariance
        from softae.analysis.eis.gates import gate_degeneracy

        class _Fit:
            model_name = "blocking_coplanar"
            covariance = FitCovariance(
                names=("R0", "R1"), values=np.array([50.0, 200.0]),
                pcov=np.array([[100.0, -99.0], [-99.0, 100.0]]),
            )

        f, Z = reference_spectrum()
        r = gate_degeneracy(f, Z, self._ctx_with_fit(_Fit()))
        assert not r.passed
        assert "sum" in r.detail

    def test_a_singular_covariance_reads_as_unidentifiable_not_as_a_good_rho(self):
        from softae.analysis.eis.fitter import FitCovariance
        from softae.analysis.eis.gates import gate_degeneracy

        class _Fit:
            model_name = "blocking_coplanar"
            covariance = FitCovariance(
                names=("R0", "R1"), values=np.array([50.0, 200.0]),
                pcov=np.full((2, 2), np.nan), singular=True,
            )

        f, Z = reference_spectrum()
        r = gate_degeneracy(f, Z, self._ctx_with_fit(_Fit()))
        assert not r.passed and "unidentifiable" in r.detail

    def test_every_front2_gate_is_inert_without_a_fit(self):
        from softae.analysis.eis.gates import FRONT2_GATES

        f, Z = reference_spectrum()
        ctx = self._ctx_with_fit(None)
        for gate in FRONT2_GATES:
            assert gate(f, Z, ctx).passed, gate.__name__


class TestGateResult:
    def test_only_point_gates_report_dropped_counts(self):
        mask = np.array([True, False, True])
        blocking = GateResult("x", BLOCK_SPECTRUM, False, "d", mask)
        pointwise = GateResult("x", BLOCK_POINT, False, "d", mask)
        assert blocking.n_dropped == 0
        assert pointwise.n_dropped == 1

    def test_the_log_entry_shape_is_the_one_the_framework_specifies(self):
        # `checked` joined the five framework keys with the gate-record split. This
        # assertion is deliberately still *closed*: `as_log_entry` is where the log
        # shape is defined, so this is the one place a silently-added key should
        # register as a change (test_eis_engine's parallel assertion is a subset
        # check precisely so that adding a key here does not read as a regression
        # there).
        entry = GateResult("x", FLAG, True, "d", np.ones(2, bool)).as_log_entry()
        assert set(entry) == {"gate", "severity", "passed", "checked", "detail",
                              "n_dropped"}
        assert entry["checked"] is True, "a gate that ran reports checked=True"

    def test_a_gate_that_could_not_check_says_so_in_the_log_entry(self):
        entry = GateResult.unchecked("x", FLAG, "no input", np.ones(2, bool)).as_log_entry()
        assert entry["checked"] is False
        assert entry["passed"] is True, "fail-open posture is preserved"

    def test_checked_defaults_true_so_the_49_positional_constructions_are_unchanged(self):
        assert GateResult("x", FLAG, True, "d", np.ones(2, bool)).checked is True

    def test_unchecked_and_refusing_is_refused_at_construction(self):
        # `passed=False, checked=False` is "the check did not run and the spectrum is
        # refused" — a state no gate is entitled to occupy. Enforcing it is what lets
        # a consumer conclude something from `checked` alone; an invariant with an
        # exception would force every reader to consult both fields.
        with pytest.raises(ValueError, match="fail open"):
            GateResult("x", FLAG, False, "d", np.ones(2, bool), checked=False)

    def test_the_invariant_permits_the_three_states_that_are_meaningful(self):
        mask = np.ones(2, bool)
        assert GateResult("x", FLAG, True, "d", mask).passed          # checked, clean
        assert not GateResult("x", FLAG, False, "d", mask).passed     # checked, found it
        assert GateResult.unchecked("x", FLAG, "d", mask).passed      # could not check


class TestValleyFeature:
    """§3.7b / R21 / F15 — the valley, never the |Z| minimum.

    A blocking-cell spectrum has two minima that are trivially confused: the |Z|
    minimum (the HF intercept ≈ R_series) and the interior local minimum of −Z''
    (the valley ≈ R_series + R_bulk). Overhaul §3.9 records taking the wrong one
    twice, differing by more than an order of magnitude on the same file, and it
    reached a published comparison before anyone noticed — because F15 has no other
    symptom. The spectrum, the fit and the residuals all look fine.
    """

    def test_a_well_formed_spectrum_finds_its_valley_and_records_both_features(self):
        from softae.analysis.eis.gates import gate_valley_feature

        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=0.5, seed=5)
        ctx = _ctx()
        res = gate_valley_feature(f, Z, ctx)
        assert res.passed and res.severity == "flag"
        assert "R_sol_valley" in res.metrics and res.metrics["R_sol_valley"] > 0
        # Both features are named in the detail so the confusion is visible in the log.
        assert "|Z|min" in res.detail and "different features" in res.detail

    def test_the_valley_recovers_the_sum_and_is_a_different_point_from_the_zmin(self):
        # The valley must track R_series + R_bulk (~2050 Ω here), and must not be the
        # |Z| minimum. How far apart the two sit is a property of the spectrum, not of
        # the gate: on this synthetic the blocking tail keeps |Z|min at ~1.6 kΩ, while
        # the real spectra of overhaul §3.9 put them an order of magnitude apart. What
        # is invariant, and what this pins, is that they are distinct features and the
        # gate reports the right one.
        from softae.analysis.eis.gates import gate_valley_feature

        f, Z = reference_spectrum(R_series=50.0, R_bulk=2000.0,
                                  noise_pct=0.2, seed=11)
        ctx = _ctx()
        res = gate_valley_feature(f, Z, ctx)
        assert res.metrics["R_sol_valley"] == pytest.approx(2050.0, rel=0.25)
        assert res.metrics["R_sol_valley"] > float(np.min(np.abs(Z)))
        assert res.metrics["valley_over_zmin"] > 1.0
        # C: the valley is reported through `metrics`, never by mutating the shared ctx.
        assert "R_sol_valley" not in ctx and "f_valley" not in ctx

    def test_a_spectrum_with_no_interior_minimum_is_rejected_not_approximated(self):
        # A pure series RC has -Z'' falling monotonically: no valley exists, and
        # falling back to the |Z| minimum is precisely the F15 error.
        from softae.analysis.eis.gates import gate_valley_feature

        f, Z = pure_series_rc()
        res = gate_valley_feature(f, Z, _ctx())
        assert not res.passed and res.severity == "block_spectrum"
        assert "do NOT fall back" in res.detail

    def test_a_band_edge_is_never_mistaken_for_a_valley(self):
        # The exact mechanism of the original error: a naive argmin over a window
        # returns the window edge when the true minimum lies outside it. Endpoints
        # must not be candidates.
        from softae.analysis.eis.gates import gate_valley_feature

        f = np.logspace(5, 1, 40)                 # descending, instrument order
        # -Z'' decreasing monotonically with index => its minimum is the last point.
        zi = np.linspace(500.0, 10.0, 40)
        Z = 100.0 - 1j * zi
        res = gate_valley_feature(f, Z, _ctx())
        assert not res.passed, "the array edge is not an interior local minimum"

    def test_the_gate_runs_before_any_fit_is_attempted(self):
        from softae.analysis.eis.gates import FRONT1_GATES, gate_valley_feature

        assert gate_valley_feature in FRONT1_GATES


class TestCrossSpectrumDuplicates:
    """§3.7c / R22 / F14 — identical values between spectra prove an instrument rail.

    Overhaul §3.8 caught this at 100 mV: two channels returning identical |Z| to 15
    significant figures. No physical measurement of two distinct samples does that.
    The remedy is a higher current range, **not** a lower amplitude — saturation
    scales with current and bites hardest at the impedance minimum, while interfacial
    nonlinearity scales with interface voltage and does the opposite.
    """

    def _pair(self, seed_a=1, seed_b=2):
        fa, Za = reference_spectrum(R_bulk=2000.0, noise_pct=1.0, seed=seed_a)
        fb, Zb = reference_spectrum(R_bulk=2500.0, noise_pct=1.0, seed=seed_b)
        return [("ch1", fa, Za), ("ch2", fb, Zb)]

    def test_two_genuinely_independent_spectra_pass(self):
        from softae.analysis.eis.gates import gate_cross_spectrum_duplicates

        res = gate_cross_spectrum_duplicates(self._pair())
        assert res.passed and res.severity == "block_spectrum"

    def test_a_single_identical_point_is_enough_to_condemn_the_set(self):
        from softae.analysis.eis.gates import gate_cross_spectrum_duplicates

        spectra = self._pair()
        (la, fa, Za), (lb, fb, Zb) = spectra
        Zb = Zb.copy()
        Zb[7] = Za[7]                      # one railed point, bitwise identical
        res = gate_cross_spectrum_duplicates([(la, fa, Za), (lb, fb, Zb)])
        assert not res.passed
        assert "instrument rail" in res.detail
        assert res.metrics["n_duplicates"] == 1.0

    def test_the_remedy_named_is_the_current_range_not_the_amplitude(self):
        from softae.analysis.eis.gates import gate_cross_spectrum_duplicates

        spectra = self._pair()
        (la, fa, Za), (lb, fb, Zb) = spectra
        Zb = Zb.copy()
        Zb[3] = Za[3]
        detail = gate_cross_spectrum_duplicates([(la, fa, Za), (lb, fb, Zb)]).detail
        assert "current range" in detail and "not the amplitude" in detail

    def test_merely_similar_values_are_not_flagged(self):
        # The tolerance is at the float-comparison floor on purpose: this gate looks
        # for a rail, not for samples that happen to measure alike.
        from softae.analysis.eis.gates import gate_cross_spectrum_duplicates

        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=0.0, seed=3)
        res = gate_cross_spectrum_duplicates(
            [("a", f, Z), ("b", f, Z * (1 + 1e-9))])
        assert res.passed

    def test_spectra_measured_at_different_frequencies_are_simply_not_compared(self):
        from softae.analysis.eis.gates import gate_cross_spectrum_duplicates

        f, Z = reference_spectrum(R_bulk=2000.0, noise_pct=0.0, seed=3)
        res = gate_cross_spectrum_duplicates([("a", f, Z), ("b", f * 1.01, Z)])
        assert res.passed

    def test_one_spectrum_alone_cannot_trip_it(self):
        # Needs two independent measurements by construction — it is series-level,
        # which is why it cannot live inside run_gates.
        from softae.analysis.eis.gates import gate_cross_spectrum_duplicates

        f, Z = reference_spectrum()
        assert gate_cross_spectrum_duplicates([("only", f, Z)]).passed


# ── The `checked=False` census ───────────────────────────────────────────────


def _failed_legacy_fit():
    """A ``FitResult`` shaped exactly as the legacy path produces on a failed fit.

    Not a stand-in: ``covariance`` is documented as ``None`` on the legacy path
    *always* (``circuit_fitting.FitResult``), ``quality`` is empty when the fit
    produced no ``z_fit`` to compare against, and ``z_fit`` is ``None`` for the same
    reason. This one object is therefore the everyday shape the shipped engine hands
    to :data:`FRONT2_GATES`, and it drives six of the census sites at once.
    """
    from softae.analysis.circuit_fitting import FitResult

    return FitResult(
        model_name="randles", parameters=np.array([]),
        R0=float("nan"), R1=float("nan"), R0_guess=1.0, R1_guess=1.0,
        z_indices=[], success=False, error_msg="fit did not converge",
    )


def _fit_with(covariance, *, success=False, R1=float("nan"), z_fit=None):
    """A production ``FitResult`` carrying *covariance* — the gated path's shape."""
    from softae.analysis.circuit_fitting import FitResult

    return FitResult(
        model_name="blocking_coplanar", parameters=np.array([50.0, 2000.0]),
        R0=50.0, R1=R1, R0_guess=50.0, R1_guess=2000.0, z_indices=[0, 1],
        success=success, z_fit=z_fit, covariance=covariance,
    )


def _checked_by_gate(results):
    return {r.name: r.checked for r in results}


class TestCouldNotCheckCensus:
    """Every site that cannot evaluate its criterion reports ``checked is False``.

    Spec §3.3 item 2 — the guard that actually holds the shape, because
    ``checked: bool = True`` means a forgotten keyword reports having checked. Per
    ``SUBAGENT_RULES`` §3.1(e) each branch is driven from **production-shaped input**
    wherever one exists, and through :func:`run_gates` wherever the runner can reach
    it, rather than from a hand-built ``ctx`` that forces the branch.

    Three sites cannot be reached that way and each says so at its own test:
    ``phase_noise_extrapolated`` and ``plateau_in_band`` are filtered out by earlier
    gates in the runner, ``residual_structure``'s ``n < 8`` needs ``min_fit_pts``
    below the shipped 8, and ``residual_structure``'s variance branch is
    arithmetically unreachable from any input at all.

    **No verdict moves.** Every site below already returned ``passed=True`` before
    this field existed, which is spec §3.8's positive control, and each test asserts
    it alongside ``checked``.
    """

    # ── Front 1, through the real runner ──────────────────────────────────────

    def test_a_two_point_survivor_set_leaves_three_front1_gates_unable_to_judge(self):
        """The measured empty-well shape: most of a 20-point sweep non-finite.

        Production-reachable, and observed on the rig — an empty well returned 4
        finite points of 20 where a cast well returned 20. With two finite points
        ``gate_finiteness`` masks the rest and the survivors reach
        ``monotonic_frequency``, ``stuck_instrument`` and the K–K ladder, none of
        which can say anything about two points.
        """
        f, Z = reference_spectrum()
        f, Z = f[:20].copy(), Z[:20].copy()
        Z[2:] = np.nan + 1j * np.nan

        _, results, log = run_gates(f, Z, _ctx())
        checked = _checked_by_gate(results)

        for name in ("monotonic_frequency", "stuck_instrument", "kk_truncation"):
            assert checked[name] is False, name
        # Fail-open preserved: none of the three refused the spectrum.
        assert all(r.passed for r in results if r.checked is False)
        # And the runner's own log carries it, which is the only route to the column.
        assert {e["gate"]: e["checked"] for e in log}["stuck_instrument"] is False

    def test_the_kk_ladder_failing_to_run_is_recorded_as_an_absence_not_a_pass(self):
        # No monkeypatch: `lin_kk` genuinely cannot build a ladder on two points, so
        # this is the gate's own "K–K test did not run" branch on real input.
        f, Z = reference_spectrum()
        r = gate_kk_truncation(f[:2], Z[:2], _ctx())
        assert "did not run" in r.detail
        assert r.passed and r.checked is False

    def test_a_record_with_no_reactance_leaves_the_topology_triad_unable_to_judge(self):
        """``Im Z ≡ 0`` starves both slope-based topology gates of usable points.

        ``tan δ`` and ``C_app`` both divide by ``|Z''|``, and ``log_slope`` needs five
        strictly-positive pairs, so ``cap_flatness`` and ``series_rc_topology`` return
        NaN slopes. Reached through the runner on a full-length sweep — the K–K
        residual limit is relaxed only to get past an unrelated gate that would
        otherwise stop the chain before the triad.

        **This is also the exhibit for the one site left deliberately alone.**
        ``tand_slope`` fails on the *same* absence at the *same* severity and returns
        the opposite verdict; see the comment at that branch in ``gates.py``.
        """
        f, Z = reference_spectrum()
        Z = np.abs(Z.real) + 0j

        _, results, _ = run_gates(f, Z, _ctx(kk_resid_pct=1e9))
        checked = _checked_by_gate(results)
        by_name = {r.name: r for r in results}

        assert checked["cap_flatness"] is False
        assert checked["series_rc_topology"] is False
        assert by_name["cap_flatness"].passed and by_name["series_rc_topology"].passed

        assert by_name["tand_slope"].checked is True, (
            "left alone deliberately — see the comment at gate_tand_slope's NaN "
            "branch; marking it unchecked would be the sole exception to the "
            "passed/checked invariant")
        assert not by_name["tand_slope"].passed, "and its verdict must not move"

    # ── Front 2, through the real runner, as engine.py calls it ───────────────

    def test_a_failed_legacy_fit_leaves_every_front2_gate_unable_to_judge(self):
        """Six sites at once, from the shape the shipped engine produces daily.

        ``engine.py`` runs ``run_gates(f_ok, Z_ok, ctx, FRONT2_GATES)`` with
        ``ctx["fit"]`` set; the legacy path carries no ``FitCovariance`` at all, so
        the three ``cov is None`` branches are not an edge case there but the norm.
        Before this field, all six reported ``passed=True`` — indistinguishable, to
        every consumer, from six gates that checked and found nothing wrong.
        """
        from softae.analysis.eis.gates import FRONT2_GATES

        f, Z = reference_spectrum()
        ctx = _ctx()
        ctx["fit"] = _failed_legacy_fit()

        _, results, _ = run_gates(f, Z, ctx, FRONT2_GATES)
        checked = _checked_by_gate(results)

        assert set(checked) == {
            "residual_norm", "residual_structure", "pegged_parameters",
            "relative_standard_error", "degeneracy", "model_free_crosscheck",
        }, "every Front-2 gate must have run — none may short-circuit the chain"
        assert all(v is False for v in checked.values()), checked
        assert all(r.passed for r in results), "fail-open, and no verdict moves"

    def test_a_singular_covariance_leaves_the_measurand_not_determined(self):
        """§3.5(i) — the site the whole ruling turns on.

        ``fit_with_covariance`` sets ``singular = not np.all(np.isfinite(pcov))`` and
        keeps the NaN ``pcov``, which is exactly the object built here. ``sum_se`` is
        then NaN, ``rel`` is NaN, and ``passed = not (rel == rel and …)`` is
        unconditionally ``True`` — "cannot check" wearing "checked and clean", with a
        detail that used to read *"determined to nan%"*.
        """
        from softae.analysis.eis.fitter import FitCovariance
        from softae.analysis.eis.gates import gate_relative_standard_error

        f, Z = reference_spectrum()
        ctx = _ctx()
        ctx["fit"] = _fit_with(FitCovariance(
            names=("R0", "R1"), values=np.array([50.0, 2000.0]),
            pcov=np.full((2, 2), np.nan), singular=True))

        r = gate_relative_standard_error(f, Z, ctx)
        assert r.passed and r.checked is False
        assert np.isnan(r.metrics["rel_se_measurand"])

    def test_the_same_singular_covariance_is_still_a_finding_for_degeneracy(self):
        """§3.5(ii) — the KEEP that stops this wave from moving a verdict.

        ``gate_degeneracy`` asks whether the series/bulk split is identifiable. A
        singular covariance is not a missing input to that question; it is the
        answer. Marking it unchecked would move every such spectrum off SUSPECT.
        """
        from softae.analysis.eis.fitter import FitCovariance
        from softae.analysis.eis.gates import gate_degeneracy

        f, Z = reference_spectrum()
        ctx = _ctx()
        ctx["fit"] = _fit_with(FitCovariance(
            names=("R0", "R1"), values=np.array([50.0, 2000.0]),
            pcov=np.full((2, 2), np.nan), singular=True))

        r = gate_degeneracy(f, Z, ctx)
        assert not r.passed and r.checked is True
        assert "unidentifiable" in r.detail

    def test_a_correlation_that_cannot_be_formed_is_an_absence_not_a_pass(self):
        # Finite `pcov` — so `singular` is False and the branch above is not the one
        # taken — but a zero variance makes `rho`'s denominator zero. curve_fit
        # returns a zero diagonal for a parameter the Jacobian does not constrain.
        from softae.analysis.eis.fitter import FitCovariance
        from softae.analysis.eis.gates import gate_degeneracy

        cov = FitCovariance(names=("R0", "R1"), values=np.array([50.0, 2000.0]),
                            pcov=np.array([[0.0, 0.0], [0.0, 100.0]]))
        assert not cov.singular and np.isnan(cov.rho("R0", "R1"))

        f, Z = reference_spectrum()
        ctx = _ctx()
        ctx["fit"] = _fit_with(cov)
        r = gate_degeneracy(f, Z, ctx)
        assert r.passed and r.checked is False and "unavailable" in r.detail

    def test_a_purely_reactive_record_makes_the_model_free_cross_check_impossible(self):
        # `1/max(Re Y)` needs a positive real admittance somewhere. A record with no
        # resistive component anywhere has none, so the cross-check has nothing to
        # compare the fit against — with the fit itself perfectly healthy.
        from softae.analysis.eis.gates import gate_model_free_crosscheck

        f, Z = reference_spectrum()
        reactive = 0.0 - 1j * np.abs(Z.imag)
        ctx = _ctx()
        ctx["fit"] = _fit_with(None, success=True, R1=2000.0)

        r = gate_model_free_crosscheck(f, reactive, ctx)
        assert r.passed and r.checked is False
        assert "unavailable" in r.detail

    # ── The runner's own exception handler ────────────────────────────────────

    def test_a_gate_that_raises_is_recorded_as_unchecked_not_as_a_pass(self):
        """§3.5(iv) — the site that justifies the field even on its own.

        A crash and a pass were the same log line. The raiser is injected here, but
        the mechanism is not hypothetical: [p74] §5(b) reported two real spectra
        crashing the engine, and nothing in their record said so.
        """
        def exploding(f, Z, ctx):
            raise RuntimeError("boom")

        f, Z = reference_spectrum()
        mask, results, log = run_gates(f, Z, _ctx(), gates=(exploding,))
        assert mask.all(), "fail-open: a broken gate must not discard a measurement"
        assert results[0].passed and results[0].checked is False
        assert log[0]["checked"] is False and "boom" in log[0]["detail"]

    # ── Sites the runner cannot reach — stated, not dressed up ────────────────

    def test_phase_noise_with_no_finite_points_is_unchecked_but_only_synthetically(self):
        """SYNTHETIC-ONLY, and the reason is a property of the runner.

        ``gate_phase_noise_extrapolated`` sits fifth in ``FRONT1_PRE_CORRECTION``.
        For its ``mag`` to have no finite entry, ``gate_finiteness`` must first have
        masked every point — and ``run_gates`` then finds an empty index set, records
        ``min_points`` and breaks before this gate is ever called. The branch is
        therefore only reachable by calling the gate directly.
        """
        from softae.analysis.eis.gates import gate_phase_noise_extrapolated

        f, Z = reference_spectrum()
        blank = np.full(f.size, np.nan + 1j * np.nan)

        _, results, _ = run_gates(f, blank, _ctx())
        assert "phase_noise_extrapolated" not in {r.name for r in results}, (
            "if the runner ever reaches it, this test's premise has changed")

        r = gate_phase_noise_extrapolated(f, blank, _ctx())
        assert r.passed and r.checked is False

    def test_plateau_with_too_few_finite_points_is_unchecked_but_only_synthetically(self):
        """SYNTHETIC-ONLY. ``gate_min_points`` runs first and blocks the chain.

        ``plateau_in_band`` is last in ``FRONT1_POST_CORRECTION`` and ``min_points``
        is third; fewer than two usable points cannot clear the shipped
        ``min_fit_pts = 8``, so the runner stops before the plateau is measured.
        """
        from softae.analysis.eis.gates import gate_plateau_in_band

        f, Z = reference_spectrum()
        r = gate_plateau_in_band(f[:1], Z[:1], _ctx())
        assert r.passed and r.checked is False
        assert "too few finite points" in r.detail

    def test_a_runs_test_on_under_eight_points_is_unchecked_but_needs_a_lowered_gate(self):
        """Reachable only with ``min_fit_pts`` below the shipped 8.

        The engine fits the points that survived ``gate_min_points``, and ``z_fit``
        is the same length, so ``n < 8`` cannot occur while that gate needs 8.
        """
        from softae.analysis.eis.gates import gate_residual_structure

        f, Z = reference_spectrum()
        f, Z = f[:6], Z[:6]
        ctx = _ctx()
        ctx["fit"] = _fit_with(None, success=True, z_fit=Z * 1.01)

        r = gate_residual_structure(f, Z, ctx)
        assert r.passed and r.checked is False
        assert "too few points" in r.detail

    def test_residual_structure_variance_branch_is_arithmetically_unreachable(self):
        """The ``var <= 0`` guard cannot be entered by any input, production or not.

        It is marked ``unchecked`` for consistency with its neighbours, and this test
        records why no census entry drives it: with ``n_tot >= 8`` and both signs
        present, ``2·n_pos·n_neg`` is minimised at ``n_pos = 1`` and equals
        ``2(n_tot − 1)``, which exceeds ``n_tot`` for every ``n_tot >= 2``. So the
        numerator ``2·n_p·n_n·(2·n_p·n_n − n_tot)`` is strictly positive, and the
        denominator ``n_tot²(n_tot − 1)`` is too. Reported as a finding rather than
        covered by a monkeypatch that would only prove the monkeypatch works.
        """
        for n_tot in range(8, 60):
            for n_pos in range(1, n_tot):
                n_neg = n_tot - n_pos
                var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_tot)) / (
                    n_tot ** 2 * (n_tot - 1))
                assert var > 0, (n_tot, n_pos)

    # ── The list itself ───────────────────────────────────────────────────────

    def test_the_unchecked_site_list_is_pinned_so_a_new_one_cannot_arrive_unnoticed(self):
        """Spec §3.5's table, counted at source.

        A census that only asserts the known sites cannot notice a *new* fail-open
        branch. Counting `GateResult.unchecked` call sites in the module is what
        turns "these nineteen are marked" into "exactly these nineteen exist", so
        adding a twentieth is a deliberate act with a test to update.
        """
        import inspect

        import softae.analysis.eis.gates as gates_module

        source = inspect.getsource(gates_module)
        assert source.count("GateResult.unchecked(") == 19, (
            "the §3.5 census moved — update this count and the tests above "
            "together, and say which site changed")


def _cov_with_rho(rho: float, *, singular: bool = False):
    """A ``FitCovariance`` whose ``R0``/``R1`` correlation is exactly *rho*.

    Unit diagonal, so ``pcov`` *is* the correlation matrix and
    ``split_identifiable``'s ``(1+|ρ|)/(1−|ρ|)`` is readable by eye: ρ = 0.99 is a
    condition number of 199, twelve decades under ``SPLIT_MAX_COND``.
    """
    from softae.analysis.eis.fitter import FitCovariance

    return FitCovariance(
        names=("R0", "R1"), values=np.array([50.0, 2000.0]),
        pcov=np.array([[1.0, float(rho)], [float(rho), 1.0]]), singular=singular,
    )


class TestDegeneracyIsTwoSided:
    """ρ = +1 is the same rank deficiency as ρ = −1, and used to pass.

    The gate's threshold is now applied to ``|ρ|``. On the 30 covariance-bearing
    spectra of ``20260825T154521Z_arrhenius_sweep`` this is strictly a tightening —
    9 PASS→FAIL, 0 FAIL→PASS — because ρ is ``+1.000000`` on 9 of them.

    These tests pin the *contract* (two-sided, on the configured magnitude, with the
    numerical predicate recorded and inert), not the arithmetic that implements it.
    """

    def _gate(self, cov, **overrides):
        from softae.analysis.eis.gates import gate_degeneracy

        f, Z = reference_spectrum()
        ctx = _ctx(**overrides)
        ctx["fit"] = _fit_with(cov)
        return gate_degeneracy(f, Z, ctx)

    def test_degeneracy_positive_unit_correlation_fails(self):
        # The documented bug: `rho > -0.95` passed this cleanly, and the split it
        # endorsed is as invented as the one at ρ = −1.
        r = self._gate(_cov_with_rho(1.0))
        assert not r.passed and r.checked is True
        assert r.metrics["rho"] == pytest.approx(1.0)
        assert "positive degeneracy" in r.detail

    def test_degeneracy_negative_unit_correlation_still_fails(self):
        # Pre-existing verdict, unmoved — the change adds a side, it does not swap one.
        r = self._gate(_cov_with_rho(-1.0))
        assert not r.passed and r.checked is True
        assert "sum" in r.detail

    @pytest.mark.parametrize("rho", [0.5, -0.5, 0.0])
    def test_degeneracy_moderate_correlation_still_passes(self, rho):
        # A tightening at the extremes only. A gate that refused every correlated pair
        # would be making a different claim than "the split is unidentifiable".
        r = self._gate(_cov_with_rho(rho))
        assert r.passed and r.checked is True

    @pytest.mark.parametrize("rho, expected_pass",
                             [(0.94, True), (-0.94, True),
                              (0.96, False), (-0.96, False)])
    def test_degeneracy_default_threshold_is_symmetric_about_zero(self, rho,
                                                                  expected_pass):
        # The shipped `rho_degenerate = -0.95` read as a magnitude: the two sides sit
        # at the same distance from zero, which is the whole content of the change.
        assert self._gate(_cov_with_rho(rho)).passed is expected_pass

    @pytest.mark.parametrize("rho, expected_pass",
                             [(0.97, True), (-0.97, True),
                              (0.995, False), (-0.995, False)])
    def test_degeneracy_configured_threshold_is_honoured_on_both_sides(
            self, rho, expected_pass):
        # `rho_degenerate` is a config key and stays one. Written negative, as the
        # shipped value is, and read as |−0.99| so its sign cannot invert the test.
        r = self._gate(_cov_with_rho(rho), rho_degenerate=-0.99)
        assert r.passed is expected_pass

    def test_degeneracy_positive_threshold_configured_reads_the_same(self):
        # An operator who drops the minus sign gets the same gate, not its inverse.
        assert not self._gate(_cov_with_rho(0.99), rho_degenerate=0.95).passed
        assert self._gate(_cov_with_rho(0.5), rho_degenerate=0.95).passed

    def test_degeneracy_singular_covariance_is_still_a_checked_refusal(self):
        # KEEP: a singular covariance is the answer to this gate's question, not a
        # missing input to it. Two-sidedness must not have turned it into an absence.
        r = self._gate(_cov_with_rho(float("nan"), singular=True))
        assert not r.passed and r.checked is True
        assert "unidentifiable" in r.detail

    def test_degeneracy_unformable_correlation_is_unchecked_not_a_verdict(self):
        # Zero variance on R0 — finite `pcov`, so `singular` is False — makes ρ's
        # denominator zero. `abs(nan) >= t` is False, so a naive two-sided test would
        # report a clean PASS here; the NaN branch must still take precedence.
        from softae.analysis.eis.fitter import FitCovariance

        cov = FitCovariance(names=("R0", "R1"), values=np.array([50.0, 2000.0]),
                            pcov=np.array([[0.0, 0.0], [0.0, 100.0]]))
        assert not cov.singular and np.isnan(cov.rho("R0", "R1"))

        r = self._gate(cov)
        assert r.passed and r.checked is False and "unavailable" in r.detail

    def test_degeneracy_split_identifiable_is_recorded_but_does_not_gate(self):
        """The test that would have caught the rejected proposal.

        ``split_identifiable`` is a float64-representability test —
        ``SPLIT_MAX_COND = 1e14`` is ``|ρ| ≥ 1 − 2e-14`` — so ρ = 0.99 (condition
        number 199) passes it comfortably while being far past any threshold about
        experiment design. Substituting it for the comparison was measured on the
        arrhenius corpus and *regressed* 4 spectra against 1 fixed.
        """
        cov = _cov_with_rho(0.99)
        assert cov.split_identifiable("R0", "R1") is True, (
            "premise: the numerical predicate must disagree with the gate here")

        r = self._gate(cov)
        assert not r.passed, "the gate's own threshold decides, not the 1e14 one"
        assert r.metrics["split_identifiable"] == 1.0

    def test_degeneracy_split_identifiable_records_zero_at_unit_correlation(self):
        # Where the two predicates agree, the metric says so — which is what makes the
        # 18-of-30 versus 30-of-30 split legible in the log.
        r = self._gate(_cov_with_rho(1.0))
        assert r.metrics["split_identifiable"] == 0.0

    def test_degeneracy_covariance_without_the_predicate_records_nan(self):
        """A covariance object that predates the method must degrade, not explode.

        ``split_identifiable`` is newer than ``FitCovariance`` itself and the metric is
        advisory, so a stand-in without it reports NaN — "not asked" — rather than
        taking down a gate whose verdict does not depend on it.
        """
        class _CovWithoutSplitTest:
            singular = False

            def rho(self, a, b):
                return -0.99

        r = self._gate(_CovWithoutSplitTest())
        assert not r.passed, "the verdict comes from ρ alone and is unaffected"
        assert np.isnan(r.metrics["split_identifiable"])


class TestPhaseNoiseFallbackIsConservative:
    def test_phase_noise_envelope_without_the_predicate_is_not_assumed_in_band(self):
        """SYNTHETIC-ONLY, and reconciliation rather than repair.

        No production path reaches this fallback: ``policy.build_context`` substitutes
        ``instrument_envelope()`` when none is passed, ``engine.analyze_spectrum`` does
        the same, and the one hand-built stand-in in the tree supplies the method. The
        test pins the *direction* of the default — matching ``report.py``'s
        ``_reporting_mode`` on the same predicate — so the permissive reading cannot
        drift back in. ``SUBAGENT_RULES`` §3.1(a): "unknown" must not be spelled with
        the same token as "checked and clean".
        """
        from types import SimpleNamespace

        from softae.analysis.eis.gates import gate_phase_noise_extrapolated

        env = SimpleNamespace()
        assert not hasattr(env, "phase_noise_valid_at"), "premise of this test"

        f, Z = reference_spectrum()
        ctx = _ctx()
        ctx["envelope"] = env

        r = gate_phase_noise_extrapolated(f, Z, ctx)
        assert not r.passed
        assert r.metrics["phase_noise_valid"] == 0.0

    def test_phase_noise_real_envelope_still_judges_by_band(self):
        # Positive control for the test above: with a real envelope the gate still
        # answers from the measurement, so the fallback is what changed and nothing
        # else.
        from softae.analysis.eis.envelope import instrument_envelope
        from softae.analysis.eis.gates import gate_phase_noise_extrapolated

        f, Z = reference_spectrum(R_bulk=1.0e4, noise_pct=0.0)
        ctx = _ctx()
        ctx["envelope"] = instrument_envelope()
        assert gate_phase_noise_extrapolated(f, Z, ctx).passed
