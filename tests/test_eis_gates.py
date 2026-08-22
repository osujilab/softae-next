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
from softae.analysis.eis.policy import build_context
from softae.analysis.eis.settings import GateSettings


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


class TestPointGates:
    def test_negative_re_z_points_are_dropped_because_they_are_artefact_not_measurement(self):
        f, Z = negative_real_part(n_points=3)
        r = gate_quadrant(f, Z, _ctx())
        assert r.severity == BLOCK_POINT
        assert r.n_dropped == 3

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
        entry = GateResult("x", FLAG, True, "d", np.ones(2, bool)).as_log_entry()
        assert set(entry) == {"gate", "severity", "passed", "detail", "n_dropped"}


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
        assert "R_sol_valley" in ctx and ctx["R_sol_valley"] > 0
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
        assert ctx["R_sol_valley"] == pytest.approx(2050.0, rel=0.25)
        assert ctx["R_sol_valley"] > float(np.min(np.abs(Z)))
        assert res.metrics["valley_over_zmin"] > 1.0

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
