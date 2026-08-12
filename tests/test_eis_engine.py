"""The engine entry point (E0) — two engines, one return shape, legacy unchanged.

The hard constraint on this whole overhaul is that the existing EIS path stays intact
and stays the default until the gated path is validated on the bench. These tests are
that constraint made mechanical rather than asserted in a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.policy import reduce_gates
from softae.analysis.eis.report import SigmaReport, SpectrumReport
from softae.analysis.eis.settings import EISSettings, GateSettings, eis_settings
from softae.analysis.quality import Verdict
from tests.eis_synthetic import (
    as_eis_result,
    pure_series_rc,
    reference_spectrum,
    stuck_instrument,
)

CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")


def _gated(enabled: bool = True) -> EISSettings:
    return EISSettings(engine="gated", gates=GateSettings(enabled=enabled))


class TestEngineSelection:
    def test_an_unconfigured_rig_selects_the_legacy_engine_because_the_gated_path_is_unvalidated(self):
        assert eis_settings().engine == "legacy"
        assert eis_settings().gates.enabled is False

    def test_an_unknown_engine_name_falls_back_to_legacy_rather_than_raising(self):
        # A typo in a config file must not stop a campaign that would otherwise have
        # run exactly as it always has.
        assert eis_settings({"engine": "gatd"}).engine == "legacy"

    def test_both_engines_return_the_same_shape_so_consumers_never_branch_on_engine(self):
        eis = as_eis_result(*reference_spectrum())
        legacy = analyze_spectrum(eis, cell=CELL, engine="legacy")
        gated = analyze_spectrum(eis, cell=CELL, settings=_gated())
        assert isinstance(legacy, SpectrumReport) and isinstance(gated, SpectrumReport)
        assert isinstance(legacy.sigma, SigmaReport)
        assert isinstance(gated.sigma, SigmaReport)


class TestLegacyEngineIsUntouched:
    def test_the_legacy_engine_runs_no_gates_at_all(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, engine="legacy")
        assert report.gate_log == ()
        assert report.mask is None
        assert report.gate_summary() == "—"

    def test_the_legacy_engine_reports_sigma_from_r1_exactly_as_before(self):
        eis = as_eis_result(*reference_spectrum())
        report = analyze_spectrum(eis, cell=CELL, engine="legacy")
        if report.fit.success:
            assert report.sigma.value == pytest.approx(CELL.sigma(report.fit.R1))
            assert report.sigma.R_basis == "split_bulk"

    def test_the_legacy_engine_does_not_reject_a_spectrum_the_gates_would_refuse(self):
        # A series-RC spectrum contains no conductivity, but the legacy path has
        # always fitted it and must keep doing so until the cutover is deliberate.
        eis = as_eis_result(*pure_series_rc())
        report = analyze_spectrum(eis, cell=CELL, engine="legacy")
        assert report.engine == "legacy"
        assert report.gate_log == ()


class TestGatedEngine:
    def test_a_clean_spectrum_is_admitted_and_keeps_every_point(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, settings=_gated())
        blocking = [e for e in report.gate_log
                    if not e["passed"]
                    and e["severity"] in ("block_spectrum", "block_session")]
        assert blocking == []
        assert report.mask.all()
        assert report.gate_summary() == "pass"

    def test_the_only_advisory_a_realistic_spectrum_trips_is_the_marginal_plateau(self):
        # Front-2 flags are diagnostics, not admission criteria. The model-free
        # 1/max(Re Y) estimator disagrees with the fit by ~40%, which is the estimator
        # degrading as the plateau is squeezed between the blocking onset and the
        # relaxation corner — the signal §3.7 asks to be flagged, not a bad fit.
        report = analyze_spectrum(
            as_eis_result(*reference_spectrum(noise_pct=1.0, seed=3)),
            cell=CELL, settings=_gated())
        flagged = {e["gate"] for e in report.gate_log
                   if not e["passed"] and e["severity"] == "flag"}
        assert flagged == {"model_free_crosscheck"}

    def test_a_noise_free_spectrum_trips_the_runs_test_by_construction(self):
        # Worth pinning rather than working around: with no noise, the residuals are
        # pure systematic fit error and therefore structured by definition — a runs
        # test has nothing random to compare against. The gate is behaving correctly;
        # it is the *input* that is unphysical. Real spectra carry noise, which is why
        # the test above uses some.
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, settings=_gated())
        runs = next(e for e in report.gate_log if e["gate"] == "residual_structure")
        assert not runs["passed"]
        assert runs["severity"] == "flag", "must never reject on this alone"

    def test_the_gated_fitter_recovers_the_series_term_the_legacy_one_absorbs(self):
        # The legacy path returns ~2900 Ω for a true 50 Ω series term, for two
        # compounding reasons: curve_fit runs unscaled and stops at iteration zero,
        # and it is unweighted so the low-frequency end dominates. The gated engine
        # fits through fit_with_covariance, which fixes both.
        from softae.analysis.circuit_fitting import fit_circuit
        from tests.eis_synthetic import DEFAULT_R_BULK, DEFAULT_R_SERIES

        eis = as_eis_result(*reference_spectrum())
        legacy = fit_circuit(eis, "simpleSalt")
        gated = analyze_spectrum(eis, cell=CELL, settings=_gated(enabled=False)).fit

        assert gated is not None and gated.success
        assert gated.R0 == pytest.approx(DEFAULT_R_SERIES, rel=1e-2)
        assert gated.R1 == pytest.approx(DEFAULT_R_BULK, rel=1e-2)
        assert abs(gated.R0 - DEFAULT_R_SERIES) < abs(legacy.R0 - DEFAULT_R_SERIES)
        assert gated.covariance is not None

    def test_an_inadmissible_spectrum_is_not_fitted_at_all(self):
        # R18: a failed gate must surface as a labelled rejection, not as a fit with
        # 1000 % residuals that still reports an R1.
        report = analyze_spectrum(as_eis_result(*pure_series_rc()),
                                  cell=CELL, settings=_gated(enabled=True))
        assert report.fit is None
        assert report.quality.verdict is Verdict.REJECT
        assert report.sigma.mode == "unavailable"
        assert report.gate_summary().startswith("REJECTED:")

    def test_observing_only_mode_still_fits_because_it_must_not_change_behaviour(self):
        # gates.enabled = false exists so thresholds can be watched against real runs.
        # A mode that silently stopped fitting would be enforcement in disguise.
        report = analyze_spectrum(as_eis_result(*pure_series_rc()),
                                  cell=CELL, settings=_gated(enabled=False))
        assert report.fit is not None
        assert report.quality.verdict is not Verdict.REJECT
        assert any("observing only" in i for i in report.quality.issues)

    def test_the_gate_log_survives_onto_the_report_for_every_spectrum(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, settings=_gated())
        assert report.gate_log
        for entry in report.gate_log:
            assert set(entry) == {"gate", "severity", "passed", "detail", "n_dropped"}

    def test_a_stuck_instrument_is_rejected_before_anything_is_fitted(self):
        report = analyze_spectrum(as_eis_result(*stuck_instrument()),
                                  cell=CELL, settings=_gated())
        assert report.fit is None
        assert "stuck_instrument" in report.gate_summary()

    def test_without_a_thickness_the_resistance_is_reported_but_sigma_is_not(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=None, settings=_gated())
        assert report.sigma.mode == "unavailable"
        assert np.isnan(report.sigma.value)

    def test_a_model_free_cross_check_accompanies_the_fit(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, settings=_gated())
        assert np.isfinite(report.sigma.model_free_R_ohm)


class TestBoundReporting:
    def test_a_spectrum_far_from_the_phase_calibration_reports_provisionally(self):
        # Phase noise is measured (0.149° at 9.9 kΩ, resistive), so a headroom
        # comparison is possible — but a film four decades higher is outside the band
        # where that number was taken, and carrying an instrument constant that far
        # without saying so is how the withdrawn Z_φ ceiling was born.
        f, Z = reference_spectrum()
        report = analyze_spectrum(as_eis_result(f, Z * 1e4), cell=CELL,
                                  settings=_gated(enabled=False))
        assert report.sigma.provisional

    def test_an_unmeasured_phase_noise_yields_a_bound_never_a_value(self):
        from softae.analysis.eis.envelope import InstrumentEnvelope

        blind = InstrumentEnvelope(phase_noise_measured=False,
                                   phase_noise_deg=float("nan"))
        report = analyze_spectrum(as_eis_result(*reference_spectrum()), cell=CELL,
                                  settings=_gated(enabled=False), envelope=blind)
        assert report.sigma.mode == "bound_unqualified"
        assert report.sigma.provisional
        assert np.isnan(report.sigma.value)
        assert np.isfinite(report.sigma.upper_bound)

    def test_a_bound_renders_as_a_ceiling_and_says_it_is_provisional(self):
        text = SigmaReport(mode="bound_unqualified", upper_bound=4e-7,
                           provisional=True).as_text()
        assert "≲" in text and "provisional" in text

    def test_a_bound_reduces_to_suspect_because_it_is_a_result_but_not_a_value(self):
        report = reduce_gates([], n_surviving=20, min_fit_pts=8, report_mode="bound")
        assert report.verdict is Verdict.SUSPECT
        assert report.ok


class TestVocabularyBridge:
    def test_dropped_points_reduce_to_suspect_because_the_spectrum_itself_survived(self):
        from softae.analysis.eis.gates import BLOCK_POINT, GateResult

        dropped = GateResult("quadrant", BLOCK_POINT, False, "3 pts",
                             np.array([False, False, False, True]))
        report = reduce_gates([dropped], n_surviving=20, min_fit_pts=8)
        assert report.verdict is Verdict.SUSPECT

    def test_too_few_survivors_reduce_to_reject_even_when_no_gate_blocked(self):
        report = reduce_gates([], n_surviving=3, min_fit_pts=8)
        assert report.verdict is Verdict.REJECT
        assert not report.ok

    def test_a_blocking_failure_reduces_to_reject_so_the_optimizer_is_told_nothing(self):
        from softae.analysis.eis.gates import BLOCK_SPECTRUM, GateResult

        blocked = GateResult("tand_slope", BLOCK_SPECTRUM, False, "series",
                             np.ones(4, bool))
        report = reduce_gates([blocked], n_surviving=20, min_fit_pts=8)
        assert report.verdict is Verdict.REJECT

    def test_gate_metrics_are_carried_into_the_report_for_later_calibration(self):
        from softae.analysis.eis.gates import FLAG, GateResult

        flagged = GateResult("cap_flatness", FLAG, False, "dispersive",
                             np.ones(4, bool), {"cap_slope": -1.3})
        report = reduce_gates([flagged], n_surviving=20, min_fit_pts=8)
        assert report.metrics["cap_slope"] == -1.3
