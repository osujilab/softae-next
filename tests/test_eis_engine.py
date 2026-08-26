"""The engine entry point (E0) — two engines, one return shape, legacy unchanged.

The hard constraint on this whole overhaul is that the existing EIS path stays intact
and stays the default until the gated path is validated on the bench. These tests are
that constraint made mechanical rather than asserted in a comment.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from softae.analysis.circuit_fitting import fit_circuit
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
    def test_an_unconfigured_rig_selects_the_engine_whose_failure_mode_is_better_understood(self):
        """The shipped default is ``legacy``, and this is provisional, not settled.

        Measured against a numpy-only physics anchor (Kása circle right-intercept plus
        low-f Re(Y) plateau, no project analysis code) on all ten real probe-3ch-v3
        spectra with ``engine`` passed explicitly, legacy is closer on 6/10: median
        error 5.54× low vs 8.75×, worst case 17.9× vs 224.3×, within 5× on 5/10 vs
        3/10. The decisive pair is ch32_002/ch32_003, whose arc sits below the sweep
        floor: legacy 4.06× and 4.83× low, gated 224.3× and 182.7× low ([a97] §1-§2).

        The test's name is the claim, and it is literal rather than rhetorical. Legacy
        is the engine that reports "arc did not close in band — R1 extrapolated" where
        gated returns a confident wrong number ([a97] §3); the failure mode is better
        understood because legacy announces it. Legacy is not thereby *right* — both
        engines read low on all ten, and on 8 of the 10 neither is fit to report R1,
        because the arc was never in the measured band.

        Gated stays the direction of travel. Its deficit is concentrated on arcs below
        the sweep floor — the mechanism `kk_truncation` ([a89]) describes — so
        extending the sweep downward, or gating on ``arc_closure()``, is what flips
        this assertion back. Expect that; do not treat this as permanent.

        The second assertion is a separate claim from the first and survives either
        setting unchanged: gates run and LOG under this default, and refuse nothing.
        """
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


# ── A fit that railed on its own bound ───────────────────────────────────────

class TestRailedFitsAreNotMeasurements:
    """335 of 1440 fits in run ``20260811T023757Z_equilibration_characterization``
    (23.3 %) came to rest on the ``simpleSalt`` R₁ floor of 100 Ω and stored
    σ = 0.5 S/cm — roughly seawater, from a dry polymer film — with ``success = 1``.
    A series-RC spectrum reproduces it exactly: the optimiser is asked for a bulk
    arc that is not there, and reports where the wall was."""

    def test_engine_railed_fit_does_not_report_success(self):
        report = analyze_spectrum(as_eis_result(*pure_series_rc()),
                                  cell=CELL, engine="legacy")
        assert report.fit.success is False
        assert "railed" in report.fit.error_msg
        # The bound and its value are named on the row, so a railed fit is
        # distinguishable from one that failed to converge without re-deriving it.
        assert "100" in report.fit.error_msg

    def test_engine_railed_fit_yields_no_sigma(self):
        report = analyze_spectrum(as_eis_result(*pure_series_rc()),
                                  cell=CELL, engine="legacy")
        assert report.sigma.mode == "unavailable"
        assert report.sigma.value != report.sigma.value       # NaN, not 0.5 S/cm
        # σ follows from R₁ everywhere it is computed, including inside
        # `record_fit`, which cannot see the demotion — so R₁ has to carry it.
        assert report.fit.R1 != report.fit.R1

    def test_engine_railed_fit_keeps_the_railed_value_as_a_diagnostic(self):
        # Demoted, not erased: the parameter vector still says *where* it railed,
        # and that is what `parameters_json` stores.
        report = analyze_spectrum(as_eis_result(*pure_series_rc()),
                                  cell=CELL, engine="legacy")
        assert report.fit.parameters[3] == pytest.approx(100.0, rel=1e-3)
        assert any("rests on" in issue for issue in report.quality.issues)

    def test_engine_converged_fit_is_unaffected(self):
        # The regression pin: 1381 of those 1440 fits were fine and must report
        # exactly what they reported before.
        eis = as_eis_result(*reference_spectrum())
        report = analyze_spectrum(eis, cell=CELL, engine="legacy")
        raw = fit_circuit(eis, "simpleSalt")

        assert report.fit.success is True
        assert report.fit.error_msg == ""
        assert report.fit.R1 == pytest.approx(raw.R1)
        assert report.fit.R0 == pytest.approx(raw.R0)
        assert report.sigma.mode == "value"
        assert report.sigma.value == pytest.approx(CELL.sigma(raw.R1))

    def test_record_fit_railed_fit_stores_no_conductivity(self, tmp_path):
        # End to end through the surface the campaign actually reads. `record_fit`
        # derives σ from `fit_result.R1` and knows nothing about railing, so this
        # is the assertion that the demotion reaches the database at all.
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "project")
        try:
            run_id = store.start_run("railed")
            eis = as_eis_result(*pure_series_rc())
            measurement_id = store.record_measurement(run_id, eis)
            fit = analyze_spectrum(eis, cell=CELL, engine="legacy").fit
            fit_id = store.record_fit(measurement_id, fit,
                                      L_cm=0.2, t_cm=0.015, w_cm=0.2)
            row = dict(store._conn.execute(
                "SELECT success, sigma_S_per_cm, R1, error_msg FROM fit_results "
                "WHERE fit_id = ?", (fit_id,)).fetchone())
        finally:
            store.close()

        assert row["success"] == 0
        assert row["sigma_S_per_cm"] is None, "0.5 S/cm from a dry film"
        assert row["R1"] is None
        assert "railed" in row["error_msg"]


class TestRailedDetectionReadsTheModelsBounds:
    """The bound is never a literal here. It is read from the registry that fitted
    the spectrum, so editing the registry moves what counts as railed."""

    @staticmethod
    def _fit(r1: float):
        return SimpleNamespace(model_name="simpleSalt", R1=r1, covariance=None)

    def test_railed_measurand_follows_the_registry_bound(self, monkeypatch):
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS
        from softae.analysis.eis.models import railed_measurand

        assert railed_measurand(self._fit(100.0))
        assert not railed_measurand(self._fit(5.0e4))

        # Drop the model's own R₁ floor to 1 Ω and 100 Ω stops being a rail — it
        # is two decades clear of the constraint and therefore set by the data.
        # Nothing in the detector had to be edited to follow it.
        lower, upper = CIRCUIT_MODELS["simpleSalt"]["bounds"]
        moved = list(lower)
        moved[CIRCUIT_MODELS["simpleSalt"]["z_indices"][1]] = 1.0
        monkeypatch.setitem(CIRCUIT_MODELS["simpleSalt"], "bounds", (moved, upper))

        assert not railed_measurand(self._fit(100.0))
        assert railed_measurand(self._fit(1.0))

    def test_railed_measurand_unbounded_model_can_never_rail(self):
        from softae.analysis.eis.models import railed_measurand

        # `flexSalt` declares no bounds, so no fit of it rests on one. Reporting a
        # rail there would be inventing a constraint the optimiser never had.
        assert not railed_measurand(
            SimpleNamespace(model_name="flexSalt", R1=100.0, covariance=None))


#: Fit metrics comfortably inside the shipped limits and outside the ones the tests
#: below set. Pinning them is what makes a threshold test turn on the threshold rather
#: than on the optimiser's last digits.
PINNED_METRICS = {"r_squared": 0.97, "residual_rms_pct": 5.0}


@pytest.fixture()
def pinned_fit_quality(monkeypatch):
    """A real fit, with a known ``quality`` dict.

    Both fitters are wrapped because the two engines use different ones — legacy fits
    through ``circuit_fitting.fit_circuit``, gated through ``fitter.fit_spectrum`` with
    ``fit_circuit`` as its fallback — and a pin that covered only one would silently
    stop pinning the moment the other ran.
    """
    from softae.analysis import circuit_fitting
    from softae.analysis.eis import fitter

    def wrap(real):
        def fit_with_pinned_quality(eis_result, model_name="simpleSalt", **kw):
            fit = real(eis_result, model_name, **kw)
            fit.quality = dict(PINNED_METRICS)
            return fit

        return fit_with_pinned_quality

    monkeypatch.setattr(circuit_fitting, "fit_circuit", wrap(circuit_fitting.fit_circuit))
    monkeypatch.setattr(fitter, "fit_spectrum", wrap(fitter.fit_spectrum))


def _quality_overlay(monkeypatch, **keys):
    """Replace ``[quality]`` at the loader, leaving every other section alone.

    The loader is the one parse point ``quality_config`` reads, so patching here
    exercises the real resolution; patching ``grade_fit``'s caller would test the test.
    Every *other* section is carried through from the real file because the gated
    engine also reads ``[eis.instrument]``, ``[eis.pregate]`` and ``[eis.fixture]`` —
    handing it a bare dict would move the gates as well as the thresholds, and the
    comparison would no longer isolate what it claims to.
    """
    from softae.config import loader

    overlaid = dict(loader.load())
    overlaid["quality"] = keys
    monkeypatch.setattr(loader, "load", lambda *a, **k: overlaid)


def _shipped_equals_defaults():
    """The premise the whole no-op property rests on, checked rather than assumed."""
    from softae.analysis.quality import (
        DEFAULT_MAX_RESIDUAL_PCT,
        DEFAULT_MIN_R_SQUARED,
        quality_config,
    )

    cfg = quality_config()
    return (cfg["min_r_squared"] == DEFAULT_MIN_R_SQUARED
            and cfg["max_residual_pct"] == DEFAULT_MAX_RESIDUAL_PCT)


class TestLegacyEngineHonoursTheQualityConfig:
    """``[quality] min_r_squared`` and ``max_residual_pct`` must reach this path.

    They did not. ``_legacy_report`` called ``grade_fit`` with neither threshold, so
    it graded against the module defaults — and the shipped config repeats those
    defaults exactly (0.95 / 15.0), so nothing ever looked wrong. An operator editing
    the file simply saw no effect, on the engine that is still the rig's default.
    """

    @staticmethod
    def _verdict():
        return analyze_spectrum(as_eis_result(*reference_spectrum()),
                                cell=CELL, engine="legacy").quality.verdict

    def test_legacy_engine_configured_min_r_squared_flips_accept_to_reject(
            self, monkeypatch, pinned_fit_quality):
        assert self._verdict() is Verdict.ACCEPT
        _quality_overlay(monkeypatch, min_r_squared=0.99)
        assert self._verdict() is Verdict.REJECT

    def test_legacy_engine_configured_max_residual_pct_flips_accept_to_reject(
            self, monkeypatch, pinned_fit_quality):
        assert self._verdict() is Verdict.ACCEPT
        _quality_overlay(monkeypatch, max_residual_pct=2.0)
        assert self._verdict() is Verdict.REJECT

    def test_legacy_engine_a_loosened_threshold_admits_what_the_default_refuses(
            self, monkeypatch, pinned_fit_quality):
        # The other direction, because a threshold that could only ever tighten would
        # be indistinguishable from a hard-coded floor plus an extra rejection rule.
        _quality_overlay(monkeypatch, min_r_squared=0.99)
        assert self._verdict() is Verdict.REJECT
        _quality_overlay(monkeypatch, min_r_squared=0.5, max_residual_pct=99.0)
        assert self._verdict() is Verdict.ACCEPT

    def test_legacy_engine_shipped_config_leaves_every_verdict_where_it_was(
            self, pinned_fit_quality):
        # The no-op property. Routing the thresholds through config is only safe
        # because the file and the defaults agree today; if someone edits
        # softae_config.toml this fails here rather than at the bench.
        from softae.analysis.quality import grade_fit

        assert _shipped_equals_defaults()
        assert self._verdict() is grade_fit(PINNED_METRICS, success=True).verdict

    def test_legacy_engine_unreadable_config_falls_back_to_the_defaults(
            self, monkeypatch, pinned_fit_quality):
        # `quality_config` already swallows a broken load; the point here is that
        # `_legacy_report` did not acquire a second config read that does not.
        from softae.config import loader

        def boom(*a, **k):
            raise OSError("config unreadable")

        monkeypatch.setattr(loader, "load", boom)
        assert self._verdict() is Verdict.ACCEPT


class TestGatedEngineHonoursTheQualityConfig:
    """The same defect, and the same fix, on the other engine.

    ``analyze_spectrum``'s gated branch graded its fit against ``grade_fit``'s defaults
    too, so the two engines could not have disagreed about a configured threshold —
    they were both ignoring it. One grading standard across both is the premise of
    comparing them at all.

    Scope: these tests pin what ``fit_report`` *says*. Who acts on it — the
    ``and gate_cfg.enabled`` conjunction below the call, a third authority flag
    distinct from ``[quality] enabled`` — is an open design question and is not
    asserted here beyond leaving it visible: the gates are enabled in these runs, which
    is the configuration under which a fit-quality rejection reaches the verdict.
    """

    @staticmethod
    def _verdict():
        return analyze_spectrum(as_eis_result(*reference_spectrum()),
                                cell=CELL, settings=_gated()).quality.verdict

    def test_gated_engine_configured_min_r_squared_flips_an_admitted_spectrum_to_reject(
            self, monkeypatch, pinned_fit_quality):
        # Both runs go through the overlay, so the *only* difference between them is
        # the threshold — the gates, envelope and fixture are identical either side.
        _quality_overlay(monkeypatch, min_r_squared=0.95, max_residual_pct=15.0)
        assert self._verdict() is not Verdict.REJECT
        _quality_overlay(monkeypatch, min_r_squared=0.99, max_residual_pct=15.0)
        assert self._verdict() is Verdict.REJECT

    def test_gated_engine_configured_max_residual_pct_flips_an_admitted_spectrum_to_reject(
            self, monkeypatch, pinned_fit_quality):
        _quality_overlay(monkeypatch, min_r_squared=0.95, max_residual_pct=15.0)
        assert self._verdict() is not Verdict.REJECT
        _quality_overlay(monkeypatch, min_r_squared=0.95, max_residual_pct=2.0)
        assert self._verdict() is Verdict.REJECT

    def test_gated_engine_shipped_config_leaves_the_verdict_where_it_was(
            self, monkeypatch, pinned_fit_quality):
        # The no-op property on this engine. The shipped file and the defaults agree,
        # so overlaying the defaults explicitly must reproduce the file's own verdict
        # exactly — which is what the unfixed code did on every spectrum.
        assert _shipped_equals_defaults()
        shipped = self._verdict()
        _quality_overlay(monkeypatch, min_r_squared=0.95, max_residual_pct=15.0)
        assert self._verdict() is shipped
        assert shipped is not Verdict.REJECT

    def test_gated_engine_the_fit_grade_is_not_the_only_authority_over_the_verdict(
            self, monkeypatch, pinned_fit_quality):
        # The boundary of this change, made mechanical. A configured threshold that
        # rejects the fit does NOT reach the verdict while [eis.gates] enabled is
        # false, because a separate flag governs that write. If someone later unifies
        # the three authority flags, this test is where they will notice.
        _quality_overlay(monkeypatch, min_r_squared=0.99)
        observing = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                     cell=CELL, settings=_gated(enabled=False))
        assert observing.quality.verdict is not Verdict.REJECT
        # …and it is not silent about it: the configured limit is named on the report,
        # which is the whole point of observing-only mode.
        assert any("0.990" in issue for issue in observing.quality.issues)


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
        # The subject is the one in the name: the log reaches the report, entry by
        # entry, for every spectrum. Exhaustiveness was never part of that. This line
        # once read `set(entry) == {...}`, and that closed set — incidental to the
        # subject — made an unrelated field added to `as_log_entry` look like a
        # regression here. The entry is an **open record**: these five keys are the
        # contract every reader relies on, and extra keys are not a breakage.
        # If the "no unexpected keys" property is wanted, it deserves its own named
        # test written against `as_log_entry` in analysis/eis/gates.py, where the
        # shape is actually defined — not a rider on this one.
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, settings=_gated())
        assert report.gate_log
        required = {"gate", "severity", "passed", "detail", "n_dropped"}
        for entry in report.gate_log:
            assert required <= set(entry), f"entry missing {required - set(entry)}"

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


# ── T7.1 part A: the pass side of the log ────────────────────────────────────
#
# Every fixture below is module-scoped, and the choice is operational rather than
# stylistic. `analyze_spectrum` in the SHADOW configuration (`engine="gated"`,
# `gates.enabled=False`) is the most expensive setting the rig has: an enforcing gate
# rejects a blocking spectrum before the fitter, an *observing* one does not, so the
# optimiser grinds a parallel-R model onto data with no arc — 78 s on a fully-blocking
# synthetic against 0.07 s on the legacy engine.
#
# So the rejected-spectrum fixture is `stuck_instrument` (2 s) rather than
# `pure_series_rc` (78 s). Both fail a `block_spectrum` gate under observation, which
# is the property every test here needs; only the gate's name differs, and the fits
# that make `pure_series_rc` expensive are already pinned by `TestGatedEngine` above.


def _analyse(eis, **kwargs):
    """One gated analysis, with its ``eis_spectrum_metrics`` event."""
    import structlog

    with structlog.testing.capture_logs() as logs:
        report = analyze_spectrum(eis, cell=CELL, **kwargs)
    events = [e for e in logs if e["event"] == "eis_spectrum_metrics"]
    return report, events


@pytest.fixture(scope="module")
def healthy_analysis():
    """A clean spectrum through the shadow configuration — the case with no log today."""
    return _analyse(as_eis_result(*reference_spectrum()), settings=_gated(enabled=False))


@pytest.fixture(scope="module")
def rejected_analysis():
    """A spectrum a ``block_spectrum`` gate refuses, *observed* rather than enforced.

    The shadow-campaign configuration on the population it exists to watch: the verdict
    reduces to SUSPECT because observing mode downgrades a rejection, and the fit
    happens anyway because observing must not change behaviour.
    """
    return _analyse(as_eis_result(*stuck_instrument()), settings=_gated(enabled=False))


@pytest.fixture(scope="module")
def enforced_analysis():
    """The same spectrum with the gates enforcing — the R18 early return.

    Free by construction: the gate rejects it before the fitter runs, which is the
    whole point of R18 and the reason observing mode is the expensive one.
    """
    return _analyse(as_eis_result(*stuck_instrument()), settings=_gated(enabled=True))


@pytest.fixture(scope="module")
def channel_11_analysis():
    f, Z = reference_spectrum()
    return _analyse(as_eis_result(f, Z, channel=11), settings=_gated(enabled=False))


def _rendered_events(processors):
    """Run one gated analysis through a real renderer and parse the text back.

    ``capture_logs`` short-circuits the processor chain, so it can never prove the
    rendered line is parseable — which is the only property the review tool depends on.
    """
    import io

    import structlog

    from softae.tools.shadow_review import parse_line

    buf = io.StringIO()
    saved = structlog.get_config()
    try:
        structlog.reset_defaults()
        kwargs = {"logger_factory": structlog.PrintLoggerFactory(buf)}
        if processors is not None:
            kwargs["processors"] = processors
        structlog.configure(**kwargs)
        analyze_spectrum(as_eis_result(*reference_spectrum()), cell=CELL,
                         settings=_gated(enabled=False))
    finally:
        structlog.configure(**saved)

    return [p for p in (parse_line(ln) for ln in buf.getvalue().splitlines())
            if p and p["event"] == "eis_spectrum_metrics"]


@pytest.fixture(scope="module")
def console_events():
    return _rendered_events(None)


@pytest.fixture(scope="module")
def json_events():
    import structlog

    return _rendered_events([structlog.processors.JSONRenderer()])


class TestSpectrumMetricsEvent:
    """``reduce_gates`` logs ``metrics=`` only where a spectrum FAILS.

    So a shadow run records the failing tail and nothing else, and every threshold rule
    worth having needs the shape of the *healthy* population — you cannot place a fence
    when you can only see what is already outside it. These pin that the event exists
    for every spectrum, carries what the recommender needs, survives the reviewer's
    parser, and changes nothing.
    """

    def test_the_metrics_event_is_emitted_once_for_a_clean_accept_spectrum(
            self, healthy_analysis):
        # The case today's log has no record of at all: nothing is emitted for a
        # spectrum that passes, so the pass-side distribution does not exist.
        _, events = healthy_analysis
        assert len(events) == 1
        assert events[0]["verdict"] in ("accept", "suspect")
        assert events[0]["enforced"] is False

    def test_the_metrics_event_is_emitted_for_a_rejected_spectrum_too(
            self, rejected_analysis, enforced_analysis):
        # Both the R18 early return and the terminal return emit, so no path escapes
        # and a spectrum cannot vanish from the population when the flag flips.
        for (_, events), enforced in ((rejected_analysis, False),
                                      (enforced_analysis, True)):
            assert len(events) == 1
            assert events[0]["enforced"] is enforced

    def test_the_r18_early_return_reports_no_report_mode_rather_than_guessing_one(
            self, enforced_analysis):
        # The value-vs-bound decision is taken after the fit, and this spectrum never
        # reached one. Naming a mode here would invent a decision nobody made.
        _, events = enforced_analysis
        assert events[0]["report_mode"] == "not_reached"
        assert events[0]["fit_ok"] is False

    def test_the_metrics_event_carries_r_squared_which_reduce_gates_never_sees(
            self, healthy_analysis):
        # The reason the emit site is engine.py rather than policy.reduce_gates:
        # `quality.metrics.update(fit_report.metrics)` runs AFTER the reduction, so an
        # event raised inside it could never recommend [quality] min_r_squared.
        _, events = healthy_analysis
        assert "r_squared" in events[0]["metrics"]
        assert "residual_rms_pct" in events[0]["metrics"]

    def test_the_metrics_event_carries_its_own_channel_rather_than_relying_on_position(
            self, channel_11_analysis):
        _, events = channel_11_analysis
        assert events[0]["channel"] == 11
        assert events[0]["spectrum_key"].startswith("c11:")

    def test_non_finite_metrics_are_dropped_so_the_mapping_stays_parseable(
            self, healthy_analysis, rejected_analysis):
        # repr(nan) is a bare `nan`, which ast.literal_eval refuses — one NaN would
        # degrade the whole rendered metrics={...} to an unparsed string downstream.
        # The rejected spectrum is the one that produces NaN metrics in quantity.
        for _, events in (healthy_analysis, rejected_analysis):
            assert events[0]["metrics"]
            assert all(np.isfinite(v) for v in events[0]["metrics"].values())

    def test_the_metrics_event_agrees_with_the_report_it_was_emitted_beside(
            self, healthy_analysis, rejected_analysis):
        # Part A is additive logging only. That the verdicts and sigma themselves did
        # not move is pinned by the thirty-odd tests ABOVE this class, which are
        # untouched by T7.1 and would fail first; what is left to check here is that
        # the event describes the same spectrum the report does, rather than a stale
        # interim taken before the Front-2 merge.
        for report, events in (healthy_analysis, rejected_analysis):
            event = events[0]
            assert event["verdict"] == report.quality.verdict.value
            assert report.mask.sum() == event["n_surviving"]
            assert report.mask.size - report.mask.sum() == event["n_dropped"]
            assert report.quality.metrics["n_surviving"] == event["n_surviving"]

    def test_the_metrics_event_names_every_gate_that_ran_and_every_one_that_failed(
            self, healthy_analysis, rejected_analysis):
        for _, events in (healthy_analysis, rejected_analysis):
            assert set(events[0]["gates_failed"]) <= set(events[0]["gates_run"])
        # An admitted spectrum runs the whole Front-1 chain, topology triad included.
        assert "tand_slope" in healthy_analysis[1][0]["gates_run"]
        # A refused one does not, and the event must say so rather than implying the
        # later gates ran and passed. `run_gates` breaks at a blocking failure, so a
        # recommender counting "spectra carrying tand_slope" must see this spectrum
        # absent from that denominator — not present with a silent zero.
        rejected = rejected_analysis[1][0]
        assert "stuck_instrument" in rejected["gates_failed"]
        assert "tand_slope" not in rejected["gates_run"]

    def test_the_metrics_event_is_proof_the_gated_engine_ran(self):
        from softae.tools.shadow_review import GATED_ONLY_EVENTS

        assert "eis_spectrum_metrics" in GATED_ONLY_EVENTS
        _, events = _analyse(as_eis_result(*reference_spectrum()), engine="legacy")
        assert events == []

    def test_two_analyze_calls_on_one_spectrum_share_a_key_and_deduplicate_to_one(
            self, healthy_analysis, console_events):
        # router.py and autonomous_wiring.py both call analyze_spectrum on the same
        # arrays. These two fixtures are genuinely independent invocations — one
        # captured, one rendered and parsed back — so the key is proved stable across
        # the render boundary as well as across calls.
        from softae.analysis.eis.recommend import SpectrumRecord, deduplicate

        captured = SpectrumRecord.from_event(dict(healthy_analysis[1][0],
                                                  event="eis_spectrum_metrics"))
        rendered = SpectrumRecord.from_event(console_events[0])
        assert captured.key == rendered.key
        assert len(deduplicate([captured, rendered])) == 1

    def test_two_different_spectra_on_one_channel_do_not_share_a_key(
            self, healthy_analysis, rejected_analysis):
        # Both are channel 1. A timestamp key would collide here; the fingerprint is
        # taken over the arrays, so it does not.
        keys = {healthy_analysis[1][0]["spectrum_key"],
                rejected_analysis[1][0]["spectrum_key"]}
        assert len(keys) == 2
        assert all(k.startswith("c01:") for k in keys)

    def test_a_console_rendered_metrics_event_round_trips_through_parse_line(
            self, console_events):
        # The reviewer parses text, not objects. The nested metrics={...} mapping rides
        # the existing brace-aware _split_kv, and this is the proof it survives.
        assert len(console_events) == 1
        assert isinstance(console_events[0]["metrics"], dict)
        assert isinstance(console_events[0]["gates_run"], list)
        assert console_events[0]["enforced"] is False
        assert "r_squared" in console_events[0]["metrics"]

    def test_a_json_rendered_metrics_event_round_trips_too(self, json_events):
        assert len(json_events) == 1
        assert isinstance(json_events[0]["metrics"], dict)
        assert "r_squared" in json_events[0]["metrics"]


class TestTelemetryNeverCostsASpectrum:
    """A spectrum came off real hardware, in a well that is now used up.

    Losing one to a defect in its own telemetry would be the mistake ``run_gates``
    already refuses when it says a broken gate must not discard a measurement. The
    "purely additive" claim in ``_log_spectrum_metrics``' docstring is enforced here
    rather than asserted there.
    """

    def test_a_failing_metrics_emit_leaves_the_analysis_untouched(
            self, monkeypatch, healthy_analysis):
        from softae.analysis.eis import engine as engine_mod

        class _Exploding:
            """Fails on the metrics event, records the fallback, passes the rest."""

            def __init__(self):
                self.warnings = []

            def info(self, event, **kw):
                if event == "eis_spectrum_metrics":
                    raise RuntimeError("structlog is on fire")

            def warning(self, event, **kw):
                self.warnings.append(event)

        broken = _Exploding()
        monkeypatch.setattr(engine_mod, "logger", broken)
        report = analyze_spectrum(as_eis_result(*reference_spectrum()), cell=CELL,
                                  settings=_gated(enabled=False))

        # Byte-for-byte the verdict the healthy fixture got with the event working.
        expected = healthy_analysis[0]
        assert report.quality.verdict is expected.quality.verdict
        assert report.quality.issues == expected.quality.issues
        assert report.sigma.mode == expected.sigma.mode
        assert report.sigma.value == pytest.approx(expected.sigma.value, nan_ok=True)
        assert np.array_equal(report.mask, expected.mask)
        # And it said so, rather than failing silently: a spectrum missing from the
        # population would otherwise bias every later threshold recommendation.
        assert "eis_spectrum_metrics_failed" in broken.warnings

    def test_a_logger_that_also_fails_on_the_warning_still_raises_nothing(
            self, monkeypatch, healthy_analysis):
        # The nested swallow. Called directly rather than through a second full
        # analysis, because the guard is what is under test, not the engine around it.
        from softae.analysis.eis import engine as engine_mod

        class _TotallyBroken:
            def info(self, *a, **kw):
                raise RuntimeError("down")

            def warning(self, *a, **kw):
                raise RuntimeError("also down")

        report = healthy_analysis[0]
        monkeypatch.setattr(engine_mod, "logger", _TotallyBroken())
        freq, Z = engine_mod._physics_complex(as_eis_result(*reference_spectrum()))
        engine_mod._log_spectrum_metrics(
            as_eis_result(*reference_spectrum()), freq=freq, Z=Z,
            quality=report.quality, results=list(report.gate_log), mask=report.mask,
            enforced=False, report_mode="value", fit_ok=True,
        )   # must return normally

    def test_a_poisoned_metric_value_is_dropped_rather_than_raising(self):
        # `_finite_metrics` is the one place a hostile value reaches: a string, a None,
        # an object with no __float__. None of them may reach the renderer.
        from softae.analysis.eis.engine import _finite_metrics

        poisoned = {"good": 1.5, "nan": float("nan"), "inf": float("inf"),
                    "text": "not a number", "none": None, "obj": object()}
        assert _finite_metrics(poisoned) == {"good": 1.5}
        assert _finite_metrics(None) == {}


# ── The legacy dropped-point count ───────────────────────────────────────────

class TestLegacyDroppedPointCountIsNotVacuous:
    """A legacy report must count the points its own fitter withheld.

    :attr:`SpectrumReport.n_dropped` sums ``n_dropped`` over the gate log, and this
    engine runs no gates — so it answered ``0`` for every spectrum. That was
    *vacuously* true only while ``fit_circuit`` fitted every point it was handed.
    Once the finiteness mask landed there the zero became a falsehood: ``ch22_003``
    carries ``fit.n_points_dropped = 1`` against a stored ``n_points_dropped`` of
    ``0`` — a row asserting that nothing was withheld from a fit that withheld a
    point. An absent count reads as absent; a wrong one does not.
    """

    @staticmethod
    def _one_non_finite_point():
        f, Z = reference_spectrum()
        Z = np.asarray(Z, dtype=complex).copy()
        Z[7] = complex(np.nan, Z[7].imag)
        return as_eis_result(f, Z)

    def test_a_clean_legacy_spectrum_still_reports_no_dropped_points(self):
        # The control, and it is not ceremony: without it a hardcoded ``1`` would
        # satisfy the test below. The property is "reads the fit", not "is non-zero".
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, engine="legacy")
        assert report.fit.n_points_dropped == 0
        assert report.n_dropped == 0

    def test_a_legacy_report_counts_the_point_its_fitter_withheld(self):
        report = analyze_spectrum(self._one_non_finite_point(),
                                  cell=CELL, engine="legacy")
        # Precondition, asserted rather than assumed: the finiteness mask really did
        # fire on this fixture. Should it stop firing, both numbers fall to 0 and the
        # test would pass while testing nothing — SUBAGENT_RULES §3's first shape. It
        # fails as a broken fixture instead.
        assert report.fit.n_points_dropped == 1
        # The falsehood itself. A revert to the gate-log sum makes this 0.
        assert report.n_dropped == 1

    def test_the_count_arrives_without_a_gate_entry_that_never_ran(self):
        # The whole reason ``_legacy_report`` returns a subclass rather than
        # synthesising a gate-log record. Carrying the number in a fabricated entry
        # would satisfy the test above and fail this one — it trades a wrong number
        # for a wrong provenance, and every reader of ``gate_log_json`` would then
        # describe a check this engine does not perform.
        report = analyze_spectrum(self._one_non_finite_point(),
                                  cell=CELL, engine="legacy")
        assert report.n_dropped == 1
        assert report.gate_log == ()
        assert report.mask is None
        # And the count stays out of the operator-facing cell, which still says only
        # that no gates ran. A non-zero ``n_dropped`` reaches ``gate_summary``'s
        # second branch on any engine but this one.
        assert report.gate_summary() == "—"
