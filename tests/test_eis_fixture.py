"""E3 — subtracting the fixture, and refusing to when the constants are not known.

The acceptance bar the plan sets for this phase is narrower than "the arithmetic is
right": *a deliberately corrupted fixture correction must produce a visible failure in
the log rather than a plausible but wrong result*. A subtraction cannot be validated by
inspecting its output — a wrong ``R_short`` yields a merely shifted spectrum, and shifted
spectra look fine — so the tests that matter most here are the ones about what happens
when it goes wrong, not when it goes right.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from softae.analysis.eis.calibration import CalibrationSet, PhaseAccuracyTable
from softae.analysis.eis.fixture import (
    CORRECTION_MODES,
    FixtureCorrection,
    apply_series_correction,
    correction_for_channel,
    fixture_impedance,
    resolve_mode,
    validate_load_blank,
)
from softae.analysis.eis.settings import FixtureSettings, eis_settings

from .eis_synthetic import log_frequencies, reference_spectrum

R_SHORT = 6.4581      # the value this rig's ch32 short blank actually derived
L_LEAD = 4.18e-6      # the overhaul's measured lead inductance


def _short_cal(**kw) -> CalibrationSet:
    base = dict(
        fixture_id="mux16",
        channels_measured=(32,),
        R_short_ohm={32: R_SHORT},
        L_lead_H={32: L_LEAD},
    )
    base.update(kw)
    return CalibrationSet(**base)


class TestTheSubtraction:
    def test_series_correction_removes_exactly_the_lead_it_was_given(self):
        f = log_frequencies()
        _, Z = reference_spectrum(f)
        contaminated = Z + fixture_impedance(f, R_short_ohm=R_SHORT, L_lead_H=L_LEAD)

        corr = FixtureCorrection(mode="series", channel=32, R_short_ohm=R_SHORT,
                                 L_lead_H=L_LEAD)
        recovered, outcome = apply_series_correction(f, contaminated, corr)

        assert outcome.applied
        assert np.allclose(recovered, Z, rtol=1e-9, atol=1e-9)

    def test_the_lead_is_inductive_so_its_imaginary_part_opposes_the_film(self):
        """The sign convention is the whole correctness of this module.

        ``derive_short`` fits ``L`` as the slope of ``Im Z`` against ``ω`` with no
        intercept, so the lead's ``Im Z`` is *positive* while a capacitive film's is
        negative. Get the sign backwards and the correction adds the lead twice
        instead of removing it — which still produces a smooth, plausible spectrum.
        """
        f = np.array([1e5, 1e3], dtype=float)
        Z_fix = fixture_impedance(f, R_short_ohm=R_SHORT, L_lead_H=L_LEAD)

        assert np.all(Z_fix.real == pytest.approx(R_SHORT))
        assert np.all(Z_fix.imag > 0)
        # ...and larger at higher frequency, because that is what an inductance does.
        assert Z_fix.imag[0] > Z_fix.imag[1]

    def test_a_non_applying_correction_returns_the_very_same_array(self):
        """Not equal values — the same object, so "uncorrected" cannot round-trip."""
        f, Z = reference_spectrum()
        out, outcome = apply_series_correction(f, Z, FixtureCorrection(mode="none"))
        assert out is np.asarray(Z, dtype=complex) or np.shares_memory(out, Z)
        assert not outcome.applied

    def test_a_few_ohms_of_fixture_barely_moves_a_megohm_film(self):
        """Sanity on magnitude: this correction should be invisible on a good sample.

        If it is not, either the sample is unusually conductive or the constant is
        wrong — which is why a large shift is logged rather than silently accepted.
        """
        f, Z = reference_spectrum(R_bulk=5.0e6)
        corr = FixtureCorrection(mode="series", channel=32, R_short_ohm=R_SHORT,
                                 L_lead_H=L_LEAD)
        _, outcome = apply_series_correction(f, Z, corr)
        assert outcome.max_shift_pct < 1.0


class TestWhenItGoesWrong:
    def test_a_corrupted_constant_announces_itself_instead_of_shifting_quietly(self):
        """The plan's acceptance criterion for this phase, made literal."""
        f, Z = reference_spectrum(R_bulk=5.0e4)
        # Two orders of magnitude larger than the true bulk: the subtraction now
        # removes more resistance than the film has.
        corrupt = FixtureCorrection(mode="series", channel=32,
                                    R_short_ohm=5.0e6, L_lead_H=L_LEAD)
        _, outcome = apply_series_correction(f, Z, corrupt)

        assert outcome.induced_nonphysical > 0
        assert outcome.suspect
        assert outcome.issues, "a suspect correction must carry a stated reason"
        assert "not consistent with this spectrum" in outcome.issues[0]

    def test_a_correct_constant_induces_nothing(self):
        """The complement — otherwise the check above would fire on every spectrum."""
        f, Z = reference_spectrum()
        corr = FixtureCorrection(mode="series", channel=32, R_short_ohm=R_SHORT,
                                 L_lead_H=L_LEAD)
        _, outcome = apply_series_correction(f, Z, corr)
        assert outcome.induced_nonphysical == 0
        assert not outcome.suspect

    def test_a_nan_constant_never_reaches_the_spectrum(self):
        """A NaN ``R_short`` would poison every point while looking like a correction."""
        cal = _short_cal(R_short_ohm={32: float("nan")})
        corr = correction_for_channel(32, calibration=cal)
        assert corr.mode == "none"
        assert not corr.applies
        assert "no short-blank constant" in corr.declined


class TestChoosingTheMode:
    def test_auto_is_none_until_a_short_blank_exists_and_says_which_command_fixes_it(self):
        mode, why = resolve_mode("auto", CalibrationSet().capabilities())
        assert mode == "none"
        assert "blank_short" in why

    def test_auto_becomes_series_the_moment_a_short_exists(self):
        mode, why = resolve_mode("auto", _short_cal().capabilities())
        assert mode == "series"
        assert why == ""

    def test_auto_never_resolves_to_osl_even_when_the_artifacts_license_it(self):
        """The clamp. ``capabilities()`` says ``osl``; this must not.

        The two disagree on purpose — one describes what the artifacts support, the
        other what the engine will apply — and a usable open must never select a
        correction that is not implemented.
        """
        cal = _short_cal(open_usable={32: True})
        assert cal.capabilities().correction_mode == "osl"

        mode, why = resolve_mode("auto", cal.capabilities())
        assert mode == "series"
        assert "not applied" in why and "F6" in why

    def test_osl_is_not_even_a_configurable_value(self):
        assert "osl" not in CORRECTION_MODES

    def test_none_is_honoured_over_an_available_short(self):
        mode, why = resolve_mode("none", _short_cal().capabilities())
        assert mode == "none"
        assert "configuration" in why

    def test_an_unknown_mode_falls_back_to_auto_rather_than_raising(self):
        mode, _ = resolve_mode("osl", _short_cal().capabilities())
        assert mode == "series"

    def test_series_requested_without_a_short_refuses_rather_than_pretending(self):
        """Explicitly asking for series does not conjure the constants it needs."""
        mode, why = resolve_mode("series", CalibrationSet().capabilities())
        assert mode == "none"
        assert "blank_short" in why

    def test_an_unimplemented_capability_fails_closed(self):
        """Adding a mode to the ladder must not silently fall through to "corrected"."""

        class FutureCaps:
            correction_mode = "quantum"

        mode, why = resolve_mode("auto", FutureCaps())
        assert mode == "none"
        assert "not implemented" in why


class TestBuildingItForAChannel:
    def test_no_calibration_declines_with_the_command_that_would_fix_it(self):
        corr = correction_for_channel(7, calibration=None)
        assert corr.mode == "none"
        assert "softae-commission" in corr.declined

    def test_an_inherited_channel_is_flagged_as_inherited(self):
        """ch1 borrows ch32's constants; the spectrum must carry that fact."""
        cal = _short_cal(
            channels_assumed=tuple(range(1, 32)),
            R_short_ohm={c: R_SHORT for c in range(1, 33)},
            L_lead_H={c: L_LEAD for c in range(1, 33)},
        )
        assert correction_for_channel(1, calibration=cal).inherited
        assert not correction_for_channel(32, calibration=cal).inherited

    def test_a_channel_with_no_constant_and_no_inheritance_declines(self):
        corr = correction_for_channel(5, calibration=_short_cal())
        assert corr.mode == "none"
        assert "--representative" in corr.declined

    def test_provenance_carries_the_short_blanks_measurement_id(self):
        cal = _short_cal(sources={"blank_short": 4242})
        assert correction_for_channel(32, calibration=cal).source_measurement_id == 4242


class TestValidatingAgainstAKnownLoad:
    """R9 — the only end-to-end check the correction has.

    Re-measuring the short proves nothing, because the short is what set the constants.
    A third component of independently known value is what separates "the arithmetic
    ran" from "the arithmetic was right".
    """

    def _load_spectrum(self, R_true: float):
        f = log_frequencies()
        Z = np.full(f.size, R_true, dtype=complex)
        return f, Z + fixture_impedance(f, R_short_ohm=R_SHORT, L_lead_H=L_LEAD)

    def test_a_good_correction_recovers_the_marked_resistor(self):
        f, Z = self._load_spectrum(1000.0)
        corr = FixtureCorrection(mode="series", channel=32, R_short_ohm=R_SHORT,
                                 L_lead_H=L_LEAD)
        ok, err, msg = validate_load_blank(f, Z, nominal_ohm=1000.0, correction=corr)
        assert ok
        assert abs(err) < 0.1
        assert "within" in msg

    def test_leaving_the_fixture_in_is_visible_on_a_small_load(self):
        """6.5 Ω of fixture against a 100 Ω load is a 6 % error — outside tolerance."""
        f, Z = self._load_spectrum(100.0)
        ok, err, _ = validate_load_blank(
            f, Z, nominal_ohm=100.0, correction=FixtureCorrection(mode="none"))
        assert not ok
        assert err > 5.0

    def test_the_same_uncorrected_error_hides_on_a_large_load(self):
        """Which is why validation needs a load comparable to the fixture, not a film.

        This is a statement about the *method*, not a defect: a 1 MΩ reference cannot
        validate a 6 Ω correction, and choosing one would make the check vacuous.
        """
        f, Z = self._load_spectrum(1.0e6)
        ok, err, _ = validate_load_blank(
            f, Z, nominal_ohm=1.0e6, correction=FixtureCorrection(mode="none"))
        assert ok and abs(err) < 0.01

    def test_no_marked_value_is_refused_rather_than_assumed(self):
        f, Z = self._load_spectrum(1000.0)
        ok, err, msg = validate_load_blank(
            f, Z, nominal_ohm=0.0, correction=FixtureCorrection(mode="none"))
        assert not ok
        assert math.isnan(err)
        assert "nothing to validate against" in msg


class TestStaleCalibrationsDoNotCorrect:
    def test_a_stale_set_corrects_nothing_because_its_constants_were_dropped(self,
                                                                            tmp_path):
        """``resolve_calibration`` strips the constants; E3 must then decline cleanly.

        The failure this prevents is the quiet one: a short blank from a different
        board subtracted from today's spectra, every number still plausible.
        """
        from softae.analysis.eis.calibration import resolve_calibration, save_calibration

        cal = _short_cal(hardware_hash="deadbeef", created_at="2026-01-01T00:00:00")
        save_calibration(cal, root=tmp_path)

        resolved = resolve_calibration("mux16", root=tmp_path)
        corr = correction_for_channel(32, calibration=resolved)

        assert corr.mode == "none"
        assert "blank_short" in corr.declined


class TestSettings:
    def test_the_shipped_default_is_auto(self):
        assert FixtureSettings().mode == "auto"

    def test_an_unknown_configured_mode_degrades_to_auto(self):
        cfg = eis_settings({"fixture": {"mode": "osl"}})
        assert cfg.fixture.mode == "auto"

    def test_the_fixture_id_is_read_and_defaults_are_not_empty(self):
        cfg = eis_settings({"fixture": {"fixture_id": "mux16"}})
        assert cfg.fixture.fixture_id == "mux16"
        assert eis_settings({"fixture": {"fixture_id": ""}}).fixture.fixture_id

    def test_a_garbage_tolerance_does_not_raise(self):
        assert eis_settings(
            {"fixture": {"load_tolerance_pct": "nope"}}).fixture.load_tolerance_pct == 5.0

    def test_the_description_names_what_will_happen(self):
        assert "series once" in FixtureSettings().describe()
        assert "off" in FixtureSettings(mode="none").describe()


class TestTheEngineIntegration:
    def _eis(self, f, Z):
        from softae.analysis.eis_data import EISResult

        return EISResult.from_arrays(channel=32, f=f, z_real=Z.real,
                                     z_imag_neg=-Z.imag)

    def test_the_legacy_engine_ignores_the_correction_entirely(self):
        """Parity is what makes the two engines comparable; correcting one breaks it."""
        from softae.analysis.eis.engine import analyze_spectrum

        f, Z = reference_spectrum()
        corr = FixtureCorrection(mode="series", channel=32, R_short_ohm=R_SHORT,
                                 L_lead_H=L_LEAD)
        with_corr = analyze_spectrum(self._eis(f, Z), engine="legacy", correction=corr)
        without = analyze_spectrum(self._eis(f, Z), engine="legacy")

        assert with_corr.correction is None
        assert not with_corr.corrected
        assert with_corr.sigma.R_reported_ohm == pytest.approx(
            without.sigma.R_reported_ohm, nan_ok=True)

    def test_the_gated_engine_records_a_declined_correction_rather_than_nothing(self):
        """"Nobody corrected this" and "nobody could" are different facts."""
        from softae.analysis.eis.engine import analyze_spectrum

        f, Z = reference_spectrum()
        report = analyze_spectrum(
            self._eis(f, Z), engine="gated",
            correction=FixtureCorrection(mode="none", channel=32,
                                         declined="no short blank"),
        )
        assert report.correction is not None
        assert not report.corrected
        assert report.correction.declined == "no short blank"

    def test_an_applied_correction_is_reported_and_measured(self):
        from softae.analysis.eis.engine import analyze_spectrum

        f, Z = reference_spectrum(R_bulk=2.0e3)
        contaminated = Z + fixture_impedance(f, R_short_ohm=R_SHORT, L_lead_H=L_LEAD)
        corr = FixtureCorrection(mode="series", channel=32, R_short_ohm=R_SHORT,
                                 L_lead_H=L_LEAD)

        report = analyze_spectrum(self._eis(f, contaminated), engine="gated",
                                  correction=corr)
        assert report.corrected
        assert report.correction_outcome.applied
        assert "fixture_shift_pct" in report.quality.metrics

    def test_a_suspect_correction_downgrades_the_verdict_it_would_otherwise_earn(self):
        from softae.analysis.quality import Verdict
        from softae.analysis.eis.engine import analyze_spectrum

        f, Z = reference_spectrum()
        corrupt = FixtureCorrection(mode="series", channel=32,
                                    R_short_ohm=5.0e6, L_lead_H=L_LEAD)
        report = analyze_spectrum(self._eis(f, Z), engine="gated", correction=corrupt)

        assert report.correction_outcome.suspect
        assert report.quality.verdict is not Verdict.ACCEPT
        assert any("not consistent" in i for i in report.quality.issues)

    def test_correcting_nothing_leaves_the_gated_result_unchanged(self):
        """E3 must be inert until a calibration exists — the same posture as E0/E1."""
        from softae.analysis.eis.engine import analyze_spectrum

        f, Z = reference_spectrum()
        plain = analyze_spectrum(self._eis(f, Z), engine="gated",
                                 correction=FixtureCorrection(mode="none"))
        none_given = analyze_spectrum(self._eis(f, Z), engine="gated", calibration=None,
                                      correction=FixtureCorrection(mode="none"))
        assert plain.sigma.R_reported_ohm == pytest.approx(
            none_given.sigma.R_reported_ohm, nan_ok=True)


class TestTheOrderingAroundTheCorrection:
    """Framework §6 puts fixture correction at step 4 — *between* the Front-1 gates.

    The split is load-bearing in both directions, and getting it wrong is invisible:
    every spectrum still produces a verdict, a fit and a number. What changes is which
    data the topology triad judged.
    """

    def _eis(self, f, Z):
        from softae.analysis.eis_data import EISResult

        return EISResult.from_arrays(channel=32, f=f, z_real=Z.real,
                                     z_imag_neg=-Z.imag)

    def test_the_two_halves_cover_every_front1_gate_exactly_once(self):
        from softae.analysis.eis.gates import (
            FRONT1_GATES,
            FRONT1_POST_CORRECTION,
            FRONT1_PRE_CORRECTION,
        )

        assert tuple(FRONT1_PRE_CORRECTION) + tuple(FRONT1_POST_CORRECTION) == \
            tuple(FRONT1_GATES)
        assert not set(FRONT1_PRE_CORRECTION) & set(FRONT1_POST_CORRECTION)

    def test_the_topology_triad_is_downstream_of_the_correction(self):
        """§6: the triad "must run on corrected, truncated data — an uncorrected
        series parasitic can invert the very slopes the triad tests." A fixture
        ``R_short`` *is* a series parasitic."""
        from softae.analysis.eis.gates import (
            FRONT1_POST_CORRECTION,
            FRONT1_PRE_CORRECTION,
            TOPOLOGY_TRIAD,
        )

        assert set(TOPOLOGY_TRIAD) <= set(FRONT1_POST_CORRECTION)
        assert not set(TOPOLOGY_TRIAD) & set(FRONT1_PRE_CORRECTION)

    def test_admission_gates_are_upstream_of_the_correction(self):
        """The other direction: a railed point stays railed however much lead you
        subtract, so a correction must never rescue a failed measurement."""
        from softae.analysis.eis.gates import (
            FRONT1_PRE_CORRECTION,
            gate_finiteness,
            gate_magnitude,
            gate_quadrant,
        )

        for gate in (gate_finiteness, gate_quadrant, gate_magnitude):
            assert gate in FRONT1_PRE_CORRECTION

    def test_the_post_stage_continues_the_masks_rather_than_resurrecting_points(self):
        """Without ``initial_mask`` the second stage restarts all-pass, undoing every
        point the admission gates dropped — with no log line to show it."""
        from softae.analysis.eis.gates import FRONT1_POST_CORRECTION, run_gates
        from softae.analysis.eis.policy import build_context
        from softae.analysis.eis.envelope import instrument_envelope
        from softae.analysis.eis.settings import GateSettings

        f, Z = reference_spectrum()
        ctx = build_context(envelope=instrument_envelope(), gates=GateSettings(),
                            cell=None)

        seeded = np.ones(f.size, dtype=bool)
        seeded[:3] = False
        mask, _, _ = run_gates(f, Z, ctx, FRONT1_POST_CORRECTION,
                               initial_mask=seeded)
        assert not mask[:3].any()

    def test_a_mismatched_initial_mask_is_refused_rather_than_silently_ignored(self):
        from softae.analysis.eis.gates import run_gates
        from softae.analysis.eis.policy import build_context
        from softae.analysis.eis.envelope import instrument_envelope
        from softae.analysis.eis.settings import GateSettings

        f, Z = reference_spectrum()
        ctx = build_context(envelope=instrument_envelope(), gates=GateSettings(),
                            cell=None)
        with pytest.raises(ValueError, match="initial_mask"):
            run_gates(f, Z, ctx, initial_mask=np.ones(3, dtype=bool))

    def test_the_triad_sees_the_corrected_spectrum_not_the_raw_one(self):
        """The behavioural claim, not just the tuple membership.

        A series resistance large enough to dominate turns the reference spectrum into
        something the loss-tangent slope reads as a series parasitic. Correct it away
        and the triad should pass; leave it in and it should not.
        """
        from softae.analysis.eis.engine import analyze_spectrum

        f, Z = reference_spectrum()
        R_par = 2.0e5   # >> R_bulk, so tan d slope inverts
        contaminated = Z + fixture_impedance(f, R_short_ohm=R_par, L_lead_H=0.0)

        uncorrected = analyze_spectrum(
            self._eis(f, contaminated), engine="gated",
            correction=FixtureCorrection(mode="none"))
        corrected = analyze_spectrum(
            self._eis(f, contaminated), engine="gated",
            correction=FixtureCorrection(mode="series", channel=32,
                                         R_short_ohm=R_par, L_lead_H=0.0))

        def triad_passed(report):
            return {e["gate"]: e["passed"] for e in report.gate_log
                    if e["gate"] in ("tand_slope", "series_rc")}

        raw, fixed = triad_passed(uncorrected), triad_passed(corrected)
        assert raw != fixed, (
            "the topology triad returned identical verdicts on raw and corrected "
            "data — the correction is not upstream of it")

    def test_a_spectrum_rejected_at_admission_is_never_corrected(self):
        """Correcting it would produce verdicts describing an inadmissible measurement."""
        from softae.analysis.eis.engine import analyze_spectrum
        from softae.analysis.eis.settings import GateSettings

        f, Z = reference_spectrum()
        Z = np.asarray(Z, dtype=complex).copy()
        Z[:] = np.nan          # nothing finite survives step 1

        report = analyze_spectrum(
            self._eis(f, Z), engine="gated",
            gates=GateSettings(enabled=True),
            correction=FixtureCorrection(mode="series", channel=32,
                                         R_short_ohm=R_SHORT, L_lead_H=L_LEAD),
        )
        assert not report.corrected
        assert report.correction is not None   # still recorded as in force


class TestProvenance:
    def test_an_uncorrected_measurement_has_no_row_rather_than_a_default_one(self,
                                                                            tmp_path):
        """Absent means uncorrected — which is the honest reading of every legacy row.

        A nullable column with a default would have to claim something about spectra
        measured before this table existed. A missing row claims nothing.
        """
        from softae.core.data_store import DataStore

        store = DataStore(str(tmp_path))
        try:
            assert store.fixture_correction_for(1) is None
        finally:
            store.close()

    def test_a_declined_correction_is_still_recorded_with_its_reason(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(str(tmp_path))
        try:
            corr = FixtureCorrection(mode="none", channel=5, fixture_id="mux16",
                                     declined="no short blank")
            store.record_fixture_correction(11, corr)
            row = store.fixture_correction_for(11)
            assert row["mode"] == "none"
            assert row["declined"] == "no short blank"
            assert row["R_short_ohm"] is None
        finally:
            store.close()

    def test_an_applied_correction_records_what_was_subtracted(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(str(tmp_path))
        try:
            f, Z = reference_spectrum()
            corr = FixtureCorrection(mode="series", channel=32, fixture_id="mux16",
                                     R_short_ohm=R_SHORT, L_lead_H=L_LEAD,
                                     inherited=True, source_measurement_id=7)
            _, outcome = apply_series_correction(f, Z, corr)
            store.record_fixture_correction(12, corr, outcome)

            row = store.fixture_correction_for(12)
            assert row["mode"] == "series"
            assert row["R_short_ohm"] == pytest.approx(R_SHORT)
            assert row["inherited"] is True
            assert row["source_measurement_id"] == 7
            assert row["induced_nonphysical"] == 0
        finally:
            store.close()

    def test_nan_shift_becomes_null_not_a_number(self, tmp_path):
        from softae.core.data_store import DataStore
        from softae.analysis.eis.fixture import CorrectionOutcome

        store = DataStore(str(tmp_path))
        try:
            store.record_fixture_correction(
                13, FixtureCorrection(mode="none"),
                CorrectionOutcome(max_shift_pct=float("nan")))
            assert store.fixture_correction_for(13)["max_shift_pct"] is None
        finally:
            store.close()


class TestTheCLIAndEngineAgreeOnWhichFixture:
    def test_the_commission_cli_takes_its_default_fixture_from_the_config(self,
                                                                         monkeypatch):
        """Commissioning ``mux16`` while the engine looks for ``default`` would produce
        a calibration nothing ever applies — with both halves behaving correctly."""
        import softae.tools.commission as commission

        monkeypatch.setattr(
            commission, "_default_fixture", lambda: "mux16", raising=True)
        parser = commission.build_parser()
        args = parser.parse_args(["status"])
        assert args.fixture == "mux16"

    def test_it_survives_an_unreadable_config(self, monkeypatch):
        import softae.tools.commission as commission
        from softae.analysis.eis import settings as settings_mod

        def boom(*a, **k):
            raise RuntimeError("no config")

        monkeypatch.setattr(settings_mod, "eis_settings", boom, raising=True)
        assert commission._default_fixture() == "default"
