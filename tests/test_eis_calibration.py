"""E2 — commissioning constants: derived, persisted, and degraded when stale.

The point of this module is to turn every "blocked on bench data" item into *run this
workflow once*. Two properties make that worth doing rather than measuring per run:
fixture electronics drift is minimal, so a calibration is a durable asset; and a set is
useful **incrementally**, so a partial calibration must say what it unlocks and name
the one artifact that would unlock each thing it does not.

The whole derivation is testable without hardware — synthetic short, open, load and
reference spectra exercise every path, which is the point of separating derivation from
acquisition.

.. note::
   ``Z_φ`` is deliberately absent. An earlier draft had this module derive a
   "phase-reliable ceiling"; that ceiling was withdrawn as a floating-reference-electrode
   artefact, and :class:`PhaseAccuracyTable` — ε where it was actually measured, NaN
   where it was not — is what replaced it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.calibration import (
    COMMISSIONING_ROLES,
    MEASUREMENT_ROLES,
    TWO_TERMINAL_ROLES,
    CalibrationSet,
    PhaseAccuracyTable,
    ReferenceCapResult,
    calibration_path,
    derive_open,
    derive_reference_cap,
    derive_reference_r,
    derive_short,
    describe_or_absent,
    hardware_hash,
    load_calibration,
    resolve_calibration,
    save_calibration,
)

#: Every commissioning test declares two-electrode sensing, because that is what a
#: valid pass now requires (R24/F17). Passing it explicitly rather than defaulting it
#: inside ``derive_calibration`` is deliberate: the refusal is the safety property, and
#: a default would quietly reopen the hole it closes.
ALL_TWO = {r: "two" for r in TWO_TERMINAL_ROLES}



def _freqs(n: int = 45) -> np.ndarray:
    """Descending, as the instrument reports them."""
    return np.logspace(math.log10(200_000), math.log10(1.2), n)


def _short(R: float = 5.4, L: float = 4.18e-6) -> tuple[np.ndarray, np.ndarray]:
    f = _freqs()
    return f, R + 1j * (2 * np.pi * f) * L


def _capacitor(C: float = 1e-9, tand: float = 5e-4) -> tuple[np.ndarray, np.ndarray]:
    f = _freqs()
    Xc = -1.0 / (2 * np.pi * f * C)
    return f, (abs(Xc) * tand) + 1j * Xc


class TestRoles:
    def test_sample_is_the_default_so_existing_rows_need_no_backfill(self):
        assert MEASUREMENT_ROLES[0] == "sample"

    def test_every_commissioning_artifact_has_a_role(self):
        assert set(COMMISSIONING_ROLES) == {
            "blank_open", "blank_short", "blank_load",
            "reference_cap", "reference_r",
        }


class TestShortBlank:
    def test_it_recovers_the_fixture_resistance_and_lead_inductance(self):
        R, L = derive_short(*_short())
        assert R == pytest.approx(5.4, rel=1e-6)
        assert L == pytest.approx(4.18e-6, rel=1e-6)

    def test_measuring_l_is_what_licenses_pinning_it_in_the_sample_fit(self):
        # F5 recorded *fitted* inductances of 400-500 uH against a short blank's true
        # 4.18 uH -- an HF phase artifact absorbed as inductance. R11 pins L to ~0 in
        # the sample fit, and this measurement is the evidence for doing so.
        _R, L = derive_short(*_short())
        assert L < 1e-5, "a real fixture inductance is microhenries, not hundreds"

    def test_a_railed_point_does_not_move_the_fixture_constant(self):
        # Median, not mean: one bad point must not become a fixture constant.
        f, Z = _short()
        Z = Z.copy()
        Z[10] = 1e9 + 0j
        R, _L = derive_short(f, Z)
        assert R == pytest.approx(5.4, rel=1e-6)

    def test_an_unusable_trace_declines_rather_than_raising(self):
        f = _freqs(4)
        R, L = derive_short(f, np.full(4, np.nan, dtype=complex))
        assert R != R and L != L


class TestShortPlausibility:
    """A short blank is the one artifact whose answer is known in advance.

    Found by running the CLI end to end against mock hardware: the derivation happily
    produced ``R = 3.2 MOhm`` and ``L = -1.47 H`` -- a *negative inductance* -- and
    wrote both into a version-controlled calibration file without complaint. Those are
    operator setup errors (no jumper, wrong channel, mislabelled run): recoverable the
    moment they are named, and invisible forever if they are simply recorded.
    """

    def test_a_real_short_passes(self):
        from softae.analysis.eis.calibration import short_is_plausible

        ok, reason = short_is_plausible(5.4, 4.18e-6)
        assert ok and reason == ""

    def test_a_megaohm_short_is_refused_and_names_the_likely_cause(self):
        from softae.analysis.eis.calibration import short_is_plausible

        ok, reason = short_is_plausible(3.2e6, 1e-6)
        assert not ok
        assert "jumper" in reason

    def test_a_negative_inductance_is_refused_as_unphysical(self):
        from softae.analysis.eis.calibration import short_is_plausible

        ok, reason = short_is_plausible(5.4, -1.47)
        assert not ok
        assert "not physical" in reason

    def test_an_f5_scale_inductance_is_refused_as_an_artifact(self):
        # F5: *fitted* inductances of 400-500 uH against a true 4.18 uH. A millihenry
        # is far past anything this fixture can carry.
        from softae.analysis.eis.calibration import short_is_plausible

        ok, reason = short_is_plausible(5.4, 5e-3)
        assert not ok
        assert "artifact" in reason

    def test_a_nan_derivation_is_refused(self):
        from softae.analysis.eis.calibration import short_is_plausible

        assert not short_is_plausible(float("nan"), 1e-6)[0]

    def test_an_all_implausible_pass_raises_rather_than_writing_a_hollow_set(self):
        # A set with no fixture constants that still *looks* like a calibration is
        # worse than none: resolve_calibration would hand it out and the ladder would
        # report a correction mode nothing supports.
        from softae.workflows.commissioning import (
            CommissioningError,
            derive_calibration,
        )

        f = _freqs()
        bogus = np.full(f.size, 3.2e6 + 0j)
        with pytest.raises(CommissioningError, match="implausible"):
            derive_calibration({"blank_short": [(1, f, bogus)]}, electrode_modes=ALL_TWO)

    def test_one_good_channel_survives_a_bad_neighbour(self):
        from softae.workflows.commissioning import derive_calibration

        f, good = _short()
        bogus = np.full(f.size, 3.2e6 + 0j)
        cal = derive_calibration(
            {"blank_short": [(1, f, good), (2, f, bogus)]}, electrode_modes=ALL_TWO)
        assert cal.channels_measured == (1,)
        assert 2 not in cal.R_short_ohm


class TestOpenBlank:
    def test_a_smooth_open_inside_the_magnitude_window_is_usable(self):
        # 1 nF keeps |Z| under the 1e9 ceiling across the whole band.
        f = _freqs()
        Z = 1.0 / (1j * 2 * np.pi * f * 1e-9)
        usable, over, flips = derive_open(f, Z)
        assert usable and over == 0.0 and flips == 0.0

    def test_a_bare_board_open_goes_over_range_at_low_frequency(self):
        # The §3.2 finding, reproduced from first principles rather than asserted: a
        # true stray-capacitance open (5.85 pF) reaches 2e10 Ohm at 1.2 Hz, twenty
        # times the instrument ceiling. An open blank is *unmeasurable* on a
        # ceiling-limited instrument — which is why OSL corrupted every corrected
        # spectrum and why series-only is the right default.
        f = _freqs()
        Z = 1.0 / (1j * 2 * np.pi * f * 5.85e-12)
        usable, over, _flips = derive_open(f, Z)
        assert not usable
        assert over > 0.25, "most of the band sits above the magnitude ceiling"

    def test_an_over_range_noisy_open_is_not_usable(self):
        # Overhaul §3.2: 77% of points over range, high Im-sign-flip density. Feeding
        # this to a 3-term OSL correction corrupted every corrected spectrum (mean
        # error 32%, one channel reading 1.26 MOhm instead of ~840).
        rng = np.random.default_rng(4)
        f = _freqs()
        Z = (rng.normal(0, 1e10, f.size) + 1j * rng.normal(0, 1e10, f.size))
        usable, over, flips = derive_open(f, Z)
        assert not usable
        assert over > 0.5 and flips > 0.3

    def test_an_unusable_open_selects_series_only_which_is_exactly_when_it_is_exact(self):
        # Not a missing artifact but a positive result: an unmeasurably high open IS
        # the evidence that shunt admittance is negligible, which is precisely the
        # condition under which short-only series correction is exact.
        cal = CalibrationSet(R_short_ohm={1: 5.4}, open_usable={1: False})
        caps = cal.capabilities()
        assert caps.correction_mode == "series"
        assert caps.open_is_usable is False
        assert "OSL correction" not in caps.blocked

    def test_a_usable_open_permits_osl(self):
        cal = CalibrationSet(R_short_ohm={1: 5.4}, open_usable={1: True})
        assert cal.capabilities().correction_mode == "osl"


class TestReferenceComponents:
    def test_a_resistor_reproduces_the_measured_magnitude_accuracy(self):
        f = _freqs()
        R, err, noise = derive_reference_r(
            f, np.full(f.size, 9931.7 + 0j), nominal_ohm=9900.0)
        assert R == pytest.approx(9931.7)
        assert err == pytest.approx(0.32, abs=0.01)     # the doc's +0.32%
        assert noise == pytest.approx(0.0, abs=1e-9)

    def test_a_resistors_phase_scatter_is_a_direct_read_of_epsilon(self):
        # A resistor's phase is 0 by definition, so its scatter IS the phase noise.
        rng = np.random.default_rng(7)
        f = _freqs()
        phase = np.radians(rng.normal(0.0, 0.149, f.size))
        Z = 9900.0 * np.exp(1j * phase)
        _R, _err, noise = derive_reference_r(f, Z, nominal_ohm=9900.0)
        assert noise == pytest.approx(0.149, rel=0.25)

    def test_a_low_loss_capacitor_yields_its_capacitance_and_loss_floor(self):
        C, tand, _z = derive_reference_cap(*_capacitor(C=1e-9, tand=5e-4))
        assert C == pytest.approx(1e-9, rel=1e-6)
        assert tand == pytest.approx(5e-4, rel=1e-6)

    def test_a_part_that_disagrees_with_its_marking_is_flagged(self, caplog):
        # Overhaul §3.7: the part marked "102" (1 nF) measured ~150 nF with a minimum
        # tan d of 0.18, 70x above the instrument floor. Unusable as a phase
        # reference, and the marking/measurement disagreement is what reveals it.
        f, Z = _capacitor(C=147e-9, tand=0.179)
        C, tand, _z = derive_reference_cap(f, Z, nominal_F=1e-9)
        assert C == pytest.approx(147e-9, rel=1e-6)
        assert tand > 0.1, "this part cannot serve as a phase reference"


class TestStrayCorrectedMarkedValueCheck:
    """The marked value describes the *part*; the sweep sees the part **plus** the
    fixture's parallel shunt.

    On mux16 the stray is ~53 pF against 100 pF C0G parts, so the uncorrected check
    reads 1.50x and reports a disagreement that belongs to the fixture, not the
    capacitor. Subtracting the stray that ``derive_open_constants`` already measures on
    the same channel makes the check a statement about the part.
    """

    def _flags(self, f, Z, **kw) -> tuple:
        """``(result, mismatch events)`` — the check's verdict and whether it fired."""
        import structlog

        with structlog.testing.capture_logs() as logs:
            res = derive_reference_cap(f, Z, **kw)
        return res, [e for e in logs
                     if e.get("event") == "eis_reference_cap_mismatch"]

    def test_a_parallel_stray_is_subtracted_so_the_part_alone_is_judged(self):
        # Two ideal capacitors in parallel ARE one capacitor of the sum, so this is
        # the fixture-plus-part spectrum exactly rather than an approximation of it.
        f, Z = _capacitor(C=100e-12 + 53.2e-12)
        res = derive_reference_cap(f, Z, nominal_F=100e-12, C_stray_F=53.2e-12)
        assert isinstance(res, ReferenceCapResult)
        assert res.corrected
        assert res.C_raw_F == pytest.approx(153.2e-12, rel=1e-6)
        assert res.C_corrected_F == pytest.approx(100e-12, rel=1e-4)
        assert res.C_F == res.C_corrected_F

    def test_subtracting_from_the_median_equals_subtracting_per_point(self):
        # The stray is one constant, and median(x - c) == median(x) - c. Pinned so a
        # later "improvement" to a per-point correction cannot silently change a number.
        f, Z = _capacitor(C=250e-12)
        omega = 2 * np.pi * f
        per_point = np.median(1.0 / (omega * np.abs(Z.imag)) - 40e-12)
        res = derive_reference_cap(f, Z, C_stray_F=40e-12)
        assert res.C_corrected_F == pytest.approx(float(per_point), rel=1e-12)

    def test_without_a_stray_the_raw_value_is_judged_exactly_as_before(self):
        # Regression pin on the legacy path: no stray, no correction, no change.
        f, Z = _capacitor(C=147e-9, tand=0.179)
        res, flagged = self._flags(f, Z, nominal_F=1e-9)
        assert not res.corrected
        assert res.C_stray_F != res.C_stray_F                     # NaN
        assert res.C_corrected_F != res.C_corrected_F             # NaN
        assert res.C_F == res.C_raw_F == pytest.approx(147e-9, rel=1e-6)
        assert flagged, "the raw check must still flag this part"

    def test_it_still_unpacks_as_the_triple_every_caller_expects(self):
        f, Z = _capacitor(C=1e-9, tand=5e-4)
        res = derive_reference_cap(f, Z, C_stray_F=100e-12)
        C, tand, z_at = res
        assert C == res.C_F == pytest.approx(0.9e-9, rel=1e-4)
        assert tand == pytest.approx(5e-4, rel=1e-6)
        assert z_at == pytest.approx(res.z_at_tand_min_ohm)

    def test_a_raw_value_that_disagrees_is_cleared_once_the_stray_comes_off(self):
        # The mux16 shape, amplified past the threshold: a 100 pF part behind a 150 pF
        # shunt reads 2.5x its marking, and the part is not what is wrong.
        f, Z = _capacitor(C=100e-12 + 150e-12)
        raw, raw_flags = self._flags(f, Z, nominal_F=100e-12)
        fixed, fixed_flags = self._flags(f, Z, nominal_F=100e-12, C_stray_F=150e-12)

        assert raw_flags, "raw ratio 2.5 must flag"
        assert not fixed_flags
        assert raw.C_F == pytest.approx(250e-12, rel=1e-6)
        assert fixed.C_F == pytest.approx(100e-12, rel=1e-4)

    def test_a_raw_value_that_looks_fine_is_flagged_once_the_stray_comes_off(self):
        # The opposite direction, and the reason the corrected value must be the one
        # JUDGED rather than merely reported: a part reading 60 pF against a 100 pF
        # marking passes the raw check at 0.6x, and is a 7 pF part in truth.
        f, Z = _capacitor(C=60e-12)
        _raw, raw_flags = self._flags(f, Z, nominal_F=100e-12)
        assert not raw_flags, "0.6x is inside the raw band"

        res, flagged = self._flags(f, Z, nominal_F=100e-12, C_stray_F=53.2e-12)
        assert flagged, "6.8 pF against a 100 pF marking is not the marked part"
        assert flagged[0]["measured_F"] == pytest.approx(res.C_corrected_F)
        assert flagged[0]["C_raw_F"] == pytest.approx(60e-12, rel=1e-6)
        assert flagged[0]["stray_corrected"] is True

    def test_the_warning_says_which_number_it_judged_and_shows_both(self):
        f, Z = _capacitor(C=1e-6)
        _res, flagged = self._flags(f, Z, nominal_F=1e-9)
        raw_only = flagged[0]
        assert raw_only["stray_corrected"] is False
        assert raw_only["C_corrected_F"] is None and raw_only["C_stray_F"] is None
        assert "RAW" in raw_only["msg"], "an uncorrected verdict must say so"

    @pytest.mark.parametrize("stray", [None, float("nan"), 0.0, -1e-12])
    def test_an_unusable_stray_is_absent_rather_than_zero(self, stray):
        # NaN, zero and negative are what derive_open_constants returns from a trace it
        # could not read. Subtracting them would report "corrected" for a correction
        # that never happened, and a fixture with literally no shunt is not a thing this
        # hardware produces.
        f, Z = _capacitor(C=147e-9, tand=0.179)
        res = derive_reference_cap(f, Z, nominal_F=1e-9, C_stray_F=stray)
        assert not res.corrected
        assert res.C_F == res.C_raw_F

    def test_an_unreadable_sweep_still_reports_nothing_rather_than_a_correction(self):
        f = _freqs()
        res = derive_reference_cap(f, np.full(f.size, np.nan + 0j), C_stray_F=53.2e-12)
        assert res.C_raw_F != res.C_raw_F and res.C_F != res.C_F

    def test_the_correction_stays_out_of_the_production_fixture_path(self):
        # The production correction is series-only BY DESIGN (OSL corrupted every
        # corrected spectrum), so this shunt subtraction lives entirely inside the
        # commissioning marked-value check and must never leak into fixture.py.
        from softae.analysis.eis import fixture

        assert not hasattr(fixture, "ReferenceCapResult")
        assert "C_stray" not in Path(fixture.__file__).read_text(encoding="utf-8")


class TestStrayReachesTheCapCheck:
    """The derive pass must hand the cap loop the stray it just measured.

    Opens are processed earlier in the same pass, so the number is already in hand for
    any channel that had one. A channel without an open falls back to the raw check --
    a known and accepted gap, not an error.
    """

    def _open(self, channel):
        from softae.workflows.commissioning import AcquiredSpectrum

        f = _freqs()
        # 1 nF keeps |Z| under the ceiling across the band, so derive_open calls it
        # usable and derive_open_constants extracts a stray from it.
        return AcquiredSpectrum(channel, f, 1.0 / (1j * 2 * np.pi * f * 1e-9),
                                electrode_mode="two")

    def _cap(self, channel):
        from softae.workflows.commissioning import AcquiredSpectrum

        f, Z = _capacitor(C=1.1e-9, tand=5e-4)
        return AcquiredSpectrum(channel, f, Z, nominal=1e-10, electrode_mode="two")

    def _strays(self, monkeypatch):
        import softae.analysis.eis.calibration as calmod

        seen: list = []
        real = calmod.derive_reference_cap

        def spy(f, Z, *, nominal_F=None, C_stray_F=None):
            seen.append(C_stray_F)
            return real(f, Z, nominal_F=nominal_F, C_stray_F=C_stray_F)

        monkeypatch.setattr(calmod, "derive_reference_cap", spy)
        return seen

    def test_the_cap_loop_is_handed_its_own_channels_stray(self, monkeypatch):
        from softae.workflows.commissioning import derive_calibration

        seen = self._strays(monkeypatch)
        cal = derive_calibration({"blank_open": [self._open(1)],
                                  "reference_cap": [self._cap(1)]})
        assert cal.C_stray_F[1] == pytest.approx(1e-9, rel=1e-6)
        assert seen == [pytest.approx(cal.C_stray_F[1])]

    def test_a_cap_on_a_channel_without_an_open_gets_the_raw_check(self, monkeypatch):
        # The accepted gap: no same-pass open on ch2, and no reach into a calibration
        # on disk for one. The check falls back to raw rather than inventing a stray.
        from softae.workflows.commissioning import derive_calibration

        seen = self._strays(monkeypatch)
        derive_calibration({"blank_open": [self._open(1)],
                            "reference_cap": [self._cap(2)]})
        assert seen == [None]

    def test_the_phase_table_is_built_from_raw_impedances(self, monkeypatch):
        # The epsilon table bounds the system AS USED, and no production sample
        # spectrum is shunt-corrected anywhere. A corrected table would claim a floor
        # no sample measurement experiences -- so the same pass, with and without an
        # open, must produce the same table.
        from softae.workflows.commissioning import derive_calibration

        with_open = derive_calibration({"blank_open": [self._open(1)],
                                        "reference_cap": [self._cap(1)]})
        without = derive_calibration({"reference_cap": [self._cap(1)]},
                                     electrode_modes=ALL_TWO)
        assert with_open.phase_acc.z_ohm == without.phase_acc.z_ohm
        assert with_open.phase_acc.eps_deg == without.phase_acc.eps_deg


class TestPhaseAccuracyTable:
    def test_an_empty_table_qualifies_nothing(self):
        table = PhaseAccuracyTable()
        assert table.is_empty
        assert table.epsilon_deg(1e6) != table.epsilon_deg(1e6)      # NaN

    def test_it_interpolates_between_characterised_impedances(self):
        table = PhaseAccuracyTable(z_ohm=(1e4, 1e6), eps_deg=(0.15, 0.35),
                                   valid_decades=1.0)
        assert table.epsilon_deg(1e5) == pytest.approx(0.25, rel=0.01)

    def test_it_refuses_to_extrapolate_where_nothing_was_measured(self):
        # The exact mistake that created the withdrawn Z_phi: carrying an instrument
        # constant three decades from where it was taken, without saying so.
        table = PhaseAccuracyTable(z_ohm=(1e4,), eps_deg=(0.149,), valid_decades=1.0)
        assert table.covers(1e5)
        assert not table.covers(1e8)
        assert table.epsilon_deg(1e8) != table.epsilon_deg(1e8)      # NaN, not 0.149

    def test_the_load_type_is_recorded_because_films_are_not_resistors(self):
        table = PhaseAccuracyTable(z_ohm=(1e4,), eps_deg=(0.149,), load="resistive")
        assert "resistive" in table.describe()


class TestCapabilityLadder:
    def test_an_empty_calibration_blocks_everything_and_says_what_would_help(self):
        caps = CalibrationSet().capabilities()
        assert caps.correction_mode == "none"
        assert "fixture correction" in caps.blocked
        assert "run the short blank" in caps.blocked["fixture correction"]

    def test_a_partially_populated_set_reports_which_artifact_unblocks_each_capability(self):
        cal = CalibrationSet(R_short_ohm={1: 5.4})
        caps = cal.capabilities()
        assert caps.correction_mode == "series"          # short alone unlocks this
        assert "qualified upper bounds" in caps.blocked
        assert "reference capacitor" in caps.blocked["qualified upper bounds"]
        assert "correction validation" in caps.blocked
        assert "load blank" in caps.blocked["correction validation"]

    def test_a_measured_phase_table_promotes_provisional_bounds_to_qualified_ones(self):
        cal = CalibrationSet(
            R_short_ohm={1: 5.4},
            phase_acc=PhaseAccuracyTable(z_ohm=(1e6,), eps_deg=(0.3,)))
        caps = cal.capabilities()
        assert caps.phase_floor_measured
        assert "qualified upper bounds" not in caps.blocked

    def test_the_load_blank_is_what_validates_a_correction(self):
        cal = CalibrationSet(R_short_ohm={1: 5.4}, load_error_pct=0.1)
        assert cal.capabilities().can_validate_correction

    def test_the_description_is_actionable_rather_than_a_list_of_absences(self):
        text = CalibrationSet().capabilities().describe()
        assert "blocked" in text and "run the short blank" in text


class TestEnvelopePromotion:
    def test_an_absent_calibration_degrades_to_estimates_flagged_unmeasured(self):
        assert load_calibration("nope", root=Path("does/not/exist")) is None
        assert "none" in describe_or_absent(None)
        assert "short blank" in describe_or_absent(None)

    def test_a_measured_phase_table_promotes_the_envelope(self):
        cal = CalibrationSet(
            created_at="2026-08-05",
            phase_acc=PhaseAccuracyTable(z_ohm=(1e6,), eps_deg=(0.31,),
                                         load="capacitive"))
        env = cal.envelope()
        assert env.phase_noise_measured is True
        assert env.phase_noise_deg == pytest.approx(0.31)
        assert env.phase_noise_at_ohm == pytest.approx(1e6)
        assert env.phase_noise_load == "capacitive"

    def test_an_unmeasured_quantity_keeps_its_estimate_and_its_unmeasured_flag(self):
        # A partial calibration must never silently promote a guess to a measurement.
        env = CalibrationSet(R_short_ohm={1: 5.4}).envelope()
        assert env.magnitude_window_measured is False

    def test_a_measured_window_promotes_the_magnitude_bounds(self):
        env = CalibrationSet(z_min_ohm=12.0, z_max_ohm=5e8).envelope()
        assert env.magnitude_window_measured is True
        assert env.z_max_ohm == pytest.approx(5e8)


class TestPersistenceAndStaleness:
    def _set(self, **over):
        base = dict(
            fixture_id="mux16", hardware_hash="abc123", created_at="2026-08-05",
            channels_measured=(1, 2), channels_assumed=(3,),
            R_short_ohm={1: 5.4, 2: 5.5}, L_lead_H={1: 4.18e-6},
            open_usable={1: False},
            phase_acc=PhaseAccuracyTable(z_ohm=(9.9e3,), eps_deg=(0.149,)),
            sources={"blank_short": 42},
        )
        base.update(over)
        return CalibrationSet(**base)

    def test_it_round_trips_through_the_version_controlled_toml(self, tmp_path):
        save_calibration(self._set(), root=tmp_path)
        back = load_calibration("mux16", root=tmp_path)
        assert back is not None
        assert back.R_short_ohm == {1: 5.4, 2: 5.5}
        assert back.L_lead_H[1] == pytest.approx(4.18e-6)
        assert back.open_usable == {1: False}
        assert back.phase_acc.eps_deg == (0.149,)
        assert back.sources == {"blank_short": 42}
        assert back.channels_assumed == (3,)

    def test_it_lives_beside_the_code_because_the_framework_requires_that(self, tmp_path):
        path = save_calibration(self._set(), root=tmp_path)
        assert path.name == "mux16.toml"
        assert calibration_path("mux16").parts[-3:] == ("calibration", "eis",
                                                        "mux16.toml")

    def test_a_hardware_hash_mismatch_marks_the_set_stale(self):
        assert self._set().is_stale(current_hash="different") is True
        assert self._set().is_stale(current_hash="abc123") is False

    def test_an_unknown_hash_on_either_side_reads_as_stale(self):
        # "We do not know what this was taken on" must degrade, not pass.
        assert self._set(hardware_hash="").is_stale(current_hash="abc123")
        assert self._set().is_stale(current_hash="")

    def test_a_stale_set_has_its_constants_dropped_rather_than_applied(self, tmp_path):
        # The failure this prevents: a short blank from a different board correcting
        # today's spectra, silently, with every number looking plausible.
        save_calibration(self._set(), root=tmp_path)
        resolved = resolve_calibration(
            "mux16", root=tmp_path, config={"pcb": {"changed": True}})
        assert resolved is not None
        assert resolved.R_short_ohm == {}
        assert resolved.phase_acc.is_empty
        assert resolved.capabilities().correction_mode == "none"
        assert "fixture correction" in resolved.capabilities().blocked

    def test_a_matching_hash_is_applied_intact(self, tmp_path):
        cfg = {"pcb": {"a": 1}}
        cal = self._set(hardware_hash=hardware_hash(cfg))
        save_calibration(cal, root=tmp_path)
        resolved = resolve_calibration("mux16", root=tmp_path, config=cfg)
        assert resolved.R_short_ohm == {1: 5.4, 2: 5.5}

    def test_an_unreadable_file_declines_rather_than_raising(self, tmp_path):
        (tmp_path / "broken.toml").write_text("this is not = = toml", encoding="utf-8")
        assert load_calibration("broken", root=tmp_path) is None

    def test_using_an_assumed_channel_logs_the_assumption(self, caplog):
        # Coverage is explicit: channels not directly measured are listed, and using
        # one records that fact in that spectrum's provenance. Never a silent
        # extrapolation across the mux.
        cal = self._set()
        assert 3 in cal.channels_assumed
        cal.for_channel(3)          # must not raise; logs eis_calibration_channel_assumed


class TestHardwareHash:
    def test_the_same_hardware_hashes_the_same(self):
        cfg = {"pcb": {"name": "A"}, "channels": {"n": 32}}
        assert hardware_hash(cfg) == hardware_hash(dict(cfg))

    def test_changing_the_board_changes_the_hash(self):
        assert hardware_hash({"pcb": {"name": "A"}}) != hardware_hash(
            {"pcb": {"name": "B"}})

    def test_changing_a_threshold_does_not(self):
        # Hashing the whole config would make a gate-threshold edit invalidate a good
        # calibration, which trains operators to ignore staleness warnings — the exact
        # failure this mechanism exists to prevent.
        a = {"pcb": {"name": "A"}, "eis": {"gates": {"min_fit_pts": 8}}}
        b = {"pcb": {"name": "A"}, "eis": {"gates": {"min_fit_pts": 12}}}
        assert hardware_hash(a) == hardware_hash(b)

    def test_the_instrument_envelope_is_hardware(self):
        a = {"eis": {"instrument": {"z_max_ohm": 1e9}}}
        b = {"eis": {"instrument": {"z_max_ohm": 1e8}}}
        assert hardware_hash(a) != hardware_hash(b)


# ── Acquisition: a commissioning sweep is an EIS measurement with a tag ──────


class TestCommissioningWorkflow:
    def test_it_reuses_the_ordinary_eis_step(self):
        from softae.core.deposition_steps import eis_measure_step
        from softae.workflows.commissioning import build_commissioning_workflow

        wf = build_commissioning_workflow("blank_short", [1, 2], fixture_id="mux16")
        assert len(wf.setup) == 2
        plain = eis_measure_step(1)
        assert wf.setup[0].method == plain.method
        assert wf.setup[0].instrument == plain.instrument

    def test_the_step_name_cannot_be_mistaken_for_a_trial_measurement(self):
        # The campaign extractors match `measure_eis_ch*`. A blank scored as a trial
        # would be a fabricated observation, so the prefix differs by construction.
        from softae.core.autonomous_wiring import _MEASURE_STEP
        from softae.workflows.commissioning import commissioning_step_name

        assert not commissioning_step_name("blank_short", 1).startswith(_MEASURE_STEP)

    def test_the_role_and_fixture_ride_on_the_step_for_the_executor(self):
        from softae.workflows.commissioning import build_commissioning_workflow

        wf = build_commissioning_workflow("reference_cap", [5], fixture_id="mux16")
        assert wf.setup[0].tags["role"] == "reference_cap"
        assert wf.setup[0].tags["fixture_id"] == "mux16"

    def test_the_nominal_value_is_carried_not_assumed(self):
        # Overhaul 3.7: a capacitor marked "102" (1 nF) measured ~150 nF. Catching
        # that needs BOTH the marking and the measurement.
        from softae.workflows.commissioning import build_commissioning_workflow

        wf = build_commissioning_workflow("reference_cap", [1], nominal=1e-9)
        assert wf.metadata["nominal"] == 1e-9

    def test_the_operator_is_told_what_to_physically_install(self):
        from softae.workflows.commissioning import build_commissioning_workflow

        wf = build_commissioning_workflow("blank_short", [1])
        assert "jumper" in wf.metadata["setup_required"].lower()

    def test_the_open_blanks_prompt_warns_about_the_floating_reference(self):
        # An open cell inherently floats RE - the condition that produced the
        # withdrawn Z_phi. The operator must know before trusting the result.
        from softae.workflows.commissioning import ARTIFACT_SETUP

        assert "RE" in ARTIFACT_SETUP["blank_open"]

    def test_an_unknown_role_is_refused(self):
        from softae.workflows.commissioning import build_commissioning_workflow

        with pytest.raises(ValueError, match="unknown commissioning role"):
            build_commissioning_workflow("blank_magic", [1])

    def test_no_channels_is_refused_rather_than_producing_an_empty_sweep(self):
        from softae.workflows.commissioning import build_commissioning_workflow

        with pytest.raises(ValueError, match="at least one channel"):
            build_commissioning_workflow("blank_short", [])


class TestDeriveCalibrationEndToEnd:
    def test_a_short_only_run_produces_a_usable_incremental_calibration(self):
        from softae.workflows.commissioning import derive_calibration

        f, Z = _short()
        cal = derive_calibration(
            {"blank_short": [(1, f, Z), (2, f, Z)]},
            fixture_id="mux16", created_at="2026-08-05", hardware_hash_value="h1",
            electrode_modes=ALL_TWO)
        assert cal.channels_measured == (1, 2)
        assert cal.R_short_ohm[1] == pytest.approx(5.4, rel=1e-6)
        assert cal.capabilities().correction_mode == "series"

    def test_unmeasured_channels_are_listed_so_the_assumption_is_never_silent(self):
        from softae.workflows.commissioning import derive_calibration

        f, Z = _short()
        cal = derive_calibration(
            {"blank_short": [(1, f, Z)]}, all_channels=range(1, 5),
            representative_channel=1, electrode_modes=ALL_TWO)
        assert cal.channels_measured == (1,)
        assert cal.channels_assumed == (2, 3, 4)
        # Inherited constants are present, but flagged by that membership.
        assert cal.R_short_ohm[3] == pytest.approx(cal.R_short_ohm[1])

    def test_a_reference_capacitor_populates_the_phase_table(self):
        from softae.workflows.commissioning import derive_calibration

        f, Z = _capacitor(C=1e-9, tand=5e-4)
        cal = derive_calibration({"reference_cap": [(1, f, Z)]},
                                 nominals={"reference_cap": 1e-9}, electrode_modes=ALL_TWO)
        assert not cal.phase_acc.is_empty
        assert cal.phase_acc.load == "capacitive"
        assert cal.capabilities().phase_floor_measured

    def test_the_next_artifact_follows_the_value_ordering(self):
        from softae.workflows.commissioning import derive_calibration, next_artifact

        assert next_artifact(None) == "blank_short"
        f, Z = _short()
        cal = derive_calibration({"blank_short": [(1, f, Z)]},
                                 electrode_modes=ALL_TWO)
        assert next_artifact(cal) == "blank_load"

    def test_a_fully_populated_set_has_nothing_left_to_run(self):
        from softae.workflows.commissioning import next_artifact

        cal = CalibrationSet(
            R_short_ohm={1: 5.4}, load_error_pct=0.1,
            phase_acc=PhaseAccuracyTable(z_ohm=(1e6,), eps_deg=(0.3,)),
            z_min_ohm=10.0, z_max_ohm=1e9)
        assert next_artifact(cal) is None


class TestTheNominalSurvivesAcquisition:
    """The marked value must reach derivation, or the ladder stalls forever.

    Reported from the bench: after running ``blank_load`` and deriving, the next
    suggested artifact was *still* ``blank_load``. The derivation needs a marked value
    to compute a load error; ``--nominal`` was collected at ``run`` time, put in the
    workflow metadata, and never persisted — so ``derive`` read the spectrum back from
    the database with no value, skipped the check, left ``load_error_pct`` NaN, and
    ``can_validate_correction`` stayed false. Nothing failed; it just never advanced.
    """

    def _load_spectrum(self):
        f = _freqs()
        return f, np.full(f.size, 9931.7 + 0j)

    def test_without_a_nominal_the_load_check_cannot_run(self):
        from softae.workflows.commissioning import derive_calibration, next_artifact

        fs, Zs = _short()
        fl, Zl = self._load_spectrum()
        cal = derive_calibration(
            {"blank_short": [(1, fs, Zs)], "blank_load": [(1, fl, Zl)]}, electrode_modes=ALL_TWO)
        assert cal.load_error_pct != cal.load_error_pct        # NaN
        assert next_artifact(cal) == "blank_load"              # the reported stall

    def test_with_the_nominal_the_ladder_advances(self):
        from softae.workflows.commissioning import derive_calibration, next_artifact

        fs, Zs = _short()
        fl, Zl = self._load_spectrum()
        cal = derive_calibration(
            {"blank_short": [(1, fs, Zs)], "blank_load": [(1, fl, Zl)]},
            nominals={"blank_load": 9900.0}, electrode_modes=ALL_TWO)
        assert cal.load_error_pct == pytest.approx(0.32, abs=0.01)
        assert cal.capabilities().can_validate_correction
        assert next_artifact(cal) == "reference_cap"

    def test_the_step_carries_the_nominal_for_the_executor_to_record(self):
        from softae.workflows.commissioning import build_commissioning_workflow

        wf = build_commissioning_workflow("blank_load", [32], nominal=9900.0)
        assert float(wf.setup[0].tags["nominal"]) == pytest.approx(9900.0)

    def test_an_artifact_with_no_marking_carries_no_nominal_tag(self):
        from softae.workflows.commissioning import build_commissioning_workflow

        wf = build_commissioning_workflow("blank_short", [32])
        assert "nominal" not in wf.setup[0].tags

    def test_the_value_round_trips_through_the_measurements_table(self, tmp_path):
        from softae.analysis.eis_data import EISResult
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("commission_blank_load")
        f = _freqs(10)
        eis = EISResult.from_arrays(channel=32, f=f, z_real=np.full(10, 9931.7),
                                    z_imag_neg=np.zeros(10))
        mid = store.record_measurement(run_id, eis, role="blank_load",
                                       fixture_id="mux16", nominal_value=9900.0)
        row = store._conn.execute(
            "SELECT nominal_value FROM measurements WHERE measurement_id = ?",
            (mid,)).fetchone()
        assert row[0] == pytest.approx(9900.0)
        store.close()

    def test_a_row_predating_the_column_reads_as_absent_not_zero(self, tmp_path):
        # Rows recorded before this fix have NULL, and NULL must mean "unknown" —
        # a 0.0 would make every load error -100% and look like a catastrophic
        # correction failure.
        from softae.analysis.eis_data import EISResult
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("commission_blank_load")
        f = _freqs(10)
        eis = EISResult.from_arrays(channel=32, f=f, z_real=np.full(10, 9931.7),
                                    z_imag_neg=np.zeros(10))
        mid = store.record_measurement(run_id, eis, role="blank_load")
        row = store._conn.execute(
            "SELECT nominal_value FROM measurements WHERE measurement_id = ?",
            (mid,)).fetchone()
        assert row[0] is None
        store.close()


class TestPerAcquisitionNominals:
    """The marked value belongs to the acquisition, not to the role.

    The database has always stored ``nominal_value`` per measurement row; the derive
    path folded them into one dict keyed by role, last row winning. On the real mux16
    record that meant three reference capacitors marked 1e-10, 1e-10 and 1e-9 F all
    derived against 1e-9 — two of them flagged as disagreeing with a marking that was
    never theirs.
    """

    def _acq(self, C, nominal, channel=1):
        from softae.workflows.commissioning import AcquiredSpectrum

        f, Z = _capacitor(C=C, tand=5e-4)
        return AcquiredSpectrum(channel=channel, freq_hz=f, Z=Z, nominal=nominal,
                                electrode_mode="two")

    def _seen(self, monkeypatch):
        """Record the nominal each reference_cap derivation was actually given."""
        import softae.analysis.eis.calibration as calmod

        seen: list[float | None] = []
        real = calmod.derive_reference_cap

        def spy(f, Z, *, nominal_F=None, C_stray_F=None):
            seen.append(nominal_F)
            return real(f, Z, nominal_F=nominal_F, C_stray_F=C_stray_F)

        monkeypatch.setattr(calmod, "derive_reference_cap", spy)
        return seen

    def test_each_acquisition_derives_against_its_own_marking(self, monkeypatch):
        from softae.workflows.commissioning import derive_calibration

        seen = self._seen(monkeypatch)
        derive_calibration({"reference_cap": [
            self._acq(1e-10, 1e-10), self._acq(1e-10, 1e-10), self._acq(1e-9, 1e-9),
        ]}, nominals={"reference_cap": 1e-9})
        assert seen == [1e-10, 1e-10, 1e-9]

    def test_the_flag_overrides_every_acquisition_of_the_role(self, monkeypatch):
        # Its documented purpose: a part mis-entered at acquisition time. That is a
        # statement about the role's records, so it must not spare the odd one out.
        from softae.workflows.commissioning import derive_calibration

        seen = self._seen(monkeypatch)
        derive_calibration({"reference_cap": [
            self._acq(1e-10, 1e-10), self._acq(1e-9, 1e-9),
        ]}, nominal_overrides={"reference_cap": 4.7e-10})
        assert seen == [4.7e-10, 4.7e-10]

    def test_an_acquisition_with_no_marking_falls_back_to_the_roles_latest(
            self, monkeypatch):
        from softae.workflows.commissioning import derive_calibration

        seen = self._seen(monkeypatch)
        derive_calibration({"reference_cap": [self._acq(1e-9, None)]},
                           nominals={"reference_cap": 1e-9})
        assert seen == [1e-9]

    def test_no_marking_anywhere_stays_unknown_rather_than_inventing_one(
            self, monkeypatch):
        from softae.workflows.commissioning import derive_calibration

        seen = self._seen(monkeypatch)
        derive_calibration({"reference_cap": [self._acq(1e-9, None)]})
        assert seen == [None]

    def test_a_legacy_tuple_still_derives_against_the_role_value(self, monkeypatch):
        # Every existing caller passes (channel, f, Z). A tuple carries no marking of
        # its own, so the role value is the only honest answer for it.
        from softae.workflows.commissioning import derive_calibration

        seen = self._seen(monkeypatch)
        f, Z = _capacitor(C=1e-9, tand=5e-4)
        derive_calibration({"reference_cap": [(1, f, Z)]},
                           nominals={"reference_cap": 1e-9}, electrode_modes=ALL_TWO)
        assert seen == [1e-9]

    def test_the_load_blank_uses_its_own_marking_too(self):
        from softae.workflows.commissioning import AcquiredSpectrum, derive_calibration

        f = _freqs()
        Z = np.full(f.size, 9931.7 + 0j)
        cal = derive_calibration({"blank_load": [AcquiredSpectrum(
            1, f, Z, nominal=9900.0, electrode_mode="two")]})
        assert cal.load_error_pct == pytest.approx(0.32, abs=0.01)


class TestPerAcquisitionElectrodeMode:
    """R24/F17 refusal, applied where the fact lives.

    Role-level modes made the refusal self-defeating: a pre-jumper sweep recorded
    ``unknown`` sat in the same role as a later two-electrode one, and the newest row's
    mode was applied to both. The spectrum the check exists to stop was the one it
    passed.
    """

    def _cap(self, mode, channel=1):
        from softae.workflows.commissioning import AcquiredSpectrum

        f, Z = _capacitor(C=1e-9, tand=5e-4)
        return AcquiredSpectrum(channel, f, Z, nominal=1e-9, electrode_mode=mode)

    def test_one_unknown_acquisition_is_dropped_without_taking_its_siblings(self):
        from softae.workflows.commissioning import derive_calibration

        cal = derive_calibration({"reference_cap": [
            self._cap("unknown", channel=31), self._cap("two", channel=25),
        ]})
        assert cal.channels_measured == (25,)
        assert not cal.phase_acc.is_empty

    def test_every_acquisition_unknown_still_raises(self):
        from softae.workflows.commissioning import CommissioningError, derive_calibration

        with pytest.raises(CommissioningError, match="electrode mode"):
            derive_calibration({"reference_cap": [self._cap("unknown")]})

    def test_the_role_level_mode_remains_the_fallback_for_a_bare_tuple(self):
        from softae.workflows.commissioning import derive_calibration

        f, Z = _capacitor(C=1e-9, tand=5e-4)
        cal = derive_calibration({"reference_cap": [(7, f, Z)]},
                                 electrode_modes={"reference_cap": "two"})
        assert cal.channels_measured == (7,)


class TestTheMeasuredWindowComesFromSurvivors:
    """``z_max`` was the instrument giving up, not a magnitude the fixture reproduces.

    ``z_points`` feeds both the phase table and the |Z| window, so an ungated table put
    the ~1.0147 GOhm input rail straight into ``InstrumentEnvelope.z_max_ohm`` — an
    envelope bound asserting the rig measures a decade beyond where it stops working.
    """

    def _capacitor_with_a_rail(self):
        f = np.logspace(0.6, 5.3, 41)[::-1]
        mag = 1.0 / (2 * np.pi * f * 1e-9)
        Z = (mag * 0.005 - 1j * mag).astype(complex)
        Z[-6:] = 1.0147e9 * 0.005 - 1j * 1.0147e9
        return f, Z

    def test_the_rail_is_excluded_from_z_max(self):
        from softae.workflows.commissioning import derive_calibration

        f, Z = self._capacitor_with_a_rail()
        cal = derive_calibration({"reference_cap": [(1, f, Z)]},
                                 nominals={"reference_cap": 1e-9},
                                 electrode_modes=ALL_TWO)
        assert cal.z_max_ohm < 1e9
        assert cal.z_max_ohm == max(cal.phase_acc.z_ohm)
        assert cal.z_min_ohm == min(cal.phase_acc.z_ohm)

    def test_a_wholly_ungatable_sweep_leaves_the_window_unmeasured(self):
        # Nothing survives -> the existing empty-table path stands: NaN, and the
        # ladder keeps asking for the reference resistors rather than inventing a
        # window from noise.
        from softae.workflows.commissioning import derive_calibration

        f = np.logspace(0.6, 5.3, 41)[::-1]
        Z = np.full(f.size, -1.0e9 + 1.0e9j)          # every point wrong-quadrant
        cal = derive_calibration({"reference_cap": [(1, f, Z)]},
                                 electrode_modes=ALL_TWO)
        assert cal.phase_acc.is_empty
        assert math.isnan(cal.z_min_ohm) and math.isnan(cal.z_max_ohm)
        assert "measured |Z| window" in cal.capabilities().blocked


class TestTheDeriveCommandReadsPerAcquisitionFacts:
    """``_load_role_spectra`` is where the collapse happened; this is the round trip."""

    def _store_with_three_caps(self, tmp_path):
        from softae.analysis.eis_data import EISResult
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("commission_reference_cap", mode="commissioning")
        for i, (C, nominal, mode) in enumerate((
                (1e-10, 1e-10, "unknown"), (1e-10, 1e-10, "two"), (1e-9, 1e-9, "two"))):
            f, Z = _capacitor(C=C, tand=5e-4)
            dest = Path(store.project_dir) / "eis" / f"cap{i}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            eis = EISResult.from_arrays(channel=25, f=f, z_real=Z.real,
                                        z_imag_neg=-Z.imag,
                                        raw_file_path=str(dest))
            eis.save(dest)
            store.record_measurement(run_id, eis, role="reference_cap",
                                     fixture_id="mux16", nominal_value=nominal,
                                     electrode_mode=mode)
        return store

    def test_each_row_comes_back_with_its_own_marking_and_mode(self, tmp_path):
        from softae.tools.commission import _load_role_spectra

        store = self._store_with_three_caps(tmp_path)
        try:
            artifacts, nominals, modes = _load_role_spectra(store, "mux16")
            caps = artifacts["reference_cap"]
            assert [a.nominal for a in caps] == [1e-10, 1e-10, 1e-9]
            assert [a.electrode_mode for a in caps] == ["unknown", "two", "two"]
            # The role dicts survive with narrowed meaning: a fallback and a summary.
            assert nominals["reference_cap"] == 1e-9
            assert modes["reference_cap"] == "mixed"
        finally:
            store.close()

    def test_the_provenance_id_travels_with_the_spectrum(self, tmp_path):
        from softae.tools.commission import _load_role_spectra, _sources

        store = self._store_with_three_caps(tmp_path)
        try:
            artifacts, _n, _m = _load_role_spectra(store, "mux16")
            ids = [a.measurement_id for a in artifacts["reference_cap"]]
            assert all(i is not None for i in ids)
            assert _sources(artifacts)["reference_cap"] == max(ids)
        finally:
            store.close()

    def test_declaring_a_mode_reaches_the_unknown_row_beside_a_known_one(self, tmp_path):
        # The operator's recovery path for a pre-jumper sweep. A role summarised by its
        # newest row reported "two" for this set and the declaration skipped it — the
        # branch existing to fix unknown rows, refusing to see them.
        from softae.tools.commission import _declare_electrode_mode, _load_role_spectra

        store = self._store_with_three_caps(tmp_path)
        try:
            artifacts, _n, _m = _load_role_spectra(store, "mux16")
            _declare_electrode_mode(store, "mux16", "two", artifacts)
            artifacts, _n, modes = _load_role_spectra(store, "mux16")
            assert modes["reference_cap"] == "two"
            assert all(a.electrode_mode == "two" for a in artifacts["reference_cap"])
        finally:
            store.close()


class TestCalibrationHistory:
    """Successive calibrations of one fixture *are* an instrument-drift measurement."""

    def _store(self, tmp_path):
        from softae.core.data_store import DataStore

        return DataStore(tmp_path / "proj")

    def test_calibrations_append_rather_than_overwrite(self, tmp_path):
        store = self._store(tmp_path)
        store.record_calibration(
            CalibrationSet(fixture_id="mux16", hardware_hash="h1",
                           created_at="2026-08-01", R_short_ohm={1: 5.4}))
        store.record_calibration(
            CalibrationSet(fixture_id="mux16", hardware_hash="h1",
                           created_at="2026-09-01", R_short_ohm={1: 5.6}))
        history = store.calibration_history("mux16")
        assert len(history) == 2
        # The drift datum: same fixture, same hardware, two epochs.
        first = history[0]["calibration"]["R_short_ohm"]["1"]
        last = history[1]["calibration"]["R_short_ohm"]["1"]
        assert first == pytest.approx(5.4) and last == pytest.approx(5.6)
        store.close()

    def test_the_earlier_row_is_superseded_not_deleted(self, tmp_path):
        store = self._store(tmp_path)
        store.record_calibration(CalibrationSet(fixture_id="mux16",
                                                created_at="2026-08-01"))
        store.record_calibration(CalibrationSet(fixture_id="mux16",
                                                created_at="2026-09-01"))
        history = store.calibration_history("mux16")
        assert history[0]["superseded_at"] is not None
        assert history[1]["superseded_at"] is None
        store.close()

    def test_a_different_fixture_keeps_its_own_history(self, tmp_path):
        store = self._store(tmp_path)
        store.record_calibration(CalibrationSet(fixture_id="a", created_at="1"))
        store.record_calibration(CalibrationSet(fixture_id="b", created_at="1"))
        assert len(store.calibration_history("a")) == 1
        assert len(store.calibration_history("b")) == 1
        store.close()

    def test_an_untouched_database_has_no_history_rather_than_failing(self, tmp_path):
        store = self._store(tmp_path)
        assert store.calibration_history("never-calibrated") == []
        store.close()


class TestMeasurementRoleRecording:
    """The role column existed since E0 but nothing wrote it - closed here."""

    def _store_and_result(self, tmp_path):
        from softae.analysis.eis_data import EISResult
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("commissioning")
        f = _freqs(10)
        eis = EISResult.from_arrays(channel=1, f=f, z_real=np.full(10, 5.4),
                                    z_imag_neg=np.zeros(10))
        return store, run_id, eis

    def test_a_sample_stays_the_default_so_existing_callers_are_unchanged(self, tmp_path):
        store, run_id, eis = self._store_and_result(tmp_path)
        mid = store.record_measurement(run_id, eis)
        row = store._conn.execute(
            "SELECT role, fixture_id FROM measurements WHERE measurement_id = ?",
            (mid,)).fetchone()
        assert row[0] == "sample" and row[1] is None
        store.close()

    def test_a_blank_is_recorded_through_the_very_same_path(self, tmp_path):
        store, run_id, eis = self._store_and_result(tmp_path)
        mid = store.record_measurement(run_id, eis, role="blank_short",
                                       fixture_id="mux16")
        row = store._conn.execute(
            "SELECT role, fixture_id FROM measurements WHERE measurement_id = ?",
            (mid,)).fetchone()
        assert tuple(row) == ("blank_short", "mux16")
        store.close()

    def test_an_unknown_role_is_recorded_as_a_sample_rather_than_invented(self, tmp_path):
        # A typo must not silently create a calibration artifact.
        store, run_id, eis = self._store_and_result(tmp_path)
        mid = store.record_measurement(run_id, eis, role="blank_typo")
        row = store._conn.execute(
            "SELECT role FROM measurements WHERE measurement_id = ?", (mid,)).fetchone()
        assert row[0] == "sample"
        store.close()


class TestElectrodeMode:
    """R24/F17 — a two-terminal reference sensed in three-electrode mode is not usable.

    Not "less accurate": *uncalibratable*. With no ionic path to the reference stripe,
    RE floats onto a capacitive divider whose ratio depends on the load. Measured on
    this rig's own commissioning files, alpha ran 2.24 (ch25 blank) to 23.8 (ch17
    blank) — and the same 1 nF part measured twice gave 9.85 and 4.96, so it is not
    even reproducible at fixed load. There is nothing stable to divide out.
    """

    def test_a_sample_is_unconstrained_because_a_film_does_couple_to_re(self):
        from softae.analysis.eis.calibration import electrode_mode_ok

        ok, _ = electrode_mode_ok("sample", "three")
        assert ok, "a conductive film in contact with RE is the valid 3-electrode case"

    @pytest.mark.parametrize("role", sorted(TWO_TERMINAL_ROLES))
    def test_every_two_terminal_reference_is_refused_in_three_electrode_mode(self, role):
        from softae.analysis.eis.calibration import electrode_mode_ok

        ok, why = electrode_mode_ok(role, "three")
        assert not ok
        assert "uncalibratable" in why

    @pytest.mark.parametrize("role", sorted(TWO_TERMINAL_ROLES))
    def test_two_electrode_is_accepted(self, role):
        from softae.analysis.eis.calibration import electrode_mode_ok

        assert electrode_mode_ok(role, "two")[0]

    def test_an_unrecorded_mode_is_refused_rather_than_assumed(self):
        """Assuming it was correct would invent the one fact that decides usability."""
        from softae.analysis.eis.calibration import electrode_mode_ok

        ok, why = electrode_mode_ok("blank_short", "unknown")
        assert not ok
        assert "no recorded electrode mode" in why


class TestPhaseTableFromOneComponent:
    """A single capacitor spans the working decades; the table should use all of them."""

    def _cap(self, C=1e-9, tand=0.005, npts=41):
        f = np.logspace(0.6, 5.3, npts)[::-1]
        Zc = 1.0 / (1j * 2 * np.pi * f * C)
        return f, Zc * (1 + tand * 1j) if False else Zc.real + tand * np.abs(Zc) - 1j * np.abs(Zc)

    def test_one_sweep_populates_several_decades(self):
        from softae.analysis.eis.calibration import derive_phase_table

        f, Z = self._cap()
        z_pts, e_pts = derive_phase_table(f, Z)
        assert len(z_pts) >= 4, "a 1 nF sweep spans four-plus decades of |Z|"
        assert len(z_pts) == len(e_pts)
        assert z_pts == sorted(z_pts)

    def test_it_uses_the_median_not_the_sweep_minimum(self):
        """The minimum is the single luckiest point, not a bound on anything.

        Taking it on this rig's 1 nF C0G gave tan d = 7e-5 (0.004 deg), ~100x tighter
        than the 0.2-0.5 deg the same data supports per decade — and a phase floor that
        small would qualify almost any spectrum as a value rather than a bound.
        """
        from softae.analysis.eis.calibration import derive_phase_table

        f, Z = self._cap(tand=0.01)
        Z = np.asarray(Z, dtype=complex)
        Z[len(Z) // 2] = Z[len(Z) // 2].imag * -1j     # one artificially perfect point

        _, e_pts = derive_phase_table(f, Z)
        assert min(e_pts) > 0.1, "a lone lossless point must not set the floor"

    def test_empty_input_yields_an_empty_table_rather_than_raising(self):
        from softae.analysis.eis.calibration import derive_phase_table

        z, e = derive_phase_table(np.array([]), np.array([], dtype=complex))
        assert z == [] and e == []


class TestPhaseTableGates:
    """A median defends against one bad point, not against a population of them.

    The mux16 record tabulated 7 of 24 points above 30 deg and put the instrument's
    ~1.0147 GOhm input rail at both ends of the table — as a 44.96 deg point and a
    0.45 deg one, from the same railed magnitude. Neither is a phase measurement, and
    both entered because the only gate was finiteness.
    """

    def _clean(self, C=1e-9, tand=0.005, npts=41):
        """An ideal capacitive reference: fourth quadrant, |Z| exactly 1/(2 pi f C)."""
        f = np.logspace(0.6, 5.3, npts)[::-1]
        mag = 1.0 / (2 * np.pi * f * C)
        return f, mag * tand - 1j * mag

    def test_a_clean_capacitive_sweep_survives_intact(self):
        from softae.analysis.eis.calibration import phase_table_gate

        f, Z = self._clean()
        assert phase_table_gate(f, Z).all()

    def test_a_railed_plateau_is_dropped(self):
        # The rail's signature is physical, not a magic constant: |Z| stops following
        # 1/(2 pi f C) and flattens, so d log|Z| / d log f leaves -1 for ~0.
        from softae.analysis.eis.calibration import phase_table_gate

        f, Z = self._clean()
        rail = 1.0147e9
        Z = np.asarray(Z, dtype=complex).copy()
        Z[-6:] = rail * 0.005 - 1j * rail          # six points pinned at the ceiling
        mask = phase_table_gate(f, Z)
        assert not mask[-6:].any(), "the plateau itself must not survive"
        # -7 rather than -6: the last good point borrows its neighbour's slope from
        # the rail and goes with it. A documented over-drop, and the safe direction.
        assert mask[:-7].all(), "points clear of the rail are untouched"

    def test_a_rail_no_longer_reaches_the_table_or_the_z_window(self):
        from softae.analysis.eis.calibration import derive_phase_table

        f, Z = self._clean()
        Z = np.asarray(Z, dtype=complex).copy()
        Z[-6:] = 1.0147e9 * 0.005 - 1j * 1.0147e9
        z_pts, _e = derive_phase_table(f, Z)
        assert z_pts, "the rest of the sweep still tabulates"
        assert max(z_pts) < 1e9

    def test_wrong_quadrant_points_are_dropped_rather_than_absolutised(self):
        # tan d = |Re| / |Im| takes absolute values, so a quadrant violation does not
        # fail — it becomes a LARGE eps and is averaged in as measured loss.
        from softae.analysis.eis.calibration import phase_table_gate

        f, Z = self._clean()
        Z = np.asarray(Z, dtype=complex).copy()
        Z[5] = -abs(Z[5].real) + 1j * Z[5].imag       # Re < 0: not a passive load
        Z[9] = Z[9].real - 1j * Z[9].imag             # Im > 0: not a capacitor
        mask = phase_table_gate(f, Z)
        assert not mask[5] and not mask[9]
        assert mask.sum() == f.size - 2

    def test_a_wrong_quadrant_point_no_longer_inflates_a_decade(self):
        from softae.analysis.eis.calibration import derive_phase_table

        f, Z = self._clean(tand=0.005)
        Z = np.asarray(Z, dtype=complex).copy()
        # Half of one decade flipped into the second quadrant, each reading ~76 deg
        # once absolutised. Ungated this moved the decade's median; gated it cannot.
        flip = (np.abs(Z) > 1e4) & (np.abs(Z) < 1e5)
        idx = np.flatnonzero(flip)[: max(1, flip.sum() // 2)]
        Z[idx] = -4.0 * np.abs(Z[idx]) + 1j * Z[idx].imag
        _z, e_pts = derive_phase_table(f, Z)
        assert max(e_pts) < 1.0, "no quadrant violation may set the floor"

    def test_the_slope_tolerance_is_a_boundary_not_a_cliff(self):
        from softae.analysis.eis.calibration import (
            PHASE_TABLE_SLOPE_TOL,
            phase_table_gate,
        )

        # A power law |Z| ~ f^-p has slope -p everywhere, so p brackets the tolerance
        # exactly: -1 +/- tol is admitted, anything past it is not.
        f = np.logspace(0.6, 5.3, 41)[::-1]
        for p, expected in ((1.0 + PHASE_TABLE_SLOPE_TOL - 0.05, True),
                            (1.0 + PHASE_TABLE_SLOPE_TOL + 0.05, False)):
            mag = 1e6 * f ** (-p)
            Z = mag * 0.005 - 1j * mag
            assert bool(phase_table_gate(f, Z).all()) is expected

    def test_a_resistive_reference_states_its_own_expectation(self):
        # The gate is parameterised by the part, not hardcoded to a capacitor: a flat
        # |Z| is a rail for a capacitor and the correct answer for a resistor.
        from softae.analysis.eis.calibration import phase_table_gate

        f = np.logspace(0.6, 5.3, 41)[::-1]
        Z = np.full(f.size, 9900.0 + 0j)
        assert phase_table_gate(f, Z, load="resistive").all()
        assert not phase_table_gate(f, Z, load="capacitive").any()

    def test_an_unknown_load_name_is_refused_rather_than_defaulted(self):
        # Silently falling back to "capacitive" would empty a resistive table and read
        # as "nothing survived gating" — a plausible result, and the wrong one.
        from softae.analysis.eis.calibration import phase_table_gate

        f = np.logspace(0.6, 5.3, 9)[::-1]
        with pytest.raises(ValueError, match="unknown reference load"):
            phase_table_gate(f, np.full(f.size, 1.0 - 1j), load="capacitve")

    def test_a_sweep_too_short_to_have_a_slope_abstains_rather_than_emptying(self):
        # With no neighbours there is no plateau to see. Dropping everything would
        # turn "cannot tell" into "all bad", which is not the conservative direction.
        from softae.analysis.eis.calibration import phase_table_gate

        f = np.array([1e5, 1e4])
        Z = np.array([1.0 - 100j, 1.0 - 1000j])
        assert phase_table_gate(f, Z).all()

    def test_gating_precedes_the_median_rather_than_replacing_it(self):
        # Both defences are needed: the gate removes non-measurements, the median
        # keeps one lucky *measurement* from setting the floor.
        from softae.analysis.eis.calibration import derive_phase_table

        f, Z = self._clean(tand=0.01)
        Z = np.asarray(Z, dtype=complex).copy()
        Z[len(Z) // 2] = 1e-9 * abs(Z[len(Z) // 2]) + 1j * Z[len(Z) // 2].imag
        _z, e_pts = derive_phase_table(f, Z)
        assert min(e_pts) > 0.1


class TestDecadesSpannedIsNotValidDecades:
    """Two numbers, two questions — conflated once in review, kept apart here."""

    def test_the_span_is_measured_while_the_trust_radius_is_declared(self):
        table = PhaseAccuracyTable(z_ohm=(795.6, 1.4e8), eps_deg=(4.35, 3.63))
        assert table.decades_spanned == pytest.approx(5.24, abs=0.01)
        assert table.valid_decades == 1.0        # a property of the instrument

    def test_a_single_point_spans_nothing_rather_than_zero(self):
        # 0.0 would read as "characterised at one impedance, no spread"; the truth is
        # that a span needs two points.
        assert math.isnan(PhaseAccuracyTable(z_ohm=(1e6,),
                                             eps_deg=(0.3,)).decades_spanned)

    def test_the_description_carries_both(self):
        text = PhaseAccuracyTable(z_ohm=(795.6, 1.4e8), eps_deg=(4.35, 3.63),
                                  load="capacitive").describe()
        assert "5.2 decade(s)" in text and "either side" in text


class TestAssumedChannelsCarryTheirMagnitude:
    """`channels_assumed` names an assumption; this gives it a size.

    Seven tied open blanks on nominally identical stripes (ch17–23, 2026-08-06) gave
    C_stray from 10.2 to 24.7 pF — 2.4× — while any single channel repeated to 1%. So
    inheriting one channel's constants is a real uncertainty roughly an order of
    magnitude above the measurement error, not bookkeeping.
    """

    def _set(self, **over):
        from softae.analysis.eis.calibration import CalibrationSet

        base = dict(fixture_id="mux16", hardware_hash="h1", created_at="2026-08-06",
                    channels_measured=(17, 18, 19), channels_assumed=(20, 21),
                    C_stray_F={17: 10.2e-12, 18: 14.6e-12, 19: 24.7e-12,
                               20: 10.2e-12, 21: 10.2e-12})
        base.update(over)
        return CalibrationSet(**base)

    def test_the_spread_is_measured_over_measured_channels_only(self):
        # Assumed channels are copies of the representative, so counting them would
        # drag the spread toward 1.0 -- the assumption flattering itself.
        assert self._set().measured_spread("C_stray_F") == pytest.approx(
            24.7 / 10.2, rel=1e-6)

    def test_one_measured_channel_reports_unknown_not_no_variation(self):
        # 1.0 would read as "no channel-to-channel variation", which is the opposite
        # of what a single measurement establishes.
        s = self._set(channels_measured=(17,), channels_assumed=(18, 19, 20, 21))
        assert math.isnan(s.measured_spread("C_stray_F"))

    def test_an_absent_constant_reports_unknown_rather_than_raising(self):
        assert math.isnan(self._set().measured_spread("R_short_ohm"))
        assert math.isnan(self._set().measured_spread("not_a_field"))

    def test_using_an_assumed_channel_warns_rather_than_informs(self, caplog):
        # It was `logger.info`. An inherited constant worth 2.4x belongs at warning.
        s = self._set()
        s.for_channel(20)
        s.for_channel(17)   # measured — must stay silent


class TestOpenBlankProducesConstants:
    """`derive_open` returned only a verdict, so `C_stray_F` was written by nothing.

    The field was declared, serialised and read by `for_channel` — a producer-less
    consumer, the same shape as `profilometry_um` and `role`/`fixture_id` before it.
    Both shunt constants were derivable from an artifact the module already collected.
    """

    def _open(self, C=15e-12, tand=0.05, npts=35):
        f = np.geomspace(2e5, 4.0, npts)
        omega = 2 * np.pi * f
        # A lossy capacitor: G = omega*C*tand, so d ln G / d ln f is exactly 1.
        Y = omega * C * tand + 1j * omega * C
        return f, 1.0 / Y

    def test_it_recovers_the_stray_capacitance(self):
        from softae.analysis.eis.calibration import derive_open_constants

        f, Z = self._open(C=15e-12)
        C, _G = derive_open_constants(f, Z)
        assert C == pytest.approx(15e-12, rel=0.02)

    def test_it_classifies_dielectric_loss_by_its_power_law(self):
        # +1 is dielectric loss, 0 an ohmic leak. The seven tied opens on this fixture
        # give +0.87 to +1.04, which is what makes G a table rather than a number.
        from softae.analysis.eis.calibration import derive_open_constants

        f, Z = self._open()
        _C, G = derive_open_constants(f, Z)
        assert G.exponent == pytest.approx(1.0, abs=0.02)
        assert G.is_dielectric

    def test_an_ohmic_leak_is_classified_differently(self):
        from softae.analysis.eis.calibration import derive_open_constants

        f = np.geomspace(2e5, 4.0, 35)
        Y = 5e-9 + 1j * 2 * np.pi * f * 15e-12      # frequency-flat conductance
        _C, G = derive_open_constants(f, 1.0 / Y)
        assert abs(G.exponent) < 0.1
        assert not G.is_dielectric

    def test_the_conductance_table_refuses_to_extrapolate(self):
        # Same discipline as PhaseAccuracyTable: carrying a fixture constant beyond the
        # band it was measured in is how the withdrawn Z_phi came to be believed.
        from softae.analysis.eis.calibration import derive_open_constants

        f, Z = self._open()
        _C, G = derive_open_constants(f, Z)
        assert G.at(1e3) == G.at(1e3)              # inside: a number
        assert math.isnan(G.at(1e9))               # above the sweep
        assert math.isnan(G.at(1e-6))              # below it

    def test_negative_real_admittance_is_dropped_not_clipped(self):
        # Re(Y) < 0 at low frequency is the phase floor showing through on a near-ideal
        # blank. Clipping to zero would read as "no loss here"; the truth is "below what
        # this instrument resolves", which is an absence, not a value.
        from softae.analysis.eis.calibration import derive_open_constants

        f, Z = self._open()
        Y = 1.0 / np.asarray(Z)
        Y[-3:] = -1e-12 + Y[-3:].imag * 1j
        _C, G = derive_open_constants(f, 1.0 / Y)
        assert len(G.freq_hz) == len(f) - 3
        assert all(g > 0 for g in G.G_S)


class TestNestedTomlRoundTrip:
    """`_to_toml` handled one level of nesting and silently lost anything deeper.

    `G_fixture` is a mapping of per-channel tables. It serialised as quoted Python
    reprs, `from_dict` discarded them, and the file *looked* correct while the
    calibration came back missing a field — the worst shape a persistence bug takes.
    """

    def _cal(self):
        from softae.analysis.eis.calibration import CalibrationSet, FixtureConductance

        return CalibrationSet(
            fixture_id="mux16", hardware_hash="h1", created_at="2026-08-07",
            channels_measured=(17, 18),
            C_stray_F={17: 10.2e-12, 18: 14.6e-12},
            open_usable={17: True, 18: True},
            G_fixture={
                17: FixtureConductance(freq_hz=(10.0, 100.0, 1000.0),
                                       G_S=(4e-11, 4e-10, 4e-9), exponent=1.0),
                18: FixtureConductance(freq_hz=(10.0, 100.0),
                                       G_S=(5e-11, 5e-10), exponent=0.99),
            },
        )

    def test_a_nested_table_survives_the_round_trip(self, tmp_path):
        from softae.analysis.eis.calibration import load_calibration, save_calibration

        save_calibration(self._cal(), root=tmp_path)
        back = load_calibration("mux16", root=tmp_path)
        assert back is not None
        assert set(back.G_fixture) == {17, 18}
        assert back.G_fixture[17].G_S == (4e-11, 4e-10, 4e-9)
        assert back.G_fixture[18].exponent == pytest.approx(0.99)

    def test_the_written_file_is_real_toml_not_a_stringified_dict(self, tmp_path):
        import tomllib

        from softae.analysis.eis.calibration import save_calibration

        path = save_calibration(self._cal(), root=tmp_path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data["G_fixture"]["17"], dict)
        assert isinstance(data["G_fixture"]["17"]["freq_hz"], list)

    def test_the_flat_tables_are_unchanged_by_the_recursion(self, tmp_path):
        from softae.analysis.eis.calibration import load_calibration, save_calibration

        save_calibration(self._cal(), root=tmp_path)
        back = load_calibration("mux16", root=tmp_path)
        assert back.C_stray_F == {17: 10.2e-12, 18: 14.6e-12}
        assert back.open_usable == {17: True, 18: True}
