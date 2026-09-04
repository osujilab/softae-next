"""The commissioned envelope, and the two decisions that put it into production.

``CalibrationSet.envelope()`` had **zero call sites in ``src/``** until
``docs/SubAgent docs/envelope_wiring.md`` decisions 1 and 2 landed: the fixture was
commissioned, ``calibration/eis/mux16.toml`` was committed beside the code, and every
gate went on reading the configured fallback. These tests are the wiring made
mechanical.

Two properties carry the weight, and neither existed before:

**A blank is not a sample.** The commissioned window is derived from reference
resistors spanning 795.6 Ω–1.4456×10⁸ Ω. Applied to its own ``sources.blank_short`` —
``measurement_id = 3490``, a jumpered channel at ~7.7 Ω — it leaves 2 of 10 points,
under a ``min_fit_pts`` of 8. Judging a commissioning artifact by a window its own
siblings produced is circular whatever the numbers say, and every path that replays
stored spectra through ``analyze_spectrum`` sees all 25 commissioning rows in the
corpus.

**The floor and the anchor come from the same row.** ``envelope()`` selected both from
``argmin(|Z|)``, which optimises neither: on the committed 19-point table
``argmax(ε)`` is both the *more conservative* floor (tan ε 0.1072 against 0.0761) and
an anchor four decades closer to where films actually sit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.calibration import (
    CalibrationSet,
    PhaseAccuracyTable,
    load_calibration,
)
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.envelope import (
    SAMPLE_ROLES,
    instrument_envelope,
    magnitude_window_applies,
)
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.policy import BOUND_MODES
from softae.analysis.eis.report import decide_report_mode
from softae.analysis.eis.settings import EISSettings, GateSettings
from tests.eis_synthetic import as_eis_result, log_frequencies, reference_spectrum

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The committed window, as commissioned 2026-09-03.
Z_MIN = 795.6011977999614
Z_MAX = 144560987.92693862

CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")


def _gated(enabled: bool = True) -> EISSettings:
    return EISSettings(engine="gated", gates=GateSettings(enabled=enabled))


def _commissioned(**over) -> CalibrationSet:
    """A calibration carrying the committed window and a two-row phase table.

    The two rows deliberately disagree about which is selected: row 0 is
    ``argmin(|Z|)``, row 1 is ``argmax(ε)``. A single-row table — which is what the
    only pre-existing selection test used — cannot see the difference.
    """
    base = dict(
        fixture_id="mux16", created_at="2026-09-03",
        z_min_ohm=Z_MIN, z_max_ohm=Z_MAX,
        phase_acc=PhaseAccuracyTable(z_ohm=(Z_MIN, 1.01178e7),
                                     eps_deg=(4.3513, 6.1200),
                                     load="capacitive", valid_decades=1.0),
    )
    base.update(over)
    return CalibrationSet(**base)


def _dropped(report, gate_name: str) -> int:
    entries = [e for e in report.gate_log if e["gate"] == gate_name]
    assert entries, f"{gate_name} never ran"
    return int(entries[0]["n_dropped"])


class TestTheRolePredicate:
    def test_only_sample_class_roles_are_judged_against_the_sample_window(self):
        assert magnitude_window_applies("sample")
        assert magnitude_window_applies("drift_repeat")
        for role in ("blank_short", "blank_open", "blank_load",
                     "reference_cap", "reference_r"):
            assert not magnitude_window_applies(role)

    def test_an_unrecognised_role_is_not_treated_as_a_sample(self):
        """An allow-list, because "undeclared" must not be spelled "checked and clean".

        ``reference_r_misplaced_lead`` is a live DataStore role — four rows — that is
        not in ``MEASUREMENT_ROLES``. An exemption list would have narrowed those
        reference resistors against a window derived from their own siblings.
        """
        assert not magnitude_window_applies("reference_r_misplaced_lead")
        assert not magnitude_window_applies("")
        assert not magnitude_window_applies(None)
        assert "sample" in SAMPLE_ROLES


class TestTheCommissionedWindowReachesTheGate:
    """Decision 1 — ``gate_magnitude`` reads the commissioned bounds, not the fallback.

    The configured fallback floor is 10 Ω, hand-mirrored in ``[eis.instrument]``; the
    commissioned floor is 795.6 Ω. A point at 300 Ω is inside one and outside the
    other, so a single spectrum distinguishes "wired" from "not running at all".
    """

    def _spectrum_with_a_point_below_the_commissioned_floor(self):
        f = log_frequencies(f_lo=20.0, f_hi=2.0e5, npts=41)
        _f, Z = reference_spectrum(f)
        Z = np.asarray(Z, dtype=complex).copy()
        Z[0] = 300.0 + 0.0j          # inside [10, z_max], outside [795.6, z_max]
        return f, Z

    def test_a_point_under_the_commissioned_floor_is_dropped_on_a_sample(self):
        f, Z = self._spectrum_with_a_point_below_the_commissioned_floor()
        report = analyze_spectrum(
            as_eis_result(f, Z), cell=CELL, settings=_gated(),
            calibration=_commissioned(), role="sample")

        assert report.envelope.z_min_ohm == pytest.approx(Z_MIN)
        assert report.envelope.z_max_ohm == pytest.approx(Z_MAX)
        assert _dropped(report, "magnitude_window") >= 1

    def test_the_same_point_survives_under_the_configured_fallback(self):
        """The control. Without it the test above passes on a gate that drops anything."""
        f, Z = self._spectrum_with_a_point_below_the_commissioned_floor()
        report = analyze_spectrum(
            as_eis_result(f, Z), cell=CELL, settings=_gated(),
            envelope=instrument_envelope(), role="sample")

        assert report.envelope.z_min_ohm == pytest.approx(10.0)
        assert _dropped(report, "magnitude_window") == 0

    def test_the_commissioned_window_flips_magnitude_window_measured(self):
        f, Z = reference_spectrum()
        report = analyze_spectrum(
            as_eis_result(f, Z), cell=CELL, settings=_gated(),
            calibration=_commissioned(), role="sample")

        assert report.envelope.magnitude_window_measured is True
        assert "not yet verified" not in report.envelope.describe()

    def test_an_uncommissioned_window_leaves_the_flag_where_it_was(self):
        f, Z = reference_spectrum()
        report = analyze_spectrum(
            as_eis_result(f, Z), cell=CELL, settings=_gated(),
            calibration=CalibrationSet(fixture_id="mux16", R_short_ohm={1: 5.4}),
            role="sample")

        assert report.envelope.magnitude_window_measured is False
        assert report.envelope.z_min_ohm == pytest.approx(10.0)


#: ``measurement_id = 3490`` — ``mux16.toml``'s own ``sources.blank_short``, ch25,
#: 2026-08-11. **Magnitudes only**, because ``gate_magnitude`` judges nothing else, and
#: reproduced here rather than loaded: the file lives under the DataStore, not under the
#: repository, so a test that read it would pass on this rig and error on a checkout.
#:
#: The shape is the whole point. Thirty points sit on the 7.7 Ω short itself and are
#: **already below the configured 10 Ω floor**; the ten that clear it are low-frequency
#: noise excursions at the bottom of the sweep. So the margin the carve-out protects is
#: 10 points against ``min_fit_pts = 8`` — two to spare — and under the commissioned
#: 795.6 Ω floor only two survive.
BLANK_SHORT_3490_MAG = (
    7.81391, 7.76849, 7.67505, 7.72728, 7.67773, 7.65523, 7.67964, 7.6772, 7.66102,
    7.62039, 7.68891, 7.7481, 7.72869, 7.78426, 7.76359, 7.72718, 7.73666, 7.76828,
    7.69024, 7.72501, 7.72006, 7.70344, 7.7515, 7.74167, 7.67388, 7.67304, 7.77247,
    7.77284, 7.7254, 7.80343, 174.296, 73.0888, 12.9986, 8.1172, 8.67683, 11.6339,
    435.946, float("nan"), float("nan"), float("nan"), 827.729, 482.15, 29.5626,
    2.02713e+06, 62.2007,
)


class TestTheBlankRoleCarveOut:
    """A blank is not a sample, and the calibration's own source proves it."""

    def _blank_short_3490(self):
        f = log_frequencies(f_lo=1.2, f_hi=2.0e5, npts=len(BLANK_SHORT_3490_MAG))
        return f, np.asarray(BLANK_SHORT_3490_MAG, dtype=complex)

    def _survivors(self, report) -> int:
        return int(report.quality.metrics["n_surviving"])

    def test_a_blank_short_is_not_judged_against_the_sample_magnitude_window(self):
        f, Z = self._blank_short_3490()
        blank = analyze_spectrum(
            as_eis_result(f, Z, channel=25), cell=None, settings=_gated(),
            calibration=_commissioned(), role="blank_short")

        assert blank.envelope.z_min_ohm == pytest.approx(10.0)
        assert blank.envelope.magnitude_window_measured is False
        assert self._survivors(blank) == 10

    def test_the_same_blank_under_the_sample_window_keeps_two_points_of_ten(self):
        """The defect the carve-out exists for, asserted rather than described.

        Without this half the test above would pass against a window that drops
        nothing — ``SUBAGENT_RULES`` §3.1(e).
        """
        f, Z = self._blank_short_3490()
        as_if_sample = analyze_spectrum(
            as_eis_result(f, Z, channel=25), cell=None, settings=_gated(),
            calibration=_commissioned(), role="sample")

        assert as_if_sample.envelope.z_min_ohm == pytest.approx(Z_MIN)
        assert self._survivors(as_if_sample) == 2

    def test_only_the_sample_window_starves_the_short_blank_of_points(self):
        """The regression pin, stated as what the window actually decides.

        ``min_fit_pts`` is 8 against ten surviving points, so this is the exact margin
        the carve-out protects: under its own role 3490 clears the point floor, under
        the sample window it does not and ``min_points`` rejects it outright.

        **It is not thereby a passing spectrum.** Under either role the gated engine
        rejects it — as a blank, on ``kk_truncation``, which is a statement about a
        jumper rather than about a film. That is the right answer for a blank pushed
        through a sample pipeline, and it is why commissioning re-derivation calls
        ``calibration_derive`` directly and runs **no** gate stack at all.
        """
        f, Z = self._blank_short_3490()
        blank = analyze_spectrum(
            as_eis_result(f, Z, channel=25), cell=None, settings=_gated(),
            calibration=_commissioned(), role="blank_short")
        as_if_sample = analyze_spectrum(
            as_eis_result(f, Z, channel=25), cell=None, settings=_gated(),
            calibration=_commissioned(), role="sample")

        def failed(report) -> set[str]:
            return {e["gate"] for e in report.gate_log if not e["passed"]}

        assert self._survivors(blank) >= _gated().gates.min_fit_pts
        assert "min_points" not in failed(blank)
        assert self._survivors(as_if_sample) < _gated().gates.min_fit_pts
        assert "min_points" in failed(as_if_sample)

    def test_the_carve_out_lives_in_envelope_so_no_call_site_can_forget_it(self):
        cal = _commissioned()
        assert cal.envelope(role="sample").magnitude_window_measured is True
        assert cal.envelope(role="blank_short").magnitude_window_measured is False
        # The phase floor is NOT carved out — only the magnitude window is.
        assert (cal.envelope(role="blank_short").phase_noise_deg
                == pytest.approx(cal.envelope(role="sample").phase_noise_deg))


class TestTheEnvelopeSelectsTheLargestEpsilon:
    """Decision 2 / spec §6 — ``argmax(ε)``, not ``argmin(|Z|)``.

    The shipped comment claimed *"the lowest characterised impedance is the
    conservative headline value"*. It is not: on the committed table the lowest-|Z|
    row gives tan ε = 0.0761 while the largest-ε row gives 0.1072, so ``argmin(|Z|)``
    was the *less* conservative of the two — a proxy for conservatism that happens to
    be nearly right on one table, and is not what the comment said.
    """

    def test_the_envelope_takes_the_largest_epsilon_not_the_lowest_impedance(self):
        env = _commissioned().envelope()
        assert env.phase_noise_deg == pytest.approx(6.1200)
        assert env.phase_noise_deg != pytest.approx(4.3513)

    def test_the_floor_and_the_anchor_come_from_the_same_row(self):
        """The property ``argmin(|Z|)`` did not have: it optimised neither field."""
        env = _commissioned().envelope()
        assert env.phase_noise_at_ohm == pytest.approx(1.01178e7)
        assert env.tand_floor == pytest.approx(math.tan(math.radians(6.1200)))

    def test_the_selected_floor_is_the_more_conservative_of_the_two_candidates(self):
        cal = _commissioned()
        selected = cal.envelope().tand_floor
        rejected = math.tan(math.radians(4.3513))
        assert selected > rejected

    def test_a_single_row_table_promotes_that_row_whichever_rule_is_used(self):
        """The pre-existing coverage, restated: it cannot see this decision at all."""
        cal = CalibrationSet(created_at="2026-08-05",
                             phase_acc=PhaseAccuracyTable(z_ohm=(1e6,),
                                                          eps_deg=(0.31,)))
        env = cal.envelope()
        assert env.phase_noise_deg == pytest.approx(0.31)
        assert env.phase_noise_at_ohm == pytest.approx(1e6)

    def test_a_table_of_unusable_epsilons_promotes_nothing(self):
        """NaN ε must not be selected and must not claim ``phase_noise_measured``."""
        cal = CalibrationSet(
            created_at="2026-09-03",
            phase_acc=PhaseAccuracyTable(z_ohm=(1e3, 1e6),
                                         eps_deg=(float("nan"), float("nan"))))
        env = cal.envelope()
        base = instrument_envelope()
        assert env.phase_noise_deg == pytest.approx(base.phase_noise_deg)
        assert env.phase_noise_at_ohm == pytest.approx(base.phase_noise_at_ohm)


class TestTheCommittedCalibrationAsset:
    """Against ``calibration/eis/mux16.toml`` itself, not a fixture of it.

    Framework §8.5 keeps the commissioning data version-controlled beside the code, so
    it is a testable asset. The 2026-09-03 reference-resistor run took the table from
    16 rows to 19 and moved ``argmax(ε)`` off a ``reference_cap`` and onto the 10 MΩ
    reference resistor — a cleaner reference, since a resistor's true phase is exactly
    zero, and a *more* conservative floor.
    """

    def _committed(self) -> CalibrationSet:
        cal = load_calibration("mux16", root=REPO_ROOT / "calibration" / "eis")
        assert cal is not None, "the committed mux16 calibration must travel with the code"
        return cal

    def test_the_committed_table_selects_the_ten_megohm_reference_resistor(self):
        env = self._committed().envelope()
        assert env.phase_noise_at_ohm == pytest.approx(1.01178e7, rel=1e-4)
        assert env.phase_noise_deg == pytest.approx(6.1200, rel=1e-4)
        assert env.phase_noise_load == "capacitive"

    def test_the_committed_anchor_is_not_the_lowest_impedance_row(self):
        cal = self._committed()
        lowest = min(cal.phase_acc.z_ohm)
        assert lowest == pytest.approx(Z_MIN)
        assert cal.envelope().phase_noise_at_ohm != pytest.approx(lowest)

    def test_the_committed_window_is_the_one_the_wiring_publishes(self):
        env = self._committed().envelope()
        assert env.z_min_ohm == pytest.approx(Z_MIN)
        assert env.z_max_ohm == pytest.approx(Z_MAX)
        assert env.magnitude_window_measured is True


class TestTheCommissionedFloorInDecideReportMode:
    """Decision 2's behavioural half — a 41× denominator changes what may be claimed.

    ``decide_report_mode`` divides the spectrum's minimum ``tan δ`` by the envelope's
    floor and calls anything under ``tand_headroom_mult`` (3.0) resolution-limited.
    The configured floor is tan(0.149°) = 0.0026; the commissioned one is
    tan(6.12°) = 0.1072. A spectrum whose minimum loss sits between 3×0.0026 and
    3×0.1072 is a value under one and a bound under the other.
    """

    FREQ = np.logspace(0.0, 5.0, 41)

    def _spectrum(self, tand: float, C: float = 1e-10) -> np.ndarray:
        w = 2.0 * np.pi * self.FREQ
        return 1.0 / (tand * w * C + 1j * w * C)

    def test_a_commissioned_floor_turns_a_marginal_value_into_a_bound(self):
        Z = self._spectrum(0.05)          # 19x the configured floor, 0.47x the commissioned
        base_mode, _, base_headroom = decide_report_mode(
            self.FREQ, Z, envelope=instrument_envelope(), cell=CELL)
        wired_mode, _, wired_headroom = decide_report_mode(
            self.FREQ, Z, envelope=_commissioned().envelope(), cell=CELL)

        assert base_mode == "value"
        assert base_headroom > 3.0
        assert wired_mode in BOUND_MODES
        assert wired_headroom < 3.0

    def test_a_genuinely_lossy_spectrum_is_still_a_value_under_the_commissioned_floor(self):
        """Negative control: a 41× floor must not turn *everything* into a bound."""
        Z = self._spectrum(5.0)
        mode, _, headroom = decide_report_mode(
            self.FREQ, Z, envelope=_commissioned().envelope(), cell=CELL)

        assert mode == "value"
        assert headroom > 3.0

    def test_the_anchor_decides_whether_a_bound_is_qualified(self):
        """``argmax(ε)`` moves the anchor to 1.0118e7 Ω, where this rig's films live.

        Same floor to within 4 %, same spectrum, opposite qualification — the effect
        spec §6 attributes entirely to the anchor rather than to the floor.
        """
        near_anchor = self._spectrum(0.05, C=5.0e-11)  # |Z| median ~1e7 Ω
        z_med = float(np.median(np.abs(near_anchor)))
        assert 1e6 < z_med < 1e8

        at_argmax = _commissioned().envelope()
        at_argmin = _commissioned(
            phase_acc=PhaseAccuracyTable(z_ohm=(Z_MIN,), eps_deg=(4.3513,),
                                         load="capacitive")).envelope()

        assert at_argmax.phase_noise_valid_at(z_med) is True
        assert at_argmin.phase_noise_valid_at(z_med) is False
        assert decide_report_mode(
            self.FREQ, near_anchor, envelope=at_argmax, cell=CELL)[0] == "bound"
        assert decide_report_mode(
            self.FREQ, near_anchor, envelope=at_argmin,
            cell=CELL)[0] == "bound_unqualified"
