"""The phase floor by load class and locality (T11.41).

One number, ``tan(6.12°) = 0.1072``, used to be the floor for every film at every
impedance, quotable only within one decade of ``1.0118×10⁷ Ω``. It came from the 10 MΩ
**reference resistor**; films sit near ``4×10⁴ Ω``, 2.37 decades out of band, and lost
their σ to ``bound_unqualified``.

Three rules are pinned here, and they are separable:

**Resistor rows set the floor.** A resistor's true phase is exactly zero, so what the
instrument reports on one is its own error. The reference capacitors were measured on an
auxiliary PCB with their own loss unknown at the 0.06° level, so they are a consistency
*check* and never a floor.

**The band is a bracketing, and the floor is the max of the bracketing pair.** No
interpolant: the live 22-row table's own interpolant crosses 4.35° → 0.18° between
adjacent rows, because those rows are different load classes.

**Permissive only.** The largest resistor ε is the 6.120° the single anchor already
used, so no floor anywhere rises, and the band's upper edge is bit-identical to today's.

The tables below are the live ``calibration/eis/mux16.toml`` rows, carrying the per-row
provenance since the 2026-09-20 re-derive (T11.41 §4) — recomputed read-only from the
stored spectra, exact float match to the asset.
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
    save_calibration,
)
from softae.analysis.eis.envelope import (
    PHASE_CLASS_CONSISTENCY_BUDGET_DEG,
    InstrumentEnvelope,
)
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.report import decide_report_mode
from softae.workflows.commissioning import PHASE_REFERENCE_MAX_EPS_DEG

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The committed 22 rows, ``(z_ohm, eps_deg, load_kind, source_measurement_id)``.
#:
#: Six resistive rows from ``reference_r`` ids 4291/4292/3898/4293/3896/3897 and sixteen
#: capacitive rows from ``reference_cap`` ids 3491/3492/3493. Id 1933 — marked 100 pF,
#: measuring 26.22 nF at ε 29-59° — contributed three rows to neither, because
#: ``phase_reference_is_plausible`` refuses it.
COMMITTED_ROWS: tuple[tuple[float, float, str, int], ...] = (
    (109.60589540190168, 0.2691305583753748, "resistive", 4291),
    (795.6011977999614, 4.351260106985991, "capacitive", 3493),
    (1001.4795094188356, 0.17660265143054396, "resistive", 4292),
    (2357.7528249776733, 1.9624261823817026, "capacitive", 3493),
    (7273.299867405578, 3.852788140514749, "capacitive", 3491),
    (7324.007438752161, 3.8528376548854966, "capacitive", 3492),
    (9917.150087272426, 0.21056721494726474, "resistive", 3898),
    (32420.280469008576, 2.2005991798567814, "capacitive", 3491),
    (32540.01864668814, 2.2103903134995866, "capacitive", 3492),
    (100088.6159931681, 0.4819655300565062, "resistive", 4293),
    (216762.2828598172, 0.602255078866865, "resistive", 3896),
    (321252.56318881316, 1.0265734366100092, "capacitive", 3491),
    (323275.56198593316, 0.9072931069176611, "capacitive", 3492),
    (539010.134674622, 2.6698254086206963, "capacitive", 3493),
    (2124712.820108402, 1.7104683730714947, "capacitive", 3492),
    (2341613.9551686863, 2.6309688826980544, "capacitive", 3493),
    (2443511.3296109233, 1.1194038476601391, "capacitive", 3491),
    (10117806.43984896, 6.119995136395174, "resistive", 3897),
    (30152649.19805681, 4.535612736754088, "capacitive", 3492),
    (36637066.315888025, 1.3443667160220403, "capacitive", 3491),
    (65108671.60990564, 1.4247068258005915, "capacitive", 3493),
    (144560987.92693862, 3.6307552193550388, "capacitive", 3493),
)

#: Where the films sit, and the floor the resistor ladder puts under them.
FILM_Z_OHM = 4.3e4
FILM_EPS_DEG = 0.4819655300565062
FILM_TAND = 0.008412083794041224

CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")


def _table(rows=COMMITTED_ROWS, **over) -> PhaseAccuracyTable:
    kw = dict(
        z_ohm=tuple(r[0] for r in rows),
        eps_deg=tuple(r[1] for r in rows),
        load="capacitive",
        load_kind=tuple(r[2] for r in rows),
        source_measurement_id=tuple(r[3] for r in rows),
    )
    kw.update(over)
    return PhaseAccuracyTable(**kw)


def _envelope(rows=COMMITTED_ROWS, **over) -> InstrumentEnvelope:
    """An envelope carrying *rows*, built the way ``CalibrationSet.envelope()`` does."""
    return CalibrationSet(
        fixture_id="mux16", created_at="2026-09-17",
        phase_acc=_table(rows, **over),
        z_min_ohm=min(r[0] for r in rows), z_max_ohm=max(r[0] for r in rows),
    ).envelope()


def _spectrum_at(z_med_ohm: float, tand: float, n: int = 41) -> tuple:
    """A constant-``tan δ`` sweep whose median ``|Z|`` is exactly *z_med_ohm*.

    ``tan δ = Re Z / |Im Z|`` is invariant under a real rescaling, so setting the median
    magnitude does not touch the numerator the decision divides.
    """
    freq = np.logspace(0.0, 5.0, n)
    w = 2.0 * np.pi * freq
    Z = 1.0 / (tand * w * 1e-10 + 1j * w * 1e-10)
    return freq, Z * (z_med_ohm / float(np.median(np.abs(Z))))


def _events(monkeypatch) -> list[tuple[str, dict]]:
    """Capture ``report``'s info events, still passing them through."""
    import softae.analysis.eis.report as mod

    seen: list[tuple[str, dict]] = []
    real = mod.logger.info

    def spy(event, **kw):
        seen.append((event, kw))
        return real(event, **kw)

    monkeypatch.setattr(mod.logger, "info", spy)
    return seen


class TestSerialisationCarriesTheProvenance:
    """The columns survive the TOML round trip, or the re-derive buys nothing."""

    def test_phase_table_round_trips_load_kind_and_source_ids(self, tmp_path):
        rows = COMMITTED_ROWS + ((5.0e8, float("nan"), "capacitive", -1),)
        cal = CalibrationSet(fixture_id="rt", hardware_hash="h", phase_acc=_table(rows))

        save_calibration(cal, root=tmp_path)
        back = load_calibration("rt", root=tmp_path)

        assert back is not None
        assert back.phase_acc.load_kind == tuple(r[2] for r in rows)
        assert back.phase_acc.source_measurement_id == tuple(r[3] for r in rows)
        assert back.phase_acc.has_load_kinds is True
        # The NaN eps and the -1 id are the two shapes a flat list could quietly drop.
        assert math.isnan(back.phase_acc.eps_deg[-1])
        assert back.phase_acc.source_measurement_id[-1] == -1
        z, eps, kind, mid = back.phase_acc.rows()[-1]
        assert (z, kind, mid) == (5.0e8, "capacitive", -1)
        assert math.isnan(eps)

    def test_an_asset_without_the_columns_loads_exactly_as_before(self, tmp_path):
        """The seam with the asset: new code + old asset is inert, never wrong."""
        plain = PhaseAccuracyTable(
            z_ohm=tuple(r[0] for r in COMMITTED_ROWS),
            eps_deg=tuple(r[1] for r in COMMITTED_ROWS),
            load="capacitive",
        )
        save_calibration(
            CalibrationSet(fixture_id="old", phase_acc=plain), root=tmp_path)
        back = load_calibration("old", root=tmp_path)

        assert back is not None
        assert back.phase_acc.has_load_kinds is False
        assert back.phase_acc.rows() == ()
        assert back.envelope().phase_rows == ()


class TestTheFallbackWhenNoRowCarriesItsClass:
    """One path serves both regimes, so the fallback is reachable from a test.

    ``tests/test_eis_envelope.py``'s ``TestTheCommissionedFloorInDecideReportMode`` is
    the *other half* of this and is deliberately left alone: its ``_commissioned()``
    two-row synthetic table carries no ``load_kind``, so its three tests exercise the
    single-anchor path unchanged and stand as the behavioural fallback regression. The
    headline floor there is still ``tan(6.12°)``; what this class adds is the proof that
    the same *call* produces it.
    """

    def test_a_table_without_load_kinds_falls_back_to_the_single_anchor(self):
        env = _envelope(load_kind=(), source_measurement_id=())

        assert env.phase_rows == ()
        for z in (1.0e3, 1.0118e7, 1.0e9):     # out of band, in band, far out
            at = env.floor_at(z)
            assert at.eps_deg == env.phase_noise_deg
            assert at.in_band is env.phase_noise_valid_at(z)
            assert at.rows_used == ()
            assert at.z_anchor_ohm == env.phase_noise_at_ohm

    def test_a_ragged_load_kind_is_refused_and_says_so(self, monkeypatch):
        """Six kinds against 22 rows must not read as "the other 16 are capacitive"."""
        import softae.analysis.eis.calibration as mod

        warned: list[str] = []
        real = mod.logger.warning
        monkeypatch.setattr(
            mod.logger, "warning",
            lambda ev, **kw: (warned.append(ev), real(ev, **kw))[1])

        env = _envelope(load_kind=("resistive",) * 6,
                        source_measurement_id=(4291,) * 6)

        assert env.phase_rows == ()
        assert "phase_table_load_kind_ragged" in warned
        at = env.floor_at(FILM_Z_OHM)
        assert at.eps_deg == env.phase_noise_deg
        assert at.rows_used == ()

    def test_a_two_row_table_without_kinds_decides_exactly_as_it_did_before(self):
        """The fallback regression, at the decision rather than at the lookup."""
        legacy = CalibrationSet(
            fixture_id="mux16",
            phase_acc=PhaseAccuracyTable(z_ohm=(795.6011977999614, 1.0117806e7),
                                         eps_deg=(4.3513, 6.1200),
                                         load="capacitive"),
        ).envelope()
        freq, Z = _spectrum_at(1.0e7, 0.05)

        decision = decide_report_mode(freq, Z, envelope=legacy, cell=CELL)

        assert legacy.phase_rows == ()
        assert decision.floor_rows_used == 0          # 0 names the legacy anchor
        assert decision.floor_tand == pytest.approx(math.tan(math.radians(6.1200)))
        assert decision.floor_z_anchor_ohm == pytest.approx(1.0117806e7)
        assert decision.floor_class_provisional is False
        assert math.isnan(decision.floor_class_gap_deg)
        assert decision.mode == "bound"

    def test_a_legacy_envelope_reports_no_class_comparison_rather_than_agreement(self):
        """NaN, not 0.0: nothing was compared, and that is not "they agree"."""
        provisional, gap, cap_eps, cap_rows = InstrumentEnvelope().class_consistency(
            FILM_Z_OHM)

        assert provisional is False
        assert math.isnan(gap)
        assert math.isnan(cap_eps)
        assert cap_rows == ()


class TestTheFloorBracketsWithinOneLoadClass:
    """Ruling 1 and ruling 3: resistor rows only, and the max of the bracketing pair."""

    def test_floor_at_brackets_and_takes_the_pair_maximum(self):
        at = _envelope().floor_at(FILM_Z_OHM)

        # The UPPER bracket wins, which excludes both a nearest-row rule and a
        # lower-bracket rule: 9917.15 Ω / 0.2106° is nearer in log |Z| and lower.
        assert at.eps_deg == pytest.approx(FILM_EPS_DEG, rel=1e-12)
        assert at.rows_used == (3898, 4293)
        assert at.z_anchor_ohm == pytest.approx(100088.6159931681, rel=1e-12)
        assert at.in_band is True
        assert _envelope().tand_floor_at(FILM_Z_OHM) == pytest.approx(
            FILM_TAND, rel=1e-12)

    def test_a_capacitor_row_never_sets_the_floor(self):
        intruder = COMMITTED_ROWS + ((FILM_Z_OHM, 3.852788140514749, "capacitive", 3491),)
        env = _envelope(intruder)

        assert env.floor_at(FILM_Z_OHM).eps_deg == pytest.approx(
            FILM_EPS_DEG, rel=1e-12)
        # POSITIVE CONTROL: the same row IS returned when asked for by its own class,
        # so the filter is running rather than the row simply unreachable.
        assert env.floor_at(FILM_Z_OHM, load_kind="capacitive").eps_deg == (
            pytest.approx(3.852788140514749, rel=1e-12))

    def test_floor_at_extends_one_decade_past_the_outermost_row(self):
        env = _envelope()

        assert env.floor_at(20.0).in_band is True
        assert env.floor_at(20.0).rows_used == (4291,)
        assert env.floor_at(5.0).in_band is False
        assert math.isnan(env.floor_at(5.0).eps_deg)

        assert env.floor_at(9e7).in_band is True
        assert env.floor_at(9e7).rows_used == (3897,)
        # Capacitor rows reach 1.4456e8 Ω and do NOT extend the resistive band.
        assert env.floor_at(2e8).in_band is False
        assert math.isnan(env.floor_at(2e8).eps_deg)

    def test_a_spectrum_beyond_the_ladder_is_reported_as_bound_unqualified(self):
        freq, Z = _spectrum_at(3e8, 0.05)
        decision = decide_report_mode(freq, Z, envelope=_envelope(), cell=CELL)

        assert decision.mode == "bound_unqualified"
        assert decision.provisional is True
        assert math.isnan(decision.headroom)

    def test_the_upper_decades_are_untouched_by_the_new_rule(self):
        """Permissive only: at 10⁷ Ω the floor and the row are today's, exactly."""
        at = _envelope().floor_at(1.0e7)

        assert at.z_anchor_ohm == pytest.approx(10117806.43984896, rel=1e-12)
        assert _envelope().tand_floor_at(1.0e7) == pytest.approx(
            0.10722215041207023, rel=1e-12)
        assert max(r[1] for r in COMMITTED_ROWS if r[2] == "resistive") == (
            pytest.approx(6.119995136395174, rel=1e-12))


class TestTheCapacitiveCheckFlagsWithoutDeciding:
    """Ruling 2: a flag on the verdict, never a floor and never a refusal."""

    def test_the_capacitive_check_flags_provisional_without_changing_the_floor(
            self, monkeypatch):
        seen = _events(monkeypatch)
        freq, Z = _spectrum_at(FILM_Z_OHM, 0.05)      # 5.9x the local floor -> a value
        resistors_only = _envelope(
            tuple(r for r in COMMITTED_ROWS if r[2] == "resistive"))

        decision = decide_report_mode(freq, Z, envelope=_envelope(), cell=CELL)
        control = decide_report_mode(freq, Z, envelope=resistors_only, cell=CELL)

        assert decision.floor_class_provisional is True
        assert decision.floor_class_gap_deg == pytest.approx(1.7284247834430804,
                                                             rel=1e-9)
        assert decision.floor_tand == pytest.approx(FILM_TAND, rel=1e-12)
        assert decision.floor_rows_used == 2
        assert decision.mode == control.mode == "value"
        assert decision.headroom == pytest.approx(control.headroom, rel=1e-12)

        events = {ev: kw for ev, kw in seen}
        assert "eis_phase_floor_class_disagreement" in events
        assert tuple(events["eis_phase_floor_class_disagreement"]
                     ["capacitive_rows"]) == (3492, 3491)

    @pytest.mark.parametrize("z_med", [1.0e6, 1.0e7])
    def test_the_capacitive_check_is_silent_where_the_resistor_floor_is_larger(
            self, monkeypatch, z_med):
        """Negative control: a flag that fired everywhere would be an annotation."""
        seen = _events(monkeypatch)
        freq, Z = _spectrum_at(z_med, 0.05)

        decision = decide_report_mode(freq, Z, envelope=_envelope(), cell=CELL)

        assert decision.floor_class_provisional is False
        assert decision.floor_class_gap_deg < 0.0
        assert "eis_phase_floor_class_disagreement" not in {ev for ev, _ in seen}

    def test_class_consistency_without_resistive_coverage_reports_the_capacitive_side(
            self):
        """One side missing is *no comparison*, and the gap must say so with NaN.

        A gap of 0.0 would read as "the two classes agree here" — which is the
        ``SUBAGENT_RULES`` §3.1(a) shape exactly, since nothing was compared. The
        capacitive numbers are still reported, because they were genuinely measured and
        discarding them would lose provenance the operator can use.
        """
        # One resistive row, 2.37 decades above the probe, so the floor is out of band;
        # the capacitive rows still bracket it at 32540.0 Ω / 321252.6 Ω.
        rows = tuple(r for r in COMMITTED_ROWS if r[2] == "capacitive") + (
            (10117806.43984896, 6.119995136395174, "resistive", 3897),)
        env = _envelope(rows)

        assert math.isnan(env.floor_at(FILM_Z_OHM).eps_deg)        # arms the branch
        provisional, gap, cap_eps, cap_rows = env.class_consistency(FILM_Z_OHM)

        assert provisional is False
        assert math.isnan(gap)
        assert cap_eps == pytest.approx(2.2103903134995866, rel=1e-12)
        assert cap_rows == (3492, 3491)

    def test_class_consistency_without_capacitive_coverage_reports_no_comparison(self):
        """The mirror: nothing to compare against, and nothing invented to fill it."""
        # Resistive rows bracket the probe; the only capacitive row is 3.5 decades away.
        rows = tuple(r for r in COMMITTED_ROWS if r[2] == "resistive") + (
            (144560987.92693862, 3.6307552193550388, "capacitive", 3493),)
        env = _envelope(rows)

        assert env.floor_at(FILM_Z_OHM).eps_deg == pytest.approx(FILM_EPS_DEG,
                                                                 rel=1e-12)
        provisional, gap, cap_eps, cap_rows = env.class_consistency(FILM_Z_OHM)

        assert provisional is False
        assert math.isnan(gap)
        assert math.isnan(cap_eps)
        assert cap_rows == ()

    def test_the_budget_is_the_rated_loss_not_the_plausibility_limit(self):
        """Pins §3.3 against silent reuse of the 15° plausibility constant."""
        env = _envelope()
        probes = (1.0e3, 1.0e4, FILM_Z_OHM, 1.0e5, 1.0e6, 1.0e7)

        # The plausibility limit can never fire: the largest capacitive eps on this rig
        # is 4.536°, so a 15° gap is unreachable -- a check that cannot fail.
        assert PHASE_REFERENCE_MAX_EPS_DEG == 15.0
        assert max(r[1] for r in COMMITTED_ROWS) < PHASE_REFERENCE_MAX_EPS_DEG
        assert not any(
            env.class_consistency(z, budget_deg=PHASE_REFERENCE_MAX_EPS_DEG)[0]
            for z in probes)

        # The rated loss discriminates over |Z|: it fires below ~3e5 and not above.
        assert PHASE_CLASS_CONSISTENCY_BUDGET_DEG == 0.06
        fired = [z for z in probes if env.class_consistency(z)[0]]
        assert fired == [1.0e3, 1.0e4, FILM_Z_OHM, 1.0e5]


class TestTheCommittedAssetOnceItIsReDerived:
    """Against ``calibration/eis/mux16.toml`` itself, since the 2026-09-20 re-derive."""

    def _committed(self) -> CalibrationSet:
        cal = load_calibration("mux16", root=REPO_ROOT / "calibration" / "eis")
        assert cal is not None, "the committed mux16 calibration must travel with the code"
        return cal

    def test_the_real_film_is_qualified_by_the_resistor_ladder(self):
        env = self._committed().envelope()

        at = env.floor_at(FILM_Z_OHM)
        assert at.in_band is True
        assert env.tand_floor_at(FILM_Z_OHM) == pytest.approx(FILM_TAND, rel=1e-12)
        # The same impedance under the single-anchor rule is 2.372 decades out of band.
        assert env.phase_noise_valid_at(FILM_Z_OHM) is False
        # ... and the upper decades do not move: same number, same row, as today.
        assert env.tand_floor_at(1.0e7) == pytest.approx(0.10722215041207023, rel=1e-12)
        assert env.floor_at(1.0e7).rows_used[-1] == 3897

    def test_the_committed_rows_carry_the_provenance_the_derive_knew(self):
        rows = self._committed().phase_acc.rows()

        assert len(rows) == 22
        resistive = {mid for _z, _e, kind, mid in rows if kind == "resistive"}
        capacitive = {mid for _z, _e, kind, mid in rows if kind == "capacitive"}
        assert resistive <= {4291, 4292, 3898, 4293, 3896, 3897}
        assert capacitive <= {3491, 3492, 3493}
        # id 1933 measured 26.22 nF against a 100 pF marking at eps 29-59 deg;
        # `phase_reference_is_plausible` refused it, so it is in neither set.
        assert 1933 not in resistive | capacitive

    def test_the_committed_table_publishes_the_headline_rows_own_load_kind(self):
        """The live asset mislabels the 6.120° anchor: it was a 10 MΩ resistor."""
        env = self._committed().envelope()

        assert env.phase_noise_deg == pytest.approx(6.1200, rel=1e-4)
        assert env.phase_noise_load == "resistive"
