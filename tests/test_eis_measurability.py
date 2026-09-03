"""S1/S2/S3 — the measurability scalars, spec ``docs/SubAgent docs/measurability_scalars.md``.

Two of the fixtures here are **real bench spectra**, not synthetics, and deliberately so:
`[p64]` §3 names "a test whose fixture is off the production manifold" as a shape this
project keeps hitting, and synthetic-only fixtures are how the loss-tangent window defect
survived. The two files are RE/CE-tied open PCB blanks taken 65 s apart on the same board
on 2026-08-06; the ch25 one is the source spectrum for the committed ``mux16.toml``
calibration, and both sit under ``derive_open``'s ``blank_flip_frac = 0.30``, so both are
*good* blanks rather than rejects.

They are **read from the durable commissioning directory rather than copied into the test
tree**, because this repo has no data-fixture directory to copy them into and the DataStore
already resolves seven ``measurements`` rows to that same directory. Copying would create a
second, drifting copy of a provenance-bearing artifact — which is the failure ``[a110]`` §2
records afl-session causing once already, by writing a session-scratchpad path into the
store. Skipped, loudly, when the directory is absent.
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis import measurability as M
from softae.analysis.eis.admittance import apparent_capacitance, loss_tangent
from softae.analysis.eis.calibration import PhaseAccuracyTable
from softae.analysis.eis_data import EISResult

# ── Real fixtures ────────────────────────────────────────────────────────────

COMMISSIONING_DIR = Path(
    r"C:\Users\Osuji\Documents\Users\Pavel\EIS_capacitance_commissioning_data")
CH25_BLANK = COMMISSIONING_DIR / "ch25_manual_open_PCB_blank_RECEcoupled.txt"
CH17_BLANK = COMMISSIONING_DIR / "ch17_manual_open_PCB_blank_multichannel_RECEcoupled.txt"

requires_real_blanks = pytest.mark.skipif(
    not (CH25_BLANK.is_file() and CH17_BLANK.is_file()),
    reason=f"commissioning blanks absent from {COMMISSIONING_DIR}",
)

MUX16 = Path(__file__).resolve().parents[1] / "calibration" / "eis" / "mux16.toml"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    result = EISResult.load(path)
    return np.asarray(result.frequency, dtype=float), result.z_complex


@pytest.fixture(scope="module")
def ch25():
    return _load(CH25_BLANK)


@pytest.fixture(scope="module")
def ch17():
    return _load(CH17_BLANK)


@pytest.fixture(scope="module")
def mux16_table() -> PhaseAccuracyTable:
    """The committed 16-point capacitive phase table — real bench data, in-repo."""
    pa = tomllib.loads(MUX16.read_text(encoding="utf-8"))["phase_acc"]
    return PhaseAccuracyTable(
        z_ohm=tuple(pa["z_ohm"]), eps_deg=tuple(pa["eps_deg"]),
        load=pa.get("load", "resistive"),
        valid_decades=float(pa.get("valid_decades", 1.0)),
    )


# ── Synthetic construction ───────────────────────────────────────────────────

F = np.logspace(0.0, 5.0, 41)


def z_from_admittance(f: np.ndarray, G, C) -> np.ndarray:
    """``Z = 1/(G + jωC)`` — builds a spectrum from the quantities S1–S3 measure.

    Constructing in admittance rather than impedance means ``Im(Y)/ω`` is exactly the
    ``C`` handed in, so a test can state the plateau it expects instead of deriving it.
    """
    f = np.asarray(f, dtype=float)
    Y = np.broadcast_to(np.asarray(G, dtype=float), f.shape) + 1j * (
        2.0 * np.pi * f * np.broadcast_to(np.asarray(C, dtype=float), f.shape))
    return 1.0 / Y


def piecewise_C(f: np.ndarray, *, plateau: float, corner_hz: float,
                low_factor: float, exponent: float = 1.0) -> np.ndarray:
    """``plateau`` above *corner_hz*; scaled toward ``low_factor × plateau`` below it."""
    f = np.asarray(f, dtype=float)
    below = f < corner_hz
    ramp = (corner_hz / np.where(below, f, corner_hz)) ** exponent
    span = np.max(ramp[below]) if below.any() else 1.0
    scale = 1.0 + (low_factor - 1.0) * (ramp - 1.0) / max(span - 1.0, 1e-12)
    return plateau * np.where(below, scale, 1.0)


# ── S1: the plateau ──────────────────────────────────────────────────────────

class TestCapacitancePlateau:
    def test_plateau_reports_the_median_of_the_widest_flat_run(self):
        Z = z_from_admittance(F, 1e-9, 1e-10)
        plateau = M.capacitance_plateau(F, Z)
        assert plateau.found
        assert plateau.C_plateau == pytest.approx(1e-10, rel=1e-6)
        assert plateau.decades == pytest.approx(5.0, abs=0.01)
        assert plateau.wide_enough

    def test_plateau_is_measured_on_im_y_over_omega_and_not_on_c_app(self):
        # tan δ held at 5 across the band, so C_app = C·(1 + tan²δ) = 26·C everywhere.
        # Both quantities are flat, so a plateau search cannot tell them apart by shape —
        # only by which one it was handed. Spec §4.2: they differ by 26× at the top of
        # this band, and commissioning uses Im(Y)/ω.
        C = 1e-10
        Z = z_from_admittance(F, 5.0 * 2.0 * np.pi * F * C, C)
        assert loss_tangent(Z) == pytest.approx(np.full(F.size, 5.0), rel=1e-9)

        plateau = M.capacitance_plateau(F, Z)
        assert plateau.C_plateau == pytest.approx(C, rel=1e-6)
        # And the decisive half: C_app really is 26× away, so the assertion above is not
        # satisfied by both definitions at once.
        assert float(np.median(apparent_capacitance(F, Z))) == pytest.approx(
            26.0 * C, rel=1e-6)

    def test_a_spectrum_with_no_flat_run_reports_no_plateau_rather_than_a_number(self):
        # Every three consecutive points span 1.6² = 2.56×, so none is within ±10 % of
        # its own median. There is no denominator for S1 here.
        C = 1e-10 * 1.6 ** np.arange(F.size)
        plateau = M.capacitance_plateau(F, z_from_admittance(F, 1e-9, C))
        assert not plateau.found
        assert math.isnan(plateau.C_plateau)
        assert plateau.decades == 0.0

    def test_a_narrow_plateau_is_found_but_reported_as_not_wide_enough(self):
        # Four flat points inside an otherwise 1.6×-per-point ramp: 0.375 decades, under
        # the 0.5 minimum. Narrow is reported, not enforced — nothing here is a gate.
        C = 1e-10 * 1.6 ** np.arange(F.size)
        C[20:24] = 1e-10
        plateau = M.capacitance_plateau(F, z_from_admittance(F, 1e-9, C))
        assert plateau.found
        assert plateau.decades == pytest.approx(0.375, abs=1e-6)
        assert not plateau.wide_enough


# ── S1: the lift, and its four outcomes ──────────────────────────────────────

class TestConductionLift:
    def test_a_low_frequency_lift_above_the_plateau_is_an_excursion(self):
        C = piecewise_C(F, plateau=1e-10, corner_hz=1e3, low_factor=20.0)
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, C))
        assert lift.outcome == "excursion"
        assert lift.judgeable
        assert lift.lift == pytest.approx(20.0, rel=0.05)

    def test_a_lift_inside_the_plateaus_own_tolerance_is_flat_not_an_excursion(self):
        # The excursion threshold is ``1 + tol_pct/100``, not 1: a point 5 % above a
        # plateau *defined* as flat to +/-10 % has not left it, and calling that an
        # excursion would report conduction from the plateau's own noise. No other test
        # occupies the 1.0 < lift <= 1.1 band -- both real blanks sit at 1.42 and 2.08 --
        # so without this the threshold could be dropped to > 1.0 with the suite green.
        #
        # A 1.05x shelf at the bottom, a deep dip above it to stop the flat run from
        # swallowing the whole band, then the plateau: lift is 1.05 with a genuine
        # below-plateau region to measure it in.
        C = np.full(F.size, 1e-10)
        C[:5] = 1.05e-10
        C[5:10] = 0.4e-10
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, C))
        assert lift.judgeable
        assert lift.lift == pytest.approx(1.05, rel=1e-9)
        assert 1.0 < lift.lift <= 1.0 + M.PLATEAU_TOL_PCT / 100.0
        assert lift.outcome == "flat"

    def test_a_below_plateau_dip_alone_is_flat_because_depth_is_not_the_criterion(self):
        # A negative control for the lift, and the reason depth is carried but not used:
        # this spectrum has the deepest below-plateau excursion in the file (0.9) and no
        # lift at all. The two statistics disagree about it, so a caller that reads
        # `below_plateau_depth` as if it were the criterion gets the opposite answer.
        C = piecewise_C(F, plateau=1e-10, corner_hz=1e3, low_factor=0.1)
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, C))
        assert lift.outcome == "flat"
        assert lift.judgeable
        assert lift.lift < 1.0
        assert lift.below_plateau_depth == pytest.approx(0.9, rel=0.05)

    def test_below_plateau_depth_is_reported_alongside_a_large_lift(self):
        # Depth is a reported metric, not a criterion, so it is carried even when the
        # outcome is decided entirely by the lift.
        C = piecewise_C(F, plateau=1e-10, corner_hz=1e3, low_factor=20.0)
        C = np.where(np.isclose(F, F[F >= 1e3][0]), 5e-11, C)
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, C))
        assert lift.outcome == "excursion"
        assert lift.below_plateau_depth > 0.0

    def test_no_plateau_is_a_distinct_outcome_and_leaves_the_lift_undefined(self):
        C = 1e-10 * 1.6 ** np.arange(F.size)
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, C))
        assert lift.outcome == "no_plateau"
        assert lift.outcome in M.UNJUDGEABLE_OUTCOMES
        assert not lift.judgeable
        assert math.isnan(lift.lift)

    def test_a_plateau_reaching_the_bottom_of_the_sweep_is_not_reported_as_flat(self):
        # A perfectly flat spectrum has a plateau spanning the whole band, so there is no
        # below-plateau region to look in. Calling that "flat" would report an absence
        # that was never looked for — the value-versus-bound confusion, one level down.
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, 1e-10))
        assert lift.outcome == "no_low_band"
        assert not lift.judgeable
        assert math.isnan(lift.lift)
        assert lift.plateau.found

    def test_the_four_outcomes_are_all_from_the_declared_vocabulary(self):
        assert set(M.UNJUDGEABLE_OUTCOMES) <= set(M.OUTCOMES)
        assert len(M.OUTCOMES) == 4

    def test_the_outcome_comes_first_so_a_positional_float_read_fails_loudly(self):
        # Structural, not documentary: a caller who unpacks this as the spec's bare
        # statistic gets a TypeError on the first arithmetic rather than a verdict.
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9, 1e-10))
        assert isinstance(lift[0], str)
        with pytest.raises(TypeError):
            lift[0] * 2.0

    def test_an_unjudgeable_lift_does_not_read_as_a_pass_against_a_threshold(self):
        lift = M.conduction_lift(F, z_from_admittance(F, 1e-9,
                                                     1e-10 * 1.6 ** np.arange(F.size)))
        assert not (lift.lift > 1.0)
        assert not (lift.lift < 1.0)


# ── S2 ───────────────────────────────────────────────────────────────────────

class TestTandMargin:
    @staticmethod
    def _table() -> PhaseAccuracyTable:
        """ε rising steeply with |Z|, so per-point and lowest-|Z| answers cannot coincide."""
        return PhaseAccuracyTable(z_ohm=(1e3, 1e5, 1e7), eps_deg=(0.1, 1.0, 4.0),
                                  load="capacitive", valid_decades=1.0)

    def test_the_denominator_is_the_table_value_at_that_points_own_impedance(self):
        table = self._table()
        # One deliberately low-loss point at a high |Z|, so argmin(tan δ) is known.
        tand = np.full(F.size, 0.5)
        tand[10] = 0.05
        C = 1e-10
        Z = z_from_admittance(F, tand * 2.0 * np.pi * F * C, C)

        margin = M.tand_margin(F, Z, table)
        assert margin.f_at_min == pytest.approx(F[10])
        assert margin.eps_deg == pytest.approx(table.epsilon_deg(margin.z_at_min))
        # CalibrationSet.envelope() would have collapsed the table to its lowest-|Z|
        # entry. That is a different number, so this test distinguishes the two routes.
        assert margin.eps_deg != pytest.approx(table.eps_deg[0])
        assert margin.margin == pytest.approx(
            0.05 / math.tan(math.radians(margin.eps_deg)), rel=1e-9)

    def test_the_numerator_is_the_minimum_loss_tangent_not_the_median(self):
        table = self._table()
        tand = np.full(F.size, 0.5)
        tand[10] = 0.05
        C = 1e-10
        Z = z_from_admittance(F, tand * 2.0 * np.pi * F * C, C)

        margin = M.tand_margin(F, Z, table)
        median_numerator = float(np.median(tand))
        assert margin.margin == pytest.approx(
            0.05 / math.tan(math.radians(margin.eps_deg)), rel=1e-9)
        assert margin.margin < median_numerator / math.tan(
            math.radians(margin.eps_deg))

    def test_an_uncharacterised_impedance_is_provisional_and_never_reads_as_a_pass(self):
        # The table covers 1 kΩ ±1 decade only; the minimum-tan δ point sits far above it.
        table = PhaseAccuracyTable(z_ohm=(1e3,), eps_deg=(0.5,), load="capacitive",
                                   valid_decades=1.0)
        tand = np.full(F.size, 0.5)
        tand[10] = 0.05
        C = 1e-10
        Z = z_from_admittance(F, tand * 2.0 * np.pi * F * C, C)

        margin = M.tand_margin(F, Z, table)
        assert math.isnan(margin.eps_deg)
        assert not margin.characterised
        assert margin.provisional
        assert math.isnan(margin.margin)
        assert not (margin.margin > 3.0)
        # The point is still reported, so a reviewer can see where the table ran out.
        assert margin.f_at_min == pytest.approx(F[10])

    def test_an_ample_loss_tangent_is_not_flagged(self):
        # Negative control for S2: a spectrum with real loss clears the floor comfortably.
        table = self._table()
        C = 1e-10
        Z = z_from_admittance(F, 5.0 * 2.0 * np.pi * F * C, C)
        margin = M.tand_margin(F, Z, table)
        assert margin.characterised
        assert margin.margin > 3.0

    def test_a_point_the_quadrant_gate_would_drop_can_still_supply_the_minimum(self):
        """Spec §6: the minimum is taken over the band *including* what ``gate_quadrant``
        drops, because masking those points first is half the shipped defect.

        The two sets are not the same set. ``tan δ = Z'/(−Z'')`` is positive whenever the
        two agree in sign, so a point at ``Re Z < 0, Im Z > 0`` has a positive ``tan δ``
        and is admitted here — while the gate, which tests ``Re Z > 0``, removes it. The
        gate is called below rather than described, so this asserts the collision instead
        of assuming it.
        """
        from softae.analysis.eis.gates import gate_quadrant

        C = 1e-10
        Z = z_from_admittance(F, 0.5 * 2.0 * np.pi * F * C, C)
        Z[10] = -1.0e5 + 1.0e6j                       # tan δ = 0.1, the band minimum

        assert not bool(gate_quadrant(F, Z, {}).mask[10])
        margin = M.tand_margin(F, Z, self._table())
        assert margin.f_at_min == pytest.approx(F[10])
        assert margin.characterised
        assert margin.margin == pytest.approx(
            0.1 / math.tan(math.radians(margin.eps_deg)), rel=1e-9)

    def test_a_spectrum_with_no_positive_loss_tangent_returns_nothing_measurable(self):
        Z = z_from_admittance(F, -1e-9, 1e-10)
        margin = M.tand_margin(F, Z, self._table())
        assert margin.provisional
        assert math.isnan(margin.f_at_min)


class TestEpsilonClamping:
    def test_a_value_inside_the_tabulated_range_is_interpolated_not_clamped(self):
        table = TestTandMargin._table()
        assert not M.eps_is_clamped(table, 1e5)

    def test_a_covered_value_past_the_last_point_is_reported_as_clamped(self):
        # np.interp clamps to the endpoint rather than extrapolating, so "no NaN" is not
        # "characterised here". Without this distinction the two are indistinguishable.
        table = TestTandMargin._table()
        assert table.covers(5e7)
        assert table.epsilon_deg(5e7) == pytest.approx(table.eps_deg[-1])
        assert M.eps_is_clamped(table, 5e7)

    def test_an_uncovered_value_is_nan_and_therefore_not_clamped(self):
        table = TestTandMargin._table()
        assert math.isnan(table.epsilon_deg(1e9))
        assert not M.eps_is_clamped(table, 1e9)


# ── S3 ───────────────────────────────────────────────────────────────────────

class TestNegativeConductance:
    def test_a_wholly_passive_spectrum_counts_none(self):
        # Negative control for S3.
        result = M.negative_conductance_count(z_from_admittance(F, 1e-9, 1e-10))
        assert result.n == 0
        assert result.frac == 0.0
        assert result.re_state == "unverified"

    def test_negative_real_admittance_is_counted(self):
        G = np.full(F.size, 1e-9)
        G[:4] = -1e-9
        result = M.negative_conductance_count(z_from_admittance(F, G, 1e-10))
        assert result.n == 4
        assert result.frac == pytest.approx(4.0 / F.size)

    def test_the_re_state_is_echoed_and_open_by_geometry_is_flagged_structural(self):
        Z = z_from_admittance(F, 1e-9, 1e-10)
        assert M.negative_conductance_count(
            Z, re_state="open_by_geometry").expected_by_construction
        assert not M.negative_conductance_count(
            Z, re_state="tied_to_ce").expected_by_construction

    def test_an_unknown_re_state_is_refused_rather_than_echoed(self):
        with pytest.raises(ValueError, match="re_state"):
            M.negative_conductance_count(z_from_admittance(F, 1e-9, 1e-10),
                                         re_state="closed")


# ── The real blanks ──────────────────────────────────────────────────────────

@requires_real_blanks
class TestRealOpenBlanks:
    """Both files are healthy RE/CE-tied open blanks; ch25 sourced ``mux16.toml``."""

    def test_s3_matches_the_counts_measured_on_the_two_real_blanks(self, ch25, ch17):
        # 6/35 and 8/35 — both under derive_open's blank_flip_frac = 0.30, so a healthy
        # open blank on this rig carries 17–23 % quadrant violation as its *normal*
        # condition. An unconditional n == 0 would reject the calibration's own source.
        a = M.negative_conductance_count(ch25[1], re_state="tied_to_ce")
        b = M.negative_conductance_count(ch17[1], re_state="tied_to_ce")
        assert (a.n, b.n) == (6, 8)
        assert a.frac == pytest.approx(6 / 35, rel=1e-6)
        assert b.frac == pytest.approx(8 / 35, rel=1e-6)
        assert not a.expected_by_construction        # tied_to_ce is a *closed* loop

    def test_both_real_blanks_carry_a_measurable_plateau(self, ch25, ch17):
        for (f, Z), expect in ((ch25, 5.0e-11), (ch17, 1.0e-11)):
            plateau = M.capacitance_plateau(f, Z)
            assert plateau.found and plateau.wide_enough
            assert plateau.decades > 2.0
            assert plateau.C_plateau == pytest.approx(expect, rel=0.1)

    def test_the_real_blanks_lift_modestly_which_no_threshold_yet_separates(
            self, ch25, ch17):
        # Recorded rather than judged. Both blanks report "excursion" — a lift is present
        # — at 1.4× and 2.1×, against the ~123× the commissioning figure's conducting
        # trace shows and the ≤1.0× its two non-conducting ones show. So the *outcome
        # label* does not discriminate on real data and must not be armed as if it did;
        # only the magnitude, against a Stage-2a distribution, can (spec §8, Stage 3).
        for (f, Z), expect in ((ch25, 1.42), (ch17, 2.08)):
            lift = M.conduction_lift(f, Z)
            assert lift.outcome == "excursion"
            assert lift.lift == pytest.approx(expect, rel=0.02)
            assert 1.0 < lift.lift < 10.0

    def test_s2_on_a_real_blank_interpolates_inside_the_committed_table(
            self, ch25, mux16_table):
        margin = M.tand_margin(ch25[0], ch25[1], mux16_table)
        assert margin.characterised
        assert not M.eps_is_clamped(mux16_table, margin.z_at_min)
        assert margin.z_at_min == pytest.approx(3.86e7, rel=0.01)
        assert margin.eps_deg == pytest.approx(1.3517, rel=1e-3)
        assert margin.margin == pytest.approx(0.593, rel=0.01)

    def test_s2_on_the_other_real_blank_is_clamped_to_the_tables_last_point(
            self, ch17, mux16_table):
        # ch17's minimum-tan δ point sits at 2.5×10⁸ Ω, above the table's largest entry
        # of 1.4456×10⁸ Ω but inside valid_decades of it — so epsilon_deg returns a
        # finite number that is the *endpoint*, not an interpolation. This is the case a
        # NaN check alone cannot see, on real data rather than a constructed table.
        margin = M.tand_margin(ch17[0], ch17[1], mux16_table)
        assert margin.characterised
        assert M.eps_is_clamped(mux16_table, margin.z_at_min)
        assert margin.z_at_min > max(mux16_table.z_ohm)
        assert margin.eps_deg == pytest.approx(mux16_table.eps_deg[-1])

    def test_s2_selects_its_minimum_over_the_unmasked_band(self, ch25, mux16_table):
        # tand_margin applies no quadrant mask of its own: the point it picks is the
        # global minimum-positive tan δ over the whole 35-point sweep, computed here
        # independently, on a spectrum that has 6 quadrant violations in it.
        f, Z = ch25
        tand = loss_tangent(Z)
        usable = np.isfinite(tand) & (tand > 0)
        expected_f = float(f[usable][int(np.argmin(tand[usable]))])

        assert M.negative_conductance_count(Z).n == 6
        assert M.tand_margin(f, Z, mux16_table).f_at_min == pytest.approx(expected_f)


class TestPlateauPosition:
    """Where ``tan δ`` peaked, and whether the band actually contained it.

    The metric exists because an edge peak and an interior peak were indistinguishable
    downstream, and that ambiguity is the difference between a measured resistance and an
    upper bound on one.
    """

    #: Where the fixture below puts its resistive plateau, to 5 significant figures.
    F_PEAK_HZ = 1584.9

    @staticmethod
    def _cell(freq):
        """A blocking coplanar cell, in physics convention (Im Z < 0).

        ``R_series + Z_block + (R_bulk ∥ C_par)``, all shunted by a fixture stray.
        **The two nuisance elements are what make an interior peak possible at all**, so
        neither may be dropped to simplify the fixture: on a bare ``R_s + (R_b ∥ C)``,
        ``−Z″ → 0`` at DC and ``tan δ`` therefore diverges at the low edge, putting the
        maximum on the first point of *every* sweep. Blocking rolls the low end off and the
        stray rolls the high end off, which is the shape real spectra from this rig have —
        and the shape the metric was written to describe.
        """
        w = 2 * np.pi * np.asarray(freq, dtype=float)
        core = 50.0 + 1 / (1j * w * 1e-7) + 5e4 / (1 + 1j * w * 5e4 * 3e-11)
        return 1 / (1 / core + 1j * w * 2e-11)

    def test_an_interior_peak_reports_its_distance_from_both_edges(self):
        freq = np.logspace(0, 6, 61)
        got = M.plateau_position(freq, self._cell(freq))
        assert got.position == "interior"
        assert got.interior
        assert got.f_hz == pytest.approx(self.F_PEAK_HZ, rel=1e-3)
        # The margin is the distance to the NEARER edge, so it cannot exceed half the
        # swept width, and must be strictly positive for an interior point.
        assert 0.0 < got.decades_to_edge <= 3.0
        assert got.n_usable == freq.size

    def test_a_peak_on_the_lowest_swept_point_is_a_low_edge_with_zero_margin(self):
        # Start the sweep at the plateau, so the maximum lands on the first point.
        freq = np.logspace(math.log10(self.F_PEAK_HZ), 6, 31)
        got = M.plateau_position(freq, self._cell(freq))
        assert got.position == "low_edge"
        assert got.decades_to_edge == 0.0
        assert not got.interior

    def test_a_peak_on_the_highest_swept_point_is_a_high_edge_with_zero_margin(self):
        # Stop the sweep at the plateau: the resistive region continues past the band,
        # so any resistance read here is an upper bound. This is the 08/21 condition.
        freq = np.logspace(0, math.log10(self.F_PEAK_HZ), 31)
        got = M.plateau_position(freq, self._cell(freq))
        assert got.position == "high_edge"
        assert got.decades_to_edge == 0.0
        assert not got.interior

    def test_too_few_usable_points_is_none_rather_than_a_position(self):
        freq = np.array([10.0, 100.0])
        got = M.plateau_position(freq, self._cell(freq))
        assert got.position == "none"
        assert got.n_usable < M.MIN_PLATEAU_POINTS
        assert math.isnan(got.f_hz)

    def test_an_all_inductive_spectrum_yields_no_position_because_tand_is_negative(self):
        # Im Z > 0 with Re Z > 0 gives tan δ < 0, which cannot enter the search —
        # the same admission rule tand_margin uses, not a restated mask.
        freq = np.logspace(1, 5, 41)
        Z = 50.0 + 1j * 2 * np.pi * freq * 1e-3
        assert (loss_tangent(Z) < 0).all()
        assert M.plateau_position(freq, Z).position == "none"

    def test_the_peak_is_the_maximum_of_tand_and_the_minimum_of_absolute_phase(self):
        """Two spellings of one quantity; a future edit must not let them diverge."""
        freq = np.logspace(0, 6, 61)
        Z = self._cell(freq)
        got = M.plateau_position(freq, Z)
        phase = np.degrees(np.angle(Z))
        assert freq[int(np.argmin(np.abs(phase)))] == pytest.approx(got.f_hz)

    def test_frequency_order_does_not_change_the_verdict(self):
        """Sweeps arrive descending as often as ascending; the edges must not swap."""
        freq = np.logspace(0, 6, 61)
        Z = self._cell(freq)
        asc = M.plateau_position(freq, Z)
        desc = M.plateau_position(freq[::-1], Z[::-1])
        assert asc.position == desc.position
        assert asc.f_hz == pytest.approx(desc.f_hz)
        assert asc.decades_to_edge == pytest.approx(desc.decades_to_edge)

    def test_it_is_a_metric_and_carries_no_verdict_machinery(self):
        """measurability.py's standing rule: nothing here gates. Held as a test."""
        freq = np.logspace(0, 6, 61)
        got = M.plateau_position(freq, self._cell(freq))
        assert not hasattr(got, "severity")
        assert not hasattr(got, "passed")
        assert not hasattr(got, "checked")
        assert got.position in M.PLATEAU_POSITIONS
