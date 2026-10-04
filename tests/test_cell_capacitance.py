"""C_cell estimator, insulating-film selector and group aggregation (slice 2, lane L2).

Spec: ``docs/SubAgent docs/regime_b_slice2_spec.md`` §2.2. Synthetic spectra are built on the
rig grid; real ones come from ``tests/data/eis_regime/`` (gitignored — those tests skip).

Each selector test makes **one rule the only guard**: the spectrum passes every other rule,
so deleting that rule must turn the test red (the mutation audit in the lane report).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from softae.analysis.eis import cell_capacitance as cc
from tests import eis_regime_synthetic as syn
from tests.eis_regime_golden import DATA

F = syn.RIG_F
W = 2 * np.pi * F


def _rc(R: float, C: float) -> np.ndarray:
    return R / (1 + 1j * W * R * C)


def _insulator(C: float = 2.5e-10, Rx: float = 1e5, seed: int = 1) -> np.ndarray:
    """Series element R_x‖C_x, then C_cell with a 1e11 Ω leak: no film arc in band."""
    return syn._noisy(_rc(Rx, 3e-12) + _rc(1e11, C), seed)


def _film(Rf: float = 1e8, seed: int = 2) -> np.ndarray:
    """A film arc R_f‖C (apex 6.4 Hz) closing below the band, then the electrode."""
    return syn._noisy(_rc(1e5, 3e-12) + _rc(Rf, 2.5e-10) + 1 / (1e-7 * (1j * W) ** 0.8), seed)


def _assess(Z, occupancy=cc.OCCUPIED, channel=3, mid=None):
    return cc.assess_spectrum(F, Z, channel=channel, occupancy=occupancy, measurement_id=mid)


def _real(name: str):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"real fixture absent: {path} (tests/data/ is gitignored)")
    from softae.analysis.eis_data import EISResult

    e = EISResult.load(path)
    return (np.asarray(e.frequency, float),
            np.asarray(e.z_real, float) - 1j * np.asarray(e.z_imag_neg, float))


# ── Estimator ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("C", [1e-10, 2.5e-10, 5e-10])
@pytest.mark.parametrize("Rx", [3e4, 1e5, 3e5])
def test_c_cell_estimator_insulator_recovers_c_within_1pct_any_series_element(C, Rx):
    assert _assess(_insulator(C, Rx)).c_cell_F == pytest.approx(C, rel=0.01)


def test_apparent_capacitance_band_only_100_to_1000_hz():
    fb, cb = cc.apparent_capacitance(F, _insulator())
    assert fb.min() >= 100 and fb.max() <= 1000 and fb.size == 10
    assert np.all(np.diff(fb) > 0) and np.all(cb > 0)


def test_foot_tand_valley_at_lowest_point_returns_zero():
    f = np.array([1.0, 10.0, 100.0])
    Z = np.array([1 - 100j, 1 - 1j, 1 - 0.5j])   # tan δ falls monotonically to LF
    assert cc.foot_tand(f, Z) == 0.0


# ── Selector: each rule as the only guard ────────────────────────────────────

def test_check_insulator_clean_insulator_qualifies():
    k = _assess(_insulator())
    assert k.qualifies and k.reasons == ()


def test_check_insulator_film_arc_below_band_refused_film_arc_present():
    k = _assess(_film())
    assert k.reasons == ("film_arc_present",) and not k.qualifies


@pytest.mark.parametrize("occupancy,reason", [(cc.EMPTY, "well_empty"),
                                              (cc.UNKNOWN, "occupancy_unrecorded")])
def test_check_insulator_occupancy_not_established_refused(occupancy, reason):
    k = _assess(_insulator(), occupancy=occupancy)
    assert k.reasons == (reason,) and not k.qualifies


@pytest.mark.parametrize("label,reason", [("U", "incoherent"), ("U", "too_many_nonphysical"),
                                          ("U", "some_future_token"), ("", "")])
def test_check_insulator_bad_data_or_unclassified_refused(label, reason):
    k = cc.check_insulator(F, _insulator(), channel=3, regime=label, regime_reason=reason,
                           occupancy=cc.OCCUPIED)
    assert k.reasons == ("regime_bad_data",)


def test_check_insulator_shape_u_allowed_regime_a_refused():
    Z = _insulator()
    shape_u = cc.check_insulator(F, Z, channel=3, regime="U", regime_reason="non_monotone",
                                 occupancy=cc.OCCUPIED)
    a = cc.check_insulator(F, Z, channel=3, regime="A", regime_reason="arc_then_cpe",
                           occupancy=cc.OCCUPIED)
    assert shape_u.qualifies and a.reasons == ("regime_a_film",)


def test_check_insulator_lf_dropped_unclosed_re_refused():
    Z = np.where(F < 20, np.nan + 0j, _insulator())   # the mostly-NaN LF of an unclosed RE
    assert _assess(Z).reasons == ("lf_dropped",)


def test_check_insulator_two_band_points_unresolved():
    keep = (F < 100) | (F > 1000) | np.isin(F, F[(F >= 100) & (F <= 1000)][:2])
    k = cc.check_insulator(F[keep], _insulator()[keep], channel=3, regime="C",
                           regime_reason="no_plateau", occupancy=cc.OCCUPIED)
    assert k.reasons == ("band_unresolved",)


def test_check_insulator_phase_and_flatness_thresholds():
    lossy = _assess(_insulator(5e-10, 3e5))          # R_x so large the band is −75°
    assert lossy.reasons == ("phase_not_capacitive",)
    f = np.array([1.0, 5.0, 150.0, 300.0, 600.0])
    Z = 1 - 1j / (2 * np.pi * f * np.array([1, 1, 1.0, 1.2, 1.4]) * 1e-10)
    k = cc.check_insulator(f, Z, channel=3, regime="C", regime_reason="no_plateau",
                           occupancy=cc.OCCUPIED)
    assert k.reasons == ("apparent_c_not_flat",) and k.flat_ratio == pytest.approx(1.4)


def test_check_insulator_empty_screen_never_raises_and_refuses():
    k = cc.check_insulator([], [], channel=3, regime="U", regime_reason="too_few_points",
                           occupancy=cc.OCCUPIED)
    assert k.c_cell_F is None and not k.qualifies and "band_unresolved" in k.reasons


def test_assess_spectrum_real_empty_well_refused_even_if_marked_occupied():
    f, Z = _real("empty_well_20260930T165408Z_ch17.txt")
    k = cc.assess_spectrum(f, Z, channel=17, occupancy=cc.OCCUPIED)
    assert "regime_bad_data" in k.reasons and not k.qualifies


def test_assess_spectrum_real_film_3b_ch1_refused_insulator_3b_ch3_qualifies():
    film = cc.assess_spectrum(*_real("rung3b_20260923T183923Z_ch1.txt"), channel=1,
                              occupancy=cc.OCCUPIED)
    ins = cc.assess_spectrum(*_real("rung3b_20260923T183923Z_ch3.txt"), channel=3,
                             occupancy=cc.OCCUPIED)
    assert "film_arc_present" in film.reasons
    assert ins.qualifies and ins.c_cell_F == pytest.approx(2.66e-10, rel=0.01)


# ── Aggregation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ch,group", [(1, (1, 8)), (8, (1, 8)), (9, (9, 16)), (24, (17, 24)),
                                      (32, (25, 32)), (0, None), (33, None)])
def test_channel_group_boundaries(ch, group):
    assert cc.channel_group(ch) == group


def _ok(ch, c, mid):
    return cc.InsulatorCheck(channel=ch, c_cell_F=c, reasons=(), measurement_id=mid)


def test_aggregate_groups_channel_median_then_group_median_with_spread():
    checks = [_ok(9, 1.0e-10, 1), _ok(9, 1.1e-10, 2), _ok(9, 1.2e-10, 3),  # ch9 → 1.1e-10
              _ok(10, 2.0e-10, 4), _ok(13, 4.0e-10, 5),
              cc.InsulatorCheck(channel=11, c_cell_F=9e-9, reasons=("film_arc_present",),
                                measurement_id=6)]
    g = cc.aggregate_groups(checks, board_id=2)
    rec = g[(9, 16)]
    assert rec.c_cell_F == pytest.approx(2.0e-10)     # median of 1.1, 2.0, 4.0 — not of 5 reads
    assert rec.spread_dec == pytest.approx(math.log10(4.0 / 1.1))
    assert rec.source_measurement_ids == (1, 2, 3, 4, 5) and rec.n_spectra == 5
    assert dict(rec.channel_values)[9] == pytest.approx(1.1e-10)
    assert rec.board_id == 2 and rec.method == cc.METHOD


def test_aggregate_groups_no_source_group_absent_never_default():
    g = cc.aggregate_groups([_ok(3, 2.6e-10, 1)], board_id=2)
    assert set(g) == {(1, 8)} and g[(1, 8)].spread_dec == 0.0
    assert cc.aggregate_groups([], board_id=2) == {}


def test_preferred_record_channel_then_group_then_none():
    group = {"channel": None, "group_lo": 9, "group_hi": 16, "c_cell_F": 1.8e-10}
    own = {"channel": 10, "group_lo": 9, "group_hi": 16, "c_cell_F": 1.9e-10}
    assert cc.preferred_record(10, [group, own]) is own
    assert cc.preferred_record(11, [group, own]) is group
    assert cc.preferred_record(3, [group, own]) is None
