"""The model-free two-feature leaf (``analysis/eis/feature_split.py``).

Spec: ``docs/SubAgent docs/regime_b_slice2_spec.md`` §1.3 and §2.1. Synthetic spectra come
from :mod:`tests.eis_two_feature_synthetic` and need nothing on disk.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from softae.analysis.eis import feature_split as fs
from tests import eis_two_feature_synthetic as tf


def _split(t: tf.TwoFeature, seed: int = 7) -> fs.FeatureSplit:
    return fs.split_features(*tf.screened(t, seed=seed))


def _dec(a: float, b: float) -> float:
    return abs(math.log10(a / b))


# ── split_features ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("R_x,R_f", [(3e4, 1e6), (1e5, 1e7), (3e5, 1e8), (1e5, 1e6)])
def test_split_features_synthetic_b_partner_within_010_dec(R_x, R_f):
    """A closed in-band arc: R_mf within 0.1 dec of R_f, R_plateau within 0.05 of R_x."""
    s = _split(tf.TwoFeature(R_x=R_x, R_f=R_f, c_ratio=1.0, Q_e=1e-7, n_e=0.8))
    assert s.valley_two_sided and s.closed
    assert s.i_foot < s.i_valley < s.i_plateau
    assert _dec(s.R_mf, R_f) <= 0.1
    assert _dec(s.R_plateau, R_x) <= 0.05


def test_split_features_insulator_not_closed_no_partner():
    """R_f = 1e11: tan δ never returns above 1 below the valley — no partner."""
    s = _split(tf.TwoFeature(R_f=1e11, Q_e=1e-7))
    assert not s.closed
    assert math.isnan(s.R_mf)
    assert s.R_chord > 0


def test_split_features_single_arc_finds_local_valley_not_electrode_floor():
    """A single film arc (route-A shape): the global tan δ minimum is the electrode floor
    at f_lo, and the two-feature valley is the *local* one above the arc."""
    t = tf.TwoFeature(R_x=1e4, C_x=2e-12, R_f=1e5, c_ratio=1.0, Q_e=1e-7, n_e=0.7)
    f, Z = tf.screened(t, seed=3)
    s = fs.split_features(f, Z)
    assert int(np.argmin(fs.tan_delta(Z))) < s.i_foot      # global min: the floor
    assert s.valley_two_sided and s.closed
    assert s.f_valley > s.f_foot
    assert _dec(s.R_mf, 1e5) <= 0.1


def test_split_features_monotone_falls_back_to_global_minimum():
    """No interior valley: the global minimum, ``two_sided`` False, no foot, not closed."""
    f = tf.F_ASC
    Z = 1e5 + 1.0 / (1j * 2 * np.pi * f * 2.5e-10)              # R_x then a bare capacitor
    s = fs.split_features(f, Z)
    assert not s.valley_two_sided
    assert s.i_valley == 0 and s.i_foot == -1
    assert not s.closed and math.isnan(s.R_mf)


def test_split_features_shallow_dip_in_plateau_not_a_valley():
    """A < 0.1 dec dip at the series plateau must not become a valley whose 'foot' is the
    plateau's own rising side (which would invent a closed arc).

    Noiseless and electrode-free on purpose: tan δ is then strictly monotone below the
    plateau, so no other two-sided valley exists and ``VALLEY_FALL_DEC`` alone decides
    (with noise, an LF wiggle is always lower and the threshold is never consulted).
    """
    f = tf.F_ASC
    w = 2 * np.pi * f
    Z = 1.0 / (1.0 / 1e5 + 1j * w * 3e-12) + 1.0 / (1j * w * 2.5e-10)
    m = int(np.argmax(fs.tan_delta(Z)))
    for k in (m - 1, m):                                         # two points: survives med3
        Z[k] = complex(Z[k].real * 0.9, Z[k].imag)               # ~0.05 dec tan δ dip
    s = fs.split_features(f, Z)
    assert not s.closed and not s.valley_two_sided


def test_split_features_too_few_points_raises():
    with pytest.raises(ValueError):
        fs.split_features(np.array([1.0, 2.0]), np.array([1 - 1j, 1 - 1j]))


def test_feature_split_r_mf_requires_tand_foot_at_least_one():
    s = _split(tf.TwoFeature())
    assert s.closed
    assert math.isnan(replace(s, tand_foot=0.99).R_mf)
    assert replace(s, tand_foot=1.0).R_mf == pytest.approx(s.R_foot - s.R_plateau)
    assert math.isnan(replace(s, i_foot=-1).R_mf)


# ── Capacitance helpers ────────────────────────────────────────────────────


def test_brug_capacitance_ideal_equals_q():
    assert fs.brug_capacitance(1e6, 2.5e-10, 1.0) == pytest.approx(2.5e-10)


@pytest.mark.parametrize("n", [0.75, 0.9, 1.0])
def test_brug_capacitance_roundtrips_generator_q(n):
    """The generator builds Q from a Brug C; the helper must give it back."""
    R, C = 1e7, 3e-10
    Q = C ** n * R ** (n - 1)
    assert fs.brug_capacitance(R, Q, n) == pytest.approx(C, rel=1e-9)


def test_apex_frequency_ideal_is_one_over_2pi_rc():
    assert fs.apex_frequency(1e6, 1e-10, 1.0) == pytest.approx(1 / (2 * math.pi * 1e-4))


def test_cell_band_position_edges_inclusive():
    c = 2.5e-10
    assert fs.cell_band_position(c / 3, c) == fs.BAND_IN
    assert fs.cell_band_position(3 * c, c) == fs.BAND_IN
    assert fs.cell_band_position(c / 3 * 0.99, c) == fs.BAND_BELOW
    assert fs.cell_band_position(3 * c * 1.01, c) == fs.BAND_ABOVE


@pytest.mark.parametrize("C,c_cell", [(float("nan"), 2e-10), (2e-10, float("nan")),
                                      (2e-10, 0.0), (0.0, 2e-10), (2e-10, -1.0)])
def test_cell_band_position_unmeasured_is_unknown_not_in(C, c_cell):
    assert fs.cell_band_position(C, c_cell) == fs.BAND_UNKNOWN


@pytest.mark.parametrize("C_x,ok", [(3e-11, True), (2e-12, True), (3.01e-11, False),
                                    (0.0, False), (float("nan"), False)])
def test_is_series_element_threshold_and_unknown(C_x, ok):
    assert fs.is_series_element(C_x) is ok


def test_pair_ratio_symmetric_and_nan_when_missing():
    assert fs.pair_ratio(2.0, 3.0) == fs.pair_ratio(3.0, 2.0) == pytest.approx(1.5)
    assert math.isnan(fs.pair_ratio(float("nan"), 1.0))
    assert math.isnan(fs.pair_ratio(0.0, 1.0))
