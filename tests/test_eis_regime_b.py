"""Regime-aware σ, slice 2 (wave 1, lane L1): the regime-B film-arc estimator.

Spec: ``docs/SubAgent docs/regime_b_slice2_spec.md`` §1–§3 and §5 (operator rulings Q1–Q10,
2026-10-02). Three layers:

* the decision table on hand-built estimates (cheap, exhaustive over the detail tokens);
* known-truth synthetic spectra from :mod:`tests.eis_two_feature_synthetic`, with a
  positive control for each identity gate (a gate that is switched off must let a false
  value through — otherwise its "never a value" test proves nothing);
* the stored rung-3b / rung-3c / board-2 spectra of spec §5.1, read **read-only** from
  ``C:\\Users\\Osuji\\softae_data\\runs`` and skipped when absent.

The route returns resistances; σ = K/R is computed here only to compare with the spec.
"""

from __future__ import annotations

import collections
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis import feature_split as fs
from softae.analysis.eis import regime_b as rb
from tests import eis_two_feature_synthetic as tf

C_CELL = tf.C_CELL


def _route(t: tf.TwoFeature, seed: int, **kw) -> rb.RegimeBOutcome:
    f, Z = tf.screened(t, seed=seed)
    return rb.regime_b_route(f, Z, c_cell=kw.pop("c_cell", C_CELL), **kw)


def _dec(a: float, b: float) -> float:
    return math.log10(a / b)


# ── The decision table on hand-built estimates ────────────────────────────────

_FIT = rb.RegimeBFit(
    R_x=1e5, Q_x=3e-12, n_x=1.0, C_x=3e-12, cx_source=rb.CX_MEASURED,
    R_f=1e7, Q_f=C_CELL, n_f=1.0, C_f=C_CELL, f_apex_f=63.7, Q_e=1e-7, n_e=0.8,
    L=2e-6, se_log_R_f=0.01, rms=0.005, starts_spread_dec=0.0, n_starts=12)
_SPLIT = fs.FeatureSplit(
    i_valley=25, i_plateau=45, i_foot=8, valley_two_sided=True, f_valley=800.0,
    f_plateau=6e4, f_foot=5.0, tand_valley=0.09, tand_plateau=3.8, tand_foot=5.0,
    R_plateau=1e5, R_foot=1.03e7, re_z_lo=1.1e7, f_lo=1.351, f_top=2e5)


def _est(fit=_FIT, split=_SPLIT, c_cell=C_CELL, **fit_kw) -> rb.RegimeBEstimates:
    return rb.RegimeBEstimates(c_cell=c_cell, split=split,
                               fit=replace(fit, **fit_kw) if fit is not None else None)


#: (estimates, expected outcome, expected detail) — spec §3.2 rows 1–8.
_TABLE = {
    "value": (_est(), rb.VALUE, rb.RESOLVED_IN_BAND),
    "c_cell_none": (_est(c_cell=float("nan")), rb.UNAVAILABLE, rb.C_CELL_UNMEASURED),
    "fit_failed": (_est(fit=None), rb.UNAVAILABLE, rb.FIT_FAILED),
    "arc_unresolved": (_est(split=replace(_SPLIT, i_valley=4)), rb.UNAVAILABLE,
                       rb.ARC_UNRESOLVED),
    "series": (_est(C_x=3.1e-11), rb.UNAVAILABLE, rb.SERIES_ELEMENT_UNIDENTIFIED),
    "series_nan": (_est(C_x=float("nan")), rb.UNAVAILABLE, rb.SERIES_ELEMENT_UNIDENTIFIED),
    "series_unresolved": (_est(C_x=float("nan"), cx_source=rb.CX_UNRESOLVED), rb.VALUE,
                          rb.RESOLVED_IN_BAND),
    "no_partner_upper": (_est(split=replace(_SPLIT, tand_foot=0.6, re_z_lo=4e8),
                              R_f=1e9, f_apex_f=0.6), rb.UPPER_BOUND, rb.NO_FILM_ARC),
    "apex_below_band_upper": (_est(split=replace(_SPLIT, re_z_lo=4e8), f_apex_f=4.0),
                              rb.UPPER_BOUND, rb.NO_FILM_ARC),
    "apex_above_band_upper": (_est(split=replace(_SPLIT, re_z_lo=4e8), f_apex_f=7e4),
                              rb.UPPER_BOUND, rb.NO_FILM_ARC),
    "electrode": (_est(split=replace(_SPLIT, tand_foot=0.6), Q_e=1e-9),
                  rb.UNAVAILABLE, rb.ELECTRODE_NOT_NEGLIGIBLE),
    "no_chord": (_est(split=replace(_SPLIT, tand_foot=0.6, re_z_lo=9e4)),
                 rb.UNAVAILABLE, rb.NO_FILM_ARC),
    "below_band": (_est(C_f=C_CELL / 3.1), rb.UNAVAILABLE, rb.ARC_BELOW_CELL_BAND),
    "interphase": (_est(C_f=C_CELL * 3.1), rb.LOWER_BOUND, rb.INTERPHASE_BAND),
    "non_ideal": (_est(n_f=0.79), rb.LOWER_BOUND, rb.ARC_NON_IDEAL),
    "pair": (_est(R_f=1e7 / 1.6), rb.LOWER_BOUND, rb.PAIR_DISAGREES),
    "se": (_est(se_log_R_f=0.11), rb.LOWER_BOUND, rb.R_F_SE_WIDE),
    "multimodal": (_est(starts_spread_dec=0.11), rb.LOWER_BOUND, rb.FIT_MULTIMODAL),
    "first_failed_wins": (_est(C_f=C_CELL * 5, n_f=0.7, se_log_R_f=0.5),
                          rb.LOWER_BOUND, rb.INTERPHASE_BAND),
}


@pytest.mark.parametrize("case", list(_TABLE))
def test_regime_b_kind_decision_table_returns_spec_token(case):
    est, kind, detail = _TABLE[case]
    assert rb.regime_b_kind(est, has_cell=True) == (kind, detail)


def test_regime_b_kind_no_cell_constant_decides_first():
    assert rb.regime_b_kind(_est(c_cell=float("nan")), has_cell=False) == (
        rb.UNAVAILABLE, rb.NO_CELL_CONSTANT)


def test_regime_b_route_states_resistance_matching_its_direction(monkeypatch):
    """Value → R_fit; lower bound → max(R_fit, R_mf); upper bound → Re Z_lo − R_x."""
    for case, R in (("value", 1e7), ("pair", 1.03e7 - 1e5),
                    ("no_partner_upper", 4e8 - 1e5)):
        est = _TABLE[case][0]
        monkeypatch.setattr(rb, "regime_b_estimates", lambda f, Z, c, _e=est: _e)
        out = rb.regime_b_route(np.zeros(3), np.zeros(3), c_cell=C_CELL)
        assert out.R_ohm == pytest.approx(R), case
    est = _TABLE["electrode"][0]
    monkeypatch.setattr(rb, "regime_b_estimates", lambda f, Z, c: est)
    assert math.isnan(rb.regime_b_route(np.zeros(3), np.zeros(3), c_cell=C_CELL).R_ohm)


# ── The fit ───────────────────────────────────────────────────────────────


def test_regime_b_model_jacobian_matches_central_difference():
    w = 2 * np.pi * tf.F_ASC
    p = np.array([-5.7, 4.9, -11.5, 0.93, 6.8, -9.6, 0.91, -7.4, 0.7])
    Z, J = rb._model_and_jac(p, w)
    for k in range(1, p.size):            # L (k=0) contributes ~1e-6 of |Z|: FD noise
        h = 1e-6
        d = (rb.two_feature_model(p + h * np.eye(p.size)[k], w)
             - rb.two_feature_model(p - h * np.eye(p.size)[k], w)) / (2 * h)
        assert np.max(np.abs(d - J[:, k])) <= 1e-6 * np.max(np.abs(J[:, k])), k


def test_regime_b_fit_recovers_truth_with_series_element_first():
    t = tf.TwoFeature(R_x=1e5, C_x=3e-12, R_f=1e7, c_ratio=1.0, n_f=1.0)
    f, Z = tf.screened(t, seed=5)
    fit = rb.fit_two_feature(f, Z, c_cell=C_CELL)
    assert fit is not None and fit.n_starts == 12 and fit.cx_source == rb.CX_MEASURED
    assert abs(_dec(fit.R_x, t.R_x)) <= 0.05 and abs(_dec(fit.R_f, t.R_f)) <= 0.05
    assert abs(_dec(fit.C_f, t.C_f)) <= 0.05 and abs(_dec(fit.C_x, t.C_x)) <= 0.2


def test_regime_b_canonical_puts_higher_apex_first():
    x = np.array([-5.7, 7.0, -10.0, 1.0, 5.0, -12.0, 1.0, -7.0, 0.8])  # apex1 16 Hz, 2 1.6 MHz
    y, swapped = rb._canonical(x)
    assert swapped and list(y[1:4]) == [5.0, -12.0, 1.0] and list(y[4:7]) == [7.0, -10.0, 1.0]
    assert rb._canonical(y)[1] is False


def test_regime_b_fit_pinned_qx_reports_cx_unresolved():
    """The HF artefact screens the top to ~80 kHz and Q_x pins: C_x is UNRESOLVED (NaN,
    ``cx_source`` says so) — not the pinned 1e-16, and not 1/(2π R_x f_top) (operator
    ruling 2026-10-03)."""
    t = tf.TwoFeature(R_x=3e4, C_x=2e-12, R_f=1e6, c_ratio=1.0, n_f=0.9, Q_e=1e-7,
                      n_e=0.6, art=True)
    f, Z = tf.screened(t, seed=8)
    fit = rb.fit_two_feature(f, Z, c_cell=C_CELL)
    assert fit.cx_source == rb.CX_UNRESOLVED
    assert math.isnan(fit.C_x)


def test_regime_b_no_c_cell_unavailable_without_fitting():
    """§3.1(a): no measured C_cell is never a default band — and the same spectrum with
    one gives a value, so the refusal is the C_cell's doing."""
    t = tf.TwoFeature()
    for c in (None, float("nan"), 0.0, -1e-10):
        out = _route(t, 1, c_cell=c)
        assert (out.kind, out.detail) == (rb.UNAVAILABLE, rb.C_CELL_UNMEASURED), c
        assert out.estimates.fit is None
    assert _route(t, 1).kind == rb.VALUE


# ── Synthetic grids (spec §5.2) ──────────────────────────────────────────────


def _run(grid):
    return [(t, _route(t, seed)) for t, seed in grid]


@pytest.fixture(scope="module")
def inband():
    return _run(tf.inband_grid())


@pytest.fixture(scope="module")
def interphase():
    return _run(tf.interphase_grid())


@pytest.fixture(scope="module")
def out_of_band():
    return _run(tf.out_of_band_grid())


def test_regime_b_kind_cell_band_inband_value_within_010_dec(inband):
    vals = [_dec(o.R_ohm, t.R_f) for t, o in inband if o.kind == rb.VALUE]
    clean = [o.kind for t, o in inband if not t.art]
    assert len(vals) >= 50
    assert clean.count(rb.VALUE) >= 0.75 * len(clean)          # measured 46/48
    assert np.percentile(np.abs(vals), 95) <= 0.1


def test_regime_b_kind_inband_refusals_never_above_truth(inband):
    """In-band arcs that do not give a value are lower bounds on the safe side, or
    unavailable — never an upper bound below the truth."""
    for t, o in inband:
        assert o.kind != rb.UPPER_BOUND, t
        if o.kind == rb.LOWER_BOUND:
            assert o.R_ohm >= t.R_f * 10 ** -0.05, t


def test_regime_b_kind_interphase_never_value(interphase):
    kinds = collections.Counter(o.kind for _, o in interphase)
    assert kinds[rb.VALUE] == 0
    assert kinds[rb.LOWER_BOUND] >= 20


def test_regime_b_kind_interphase_lower_bound_never_above_truth(interphase):
    """K/max(R) ≤ K/R_f, to within the fit's noise (≤ 0.05 dec; measured worst 0.043)."""
    lows = [(t, o) for t, o in interphase if o.kind == rb.LOWER_BOUND]
    assert lows
    assert all(o.detail == rb.INTERPHASE_BAND for _, o in lows)
    for t, o in lows:
        assert o.R_ohm >= t.R_f * 10 ** -0.05, t


def test_regime_b_kind_non_ideal_never_value():
    outs = _run(tf.non_ideal_grid())
    assert not [o for _, o in outs if o.kind == rb.VALUE]
    assert sum(o.detail == rb.ARC_NON_IDEAL for _, o in outs) >= 20


def test_regime_b_kind_out_of_band_never_value(out_of_band):
    assert not [o for _, o in out_of_band if o.kind == rb.VALUE]


def test_regime_b_kind_out_of_band_upper_bound_covers_truth(out_of_band):
    ups = [(t, o) for t, o in out_of_band if o.kind == rb.UPPER_BOUND]
    assert len(ups) >= 15
    covered = sum(o.R_ohm <= t.R_f for t, o in ups)
    assert covered >= 0.995 * len(ups)


def test_regime_b_kind_insulator_never_value(out_of_band):
    ins = [o for t, o in out_of_band if t.R_f >= 1e11]
    assert ins and all(o.kind != rb.VALUE for o in ins)


def test_regime_b_kind_hf_film_series_rule_refuses(monkeypatch):
    """C_x = 3e-10 (the HF feature is film-like) never gives a value. Positive control:
    with rule 4 switched off, the same spectra give false values."""
    grid = list(tf.series_film_grid())[:12]
    assert not [o for _, o in _run(grid) if o.kind == rb.VALUE]
    monkeypatch.setattr(fs, "is_series_element", lambda C, c_max=0.0: True)
    assert [o for _, o in _run(grid) if o.kind == rb.VALUE]


def test_regime_b_kind_band_disabled_interphase_gives_false_value(monkeypatch):
    """Positive control for rule 7's C band: without it, interphase arcs become values."""
    grid = [(t, s) for t, s in tf.interphase_grid() if t.c_ratio == 5.0 and not t.art][:10]
    assert not [o for _, o in _run(grid) if o.kind == rb.VALUE]
    monkeypatch.setattr(fs, "cell_band_position", lambda C, c, factor=3.0: fs.BAND_IN)
    assert [o for _, o in _run(grid) if o.kind == rb.VALUE]


def test_film_arc_resistance_identified_or_none():
    """The settle observable: R_f for a closed in-band arc (value gates not applied),
    ``None`` when unidentified — never R_x."""
    t = tf.TwoFeature(R_f=1e7)
    f, Z = tf.screened(t, seed=2)
    assert abs(_dec(rb.film_arc_resistance(f, Z, C_CELL), 1e7)) <= 0.1
    assert rb.film_arc_resistance(f, Z, None) is None
    hi = tf.TwoFeature(R_x=1e5, R_f=1e6, c_ratio=5.0, Q_e=1e-7)       # interphase arc
    R = rb.film_arc_resistance(*tf.screened(hi, seed=4), C_CELL)
    assert R is not None and abs(_dec(R, 1e6)) <= 0.1
    ins = tf.TwoFeature(R_f=1e11, Q_e=1e-6)
    assert rb.film_arc_resistance(*tf.screened(ins, seed=6), C_CELL) is None


# ── Stored data (spec §5.1) — read-only, skipped when absent ──────────────────

RUNS = Path(r"C:\Users\Osuji\softae_data\runs")
RUNG3B = RUNS / "20260923T183923Z_rung3b_bench" / "eis"
RUNG3C = RUNS / "20261002T053100Z_rung3c_paa_bench" / "eis"
BOARD2 = RUNS / "20260929T150528Z_manual_eis" / "eis"
#: Predicted 160.395 µm, L_gap = L_stripe = 0.2 cm (``regime_b_two_feature_analysis.md``).
K = 62.34595340804066
#: The mux16 series terms (6.46 Ω, 2.54 µH), subtracted as the analysis did.
R_SHORT, L_LEAD = 6.4581, 2.539736206815413e-06
#: Per-group C_cell on board 2: lane L2's values under the rules as written (operator
#: ruling 2026-10-03; group 9–16 is ch11 + ch13).
C_CELL_GROUP = {1: 2.66e-10, 2: 1.71e-10, 3: 3.08e-10}


def _stored_path(rung: str, ch: int) -> Path:
    if rung == "b2":
        return BOARD2 / f"ch{ch:02d}_manual.txt"
    d = RUNG3B if rung == "3b" else RUNG3C
    return d / f"production_measure_eis_ch{ch}_ch{ch}.txt"


def _load_stored(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, comments="#")                            # read-only
    f, Z = a[:, 0], a[:, 3] - 1j * a[:, 4]
    Z = Z - R_SHORT - 1j * 2 * np.pi * f * L_LEAD
    o = np.argsort(f)
    return f[o], Z[o]


STORED = [("3b", c) for c in range(1, 9)] + [("3c", c) for c in (18, 19, 20, 23)] \
    + [("b2", c) for c in range(9, 17)]


@pytest.fixture(scope="module")
def stored():
    from softae.analysis.eis.regime import classify_regime

    paths = {k: _stored_path(*k) for k in STORED}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"stored spectra absent (read-only DataStore runs): {missing[:2]}")
    out = {}
    for (rung, ch), p in paths.items():
        v = classify_regime(*_load_stored(p))
        group = 1 if ch <= 8 else (2 if ch <= 16 else 3)
        out[(rung, ch)] = (v, rb.regime_b_route(v.screen.f, v.screen.Z,
                                                 c_cell=C_CELL_GROUP[group]))
    return out


def _sigma(o: rb.RegimeBOutcome) -> float:
    return K / o.R_ohm


@pytest.mark.parametrize("ch,expected", [(1, 3.1e-6), (5, 1.5e-6), (2, 1.2e-5),
                                         (6, 3.2e-6), (4, 5.2e-6), (8, 1.1e-6)])
def test_regime_b_route_stored_rung3b_value_within_010_dec(stored, ch, expected):
    v, o = stored[("3b", ch)]
    assert v.label == "B"
    assert (o.kind, o.detail) == (rb.VALUE, rb.RESOLVED_IN_BAND)
    assert abs(_dec(_sigma(o), expected)) <= 0.1


@pytest.mark.parametrize("ch,label,expected", [(18, "B", 4.9e-5), (23, "U", 1.04e-4)])
def test_regime_b_route_stored_rung3c_value_within_010_dec(stored, ch, label, expected):
    """23 is the pin against the double-layer capture failure (spec §1.4)."""
    v, o = stored[("3c", ch)]
    assert v.label == label
    assert o.kind == rb.VALUE
    assert abs(_dec(_sigma(o), expected)) <= 0.1


def test_regime_b_route_stored_rung3c_19_interphase_lower_bound(stored):
    """σ ≥ K/max(3.2e5, 3.7e5) ≈ 1.7e-4 — today's told 5.7e-4 is 0.5 dec high."""
    v, o = stored[("3c", 19)]
    assert (v.label, v.reason) == ("U", "no_cpe_drop")
    assert (o.kind, o.detail) == (rb.LOWER_BOUND, rb.INTERPHASE_BAND)
    assert abs(_dec(_sigma(o), 1.7e-4)) <= 0.1
    assert _sigma(o) < 5.7e-4 / 10 ** 0.4


@pytest.mark.parametrize("rung,ch,expected", [("3b", 3, 1.6e-7), ("3c", 20, 2.0e-7)])
def test_regime_b_route_stored_insulator_upper_bound_informative(stored, rung, ch, expected):
    """Within 0.1 dec of the spec's bound, and ≥ 1 dec below K/R_x — otherwise it is the
    channel constant in disguise (spec §5.1 informativeness check)."""
    _, o = stored[(rung, ch)]
    assert (o.kind, o.detail) == (rb.UPPER_BOUND, rb.NO_FILM_ARC)
    assert abs(_dec(_sigma(o), expected)) <= 0.1
    assert _dec(K / o.estimates.fit.R_x, _sigma(o)) >= 1.0


@pytest.mark.parametrize("ch", [9, 10, 11, 12, 13, 14])
def test_regime_b_route_stored_board2_9_to_14_never_value(stored, ch):
    _, o = stored[("b2", ch)]
    assert o.kind in (rb.UPPER_BOUND, rb.UNAVAILABLE)


@pytest.mark.parametrize("ch,route_a", [(15, 6.32e-4), (16, 5.62e-4)])
def test_regime_b_route_stored_board2_15_16_forced_value_matches_route_a(stored, ch, route_a):
    """Cross-route check: B forced onto the two A films gives a value within 0.05 dec of
    route A. Their shunt is unresolved (R_x ≈ 1e4, screened f_top 27.9 kHz), which rule 4
    no longer refuses (operator ruling 2026-10-03); the film is identified by C_f alone."""
    v, o = stored[("b2", ch)]
    assert v.label == "A"
    assert o.estimates.fit.cx_source == rb.CX_UNRESOLVED
    assert (o.kind, o.detail) == (rb.VALUE, rb.RESOLVED_IN_BAND)
    assert abs(_dec(_sigma(o), route_a)) <= 0.05
    assert fs.cell_band_position(o.estimates.fit.C_f, C_CELL_GROUP[2]) == fs.BAND_IN


def test_regime_b_route_reached_by_real_fixtures(stored):
    """§3.2 counter (``SUBAGENT_RULES`` §3.2): real data enters value, lower and upper."""
    kinds = collections.Counter(o.kind for _, o in stored.values())
    assert kinds[rb.VALUE] >= 8
    assert kinds[rb.LOWER_BOUND] >= 1
    assert kinds[rb.UPPER_BOUND] >= 2
