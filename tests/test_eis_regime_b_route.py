"""The regime-B route through the real engine (slice 2, wave 2).

Spec: ``docs/SubAgent docs/regime_b_slice2_spec.md`` §3.2–§3.4 and §5. The estimator itself
is pinned in ``tests/test_eis_regime_b.py``; this file pins what ``analyze_spectrum`` does
with it: the shadow in every flag state, the activation gate (``regime_aware`` **and**
``regime_b``), the mapping of the four outcomes onto slice 1's report conventions, and
identity to HEAD's fields whenever the route is not live. Stored wells are read
**read-only** from ``C:\\Users\\Osuji\\softae_data\\runs`` and skipped when absent.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import structlog

from softae.analysis.eis import regime_b as rb
from softae.analysis.eis import regime_route as rr
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.fixture import FixtureCorrection
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.observation import LOWER_BOUND, UPPER_BOUND, VALUE, sigma_observation
from softae.analysis.eis.report import (
    REGIME_A_FIT_RB,
    REGIME_B_ARC_LOWER,
    REGIME_B_FILM_ARC,
    REGIME_B_NO_ARC,
)
from softae.analysis.equilibration import (
    BASIS_FILM_ARC,
    BASIS_FILM_RB,
    BASIS_FILM_UNIDENTIFIED,
)
from tests import eis_regime_golden as golden
from tests import eis_regime_synthetic as syn
from tests import eis_two_feature_synthetic as tf

FLAG_OFF = rr.RegimeSettings(enabled=False, regime_b=False)
FLAG_OFF_B_ON = rr.RegimeSettings(enabled=False, regime_b=True)
B_OFF = rr.RegimeSettings(enabled=True, regime_b=False)
ON = rr.RegimeSettings(enabled=True, regime_b=True)
K = golden.golden_inputs(15)["cell"].K_per_cm
#: The fields slice 2 added; every other SigmaReport field is HEAD's.
NEW_FIELDS = {"regime_b_mode", "regime_b_detail", "regime_b_sigma", "regime_cx_source",
              "regime_film_R_ohm", "regime_film_basis"}

#: Synthetic two-feature spectra the classifier routes to B, one per outcome, with truth.
#: ``lower`` is the first in-band grid point: U/no_cpe_drop, whose pair disagrees.
_LOWER_T, _LOWER_SEED = next(iter(tf.inband_grid()))
SYNTH = {
    "value": (tf.TwoFeature(), 1),
    "upper": (tf.TwoFeature(R_f=1e10), 5),
    "lower": (_LOWER_T, _LOWER_SEED),
}


def _dec(a: float, b: float) -> float:
    return math.log10(a) - math.log10(b)


def _same(x, y) -> bool:
    return x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)


def _head_fields(report) -> dict:
    """Every HEAD field of the report, plus what the engine decided beside σ."""
    s = dataclasses.asdict(report.sigma)
    return dict({k: v for k, v in s.items() if k not in NEW_FIELDS},
                fitter=report.fitter, issues=list(report.quality.issues),
                verdict=str(report.quality.verdict))


def _diff(a, b) -> set:
    fa, fb = _head_fields(a), _head_fields(b)
    return {k for k in fa if not _same(fa[k], fb[k])}


@pytest.fixture(autouse=True)
def _pinned_config(monkeypatch):
    from softae.config import loader

    monkeypatch.setattr(loader, "load", lambda *a, **k: {})


def _analyse(name: str, flags: rr.RegimeSettings, *, c_cell: float | None = tf.C_CELL,
             **overrides):
    """One synthetic spectrum through the gated engine (golden inputs + *overrides*)."""
    t, seed = SYNTH[name]
    inputs = {**golden.golden_inputs(15), **overrides}
    return analyze_spectrum(syn.as_eis(*tf.spectrum(t, seed=seed)), **inputs, regime=flags,
                            cell_capacitance=c_cell)


_CACHE: dict = {}


def _run(name: str, flags: rr.RegimeSettings, *, c_cell: float | None = tf.C_CELL):
    """:func:`_analyse`, once per (spectrum, flags, C_cell) — a gated fit each."""
    key = (name, flags, c_cell)
    if key not in _CACHE:
        _CACHE[key] = _analyse(name, flags, c_cell=c_cell)
    return _CACHE[key]


# ── The four outcomes, mapped onto slice 1's conventions ─────────────────────────


def test_engine_regime_b_value_reports_film_arc_within_010_dec():
    t, _ = SYNTH["value"]
    s = _run("value", ON).sigma
    assert (s.regime, s.regime_active, s.mode, s.regime_mode) == ("B", True, "value", "value")
    assert (s.regime_b_mode, s.regime_b_detail) == (rb.VALUE, rb.RESOLVED_IN_BAND)
    assert s.R_basis == REGIME_B_FILM_ARC
    assert abs(_dec(s.value, K / t.R_f)) <= 0.1
    assert s.value == s.regime_sigma == s.regime_b_sigma
    assert (s.regime_film_basis, s.regime_cx_source) == (BASIS_FILM_ARC, rb.CX_MEASURED)
    assert s.regime_film_R_ohm == s.R_reported_ohm
    assert 0 <= s.cross_check_pct < 50.0             # the pair agreed within ×1.5
    assert math.isnan(s.regime_sigma_lower) and s.upper_bound_basis == "unavailable"


def test_engine_regime_b_upper_bound_covers_truth_basis_no_arc():
    t, _ = SYNTH["upper"]
    s = _run("upper", ON).sigma
    assert (s.regime, s.regime_reason) == ("U", "non_monotone")      # a routed U
    assert (s.regime_active, s.mode, s.regime_mode) == (True, "bound", "bound")
    assert (s.regime_b_mode, s.regime_b_detail) == (rb.UPPER_BOUND, rb.NO_FILM_ARC)
    assert s.upper_bound_basis == s.R_basis == REGIME_B_NO_ARC
    assert s.upper_bound >= K / t.R_f
    assert s.provisional is True                     # model-conditional (spec §3.2)
    assert math.isnan(s.upper_bound_f_hz) and math.isnan(s.regime_sigma_lower)
    assert (s.regime_film_basis, s.regime_film_R_ohm == s.regime_film_R_ohm) == (
        BASIS_FILM_UNIDENTIFIED, False)


def test_engine_regime_b_lower_bound_spelled_as_foot_lower_never_above_truth():
    """Spelled exactly as route A's foot lower bound — *unavailable* with
    ``regime_sigma_lower`` — so observation admits it as a censored lower bound."""
    t, _ = SYNTH["lower"]
    s = _run("lower", ON).sigma
    assert (s.regime_active, s.mode, s.regime_mode) == (True, "unavailable", "unavailable")
    assert s.regime_b_mode == rb.LOWER_BOUND
    assert s.R_basis == REGIME_B_ARC_LOWER
    assert s.regime_sigma_lower <= K / t.R_f
    assert s.regime_sigma_lower == s.regime_b_sigma
    assert s.regime_R_foot_ohm == s.R_reported_ohm
    assert math.isnan(s.value) and math.isnan(s.upper_bound)
    assert s.regime_film_basis == BASIS_FILM_ARC     # identified: settle tracks it (§4)


def test_engine_regime_b_no_c_cell_unavailable_and_today_stands():
    """§3.1(a): no C_cell is *unmeasured*, never a default band — and nothing is fitted."""
    on, off = _run("value", ON, c_cell=None), _run("value", B_OFF, c_cell=None)
    s = on.sigma
    assert (s.regime_b_mode, s.regime_b_detail) == (rb.UNAVAILABLE, rb.C_CELL_UNMEASURED)
    assert s.regime_active is False and s.regime_cx_source == ""
    assert s.regime_film_basis == BASIS_FILM_UNIDENTIFIED
    assert _diff(on, off) == set()


def test_engine_regime_b_no_cell_constant_unavailable_film_still_identified():
    """No K: nothing stated; the film R is still identified — the settle observable is
    1/R_f and needs no K (``regime_b.film_arc_resistance`` judges it the same way)."""
    t, _ = SYNTH["value"]
    s = _analyse("value", ON, cell=None).sigma
    assert (s.regime_b_mode, s.regime_b_detail) == (rb.UNAVAILABLE, rr.NO_CELL_CONSTANT)
    assert s.regime_active is False and s.mode == "unavailable"
    assert s.regime_film_basis == BASIS_FILM_ARC
    assert abs(_dec(s.regime_film_R_ohm, t.R_f)) <= 0.1


# ── The activation gate and the shadow ───────────────────────────────────────────


@pytest.mark.parametrize("flags, live", [(FLAG_OFF, False), (FLAG_OFF_B_ON, False),
                                         (B_OFF, False), (ON, True)],
                         ids=["off", "off_b_on", "b_off", "on"])
def test_engine_regime_b_route_live_only_when_both_flags_on(flags, live):
    """The shadow fields are the same in every flag state; only ``ON`` reports them."""
    report = _run("value", flags)
    on = _run("value", ON).sigma
    s = report.sigma
    assert s.regime_active is live
    assert {k: getattr(s, k) for k in NEW_FIELDS} == {k: getattr(on, k) for k in NEW_FIELDS}
    if not live:
        assert s.R_basis != REGIME_B_FILM_ARC
        assert s.regime_mode == "" and math.isnan(s.regime_sigma)


@pytest.mark.parametrize("name", sorted(SYNTH))
@pytest.mark.parametrize("flags", [FLAG_OFF, B_OFF], ids=["flag_off", "b_off"])
def test_engine_regime_b_not_live_head_fields_match_engine_without_b_route(
        monkeypatch, name, flags):
    """``regime_b`` off (or ``regime_aware`` off) is HEAD: every HEAD field equals the
    engine with the B route removed outright — the pre-slice-2 engine by construction.
    Also checked once against a capture from HEAD source (report, 2026-10-04)."""
    report = _run(name, flags)
    monkeypatch.setattr(rr, "b_routed", lambda verdict: False)
    baseline = _analyse(name, flags)
    assert baseline.sigma.regime_b_mode == ""                 # the route really is gone
    assert _diff(report, baseline) == set()


def test_engine_regime_b_live_moves_report_positive_control(monkeypatch):
    """POSITIVE CONTROL for the identity above: live, the same comparison differs."""
    report = _run("value", ON)
    monkeypatch.setattr(rr, "b_routed", lambda verdict: False)
    baseline = _analyse("value", ON)
    assert _diff(report, baseline) != set()


def test_engine_regime_b_route_error_today_stands_regime_label_kept(monkeypatch):
    """A raising route is caught at the B route: the label and route A's fields survive."""
    def boom(*_a, **_k):
        raise RuntimeError("route disabled")

    monkeypatch.setattr(rb, "regime_b_route", boom)
    report = _analyse("value", ON)
    s = report.sigma
    assert (s.regime, s.regime_b_mode, s.regime_b_detail) == (
        "B", rb.UNAVAILABLE, rr.B_ROUTE_ERROR)
    assert s.regime_active is False
    assert _diff(report, _run("value", B_OFF)) == set()


def test_engine_regime_b_shadow_logged_with_b_prefix_in_both_states():
    for flags in (B_OFF, ON):
        with structlog.testing.capture_logs() as logs:
            _analyse("value", flags)
        ev = [e for e in logs if e["event"] == "sigma_regime_shadow"]
        assert len(ev) == 1, flags
        assert ev[0]["b_kind"] == rb.VALUE and ev[0]["b_c_cell"] == tf.C_CELL, flags
        assert ev[0]["route_active"] is (flags is ON), flags
        assert ev[0]["regime_basis"] == (REGIME_B_FILM_ARC if flags is ON else ""), flags


def test_engine_regime_b_c_and_a_spectra_never_routed():
    """C keeps no film basis (today's R₁ tracks it); A carries route A's film R."""
    corpus = dict((n, Z) for n, _, Z in golden.synthetic_corpus())
    c = analyze_spectrum(syn.as_eis(syn.RIG_F, corpus["synth_C_0"]),
                         **golden.golden_inputs(15), regime=ON, cell_capacitance=tf.C_CELL)
    assert c.sigma.regime == "C" and c.sigma.regime_b_mode == ""
    assert (c.sigma.regime_film_basis, c.sigma.regime_active) == ("", False)
    a = analyze_spectrum(syn.as_eis(syn.RIG_F, corpus["synth_A_inband_0"]),
                         **golden.golden_inputs(15), regime=ON, cell_capacitance=tf.C_CELL)
    s = a.sigma
    assert (s.regime, s.regime_b_mode, s.R_basis) == ("A", "", REGIME_A_FIT_RB)
    assert s.regime_film_basis == BASIS_FILM_RB and s.regime_film_R_ohm == s.R_reported_ohm


# ── Stored wells (spec §5.1) — read-only, skipped when absent ──────────────────────

RUNS = Path(r"C:\Users\Osuji\softae_data\runs")
DIRS = {"3b": RUNS / "20260923T183923Z_rung3b_bench" / "eis",
        "3c": RUNS / "20261002T053100Z_rung3c_paa_bench" / "eis",
        "b2": RUNS / "20260929T150528Z_manual_eis" / "eis"}
#: Predicted 160.395 µm, L_gap = L_stripe = 0.2 cm; the mux16 series terms.
STORED_CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.0160395,
                           thickness_method="predicted")
R_SHORT, L_LEAD = 6.4581, 2.539736206815413e-06
#: Per-group C_cell on board 2 (operator ruling 2026-10-03).
C_CELL_GROUP = {1: 2.66e-10, 2: 1.71e-10, 3: 3.08e-10}
#: b2 11 and 12 are left to the route-level test: today's gated fit costs ~63 s on each.
STORED = [("3b", c) for c in range(1, 9)] + [("3c", c) for c in (18, 19, 20, 23)] \
    + [("b2", c) for c in (9, 13, 14, 15, 16)]


def _stored_eis(rung: str, ch: int):
    p = DIRS[rung] / (f"ch{ch:02d}_manual.txt" if rung == "b2"
                      else f"production_measure_eis_ch{ch}_ch{ch}.txt")
    a = np.loadtxt(p, comments="#")                               # read-only
    return syn.as_eis(a[:, 0], a[:, 3] - 1j * a[:, 4], channel=ch)


def _stored_run(rung: str, ch: int, flags: rr.RegimeSettings):
    inputs = dict(golden.golden_inputs(ch), cell=STORED_CELL, correction=FixtureCorrection(
        mode="series", channel=ch, fixture_id="mux16", R_short_ohm=R_SHORT, L_lead_H=L_LEAD))
    group = 1 if ch <= 8 else (2 if ch <= 16 else 3)
    return analyze_spectrum(_stored_eis(rung, ch), **inputs, regime=flags,
                            cell_capacitance=C_CELL_GROUP[group])


@pytest.fixture(scope="module")
def stored():
    missing = [d for d in DIRS.values() if not d.exists()]
    if missing:
        pytest.skip(f"stored spectra absent (read-only DataStore runs): {missing[0]}")
    with pytest.MonkeyPatch.context() as mp:
        from softae.config import loader

        mp.setattr(loader, "load", lambda *a, **k: {})
        return {k: _stored_run(*k, ON) for k in STORED}


@pytest.mark.parametrize("rung, ch, expected", [
    ("3b", 1, 3.1e-6), ("3b", 5, 1.5e-6), ("3b", 2, 1.2e-5), ("3b", 6, 3.2e-6),
    ("3b", 4, 5.2e-6), ("3b", 8, 1.1e-6), ("3c", 18, 4.9e-5), ("3c", 23, 1.04e-4)])
def test_engine_regime_b_stored_values_within_010_dec_and_told(stored, rung, ch, expected):
    report = stored[(rung, ch)]
    s, o = report.sigma, sigma_observation(report)
    assert (s.regime_active, s.mode, s.R_basis) == (True, "value", REGIME_B_FILM_ARC)
    assert abs(_dec(s.value, expected)) <= 0.1
    assert (o.kind, o.basis, o.admitted) == (VALUE, REGIME_B_FILM_ARC, True)


def test_engine_regime_b_stored_3c_19_lower_bound_admitted_censored(stored):
    """σ ≥ K/max(3.2e5, 3.7e5) ≈ 1.7e-4; today's told 5.7e-4 is 0.5 dec high."""
    report = stored[("3c", 19)]
    s, o = report.sigma, sigma_observation(report)
    assert (s.regime_b_mode, s.regime_b_detail) == (rb.LOWER_BOUND, rb.INTERPHASE_BAND)
    assert abs(_dec(s.regime_sigma_lower, 1.7e-4)) <= 0.1
    assert (o.kind, o.basis, o.R_film_basis, o.admitted) == (
        LOWER_BOUND, REGIME_B_ARC_LOWER, REGIME_B_ARC_LOWER, True)


@pytest.mark.parametrize("rung, ch, expected", [("3b", 3, 1.6e-7), ("3c", 20, 2.0e-7)])
def test_engine_regime_b_stored_insulator_upper_bound_recorded_not_admitted(
        stored, rung, ch, expected):
    report = stored[(rung, ch)]
    s, o = report.sigma, sigma_observation(report)
    assert (s.mode, s.upper_bound_basis) == ("bound", REGIME_B_NO_ARC)
    assert abs(_dec(s.upper_bound, expected)) <= 0.1
    assert (o.kind, o.basis, o.admitted) == (UPPER_BOUND, REGIME_B_NO_ARC, False)
    assert o.classified and o.counts_as_measured                  # never a park


@pytest.mark.parametrize("ch", [9, 13, 14])
def test_engine_regime_b_stored_board2_insulators_never_value(stored, ch):
    s = stored[("b2", ch)].sigma
    assert s.regime_b_mode in (rb.UPPER_BOUND, rb.UNAVAILABLE)
    assert not (s.regime_active and s.mode == "value")
    assert not sigma_observation(stored[("b2", ch)]).admitted


@pytest.mark.parametrize("ch, expected", [(15, 6.32e-4), (16, 5.64e-4)])
def test_engine_regime_b_stored_board2_a_films_route_a_unchanged(stored, ch, expected):
    s = stored[("b2", ch)].sigma
    assert (s.regime, s.regime_b_mode, s.R_basis) == ("A", "", REGIME_A_FIT_RB)
    assert abs(_dec(s.value, expected)) <= 0.01
    assert s.regime_film_basis == BASIS_FILM_RB


def test_engine_regime_b_stored_route_reached_by_real_data(stored):
    """§3.2 counter (``SUBAGENT_RULES`` §3.2): through the engine, real wells enter all
    three stated outcomes and the observation layer admits each as the matrix says."""
    taken = [r for r in stored.values() if r.sigma.regime_active and r.sigma.regime != "A"]
    kinds = {k: sum(1 for r in taken if r.sigma.regime_b_mode == k)
             for k in (rb.VALUE, rb.LOWER_BOUND, rb.UPPER_BOUND)}
    assert kinds[rb.VALUE] >= 8 and kinds[rb.LOWER_BOUND] >= 1 and kinds[rb.UPPER_BOUND] >= 2
    told = [sigma_observation(r) for r in taken]
    assert sum(o.admitted and o.kind == VALUE for o in told) == kinds[rb.VALUE]
    assert not any(o.admitted for o in told if o.kind == UPPER_BOUND)


@pytest.mark.parametrize("rung, ch", [("3b", 1), ("3c", 19), ("3c", 20), ("3b", 7)])
def test_engine_regime_b_stored_b_off_matches_engine_without_b_route(stored, rung, ch):
    """``regime_b`` off with C_cell passed is the HEAD report on real wells (and 3b 7, a C,
    is untouched even with the route live)."""
    with pytest.MonkeyPatch.context() as mp:
        from softae.config import loader

        mp.setattr(loader, "load", lambda *a, **k: {})
        off = _stored_run(rung, ch, B_OFF)
        mp.setattr(rr, "b_routed", lambda verdict: False)
        baseline = _stored_run(rung, ch, B_OFF)
    assert _diff(off, baseline) == set()
    if (rung, ch) == ("3b", 7):
        assert stored[(rung, ch)].sigma.regime == "C"
        assert _diff(stored[(rung, ch)], off) == set()
