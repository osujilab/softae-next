"""Regime-aware σ, slice 1: the screen, the classifier, the regime-A route and its flag.

Spec: ``docs/SubAgent docs/eis_regime_slice1_spec.md`` (rev 2) and the operator rulings of
2026-09-30 at its top. Synthetic spectra come from :mod:`tests.eis_regime_synthetic` and
need nothing on disk. Real spectra live in ``tests/data/eis_regime/``, which is
**gitignored**: every test that reads one skips, naming the missing path, on a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import structlog

from softae.analysis.eis import regime as rg
from softae.analysis.eis import regime_route as rr
from softae.analysis.eis.envelope import InstrumentEnvelope
from softae.analysis.eis.geometry import CellConstant
from tests import eis_regime_synthetic as syn
from tests.eis_regime_golden import DATA, golden_inputs

F = syn.RIG_F
CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")
ENV = InstrumentEnvelope()
BAND_DEC = math.log10(rr.PAIR_RATIO)
#: Criterion 1: a reported regime-A value within 10 % of truth.
CRITERION_1_DEC = math.log10(1.10)

E148 = {  # RH step -> (ch15 file, ch16 file)
    22: ("e148_rh22_20260929T150528Z_ch15.txt", "e148_rh22_20260929T150528Z_ch16.txt"),
    25: ("e148_rh25_20260930T005809Z_ch15.txt", "e148_rh25_20260930T005809Z_ch16.txt"),
    47: ("e148_rh47_20260930T013854Z_ch15.txt", "e148_rh47_20260930T013854Z_ch16.txt"),
    53: ("e148_rh53_20260930T044541Z_ch15.txt", "e148_rh53_20260930T044541Z_ch16.txt"),
}
EMPTY_WELL = "empty_well_20260930T165408Z_ch17.txt"
JAGGED = "jagged_20260820T142833Z_ch25.txt"
OPEN_ARC = "aug_20260825T154521Z_ch9_T50_RH0.txt"
#: B, U, U, U. The C fixture (``OPEN_ARC``) is deliberately absent: its gated fit grinds
#: ~10–45 s per run, and the HEAD golden already pins it flag-off.
NON_A = ("rung3b_20260923T183923Z_ch1.txt", "rung3b_20260923T183923Z_ch3.txt",
         "aug_20260825T154521Z_ch7_T40_RH0.txt", EMPTY_WELL, JAGGED)


def _load(name: str):
    """A real fixture, or a skip naming the missing path (``tests/data`` is gitignored)."""
    path = DATA / name
    if not path.exists():
        pytest.skip(f"real fixture absent: {path} (tests/data/ is gitignored)")
    from softae.analysis.eis_data import EISResult

    return EISResult.load(path)


def _fz(e) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(e.frequency, float),
            np.asarray(e.z_real, float) - 1j * np.asarray(e.z_imag_neg, float))


def _same(x, y) -> bool:
    """Equal, or both NaN — a field-by-field identity that NaN does not defeat."""
    return x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)


def _analyse(eis, *, flag: bool, **kw):
    from softae.analysis.eis.engine import analyze_spectrum

    ch = int(getattr(eis, "channel", 15) or 15)
    return analyze_spectrum(eis, **golden_inputs(ch),
                            regime=rr.RegimeSettings(enabled=flag), **kw)


# ── Classifier ───────────────────────────────────────────────────────────────


def test_classify_regime_synthetic_a_returns_a():
    """≥ 99 % A on Step-1 A with α ≤ 0.8 (measured 216/216); the R_s grid never B or C."""
    step1 = [rg.classify_regime(F, Z).label
             for p, Z in syn.step1_regime_a() if p["alpha"] <= 0.8]
    assert step1.count("A") / len(step1) >= 0.99

    grid = [rg.classify_regime(F, Z).label for _, Z in syn.rs_grid()]
    assert set(grid) <= {"A", "U"}
    assert grid.count("A") / len(grid) >= 0.85          # measured 2509 / 2880


def test_classify_regime_synthetic_a_alpha_090_never_returns_a():
    labels = {rg.classify_regime(F, Z).label
              for p, Z in syn.step1_regime_a() if p["alpha"] == 0.9}
    assert "A" not in labels


def test_classify_regime_synthetic_b_and_c_never_return_a():
    labels = [rg.classify_regime(F, Z).label
              for gen in (syn.step1_regime_b, syn.step1_regime_c) for _, Z in gen()]
    assert "A" not in labels


def test_classify_regime_floor_guard_disabled_open_arc_returns_a(monkeypatch):
    """POSITIVE CONTROL for ``PHI_A_LO`` (the load-bearing −75° guard).

    With the guard widened to −90° the α = 0.9 synthetics — which the test above pins as
    never A — are admitted. If this goes green with the guard in place, the guard is not
    what keeps them out.
    """
    a09 = [Z for p, Z in syn.step1_regime_a() if p["alpha"] == 0.9]
    monkeypatch.setattr(rg, "PHI_A_LO", -90.0)
    assert any(rg.classify_regime(F, Z).label == "A" for Z in a09)


def test_classify_regime_e148_all_four_steps_return_a():
    for files in E148.values():
        for name in files:
            assert rg.classify_regime(*_fz(_load(name))).label == "A", name


def test_classify_regime_empty_well_returns_u_incoherent():
    v = rg.classify_regime(*_fz(_load(EMPTY_WELL)))
    assert (v.label, v.reason) == ("U", "incoherent")


def test_classify_regime_open_arc_fixture_plateau_returns_c():
    v = rg.classify_regime(*_fz(_load(OPEN_ARC)))
    assert (v.label, v.reason) == ("C", "plateau_then_open_arc")


def test_classify_regime_jagged_phase_ch25_returns_u():
    """Operator ruling 2026-09-30: ``20260820T142833Z`` ch25 must be U, not A."""
    v = rg.classify_regime(*_fz(_load(JAGGED)))
    assert (v.label, v.reason) == ("U", "phase_incoherent")


def test_classify_regime_phase_guard_disabled_jagged_ch25_returns_a(monkeypatch):
    """POSITIVE CONTROL: without the phase-roughness guard the jagged ch25 reads A."""
    f, Z = _fz(_load(JAGGED))
    monkeypatch.setattr(rg, "PHASE_ROUGH_MAX", float("inf"))
    assert rg.classify_regime(f, Z).label == "A"


def test_screen_points_hf_artefact_band_drops_above_interior_inductive():
    """[e148]'s shape: capacitive above an inductive run inside the top decade."""
    f = np.array([1e5, 1.3e5, 1.6e5, 2e5])
    Z = np.array([100 - 10j, 100 + 5j, 100 - 3j, 100 - 4j])
    s = rg.screen_points(f, Z)
    assert s.n_hf_band == 3 and list(s.f) == [1e5]


def test_screen_points_rail_negative_real_dropped_not_refused():
    """Rail-adjacent Re Z < 0 is dropped (§3.3c) — counted, never a refusal on its own."""
    Z = syn.regime_a_rs("inband", 3e3, 0.7, 1.0, 0.0, 1e5, 0.0, False, seed=11)
    Z = Z.copy()
    Z[-1] = -abs(Z[-1].real) + 1j * Z[-1].imag
    s = rg.screen_points(F, Z)
    assert s.n_nonphys == 1 and s.n_ok == F.size - 1
    assert rg.classify_regime(F, Z).label == "A"


@pytest.mark.parametrize("f, Z", [
    (np.array([]), np.array([], complex)),
    (np.full(5, np.nan), np.full(5, np.nan + 0j)),
    (np.array([1.0, 2.0, 3.0]), np.array([1 - 1j, 1 - 1j, 1 - 1j])),
    (np.logspace(0, 5, 30), np.full(30, 1 + 1j)),                    # all inductive
    (np.logspace(0, 5, 30), np.zeros(30, complex)),
])
def test_screen_points_degenerate_inputs_return_u_without_raising(f, Z):
    assert rg.classify_regime(f, Z).label == "U"


def test_classify_regime_sweep_order_invariant():
    Z = syn.regime_a_rs("inband", 3e3, 0.7, 1.0, 1e4, 1e5, 5e-6, True, seed=12)
    a = rg.classify_regime(F, Z)
    b = rg.classify_regime(F[::-1], Z[::-1])
    perm = np.random.default_rng(3).permutation(F.size)
    c = rg.classify_regime(F[perm], Z[perm])
    assert a.label == b.label == c.label == "A"
    assert a.features == b.features == c.features


# ── Regime-A route on the synthetic subset ─────────────────────────────────────

#: Both shapes × both corner frequencies × R_s ∈ {0, 1e4, 1e5} × artefact ∈ {0, 1}:
#: 24 spectra, R_b = 1e5, α_el 0.7, ideal C_geo, L = 5 µH.
SUBSET = [(sh, k, Rs, art) for sh in ("inband", "compressed") for k in (0, 1)
          for Rs in (0.0, 1e4, 1e5) for art in (0, 1)]
RB = 1e5


@pytest.fixture(scope="module")
def subset_outcomes():
    out = []
    for j, (sh, k, Rs, art) in enumerate(SUBSET):
        Z = syn.regime_a_rs(sh, syn.FC[sh][k], 0.7, 1.0, Rs, RB, 5e-6, art, seed=700 + j)
        v = rg.classify_regime(F, Z)
        assert v.is_a, (sh, k, Rs, art, v.reason)
        est = rr.regime_a_estimates(v)
        pl = rr.plateau_decision(v, envelope=ENV, cell=CELL, tand_headroom_mult=3.0)
        out.append(dict(shape=sh, Rs=Rs, art=art, est=est, plateau=pl.mode,
                        kind=rr.regime_a_kind(est, pl.mode)[0],
                        kind_nospan=rr.regime_a_kind(est, pl.mode, span_min=0.0)[0]))
    return out


def test_regime_a_route_inband_value_within_010_dec(subset_outcomes):
    """Criterion 1 (10 % = 0.041 dec) on the in-band values of the subset."""
    vals = [o for o in subset_outcomes if o["shape"] == "inband" and o["kind"] == "value"]
    assert len(vals) >= 10
    for o in vals:
        assert abs(math.log10(o["est"].R_fit / RB)) <= CRITERION_1_DEC, o


def test_regime_a_route_passive_bound_covers_truth(subset_outcomes):
    for o in subset_outcomes:
        assert o["est"].R_b_min <= RB, o


def test_regime_a_route_span_gate_disabled_compressed_artefact_gives_false_value(
        subset_outcomes):
    """POSITIVE CONTROL for the span gate: without it a compressed + artefact spectrum
    reports a value outside the ruled band; with it, none does."""
    def false_value(o, key):
        return (o[key] == "value"
                and abs(math.log10(o["est"].R_fit / RB)) > BAND_DEC)

    assert any(false_value(o, "kind_nospan") for o in subset_outcomes
               if o["shape"] == "compressed" and o["art"])
    assert not any(false_value(o, "kind") for o in subset_outcomes)


def test_regime_a_route_pair_disagreement_never_value(subset_outcomes):
    for o in subset_outcomes:
        if not (o["est"].pair_ratio <= rr.PAIR_RATIO):
            assert o["kind"] != "value", o


def test_regime_a_kind_agreement_disabled_admits_disagreeing_pair(monkeypatch):
    """POSITIVE CONTROL for the pair test: widen the band and a ×2 pair becomes a value."""
    est = rr.RegimeAEstimates(R_fit=2e5, R_mf=1e5, R_b_min=9e4, R_foot=2.2e5, span=2.0)
    assert rr.regime_a_kind(est, "value")[0] == "bound"
    monkeypatch.setattr(rr, "PAIR_RATIO", float("inf"))
    assert rr.regime_a_kind(est, "value")[0] == "value"


def test_regime_a_route_compressed_rs_gt_rb_never_false_value():
    """R_s > R_b on the compressed sub-shape: a value, where one is reported, is right.

    **Not "never a value"** (the spec's §7 name): with R_s > R_b and no artefact the
    series lifts the HF end onto the real axis, the arc becomes visible above the foot,
    and the route reports it correctly (≤ 0.02 dec on these seeds). What must never
    happen is a value outside the ruled band — the full-grid rate is in the report.
    """
    for j, (k, Rs, Rb, art) in enumerate(
            [(k, Rs, Rb, art) for k in (0, 1) for Rs, Rb in ((1e4, 1e3), (1e5, 1e4),
                                                             (1e5, 1e3), (3e4, 1e4))
             for art in (0, 1)]):
        Z = syn.regime_a_rs("compressed", syn.FC["compressed"][k], 0.7, 1.0, Rs, Rb,
                            0.0, art, seed=900 + j)
        v = rg.classify_regime(F, Z)
        if not v.is_a:
            continue
        est = rr.regime_a_estimates(v)
        pl = rr.plateau_decision(v, envelope=ENV, cell=CELL, tand_headroom_mult=3.0)
        if rr.regime_a_kind(est, pl.mode)[0] == "value":
            assert abs(math.log10(est.R_fit / Rb)) <= BAND_DEC, (k, Rs, Rb, art)


def test_regime_a_route_unmeasured_phase_floor_never_value(subset_outcomes):
    """An unmeasured ε makes the plateau decision ``bound_unqualified`` — never a value."""
    env = InstrumentEnvelope(phase_noise_measured=False)
    sh, k, Rs, art = SUBSET[0]
    Z = syn.regime_a_rs(sh, syn.FC[sh][k], 0.7, 1.0, Rs, RB, 5e-6, art, seed=700)
    v = rg.classify_regime(F, Z)
    pl = rr.plateau_decision(v, envelope=env, cell=CELL, tand_headroom_mult=3.0)
    assert pl.mode == "bound_unqualified"
    assert subset_outcomes[0]["kind"] == "value"                # measured ε: a value
    assert rr.regime_a_kind(rr.regime_a_estimates(v), pl.mode)[0] != "value"


# ── Settings ─────────────────────────────────────────────────────────────────


def test_regime_settings_missing_key_off():
    assert rr.regime_settings({}).enabled is False
    assert rr.regime_settings({"pregate": {}}).enabled is False


@pytest.mark.parametrize("raw", ["false", "true", 1, "yes"])
def test_regime_settings_string_false_off(raw):
    assert rr.regime_settings({"regime_aware": raw}).enabled is False


def test_regime_settings_true_arms():
    assert rr.regime_settings({"regime_aware": True}).enabled is True
    assert rr.regime_settings({"regime_aware": False}).enabled is False


# ── Engine: shadow, flag, and the real fixtures ─────────────────────────────────


def _shadow_events(eis, *, flag: bool):
    with structlog.testing.capture_logs() as logs:
        report = _analyse(eis, flag=flag)
    return report, logs


def test_regime_shadow_logged_when_flag_off():
    Z = syn.regime_a_rs("inband", 3e3, 0.7, 1.0, 1e4, 1e5, 5e-6, False, seed=21)
    report, logs = _shadow_events(syn.as_eis(F, Z), flag=False)
    shadow = [e for e in logs if e["event"] == "sigma_regime_shadow"]
    assert len(shadow) == 1
    ev = shadow[0]
    assert ev["regime"] == "A" and ev["route_active"] is False
    assert ev["regime_mode"] == "value" and ev["sigma_value"] == ev["regime_sigma"]
    assert ev["today_mode"] == report.sigma.mode
    assert report.sigma.regime == "A" and report.sigma.regime_active is False


def test_regime_a_route_skips_arc_open_refusal():
    """On the A route refusal (a) is bypassed: no ``eis_arc_open_bound`` announcement and
    the reported σ is the route's. Flag off, the same spectrum is refused today."""
    Z = syn.regime_a_rs("inband", 1e4, 0.8, 0.85, 1e4, 1e5, 5e-6, True, seed=501)
    eis = syn.as_eis(F, Z)
    off, off_logs = _shadow_events(eis, flag=False)
    on, on_logs = _shadow_events(eis, flag=True)
    assert any(e["event"] == "eis_arc_open_bound" for e in off_logs)
    assert off.sigma.is_bound
    assert not any(e["event"] == "eis_arc_open_bound" for e in on_logs)
    assert on.sigma.regime_active and on.sigma.mode == "value"
    assert on.sigma.R_basis == rr.REGIME_A_FIT_RB


def test_regime_flag_on_non_a_fixtures_unchanged():
    """Criterion 4: B, C and U spectra report identically in both flag states."""
    import dataclasses

    synth = [syn.as_eis(F, Z) for gen in (syn.step1_regime_b, syn.step1_regime_c)
             for _, Z in list(gen())[:1]]
    real = [_load(n) for n in NON_A if (DATA / n).exists()]
    for eis in synth + real:
        off, on = _analyse(eis, flag=False), _analyse(eis, flag=True)
        assert on.sigma.regime != "A" and not on.sigma.regime_active
        a, b = dataclasses.asdict(off.sigma), dataclasses.asdict(on.sigma)
        assert {k for k in a if not _same(a[k], b[k])} == set()


@pytest.fixture(scope="module")
def e148_reports():
    if not all((DATA / n).exists() for files in E148.values() for n in files):
        pytest.skip(f"[e148] fixtures absent under {DATA} (tests/data/ is gitignored)")
    return {(rh, ch): _analyse(_load(name), flag=True)
            for rh, files in E148.items() for ch, name in zip((15, 16), files)}


def test_regime_a_route_e148_22_25_pairs_values_agree_within_015_dec(e148_reports):
    """Criterion 2: the two channels of one board agree at each in-band step."""
    for rh in (22, 25):
        a, b = e148_reports[(rh, 15)].sigma, e148_reports[(rh, 16)].sigma
        assert a.mode == b.mode == "value" and a.regime_active and b.regime_active
        assert abs(math.log10(a.value / b.value)) <= 0.15, rh


def test_regime_a_route_e148_47_53_no_value(e148_reports):
    """The compressed steps carry no value — which is why criterion 3 is open."""
    for rh in (47, 53):
        for ch in (15, 16):
            s = e148_reports[(rh, ch)].sigma
            assert s.regime == "A" and s.mode != "value", (rh, ch)
            if s.mode == "unavailable":
                assert s.regime_reason not in ("", rr.SERIES_NOT_SEPARABLE), (rh, ch)
                assert s.regime_route_detail, (rh, ch)
                assert s.regime_sigma_lower > 0
                assert s.R_basis == rr.REGIME_A_FOOT_LOWER
                assert s.R_reported_ohm == s.regime_R_foot_ohm > 0


@pytest.mark.xfail(strict=True, reason="criterion 3 (σ monotone in RH over four steps) is "
                   "OPEN by operator ruling Q3: it needs an in-band RH series")
def test_regime_a_route_e148_sigma_monotone_in_rh_criterion_3(e148_reports):
    for ch in (15, 16):
        sig = [e148_reports[(rh, ch)].sigma for rh in sorted(E148)]
        assert all(s.mode == "value" for s in sig)
        vals = [s.value for s in sig]
        assert vals == sorted(vals)


#: Measured criterion-1 misses on the full R_s grid (65 of 1086 in-band values, 6 %):
#: grid index -> its parameters, regenerated with the grid's own seed. They are NOT at
#: L = 20 µH, R_b = 1e3 (the brief's expectation: no value there misses); they sit under
#: the HF artefact with a dispersive CPE_geo (n_g 0.85), plus two at R_b = 1e3, R_s = 3e4.
CRITERION_1_MISSES = (2299, 2365, 2371, 2714, 2832)


@pytest.mark.xfail(strict=True, reason="known synthetic misses: criterion 1 (10 %) on "
                   "in-band values under the HF artefact with n_g 0.85, and at R_b = 1e3 "
                   "with R_s = 3e4 — open, pinned rather than relaxed")
def test_regime_a_route_inband_known_misses_value_within_010_dec_criterion_1():
    import itertools

    grid = list(itertools.product(*syn.RS_GRID_AXES))
    errs = []
    for idx in CRITERION_1_MISSES:
        shape, k, a, ng, Rs, Rb, L, art = grid[idx]
        Z = syn.regime_a_rs(shape, syn.FC[shape][k], a, ng, Rs, Rb, L, art,
                            seed=20260930 + idx)
        v = rg.classify_regime(F, Z)
        est = rr.regime_a_estimates(v)
        pl = rr.plateau_decision(v, envelope=ENV, cell=CELL, tand_headroom_mult=3.0)
        assert v.is_a and rr.regime_a_kind(est, pl.mode)[0] == "value", idx
        errs.append(abs(math.log10(est.R_fit / Rb)))
    assert max(errs) <= CRITERION_1_DEC


def test_regime_route_reached_by_real_fixtures():
    """§3.2: real data enters every branch of the A route, counted."""
    names = [n for files in E148.values() for n in files] + [
        "a_inband_20260821T122736Z_ch19.txt", "a_compressed_20260818T002915Z_ch29.txt"]
    seen: dict[str, int] = {}
    for name in names:
        v = rg.classify_regime(*_fz(_load(name)))
        assert v.is_a, name
        est = rr.regime_a_estimates(v)
        pl = rr.plateau_decision(v, envelope=ENV, cell=CELL, tand_headroom_mult=3.0)
        kind = rr.regime_a_kind(est, pl.mode)[0]
        seen[kind] = seen.get(kind, 0) + 1
    assert set(seen) == {"value", "bound", "unavailable"}, seen


# ── R7, slice 1a: the lower-bound basis, R_foot, and the route's own why ─────────────


def _route(cell, *pick):
    """``_regime_a_sigma`` on one synthetic A's real verdict; ``(verdict, sigma, ceiling)``."""
    from softae.analysis.eis.engine import _regime_a_sigma
    from softae.analysis.eis.report import SigmaReport

    *args, seed = pick
    v = rg.classify_regime(F, syn.regime_a_rs(*args, seed=seed))
    assert v.is_a
    sigma, ceiling, _ = _regime_a_sigma(v, SigmaReport(), cell=cell, envelope=ENV,
                                        tand_headroom_mult=3.0, R_engine=float("nan"))
    return v, sigma, ceiling


#: Compressed, R_s = R_b: the route can state only ``σ ≥ K/R_foot``.
COMPRESSED = ("compressed", 1.5e5, 0.6, 0.85, 1e5, 1e5, 20e-6, True, 503)
INBAND = ("inband", 3e3, 0.7, 1.0, 0.0, 1e5, 0.0, False, 500)


def test_regime_a_sigma_lower_bound_reports_foot_basis_and_r_foot():
    v, s, ceiling = _route(CELL, *COMPRESSED)
    assert s.mode == "unavailable" and s.regime_mode == "unavailable"
    assert s.R_basis == rr.REGIME_A_FOOT_LOWER
    assert s.R_reported_ohm == s.regime_R_foot_ohm > 0
    assert s.regime_sigma_lower == pytest.approx(CELL.sigma(s.regime_R_foot_ohm))
    assert ceiling.reason == rr.SERIES_NOT_SEPARABLE


def test_regime_a_sigma_lower_bound_keeps_classifier_reason_route_detail_separate():
    v, s, _ = _route(CELL, *COMPRESSED)
    assert s.regime_reason == v.reason != rr.SERIES_NOT_SEPARABLE
    assert s.regime_route_detail == rr.PAIR_UNAVAILABLE      # no circle: R_mf is NaN


def test_regime_a_sigma_value_carries_r_foot_and_detail():
    _, s, _ = _route(CELL, *INBAND)
    assert s.mode == "value" and s.R_basis == rr.REGIME_A_FIT_RB
    assert s.regime_route_detail == "resolved_in_band"
    assert s.regime_R_foot_ohm > 0


@pytest.mark.parametrize("pick", [COMPRESSED, INBAND], ids=["compressed", "inband"])
def test_regime_a_sigma_no_cell_constant_states_no_lower_bound(pick):
    v, s, ceiling = _route(None, *pick)
    assert s.mode == "unavailable" and s.regime_route_detail == rr.NO_CELL_CONSTANT
    assert s.regime_reason == v.reason
    assert s.R_basis != rr.REGIME_A_FOOT_LOWER and s.R_reported_ohm != s.R_reported_ohm
    assert s.regime_sigma_lower != s.regime_sigma_lower
    assert ceiling.reason == "no cell constant"
    assert rr.route_basis(s.regime_mode, s.regime_sigma_lower) == ""


def test_route_basis_unavailable_names_foot_only_with_a_number():
    assert rr.route_basis("unavailable", 1e-4) == rr.REGIME_A_FOOT_LOWER
    assert rr.route_basis("unavailable") == ""
    assert rr.route_basis("bound", 1e-4) == rr.REGIME_A_PASSIVE
    assert rr.route_basis("value") == rr.REGIME_A_FIT_RB
    assert rr.route_basis("") == ""


def test_regime_shadow_flag_off_carries_r_foot_and_detail_not_basis():
    """The new shadow fields flow flag-off; the reported basis stays today's."""
    *args, seed = COMPRESSED
    off = _analyse(syn.as_eis(F, syn.regime_a_rs(*args, seed=seed)), flag=False)
    s = off.sigma
    assert s.regime == "A" and not s.regime_active
    assert s.regime_route_detail == rr.PAIR_UNAVAILABLE and s.regime_R_foot_ohm > 0
    assert s.R_basis != rr.REGIME_A_FOOT_LOWER


# ── Arming wave (eis_regime_arming_review.md F4, F5) ─────────────────────────────────


@pytest.mark.parametrize("R_fit, R_mf", [(2e5, float("nan")), (float("nan"), 1e5),
                                         (2e5, -3e4)], ids=["mf_nan", "fit_nan", "mf_neg"])
def test_regime_a_kind_pair_not_comparable_detail_pair_unavailable(R_fit, R_mf):
    """F4: *could not compare* is not spelled *compared and disagreed* (§3.1(a))."""
    for R_b_min, kind in ((9e4, "bound"), (-8e3, "unavailable")):
        est = rr.RegimeAEstimates(R_fit=R_fit, R_mf=R_mf, R_b_min=R_b_min, R_foot=2.2e5,
                                  span=2.0)
        assert rr.regime_a_kind(est, "value") == (kind, rr.PAIR_UNAVAILABLE)


def test_regime_a_kind_pair_compared_outside_band_detail_pair_disagrees():
    est = rr.RegimeAEstimates(R_fit=2e5, R_mf=1e5, R_b_min=9e4, R_foot=2.2e5, span=2.0)
    assert rr.regime_a_kind(est, "value") == ("bound", "pair_disagrees")


def test_regime_a_sigma_value_cross_check_against_route_partner_not_today():
    """F5: ``cross_check_pct`` is R_fit against the route's own R_mf; today's withdrawn
    ``model_free_R_ohm`` (1/max Re Y) must not enter it, whatever it holds."""
    from softae.analysis.eis.engine import _regime_a_sigma
    from softae.analysis.eis.report import SigmaReport

    *args, seed = INBAND
    v = rg.classify_regime(F, syn.regime_a_rs(*args, seed=seed))
    est = rr.regime_a_estimates(v)
    expected = abs(est.R_fit - est.R_mf) / est.R_fit * 100.0
    for today_mf in (float("nan"), 1.0, 1e9):
        s, _, _ = _regime_a_sigma(v, SigmaReport(model_free_R_ohm=today_mf), cell=CELL,
                                  envelope=ENV, tand_headroom_mult=3.0,
                                  R_engine=float("nan"))
        assert s.mode == "value"
        assert s.cross_check_pct == pytest.approx(expected), today_mf
    assert 0 < expected < 50.0          # a value implies the pair agreed within x1.5


def test_regime_settings_repo_config_armed_as_boolean():
    """Operator ruling 2026-10-01: the repo config arms the route, as a TOML boolean
    (a string ``"true"`` would leave it off — see ``test_regime_settings_string_false_off``)."""
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "softae_config.toml")
                        .read_text(encoding="utf-8"))
    assert cfg["eis"]["regime_aware"] is True
    assert rr.regime_settings(cfg["eis"]).enabled is True
