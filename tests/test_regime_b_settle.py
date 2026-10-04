"""Settle under ``[eis] regime_b`` (slice 2 wave 2, spec §4; operator ruling Q8).

With ``regime_aware`` AND ``regime_b`` on, a settle round tracks the *film* resistance the
gated engine's route identified (``regime_film_R_ohm`` under ``regime_film_basis``); a
round whose film is unidentified is unjudgeable and is **never** answered with 1/R₁, which
on regime B is the flat series element. With either flag off the round is exactly
[a265]'s legacy 1/R₁, which is what the identity tests pin.

Two layers: a spy on ``_spectrum_report_from_raw`` returning reports built to make the
two answers (film R vs R₁) different numbers, so a fallback cannot hide; and the real
engine on known-truth two-feature spectra, so the branch is shown reachable (§3.2).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import softae.core.autonomous_wiring as wiring
from softae.analysis.eis.regime_route import RegimeSettings
from softae.analysis.equilibration import (
    BASIS_ABSENT,
    BASIS_FILM_ARC,
    BASIS_FILM_RB,
    BASIS_FILM_UNIDENTIFIED,
    BASIS_FIT_FAILED,
    BASIS_FITTED,
    EXCLUDED_SIGMA_NULL,
    RoundFit,
    settle_check,
)

R1_SERIES = 1.0e5          # what a legacy fit's R₁ follows on B: the series element
R_FILM = 1.0e8             # the film arc's resistance
C_CELL = 2.5e-10
ON = RegimeSettings(enabled=True, regime_b=True)
OFF_STATES = [None, RegimeSettings(enabled=True), RegimeSettings(enabled=False),
              RegimeSettings(enabled=False, regime_b=True)]


def _report(*, film_basis: str = "", film_r: float = float("nan"),
            r1: float = R1_SERIES, success: bool = True):
    """A report carrying BOTH a successful R₁ fit and the film fields."""
    fit = SimpleNamespace(success=success, R1=r1, n_points_dropped=0,
                          arc_closure=SimpleNamespace(state="closed"))
    sigma = SimpleNamespace(mode="value", regime_film_basis=film_basis,
                            regime_film_R_ohm=film_r)
    quality = SimpleNamespace(verdict=SimpleNamespace(value="accept"))
    return SimpleNamespace(fit=fit, sigma=sigma, quality=quality)


@pytest.fixture
def spy(monkeypatch):
    """Patch the one raw → report hop; ``spy.reports[engine]`` is what each engine returns."""
    calls: list[dict] = []
    reports: dict[str, object] = {}

    def fake(raw, **kwargs):
        calls.append(kwargs)
        return reports.get(kwargs.get("engine"))

    monkeypatch.setattr(wiring, "_spectrum_report_from_raw", fake)
    return SimpleNamespace(calls=calls, reports=reports)


def _fits(regime, channels=(7,), c_cell=lambda ch: C_CELL):
    return wiring.settle_round_fits({ch: object() for ch in channels}, list(channels),
                                    thickness_for=lambda ch: 50.0, regime=regime,
                                    cell_capacitance_for=c_cell)


# ── regime_b on: the film resistance is the tracked quantity ────────────────────

@pytest.mark.parametrize("basis", [BASIS_FILM_ARC, BASIS_FILM_RB])
def test_settle_round_fits_regime_b_film_basis_tracks_film_resistance(spy, basis):
    spy.reports["gated"] = _report(film_basis=basis, film_r=R_FILM)
    spy.reports["legacy"] = _report()

    fit, = _fits(ON)

    assert fit.sigma == pytest.approx(1.0 / R_FILM)
    assert fit.basis == basis and fit.r1_ohms is None
    assert fit.arc_state == "closed" and fit.sigma_mode == "value"
    assert [c["engine"] for c in spy.calls] == ["gated"]
    assert spy.calls[0]["regime"] is ON and spy.calls[0]["cell_capacitance"] == C_CELL
    assert spy.calls[0]["thickness_um"] == 50.0


@pytest.mark.parametrize("report", [
    _report(film_basis=BASIS_FILM_UNIDENTIFIED),
    _report(film_basis=BASIS_FILM_ARC, film_r=float("nan")),
    _report(film_basis=BASIS_FILM_ARC, film_r=-1.0),
    _report(film_basis="film_something_new", film_r=R_FILM),
], ids=["unidentified", "arc_without_r", "arc_negative_r", "unknown_token"])
def test_settle_round_fits_regime_b_unidentified_unjudgeable_never_r1(spy, report):
    """The fallback ban: R₁ is on the gated report AND the legacy one, and neither is read."""
    spy.reports["gated"] = report
    spy.reports["legacy"] = _report()

    fit, = _fits(ON)

    assert fit.sigma is None
    assert fit.basis == BASIS_FILM_UNIDENTIFIED
    assert math.isnan(fit.r1_ohms)
    assert [c["engine"] for c in spy.calls] == ["gated"]


def test_settle_round_fits_regime_b_report_unavailable_unjudgeable_never_r1(spy):
    spy.reports["legacy"] = _report()                        # gated → None

    fit, = _fits(ON)

    assert (fit.sigma, fit.basis) == (None, BASIS_FIT_FAILED)
    assert math.isnan(fit.r1_ohms)
    assert [c["engine"] for c in spy.calls] == ["gated"]


def test_settle_round_fits_regime_b_no_film_route_takes_legacy_r1(spy):
    """C, bad-data U, unclassified: no film route, so today's 1/R₁ round (spec §4 table)."""
    spy.reports["gated"] = _report(film_basis="")
    spy.reports["legacy"] = _report()

    fit, = _fits(ON)

    assert fit.sigma == pytest.approx(1.0 / R1_SERIES)
    assert (fit.basis, fit.r1_ohms) == (BASIS_FITTED, R1_SERIES)
    assert [c["engine"] for c in spy.calls] == ["gated", "legacy"]
    assert spy.calls[1]["cell_capacitance"] == C_CELL


def test_settle_round_fits_regime_b_absent_round_not_analysed(spy):
    fit, = wiring.settle_round_fits({}, [7], regime=ON,
                                    cell_capacitance_for=lambda ch: C_CELL)
    assert (fit.sigma, fit.r1_ohms, fit.basis) == (None, None, BASIS_ABSENT)
    assert spy.calls == []


def test_settle_round_fits_regime_b_unmeasured_c_cell_passes_none(spy):
    spy.reports["gated"] = _report(film_basis=BASIS_FILM_UNIDENTIFIED)

    _fits(ON, c_cell=lambda ch: None)
    _fits(ON, c_cell=None)

    assert [c["cell_capacitance"] for c in spy.calls] == [None, None]


def test_settle_unidentified_round_excluded_as_sigma_null_not_absent(spy):
    """NaN, not None, in ``r1_ohms``: the sweep happened, so the criterion must say
    *sigma_null* — an all-None round reads as *absent*, a different finding."""
    spy.reports["gated"] = _report(film_basis=BASIS_FILM_UNIDENTIFIED)
    window = [_fits(ON) for _ in range(3)]

    check = settle_check(window, min_channels=1, board_minimum=None)

    assert check.excluded == {7: EXCLUDED_SIGMA_NULL}


# ── regime_b off: byte-identical to HEAD ────────────────────────────────────────

@pytest.mark.parametrize("regime", OFF_STATES,
                         ids=["none", "aware_only", "all_off", "b_without_aware"])
def test_settle_round_fits_regime_b_off_ignores_film_fields_legacy_r1(spy, regime):
    """Every legacy report here ALSO carries a film arc, so reading it would show."""
    spy.reports["legacy"] = _report(film_basis=BASIS_FILM_ARC, film_r=R_FILM)
    spy.reports["gated"] = _report(film_basis=BASIS_FILM_ARC, film_r=R_FILM)

    fit, = _fits(regime)

    assert fit == RoundFit(channel=7, sigma=1.0 / R1_SERIES, r1_ohms=R1_SERIES,
                           basis=BASIS_FITTED, arc_state="closed", n_points_dropped=0,
                           sigma_mode="value", quality_verdict="accept")
    assert [c["engine"] for c in spy.calls] == ["legacy"]


def _closed_arc_raw():
    from tests.eis_synthetic import reference_spectrum
    return _eis_raw(*reference_spectrum(Q=0.0, R_series=500.0, R_bulk=50000.0,
                                        C_par=1e-9))


def _eis_raw(f, Z):
    return [np.column_stack([f, np.abs(Z), np.degrees(np.angle(Z)), Z.real, -Z.imag])]


@pytest.mark.parametrize("regime", OFF_STATES[1:], ids=["aware_only", "all_off",
                                                        "b_without_aware"])
def test_settle_round_fits_regime_b_off_real_spectrum_matches_no_regime(regime):
    """The real engine: with regime_b off the round equals the pre-slice call's."""
    raws = {7: _closed_arc_raw()}
    head, = wiring.settle_round_fits(raws, [7])
    now, = wiring.settle_round_fits(raws, [7], regime=regime,
                                    cell_capacitance_for=lambda ch: C_CELL)
    assert now == head and now.basis == BASIS_FITTED


# ── The real engine: the film route is reachable (§3.2) ─────────────────────────

def _two_feature_raw(**kw):
    from tests import eis_two_feature_synthetic as tf

    t = tf.TwoFeature(**kw)
    return _eis_raw(*tf.spectrum(t, seed=3)), t


@pytest.mark.parametrize("R_f", [1e7, 1e8])
def test_settle_round_fits_regime_b_real_engine_film_arc_within_010_dec(R_f):
    """A film evolving a decade under a fixed series element: each round reads the film."""
    from tests import eis_two_feature_synthetic as tf

    raw, truth = _two_feature_raw(R_f=R_f)
    fit, = wiring.settle_round_fits({5: raw}, [5], thickness_for=lambda ch: 150.0,
                                    regime=ON, cell_capacitance_for=lambda ch: tf.C_CELL)

    assert fit.basis == BASIS_FILM_ARC, fit
    assert fit.r1_ohms is None
    assert abs(math.log10(1.0 / fit.sigma) - math.log10(truth.R_f)) <= 0.1


def test_settle_round_fits_regime_b_real_engine_no_c_cell_unjudgeable():
    raw, _truth = _two_feature_raw(R_f=1e7)

    fit, = wiring.settle_round_fits({5: raw}, [5], thickness_for=lambda ch: 150.0,
                                    regime=ON, cell_capacitance_for=lambda ch: None)

    assert (fit.sigma, fit.basis) == (None, BASIS_FILM_UNIDENTIFIED)


# ── The basis-change event ──────────────────────────────────────────────────────

def test_settle_basis_watch_emits_once_per_change_ignoring_sigma_less_rounds():
    seen: list[dict] = []
    watch = wiring.SettleBasisWatch(lambda **kw: seen.append(kw))
    rounds = [
        [RoundFit(channel=3, sigma=1e-5, basis=BASIS_FITTED)],
        [RoundFit(channel=3, sigma=None, basis=BASIS_FILM_UNIDENTIFIED)],
        [RoundFit(channel=3, sigma=1e-8, basis=BASIS_FILM_ARC)],
        [RoundFit(channel=3, sigma=2e-8, basis=BASIS_FILM_ARC)],
    ]
    for fits in rounds:
        assert watch(fits) is fits

    assert seen == [{"channel": 3, "previous": BASIS_FITTED, "basis": BASIS_FILM_ARC}]


def test_settle_basis_watch_single_basis_silent():
    seen: list[dict] = []
    watch = wiring.SettleBasisWatch(lambda **kw: seen.append(kw))
    for _ in range(4):
        watch([RoundFit(channel=3, sigma=1e-5, basis=BASIS_FITTED),
               RoundFit(channel=4, sigma=None, basis=BASIS_FIT_FAILED)])
    assert seen == []


# ── eis_validate_hold reads the film bases as fits ([e158] L4 note) ─────────────

def test_validate_hold_n_modelled_counts_film_bases():
    from softae.tools.eis_validate_hold import _n_modelled

    fits = [RoundFit(channel=1, sigma=1e-5, basis=BASIS_FITTED),
            RoundFit(channel=2, sigma=1e-8, basis=BASIS_FILM_ARC),
            RoundFit(channel=3, sigma=1e-7, basis=BASIS_FILM_RB),
            RoundFit(channel=4, sigma=None, basis=BASIS_FILM_UNIDENTIFIED),
            RoundFit(channel=5, sigma=None, basis=BASIS_FIT_FAILED)]
    assert _n_modelled(fits) == 3


def test_validate_hold_announce_basis_film_bases_not_no_fit(capsys):
    from softae.tools.eis_validate_hold import _announce_basis

    fits = [RoundFit(channel=2, sigma=1e-8, basis=BASIS_FILM_ARC),
            RoundFit(channel=3, sigma=1e-7, basis=BASIS_FILM_RB),
            RoundFit(channel=4, sigma=None, r1_ohms=float("nan"),
                     basis=BASIS_FILM_UNIDENTIFIED),
            RoundFit(channel=5, sigma=None, r1_ohms=float("nan"),
                     basis=BASIS_FIT_FAILED)]
    _announce_basis(fits, None, 1, "simpleSalt", {})
    out = capsys.readouterr().out

    assert "ch2" not in out and "ch3" not in out
    assert "ch4 NO FILM ARC" in out and "ch4 NO FIT" not in out
    assert "ch5 NO FIT" in out
