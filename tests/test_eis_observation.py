"""``sigma_observation``: one SpectrumReport → value / lower bound / upper bound (R3, R7).

Reports are built from the real ``SigmaReport`` / ``SpectrumReport`` constructors; the
last section runs the engine itself so the mapping is checked on a report the production
path builds, not only on hand-made ones (``SUBAGENT_RULES`` §3.2).
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest

from softae.analysis.eis import observation as ob
from softae.analysis.eis.report import (
    REGIME_A_FIT_RB,
    REGIME_A_FOOT_LOWER,
    REGIME_A_PASSIVE,
    SigmaReport,
    SpectrumReport,
)

NAN = float("nan")


def _report(ok: bool = True, **sigma) -> SpectrumReport:
    quality = None if ok else SimpleNamespace(ok=False)
    return SpectrumReport(engine="gated", sigma=SigmaReport(**sigma), quality=quality)


#: Route-A states as the engine builds them (``engine._regime_a_sigma``).
A_VALUE = dict(mode="value", value=2e-4, R_reported_ohm=1e5, R_basis=REGIME_A_FIT_RB,
               regime="A", regime_mode="value", regime_sigma=2e-4,
               regime_sigma_lower=1e-4, regime_R_foot_ohm=2e5, regime_active=True)
A_BOUND = dict(mode="bound", upper_bound=5e-4, upper_bound_basis=REGIME_A_PASSIVE,
               R_reported_ohm=4e4, R_basis=REGIME_A_PASSIVE, regime="A",
               regime_mode="bound", regime_sigma=5e-4, regime_sigma_lower=1e-4,
               regime_R_foot_ohm=2e5, regime_active=True)
A_LOWER = dict(mode="unavailable", R_reported_ohm=2e5, R_basis=REGIME_A_FOOT_LOWER,
               regime="A", regime_mode="unavailable", regime_sigma_lower=1e-4,
               regime_R_foot_ohm=2e5, regime_active=True)
LEGACY_BOUND = dict(mode="bound", upper_bound=3e-7, upper_bound_basis="loss_at_numerator",
                    R_reported_ohm=8e5, R_basis="split_bulk", regime="A",
                    regime_sigma_lower=1e-4, regime_R_foot_ohm=2e5)


# ── The mapping, row by row ─────────────────────────────────────────────────────


def test_sigma_observation_value_admitted_with_r_basis():
    o = ob.sigma_observation(_report(**A_VALUE))
    assert (o.kind, o.reported, o.lower) == (ob.VALUE, 2e-4, None)
    assert (o.basis, o.R_film_ohm, o.R_film_basis) == (REGIME_A_FIT_RB, 1e5, REGIME_A_FIT_RB)
    assert o.admitted and o.classified and o.regime == "A"


def test_sigma_observation_legacy_value_flag_off_admitted():
    o = ob.sigma_observation(_report(mode="value", value=1e-5, R_reported_ohm=3e6))
    assert o.kind == ob.VALUE and o.admitted and o.basis == "split_bulk"
    assert not o.classified and o.regime == ""


def test_sigma_observation_legacy_upper_bound_recorded_not_admitted():
    o = ob.sigma_observation(_report(**LEGACY_BOUND))
    assert (o.kind, o.reported) == (ob.UPPER_BOUND, 3e-7)
    assert o.basis == "loss_at_numerator" and o.R_film_basis == "split_bulk"
    assert o.lower is None            # no interval without the active route
    assert not o.admitted and o.classified


@pytest.mark.parametrize("mode", ["bound", "bound_unqualified"])
def test_sigma_observation_route_a_passive_bound_admitted_with_lower(mode):
    o = ob.sigma_observation(_report(**{**A_BOUND, "mode": mode}))
    assert (o.kind, o.reported, o.lower) == (ob.UPPER_BOUND, 5e-4, 1e-4)
    assert (o.basis, o.R_film_ohm) == (REGIME_A_PASSIVE, 4e4)
    assert o.admitted


def test_sigma_observation_active_bound_non_route_basis_not_admitted():
    """Admission keys on the basis too: an active report whose ceiling is not route A's."""
    o = ob.sigma_observation(_report(**{**A_BOUND, "upper_bound_basis": "arc_open"}))
    assert o.kind == ob.UPPER_BOUND and not o.admitted


def test_sigma_observation_foot_lower_bound_admitted():
    o = ob.sigma_observation(_report(**A_LOWER))
    assert (o.kind, o.reported, o.lower) == (ob.LOWER_BOUND, 1e-4, None)
    assert (o.basis, o.R_film_ohm, o.R_film_basis) == (
        REGIME_A_FOOT_LOWER, 2e5, REGIME_A_FOOT_LOWER)
    assert o.admitted and o.counts_as_measured


def test_sigma_observation_flag_off_lower_bound_never_appears():
    """The flag-off shadow carries a finite regime_sigma_lower; it must never surface."""
    o = ob.sigma_observation(_report(**{**A_LOWER, "regime_active": False}))
    assert o.kind is None and o.reported is None and not o.admitted
    assert o.classified                   # still a measured spectrum for the park counter


def test_sigma_observation_unavailable_without_lower_is_unmeasured():
    o = ob.sigma_observation(_report(**{**A_LOWER, "regime_sigma_lower": NAN}))
    assert o == ob.unmeasured("A")


def test_sigma_observation_not_ok_recorded_not_admitted():
    for state, kind in ((A_VALUE, ob.VALUE), (A_BOUND, ob.UPPER_BOUND),
                        (A_LOWER, ob.LOWER_BOUND)):
        o = ob.sigma_observation(_report(ok=False, **state))
        assert o.kind == kind and o.reported is not None, kind
        assert not o.admitted, kind
        assert o.classified, kind


@pytest.mark.parametrize("bad", [NAN, math.inf, 0.0, -1e-4])
def test_sigma_observation_non_finite_or_nonpositive_reported_is_none(bad):
    for state, field in ((A_VALUE, "value"), (A_BOUND, "upper_bound"),
                         (A_LOWER, "regime_sigma_lower")):
        o = ob.sigma_observation(_report(**{**state, field: bad}))
        assert o.kind is None and o.reported is None and not o.admitted, field
        assert o.regime == "A" and o.classified, field


@pytest.mark.parametrize("regime, classified",
                         [("A", True), ("B", True), ("C", True), ("U", False), ("", False)])
def test_sigma_observation_regime_label_sets_classified(regime, classified):
    o = ob.sigma_observation(_report(mode="unavailable", regime=regime))
    assert o.kind is None and o.regime == regime
    assert o.classified is classified and o.counts_as_measured is classified


def test_sigma_observation_malformed_report_unmeasured_never_raises():
    assert ob.sigma_observation(None) == ob.unmeasured()
    broken = SimpleNamespace(ok=True, sigma=SimpleNamespace(regime="B"))
    assert ob.sigma_observation(broken) == ob.unmeasured("B")


def test_unmeasured_is_nothing_stateable():
    o = ob.unmeasured()
    assert (o.kind, o.reported, o.lower, o.R_film_ohm) == (None, None, None, None)
    assert (o.regime, o.basis, o.admitted, o.classified) == ("", "", False, False)
    assert not o.counts_as_measured


@pytest.mark.parametrize("kind, admitted, classified, expected", [
    (ob.VALUE, True, False, True),
    (ob.UPPER_BOUND, False, False, False),
    (None, True, False, False),
    (None, False, True, True),
    (ob.UPPER_BOUND, False, True, True),
    (None, False, False, False),
])
def test_sigma_observation_counts_as_measured_truth_table(kind, admitted, classified,
                                                          expected):
    o = dataclasses.replace(ob.unmeasured(), kind=kind, admitted=admitted,
                            classified=classified)
    assert o.counts_as_measured is expected


def test_sigma_observation_kinds_are_the_store_vocabulary():
    assert ob.KINDS == ("value", "lower_bound", "upper_bound")


# ── On a report the engine built ──────────────────────────────────────────────────


def test_sigma_observation_engine_compressed_a_lower_only_with_flag_on():
    """The production path: flag on → an admitted foot lower bound; flag off → not."""
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.regime_route import RegimeSettings
    from tests import eis_regime_synthetic as syn
    from tests.eis_regime_golden import golden_inputs

    eis = syn.as_eis(syn.RIG_F, syn.regime_a_rs(
        "compressed", 1.5e5, 0.6, 0.85, 1e5, 1e5, 20e-6, True, seed=503))
    on, off = (ob.sigma_observation(analyze_spectrum(
        eis, **golden_inputs(15), regime=RegimeSettings(enabled=flag)))
        for flag in (True, False))
    assert on.kind == ob.LOWER_BOUND and on.basis == REGIME_A_FOOT_LOWER
    assert on.R_film_ohm is not None and on.R_film_ohm > 0
    assert off.kind != ob.LOWER_BOUND and off.classified
