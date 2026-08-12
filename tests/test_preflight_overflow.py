"""Tests for the Live-BO pre-flight overflow scan (autonomous_wiring)."""

from __future__ import annotations

import pytest

from softae.core.autonomous_wiring import (
    CampaignSpec,
    GeneralFormulation,
    campaign_well_capacity_uL,
    preflight_overflow,
)
from softae.core.formulation import ChemicalCatalog


def _legacy_spec(hi: float = 100.0, **over) -> CampaignSpec:
    """Legacy identity-map spec: searched params ARE per-pump volumes (µL).

    Uses the 4-stripe board, so the total cast volume is just the sum of the two
    axes and overflow is easy to reason about. Its capacity is now *derived* from
    the declared well (4.88 mm across, 6.35 mm deep) rather than hand-typed, so it
    is 118.769 µL — the brim volume — not the 120 µL that was written down.
    """
    base = dict(
        name="legacy_overflow",
        channels=(21,),
        pcb_name="SoftAE_EIS_4Stripe",  # capacity derived: 118.769 uL
        parameter_space={
            "v0": {"type": "float", "low": 0.0, "high": hi},
            "v1": {"type": "float", "low": 0.0, "high": hi},
        },
        vol_params=("v0", "v1"),
        pump_ids=(0, 1),
        optimizer="random",
        time_scale=0.0,
        budget=4,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


def test_capacity_from_board_when_no_explicit_budget():
    """The board's own well, not a number someone typed beside it.

    It was 120 uL against a brim volume of 118.769 -- the guard permitted 1 % more
    than the walls hold. The capacity now derives from the declared geometry.
    """
    assert campaign_well_capacity_uL(_legacy_spec()) == pytest.approx(118.769,
                                                                     abs=1e-3)


def test_capacity_prefers_explicit_general_budget():
    spec = _legacy_spec()
    spec.general_formulation = GeneralFormulation(
        stocks={}, catalog=ChemicalCatalog(), pump_assignment={},
        target_deposition_uL=6.0, build_targets=lambda p: [], budget_uL=42.0,
    )
    assert campaign_well_capacity_uL(spec) == pytest.approx(42.0)


def test_preflight_flags_the_overflowing_subregion():
    # 3x3 grid over {0, 50, 100} per axis; total = v0 + v1 vs the well's 118.769.
    result = preflight_overflow(_legacy_spec(hi=100.0), steps=3)
    assert result.n_points == 9
    assert result.capacity_uL == pytest.approx(118.769, abs=1e-3)
    # sums over capacity: (50,100)=150, (100,50)=150, (100,100)=200 -> 3 overflow.
    assert result.n_overflow == 3
    assert result.overflow_fraction == pytest.approx(3 / 9)
    assert result.max_total_uL == pytest.approx(200.0)
    worst_point, worst = result.worst
    assert worst.total_uL == pytest.approx(200.0)
    assert worst.headroom_uL == pytest.approx(-81.231, abs=1e-3)


def test_preflight_all_clear_when_space_fits():
    # Axes capped at 10 uL -> max total 20 uL, well inside the 118.769 uL well.
    result = preflight_overflow(_legacy_spec(hi=10.0), steps=3)
    assert result.n_points == 9
    assert not result.any_overflow
    assert result.overflow_fraction == 0.0


def test_preflight_does_not_raise_on_infeasible_region():
    """A pre-flight *flags* overflow; it never raises FormulationInfeasibleError."""
    result = preflight_overflow(_legacy_spec(hi=100.0), steps=4)
    assert result.any_overflow  # some region overflows, but the call returned
