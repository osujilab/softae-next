"""Unit tests for the two-phase dropcast rate/volume split."""

from __future__ import annotations

import pytest

from softae.core.dropcast_plan import (
    DropcastPlan,
    PhaseParams,
    build_dropcast_plan,
    split_rate,
)


# ── split_rate ───────────────────────────────────────────────────────────────

def test_split_rate_is_proportional_and_sums_to_total():
    rates = split_rate(100.0, [10.0, 30.0])
    assert rates == [pytest.approx(25.0), pytest.approx(75.0)]
    assert sum(rates) == pytest.approx(100.0)


def test_split_rate_gives_equal_duration():
    volumes = [8.0, 2.0, 5.0]
    rates = split_rate(60.0, volumes)
    # duration = vol/rate is identical across pumps (the whole point of the split).
    durations = [v / r for v, r in zip(volumes, rates)]
    assert durations[0] == pytest.approx(durations[1]) == pytest.approx(durations[2])


def test_split_rate_zero_total_volume_gives_zero_rates():
    assert split_rate(500.0, [0.0, 0.0]) == [0.0, 0.0]


def test_split_rate_zero_volume_pump_gets_zero_rate():
    rates = split_rate(100.0, [10.0, 0.0])
    assert rates == [pytest.approx(100.0), 0.0]


# ── PhaseParams ──────────────────────────────────────────────────────────────

def test_phase_duration_min():
    phase = PhaseParams(volumes_uL=[10.0, 30.0], rates_uL_min=[25.0, 75.0])
    # total 40 uL at total 100 uL/min → 0.4 min.
    assert phase.duration_min() == pytest.approx(0.4)


def test_phase_duration_zero_when_no_flow():
    phase = PhaseParams(volumes_uL=[10.0], rates_uL_min=[0.0])
    assert phase.duration_min() == 0.0


# ── build_dropcast_plan ──────────────────────────────────────────────────────

def test_build_plan_splits_both_phases():
    plan = build_dropcast_plan(
        [10.0, 30.0],
        dispense_rate_total=100.0,
        flush_rate_total=500.0,
        flush_factor=3.0,
        settle_factor=2.0,
    )
    assert isinstance(plan, DropcastPlan)
    assert plan.deposition.rates_uL_min == [pytest.approx(25.0), pytest.approx(75.0)]
    assert plan.flush_rates_uL_min == [pytest.approx(125.0), pytest.approx(375.0)]
    assert plan.flush_factor == 3.0


def test_build_plan_deposition_and_flush_share_proportions():
    plan = build_dropcast_plan(
        [8.0, 2.0],
        dispense_rate_total=50.0,
        flush_rate_total=500.0,
        flush_factor=2.0,
        settle_factor=1.0,
    )
    dep = plan.deposition.rates_uL_min
    flush = plan.flush_rates_uL_min
    # Same volume proportions → same rate proportions in both phases.
    assert dep[0] / dep[1] == pytest.approx(flush[0] / flush[1])


def test_build_plan_settle_wait_scales_with_volume_and_factor():
    # 40 uL at 100 uL/min = 0.4 min = 24 s; × settle_factor 2 = 48 s.
    plan = build_dropcast_plan(
        [10.0, 30.0],
        dispense_rate_total=100.0,
        flush_rate_total=500.0,
        flush_factor=3.0,
        settle_factor=2.0,
    )
    assert plan.settle_wait_s == pytest.approx(48.0)


def test_build_plan_settle_base_is_added():
    plan = build_dropcast_plan(
        [10.0, 30.0],
        dispense_rate_total=100.0,
        flush_rate_total=500.0,
        flush_factor=3.0,
        settle_factor=2.0,
        settle_base_s=10.0,
    )
    assert plan.settle_wait_s == pytest.approx(58.0)


def test_build_plan_preload_volumes():
    plan = build_dropcast_plan(
        [10.0, 30.0],
        dispense_rate_total=100.0,
        flush_rate_total=500.0,
        flush_factor=3.0,
        settle_factor=2.0,
    )
    assert plan.preload_volumes_uL() == [pytest.approx(30.0), pytest.approx(90.0)]
    assert plan.volumes_uL == [10.0, 30.0]


def test_build_plan_zero_formulation_is_safe():
    plan = build_dropcast_plan(
        [0.0, 0.0],
        dispense_rate_total=100.0,
        flush_rate_total=500.0,
        flush_factor=3.0,
        settle_factor=2.0,
    )
    assert plan.deposition.rates_uL_min == [0.0, 0.0]
    assert plan.flush_rates_uL_min == [0.0, 0.0]
    assert plan.settle_wait_s == 0.0
