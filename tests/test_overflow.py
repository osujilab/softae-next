"""Tests for the shared overflow guard (softae.core.overflow)."""

from __future__ import annotations

import pytest

from softae.core.overflow import (
    OverflowSweepResult,
    enumerate_space,
    sweep_overflow,
    well_overflow,
)


# ── well_overflow: the single-formulation verdict ───────────────────────────

def test_under_capacity_does_not_overflow():
    v = well_overflow(18.0, 20.0)
    assert not v.overflows
    assert v.headroom_uL == pytest.approx(2.0)


def test_over_capacity_overflows_with_negative_headroom():
    v = well_overflow(25.0, 20.0)
    assert v.overflows
    assert v.headroom_uL == pytest.approx(-5.0)


def test_exactly_at_capacity_is_not_overflow():
    """A total exactly at capacity fits (boundary is inclusive)."""
    v = well_overflow(20.0, 20.0)
    assert not v.overflows
    assert v.headroom_uL == pytest.approx(0.0)


def test_float_noise_at_capacity_is_not_overflow():
    v = well_overflow(20.0 + 1e-12, 20.0)
    assert not v.overflows


# ── enumerate_space: pool passthrough + bounds-dict grid ────────────────────

def test_enumerate_passes_through_an_explicit_pool():
    pool = [{"x_a": 0.1, "x_b": 0.9}, {"x_a": 0.5, "x_b": 0.5}]
    out = enumerate_space(pool)
    assert out == pool
    assert out is not pool  # copied, not aliased


def test_enumerate_grid_samples_a_bounds_dict():
    space = {
        "eo_li_ratio": {"type": "float", "low": 10.0, "high": 20.0},
        "silica_vol_frac": {"type": "float", "low": 0.0, "high": 0.2},
    }
    pool = enumerate_space(space, steps=3)
    assert len(pool) == 3 * 3
    ratios = sorted({p["eo_li_ratio"] for p in pool})
    assert ratios == pytest.approx([10.0, 15.0, 20.0])
    fracs = sorted({p["silica_vol_frac"] for p in pool})
    assert fracs == pytest.approx([0.0, 0.1, 0.2])


def test_enumerate_int_axis_rounds_and_dedups():
    pool = enumerate_space({"n": {"type": "int", "low": 1, "high": 3, "steps": 5}})
    ns = sorted({p["n"] for p in pool})
    assert ns == [1, 2, 3]
    assert all(isinstance(p["n"], int) for p in pool)


def test_enumerate_categorical_axis():
    pool = enumerate_space({"mode": {"type": "categorical", "choices": ["a", "b"]}})
    assert sorted(p["mode"] for p in pool) == ["a", "b"]


def test_enumerate_reuses_doe_candidate_pool_shape():
    """A DOE candidate pool (list of x_* dicts) flows through unchanged."""
    from softae.campaigns.doe import ExperimentDesign, ParamScale

    design = ExperimentDesign(
        components=["water", "polymer"],
        param_scales=[ParamScale("x_water", "linear", start=0.2, stop=0.8, steps=3)],
    )
    pool = design.candidate_pool()
    assert enumerate_space(pool) == pool


# ── sweep_overflow: flag overflow across the whole space ────────────────────

def _total_from_frac(point):
    """Toy mapper: total cast volume grows with the 'load' axis."""
    return 10.0 + 40.0 * point["load"]


def test_sweep_flags_the_overflowing_subregion():
    space = {"load": {"low": 0.0, "high": 1.0, "steps": 5}}  # totals 10..50 µL
    points = enumerate_space(space)
    result = sweep_overflow(points, _total_from_frac, capacity_uL=30.0)

    assert isinstance(result, OverflowSweepResult)
    assert result.n_points == 5
    # totals: 10, 20, 30, 40, 50 → only 40 and 50 exceed 30.
    assert result.n_overflow == 2
    assert result.overflow_fraction == pytest.approx(0.4)
    assert result.any_overflow and not result.all_overflow
    assert result.max_total_uL == pytest.approx(50.0)

    worst_point, worst = result.worst
    assert worst_point["load"] == pytest.approx(1.0)
    assert worst.total_uL == pytest.approx(50.0)
    assert worst.headroom_uL == pytest.approx(-20.0)


def test_sweep_all_clear_when_capacity_is_generous():
    points = enumerate_space({"load": {"low": 0.0, "high": 1.0, "steps": 4}})
    result = sweep_overflow(points, _total_from_frac, capacity_uL=1000.0)
    assert not result.any_overflow
    assert result.overflow_fraction == 0.0
    assert result.overflowing_points() == []


def test_sweep_reuses_well_geometry_capacity():
    """Integration: capacity sourced from WellGeometry, as the panel does."""
    from softae.core.deposition import WellGeometry

    well = WellGeometry(diameter_mm=3.0, depth_mm=1.0)
    points = enumerate_space({"load": {"low": 0.0, "high": 1.0, "steps": 3}})
    result = sweep_overflow(points, _total_from_frac, capacity_uL=well.capacity_uL)
    assert result.capacity_uL == pytest.approx(well.capacity_uL)
    assert result.n_points == 3
