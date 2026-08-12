"""The temperature-source authority: which thermometer, and can it say so.

The rig has two temperature reads and the column names hide that fact. These tests
pin the precedence and — with the real numbers off run ``20260811T023757Z`` — the
divergence that made the precedence necessary.
"""

from __future__ import annotations

import math

import pytest

from softae.analysis.conditions import (
    ABSOLUTE_ZERO_C,
    TEMPERATURE_MIXED,
    TEMPERATURE_SOURCES,
    TEMPERATURE_UNAVAILABLE,
    combine_temperature_sources,
    resolve_temperature_C,
)


class TestPrecedence:
    def test_resolve_temperature_stage_pv_present_wins_over_every_other_source(self):
        value, source = resolve_temperature_C(
            stage_pv_C=65.0, stage_sp_C=65.0, chamber_air_C=36.2)
        assert value == pytest.approx(65.0)
        assert source == "stage_pv"

    def test_resolve_temperature_no_stage_pv_falls_back_to_the_setpoint(self):
        value, source = resolve_temperature_C(stage_sp_C=45.0, chamber_air_C=29.1)
        assert value == pytest.approx(45.0)
        assert source == "stage_sp"

    def test_resolve_temperature_only_chamber_air_returns_it_labelled_as_air(self):
        # Last resort, and it must never be mistaken for the sample's temperature.
        value, source = resolve_temperature_C(chamber_air_C=22.8)
        assert value == pytest.approx(22.8)
        assert source == "chamber_air"

    def test_temperature_sources_ordering_is_the_resolvers_precedence(self):
        assert TEMPERATURE_SOURCES == ("stage_pv", "stage_sp", "chamber_air")
        for i, best in enumerate(TEMPERATURE_SOURCES):
            worse = {name: 10.0 * (j + 1)
                     for j, name in enumerate(TEMPERATURE_SOURCES) if j >= i}
            kwargs = {"stage_pv_C": worse.get("stage_pv"),
                      "stage_sp_C": worse.get("stage_sp"),
                      "chamber_air_C": worse.get("chamber_air")}
            assert resolve_temperature_C(**kwargs)[1] == best


class TestRefusal:
    def test_resolve_temperature_nothing_known_returns_nan_and_unavailable(self):
        value, source = resolve_temperature_C()
        assert math.isnan(value)
        assert source == TEMPERATURE_UNAVAILABLE

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_resolve_temperature_non_finite_stage_pv_is_rejected_not_passed_through(
            self, bad):
        value, source = resolve_temperature_C(stage_pv_C=bad, chamber_air_C=22.8)
        assert value == pytest.approx(22.8)
        assert source == "chamber_air"

    @pytest.mark.parametrize("bad", [ABSOLUTE_ZERO_C, -300.0, -1000.0])
    def test_resolve_temperature_below_absolute_zero_is_rejected_not_passed_through(
            self, bad):
        # An uninitialised Modbus register, not a measurement.
        value, source = resolve_temperature_C(stage_pv_C=bad, stage_sp_C=45.0)
        assert value == pytest.approx(45.0)
        assert source == "stage_sp"

    def test_resolve_temperature_all_sources_unusable_returns_unavailable(self):
        value, source = resolve_temperature_C(
            stage_pv_C=float("nan"), stage_sp_C=None, chamber_air_C=-500.0)
        assert math.isnan(value)
        assert source == TEMPERATURE_UNAVAILABLE

    def test_resolve_temperature_non_numeric_input_is_skipped_not_raised(self):
        value, source = resolve_temperature_C(stage_pv_C="warm", stage_sp_C=27.5)
        assert value == pytest.approx(27.5)
        assert source == "stage_sp"


class TestTheMeasuredDivergence:
    # Run 20260811T023757Z, top of the up leg: the stage controller and the
    # humidity sensor's air probe disagree by 42.1 C at the same setpoint.
    def test_resolve_temperature_at_the_85C_setpoint_returns_the_stage_not_the_air(self):
        value, source = resolve_temperature_C(stage_pv_C=85.0, chamber_air_C=42.9)
        assert value == pytest.approx(85.0)
        assert source == "stage_pv"

    @pytest.mark.parametrize("stage_pv, chamber_air", [
        (27.6, 22.8), (45.4, 26.2), (65.0, 36.2), (85.0, 42.9),
    ])
    def test_resolve_temperature_every_setpoint_of_the_real_run_returns_the_stage(
            self, stage_pv, chamber_air):
        value, source = resolve_temperature_C(
            stage_pv_C=stage_pv, stage_sp_C=None, chamber_air_C=chamber_air)
        assert value == pytest.approx(stage_pv)
        assert source == "stage_pv"

    def test_resolve_temperature_the_air_probe_would_shrink_the_inverse_T_span(self):
        # Why this module exists: the compression, not the offset. ln(sigma) is
        # unchanged, so a 2.4x smaller 1/T span is a 2.4x inflated activation energy
        # with an unharmed R^2.
        def span(low_C, high_C):
            return 1.0 / (low_C + 273.15) - 1.0 / (high_C + 273.15)

        stage_span = span(27.6, 85.0)
        air_span = span(22.8, 42.9)
        assert stage_span == pytest.approx(5.33e-4, rel=0.02)
        assert air_span == pytest.approx(2.15e-4, rel=0.02)
        assert stage_span / air_span > 2.0


class TestSourceCombination:
    def test_combine_temperature_sources_one_source_throughout_keeps_that_label(self):
        assert combine_temperature_sources(["stage_pv"] * 15) == "stage_pv"

    def test_combine_temperature_sources_two_thermometers_is_labelled_mixed(self):
        assert combine_temperature_sources(
            ["stage_pv", "stage_pv", "chamber_air"]) == TEMPERATURE_MIXED

    def test_combine_temperature_sources_unavailable_rounds_do_not_make_it_mixed(self):
        # A series that lost the stage read for two rounds is still a stage series.
        assert combine_temperature_sources(
            ["stage_pv", TEMPERATURE_UNAVAILABLE, "stage_pv"]) == "stage_pv"

    def test_combine_temperature_sources_nothing_known_is_unavailable(self):
        assert combine_temperature_sources([]) == TEMPERATURE_UNAVAILABLE
        assert combine_temperature_sources(
            [TEMPERATURE_UNAVAILABLE]) == TEMPERATURE_UNAVAILABLE
