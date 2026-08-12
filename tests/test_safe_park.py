"""Tests for the canonical safe-park sequence (core, GUI-free)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from softae.core.safe_park import (
    DEFAULT_SAFE_TEMP_C,
    safe_park,
    safe_park_async,
)


def _manager(**overrides):
    """A manager whose instruments are all connected MagicMocks."""
    insts: dict[str, MagicMock] = {}
    for name in ("syringe", "temp_controller", "lamp"):
        m = MagicMock()
        m.is_connected = True
        insts[name] = m
    insts.update(overrides)

    mgr = MagicMock()
    mgr.get.side_effect = lambda n: insts[n]  # KeyError for anything else
    mgr._insts = insts
    return mgr


class TestHappyPath:
    def test_all_subsystems_made_safe(self):
        mgr = _manager()
        result = safe_park(mgr, reason="unit test")

        assert result.ok
        assert result.errors == []
        syr = mgr._insts["syringe"]
        syr.head_retract.assert_called_once()
        assert syr.single_pump.call_count == 3          # pumps 0,1,2 halted
        mgr._insts["temp_controller"].write_sp.assert_called_once_with(
            DEFAULT_SAFE_TEMP_C, print_flag=0
        )
        mgr._insts["lamp"].off.assert_called_once()

    def test_head_retracts_before_anything_else(self):
        """Ordering matters: the head must clear the board first."""
        order: list[str] = []
        mgr = _manager()
        mgr._insts["syringe"].head_retract.side_effect = lambda: order.append("head")
        mgr._insts["syringe"].single_pump.side_effect = lambda *a: order.append("pump")
        mgr._insts["temp_controller"].write_sp.side_effect = (
            lambda *a, **k: order.append("temp")
        )
        mgr._insts["lamp"].off.side_effect = lambda: order.append("lamp")

        safe_park(mgr)
        assert order[0] == "head"
        assert order.index("pump") < order.index("temp") < order.index("lamp")

    def test_custom_temp_and_pump_ids(self):
        mgr = _manager()
        safe_park(mgr, pump_ids=(0, 1), safe_temp_C=25.0)
        assert mgr._insts["syringe"].single_pump.call_count == 2
        mgr._insts["temp_controller"].write_sp.assert_called_once_with(
            25.0, print_flag=0
        )


class TestBestEffort:
    def test_one_failure_does_not_block_the_others(self):
        """A refusing subsystem must not prevent the rest from going safe."""
        mgr = _manager()
        mgr._insts["syringe"].head_retract.side_effect = RuntimeError("stuck")

        result = safe_park(mgr)

        assert not result.ok
        assert any("syringe head" in e for e in result.errors)
        # Everything downstream still ran.
        mgr._insts["temp_controller"].write_sp.assert_called_once()
        mgr._insts["lamp"].off.assert_called_once()

    def test_never_raises_even_if_everything_fails(self):
        mgr = _manager()
        for name in ("syringe", "temp_controller", "lamp"):
            for attr in ("head_retract", "single_pump", "write_sp", "off"):
                getattr(mgr._insts[name], attr).side_effect = RuntimeError("dead")

        result = safe_park(mgr)          # must not raise
        assert not result.ok
        assert len(result.errors) >= 3

    def test_disconnected_instruments_are_skipped_not_errors(self):
        mgr = _manager()
        mgr._insts["lamp"].is_connected = False

        result = safe_park(mgr)

        assert result.ok                              # skipping is not a failure
        assert any("lamp" in s for s in result.skipped)
        mgr._insts["lamp"].off.assert_not_called()

    def test_absent_instrument_is_skipped(self):
        mgr = MagicMock()
        mgr.get.side_effect = KeyError("nope")

        result = safe_park(mgr)

        assert result.ok
        assert len(result.skipped) == 3               # syringe, temp, lamp
        assert result.actions == []

    def test_partial_pump_failure_records_both(self):
        mgr = _manager()
        calls = {"n": 0}

        def flaky(*a):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("pump 1 jammed")

        mgr._insts["syringe"].single_pump.side_effect = flaky
        result = safe_park(mgr)

        assert any("pump 1 stop" in e for e in result.errors)
        assert any("pumps" in a for a in result.actions)   # the others halted


class TestSummaryAndAsync:
    def test_summary_is_human_readable(self):
        mgr = _manager()
        mgr._insts["lamp"].is_connected = False
        s = safe_park(mgr).summary()
        assert "ok" in s and "skipped" in s

    @pytest.mark.asyncio
    async def test_async_wrapper_matches_sync(self):
        mgr = _manager()
        result = await safe_park_async(mgr, reason="async test")
        assert result.ok
        mgr._insts["syringe"].head_retract.assert_called_once()
