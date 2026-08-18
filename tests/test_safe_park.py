"""Tests for the canonical safe-park sequence (core, GUI-free).

Three claims are load-bearing here and each has a test that fails against the
previous implementation:

* a park triggered *by* reservoir depletion still halts the pumps (it used to be
  refused by the depletion interlock, three times);
* an automatic park issues **no** head motion (it used to flip the head, which
  drives it *down* whenever the belief was stale in the unlucky direction);
* the result never claims verification it does not have.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from softae.core.reservoir import ReservoirLedger
from softae.core.safe_park import (
    DEFAULT_SAFE_TEMP_C,
    SafeParkResult,
    safe_park,
    safe_park_async,
)
from softae.drivers.mock_syringe import MockSyringe
from softae.server.base_instrument import InstrumentState


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


def _real_syringe_manager(*, head_up: bool, ledger: ReservoirLedger | None = None):
    """A manager holding a genuine :class:`MockSyringe` with a chosen head belief."""
    syr = MockSyringe(config={"min_rate": 0.05})   # the shipped rig's floor
    syr._state = InstrumentState.CONNECTED
    syr.set_head_state(head_up)
    if ledger is not None:
        syr.reservoir_ledger = ledger
    return _manager(syringe=syr), syr


# ── The pump halt ────────────────────────────────────────────────────────────

class TestPumpHalt:
    def test_the_park_halts_pumps_without_dispensing(self):
        """The halt is ``halt_pump``, never a near-zero ``single_pump``."""
        mgr = _manager()
        safe_park(mgr, reason="unit test")
        syr = mgr._insts["syringe"]
        assert syr.halt_pump.call_count == 3
        syr.single_pump.assert_not_called()

    def test_a_depletion_park_still_halts_the_pumps(self):
        """The park's own halt must not be refusable by the depletion interlock.

        A reservoir at its hard stop is a *park trigger*. Routed through
        ``single_pump`` the 0.001 µL halt raised ``SafetyError`` on every pump —
        so the one park the ledger causes was the one park that could not stop
        the pumps. Against the previous implementation this produces three errors.
        """
        ledger = ReservoirLedger(soft_warn_uL=500.0, hard_stop_uL=200.0)
        for pump_id in (0, 1, 2):
            ledger.refill(pump_id, 200.0)          # exactly at the hard stop
        mgr, syr = _real_syringe_manager(head_up=True, ledger=ledger)

        result = safe_park(mgr, reason="reservoir depleted")

        assert result.errors == []
        assert syr._halted == [0, 1, 2]

    def test_halt_pump_does_not_debit_the_stock_ledger(self):
        """A safety action must not spend the budget the interlock protects."""
        ledger = ReservoirLedger(soft_warn_uL=5_000.0, hard_stop_uL=200.0)
        ledger.refill(0, 4_321.0)
        mgr, _ = _real_syringe_manager(head_up=True, ledger=ledger)

        safe_park(mgr)

        assert ledger.remaining_uL(0) == 4_321.0

    def test_a_driver_with_no_halt_is_an_error_not_a_silent_fallback(self):
        """Falling back to ``single_pump`` would restore the refusal invisibly."""
        syr = MagicMock(spec=["is_connected", "head_retract", "single_pump"])
        syr.is_connected = True
        result = safe_park(_manager(syringe=syr))

        assert any("halt_pump" in e for e in result.errors)
        syr.single_pump.assert_not_called()

    def test_one_jammed_pump_does_not_stop_the_others(self):
        mgr = _manager()
        calls = {"n": 0}

        def flaky(_pump_id):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("pump 1 jammed")

        mgr._insts["syringe"].halt_pump.side_effect = flaky
        result = safe_park(mgr)

        assert any("pump 1 stop" in e for e in result.errors)
        assert any("pumps" in c for c in result.commanded)


# ── Head policy ──────────────────────────────────────────────────────────────

class TestHeadPolicy:
    def test_an_automatic_park_does_not_move_the_head(self):
        """Belief says DOWN. The old default flipped it — i.e. drove it down."""
        mgr, syr = _real_syringe_manager(head_up=False)

        result = safe_park(mgr, reason="hard fault")

        assert syr.is_head_up() is False          # untouched, not flipped
        assert not any("head" in c.lower() for c in result.commanded)

    def test_an_automatic_park_does_not_move_a_head_believed_up_either(self):
        mgr, syr = _real_syringe_manager(head_up=True)
        safe_park(mgr, reason="hard fault")
        assert syr.is_head_up() is True

    @pytest.mark.parametrize("head_up", [True, False])
    def test_the_park_result_reports_head_state_as_unverifiable(self, head_up):
        """Never ``verified``, in either belief state, under any policy."""
        for retract in (None, True, False):
            mgr, _ = _real_syringe_manager(head_up=head_up)
            result = safe_park(mgr, retract_head=retract)
            assert any("head" in u.lower() for u in result.unverifiable)
            assert result.verified == []

    def test_an_explicit_retract_is_commanded_not_asserted(self):
        """The operator asked; the wording still does not claim it happened."""
        mgr = _manager()
        result = safe_park(mgr, retract_head=True)

        mgr._insts["syringe"].head_retract.assert_called_once()
        assert any("commanded" in c for c in result.commanded if "head" in c)
        assert "head retracted" not in result.commanded

    def test_leaving_the_head_lowered_is_recorded_as_a_choice(self):
        mgr = _manager()
        result = safe_park(mgr, retract_head=False)

        mgr._insts["syringe"].head_retract.assert_not_called()
        assert any("lowered" in c for c in result.commanded)

    def test_a_failing_retract_does_not_block_the_rest(self):
        mgr = _manager()
        mgr._insts["syringe"].head_retract.side_effect = RuntimeError("stuck")

        result = safe_park(mgr, retract_head=True)

        assert any("syringe head" in e for e in result.errors)
        mgr._insts["temp_controller"].write_sp.assert_called_once()
        mgr._insts["lamp"].off.assert_called_once()


# ── Ordering, thermal and optical load ───────────────────────────────────────

class TestSequence:
    def test_all_subsystems_are_commanded(self):
        mgr = _manager()
        result = safe_park(mgr, reason="unit test")

        assert result.ok
        assert result.errors == []
        assert mgr._insts["syringe"].halt_pump.call_count == 3
        mgr._insts["temp_controller"].write_sp.assert_called_once_with(
            DEFAULT_SAFE_TEMP_C, print_flag=0
        )
        mgr._insts["lamp"].off.assert_called_once()

    def test_the_head_is_handled_before_anything_else(self):
        """Ordering matters: the head must clear the board first."""
        order: list[str] = []
        mgr = _manager()
        mgr._insts["syringe"].head_retract.side_effect = lambda: order.append("head")
        mgr._insts["syringe"].halt_pump.side_effect = lambda *a: order.append("pump")
        mgr._insts["temp_controller"].write_sp.side_effect = (
            lambda *a, **k: order.append("temp")
        )
        mgr._insts["lamp"].off.side_effect = lambda: order.append("lamp")

        safe_park(mgr, retract_head=True)
        assert order[0] == "head"
        assert order.index("pump") < order.index("temp") < order.index("lamp")

    def test_custom_temp_and_pump_ids(self):
        mgr = _manager()
        safe_park(mgr, pump_ids=(0, 1), safe_temp_C=25.0)
        assert mgr._insts["syringe"].halt_pump.call_count == 2
        mgr._insts["temp_controller"].write_sp.assert_called_once_with(
            25.0, print_flag=0
        )


class TestBestEffort:
    def test_never_raises_even_if_everything_fails(self):
        mgr = _manager()
        for name in ("syringe", "temp_controller", "lamp"):
            for attr in ("head_retract", "halt_pump", "write_sp", "off"):
                getattr(mgr._insts[name], attr).side_effect = RuntimeError("dead")

        result = safe_park(mgr, retract_head=True)      # must not raise
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


# ── The vocabulary of the result ─────────────────────────────────────────────

class TestResultVocabulary:
    def test_nothing_on_this_rig_is_ever_verified(self):
        assert safe_park(_manager()).verified == []

    def test_actions_is_commanded_plus_verified(self):
        """Kept for the callers outside this module that log it."""
        result = SafeParkResult(commanded=["a"], verified=["b"])
        assert result.actions == ["a", "b"]

    def test_ok_still_means_nothing_raised(self):
        assert SafeParkResult(commanded=["a"]).ok is True
        assert SafeParkResult(errors=["boom"]).ok is False

    def test_describe_says_what_was_checked_and_never_says_safe(self):
        text = safe_park(_manager()).describe()
        assert "Commanded" in text
        assert "NOT verifiable" in text
        assert "Nothing was verified" in text
        assert "safe" not in text.lower()

    def test_summary_names_the_grades(self):
        mgr = _manager()
        mgr._insts["lamp"].is_connected = False
        s = safe_park(mgr).summary()
        assert "commanded" in s and "unverifiable" in s and "skipped" in s

    def test_the_result_is_constructible_by_keyword(self):
        """The headless shutdown path builds one directly on a raised park."""
        assert SafeParkResult(errors=["safe_park raised: x"]).ok is False


class TestTheAntiClogConsequence:
    """A park suspends the P8 purge harness. That cost must not be invisible."""

    def test_a_park_says_that_purging_is_suspended(self):
        result = safe_park(_manager())
        assert any("purging is refused" in n for n in result.notes)

    def test_the_note_is_not_an_error_and_not_a_claim_about_hardware(self):
        result = safe_park(_manager())
        assert result.ok
        assert result.notes and not any(n in result.commanded for n in result.notes)

    def test_the_note_reaches_the_operator_paragraph(self):
        assert "Consequences of the park" in safe_park(_manager()).describe()

    def test_it_is_unconditional_because_the_latch_causes_it_not_the_head(self):
        """It held when the park retracted too — the refusal is the park latch's,
        checked ahead of any pose."""
        for retract in (None, True, False):
            assert safe_park(_manager(), retract_head=retract).notes

    def test_a_park_from_idle_rest_leaves_the_tip_where_it_was(self):
        """Idle rest is head-down in the flush basin, deliberately, so the tip
        stays wet. The old default pulled it *out* of the basin; this one does
        not — better for a particulate line, not worse."""
        mgr, syr = _real_syringe_manager(head_up=False)
        safe_park(mgr, reason="hard fault")
        assert syr.is_head_up() is False


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_wrapper_matches_sync(self):
        mgr = _manager()
        result = await safe_park_async(mgr, reason="async test")
        assert result.ok
        assert mgr._insts["syringe"].halt_pump.call_count == 3
        mgr._insts["syringe"].head_retract.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_wrapper_forwards_an_explicit_retract(self):
        mgr = _manager()
        await safe_park_async(mgr, retract_head=True)
        mgr._insts["syringe"].head_retract.assert_called_once()
