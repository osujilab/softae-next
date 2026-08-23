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
    HEADLINE_COMMANDED,
    HEADLINE_NOTHING,
    HEADLINE_PARTIAL,
    RH_DEADMAN_S,
    SafeParkResult,
    dry_purge_humidifier,
    safe_park,
    safe_park_async,
)
from softae.drivers.mock_syringe import MockSyringe
from softae.server.base_instrument import InstrumentState


def _manager(**overrides):
    """A manager whose instruments are all connected MagicMocks."""
    insts: dict[str, MagicMock] = {}
    for name in ("syringe", "temp_controller", "lamp", "rh_controller"):
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
        assert len(result.skipped) == 4       # syringe, temp, rh_controller, lamp
        assert result.actions == []


# ── The humidifier ───────────────────────────────────────────────────────────

class TestHumidifier:
    """A parked rig must not keep humidifying.

    The safe state is not invented here: the operator ruled it is the one the
    driver already writes on a clean stop — duty 0. What is new is that the park
    reaches it at all, on the paths where nothing did.
    """

    def test_the_park_turns_the_humidifier_off(self):
        mgr = _manager()
        result = safe_park(mgr, reason="unit test")

        mgr._insts["rh_controller"].safe_off.assert_called_once()
        assert any("humidifier off" in c for c in result.commanded)

    def test_the_park_never_calls_stop_instead_of_safe_off(self):
        """``stop()`` writes nothing when ``start()`` was never called, and
        returns cleanly having written nothing when the loop is wedged past its
        five-second join. Pinned, because it is the tempting substitution."""
        mgr = _manager()
        safe_park(mgr)

        mgr._insts["rh_controller"].stop.assert_not_called()

    def test_an_absent_rh_controller_is_skipped_not_an_error(self):
        mgr = _manager()
        mgr._insts.pop("rh_controller")

        result = safe_park(mgr)

        assert result.ok
        assert any("rh_controller: not registered" in s for s in result.skipped)

    def test_a_disconnected_rh_controller_is_skipped(self):
        mgr = _manager()
        mgr._insts["rh_controller"].is_connected = False

        result = safe_park(mgr)

        assert result.ok
        assert any("rh_controller: not connected" in s for s in result.skipped)
        mgr._insts["rh_controller"].safe_off.assert_not_called()
        assert not any("humidifier" in c for c in result.commanded)

    def test_a_driver_with_no_safe_off_is_an_error_not_a_silent_skip(self):
        """Mirrors the pump halt: a registered driver that cannot be turned off
        is a finding, not a non-event."""
        rh = MagicMock(spec=["is_connected", "stop"])
        rh.is_connected = True

        result = safe_park(_manager(rh_controller=rh))

        assert any("safe_off" in e for e in result.errors)
        rh.stop.assert_not_called()

    def test_a_raising_safe_off_does_not_block_the_lamp(self):
        mgr = _manager()
        mgr._insts["rh_controller"].safe_off.side_effect = RuntimeError("wedged")

        result = safe_park(mgr)

        assert any("humidity: wedged" in e for e in result.errors)
        mgr._insts["lamp"].off.assert_called_once()

    def test_a_failed_duty_write_is_reported_as_an_error_not_commanded(self):
        """``safe_off`` swallows a comms failure to keep the never-raise
        contract, so a park that only watched for an exception would claim a
        write that never landed."""
        mgr = _manager()
        mgr._insts["rh_controller"].last_safe_off_error = (
            "Failed to send duty cycle: port went away")

        result = safe_park(mgr)

        assert any("port went away" in e for e in result.errors)
        assert not any("humidifier off" in c for c in result.commanded)

    def test_the_humidifier_is_zeroed_before_the_lamp(self):
        order: list[str] = []
        mgr = _manager()
        mgr._insts["temp_controller"].write_sp.side_effect = (
            lambda *a, **k: order.append("temp"))
        mgr._insts["rh_controller"].safe_off.side_effect = lambda: order.append("rh")
        mgr._insts["lamp"].off.side_effect = lambda: order.append("lamp")

        safe_park(mgr)

        assert order == ["temp", "rh", "lamp"]

    def test_a_mock_rh_controller_is_turned_off_too(self):
        """``safe_park`` cannot tell a mock from a real driver, so the mock's own
        ``safe_off`` is what makes every simulated park honest."""
        from softae.drivers.mock_rh_controller import MockRHController

        rh = MockRHController(name="rh_controller")
        rh._state = InstrumentState.CONNECTED
        rh.set_setpoint(45.0)
        rh.start()

        result = safe_park(_manager(rh_controller=rh))

        assert result.ok
        assert rh._duty == 0.0
        assert rh._running is False
        assert rh._setpoint == 0.0


# ── The dry-purge park ───────────────────────────────────────────────────────

class TestDryPurgePark:
    """The other humidifier end state, and it is **opt-in**.

    ``ctrl`` near 0 is dry air; ``ctrl == 0`` exactly is a firmware special case
    that shuts both Aalborg PSVs, so a park to duty 0 leaves no flow and room air
    wins — a chamber held at 10 %RH goes to ~50 %RH in tens of seconds and every
    restart costs a re-drying. ``rh_dry_purge=True`` parks to ``out_min`` instead
    and lets the Trinket's ~25 s deadman close the valves.

    Every existing caller keeps duty 0. An E-stop that leaves gas flowing for
    25 s is not an E-stop, and that is an operator ruling, not a default.
    """

    def _mock_rh(self, **config):
        from softae.drivers.mock_rh_controller import MockRHController

        rh = MockRHController(name="rh_controller", config=config or None)
        rh._state = InstrumentState.CONNECTED
        rh.set_setpoint(45.0)
        rh.start()
        return rh

    # -- the default path is untouched ----------------------------------------

    def test_the_park_defaults_to_zeroing_and_never_dry_purges(self):
        """Additive means additive: the shipped call is byte-for-byte itself."""
        mgr = _manager()

        result = safe_park(mgr, reason="unit test")

        mgr._insts["rh_controller"].safe_off.assert_called_once()
        mgr._insts["rh_controller"].safe_dry.assert_not_called()
        assert "humidifier off (PID stopped, duty 0)" in result.commanded

    def test_the_default_park_leaves_a_real_driver_at_duty_zero(self):
        rh = self._mock_rh()

        result = safe_park(_manager(rh_controller=rh))

        assert result.ok
        assert rh._duty == 0.0
        assert rh.last_safe_dry_duty == 0.0

    # -- the opt-in path -------------------------------------------------------

    def test_the_dry_purge_park_calls_safe_dry_and_not_safe_off(self):
        mgr = _manager()

        safe_park(mgr, rh_dry_purge=True)

        mgr._insts["rh_controller"].safe_dry.assert_called_once()
        mgr._insts["rh_controller"].safe_off.assert_not_called()

    def test_the_dry_purge_park_leaves_a_real_driver_at_out_min(self):
        rh = self._mock_rh()

        result = safe_park(_manager(rh_controller=rh), rh_dry_purge=True)

        assert result.ok
        assert rh._duty == pytest.approx(0.01)
        assert rh._running is False
        assert rh._setpoint == 0.0

    def test_the_dry_purge_report_names_the_duty_and_the_deadman(self):
        rh = self._mock_rh(out_min=0.05)

        result = safe_park(_manager(rh_controller=rh), rh_dry_purge=True)

        (line,) = [c for c in result.commanded if "DRY-PURGED" in c]
        assert "0.05" in line
        assert f"{RH_DEADMAN_S:g} s" in line

    def test_the_dry_purge_report_cannot_be_read_as_the_failure_message(self):
        """It deliberately leaves the device commanded, so it must read as its
        own success — never as a near-miss of ``HUMIDIFIER WAS NOT TURNED OFF``.

        The two describe opposite situations: that one is a humidifier nobody
        could turn off; this one is dry air left flowing on purpose.
        """
        rh = self._mock_rh()

        result = safe_park(_manager(rh_controller=rh), rh_dry_purge=True)

        assert result.ok
        assert result.errors == []
        assert result.headline() == (HEADLINE_COMMANDED, False)
        (line,) = [c for c in result.commanded if "DRY-PURGED" in c]
        assert "DELIBERATE" in line
        assert "not a humidifier left on" in line
        # None of the vocabulary the genuine failures use.
        for forbidden in ("NOT TURNED OFF", "was not zeroed", "could not be"):
            assert forbidden not in line
        # And it is filed as a claim, not as a fault: the paragraph an operator
        # reads has a Commanded block carrying this line and no Failed block at
        # all. Asserted on the rendered text because that is the surface the
        # confusion would actually happen on.
        paragraph = result.describe()
        assert "Failed:" not in paragraph
        commanded_block = paragraph.split("Commanded (sent")[1]
        assert line in commanded_block

    def test_the_dry_purge_park_still_happens_before_the_lamp(self):
        order: list[str] = []
        mgr = _manager()
        mgr._insts["temp_controller"].write_sp.side_effect = (
            lambda *a, **k: order.append("temp"))
        mgr._insts["rh_controller"].safe_dry.side_effect = lambda: order.append("rh")
        mgr._insts["lamp"].off.side_effect = lambda: order.append("lamp")

        safe_park(mgr, rh_dry_purge=True)

        assert order == ["temp", "rh", "lamp"]

    # -- the ways it can go wrong ---------------------------------------------

    def test_a_degenerate_out_min_is_an_error_not_a_silent_dry_purge(self):
        """The driver's fallback leaves the hardware safe; the *report* still
        refuses to call it a dry purge, because a one-character config mistake
        that silently disables the feature is met months later as unexplained
        RH collapses with nothing naming the cause."""
        rh = self._mock_rh(out_min=0.0)

        result = safe_park(_manager(rh_controller=rh), rh_dry_purge=True)

        assert not result.ok
        assert any("out_min" in e for e in result.errors)
        assert not any("DRY-PURGED" in c for c in result.commanded)
        # ...and the hardware really did end up safe, which the message says.
        assert rh._duty == 0.0
        assert any("zeroed instead" in e for e in result.errors)

    def test_a_driver_with_no_safe_dry_is_zeroed_and_the_gap_is_reported(self):
        """Unlike the pump halt's refusal to fall back: there the fallback was a
        dispense, here it is the strictly safer end state."""
        rh = MagicMock(spec=["is_connected", "safe_off", "last_safe_off_error"])
        rh.is_connected = True
        rh.last_safe_off_error = ""

        result = safe_park(_manager(rh_controller=rh), rh_dry_purge=True)

        rh.safe_off.assert_called_once()
        assert any("no safe_dry()" in e for e in result.errors)
        assert "humidifier off (PID stopped, duty 0)" in result.commanded

    def test_a_raising_safe_dry_does_not_block_the_lamp(self):
        mgr = _manager()
        mgr._insts["rh_controller"].safe_dry.side_effect = RuntimeError("wedged")

        result = safe_park(mgr, rh_dry_purge=True)

        assert any("humidity: wedged" in e for e in result.errors)
        mgr._insts["lamp"].off.assert_called_once()

    def test_a_failed_dry_write_is_reported_as_an_error_not_commanded(self):
        """``safe_dry`` swallows a comms failure to keep the never-raise
        contract, so a park that only watched for an exception would claim a
        write that never landed."""
        mgr = _manager()
        mgr._insts["rh_controller"].last_safe_dry_error = "port went away"

        result = safe_park(mgr, rh_dry_purge=True)

        assert any("port went away" in e for e in result.errors)
        assert not any("DRY-PURGED" in c for c in result.commanded)

    def test_a_disconnected_rh_controller_is_skipped_on_the_dry_path_too(self):
        mgr = _manager()
        mgr._insts["rh_controller"].is_connected = False

        result = safe_park(mgr, rh_dry_purge=True)

        assert result.ok
        mgr._insts["rh_controller"].safe_dry.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_async_park_forwards_the_dry_purge_flag(self):
        mgr = _manager()

        await safe_park_async(mgr, rh_dry_purge=True)

        mgr._insts["rh_controller"].safe_dry.assert_called_once()


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


# ── Did the park reach the rig at all? ───────────────────────────────────────

class TestCommandedAnything:
    """The question ``ok`` was being asked and cannot answer.

    ``ok`` is a statement about exceptions and stays one (see
    ``test_ok_still_means_nothing_raised``, which is the pin these tests sit
    beside rather than replace). ``commanded_anything`` is the statement about
    whether anything was actually sent.
    """

    def test_result_with_no_instruments_commanded_anything_false(self):
        assert SafeParkResult(skipped=["lamp: not connected"]).commanded_anything \
            is False

    def test_result_with_a_commanded_write_commanded_anything_true(self):
        assert SafeParkResult(commanded=["lamp off"]).commanded_anything is True

    def test_result_with_only_a_verified_axis_commanded_anything_true(self):
        """Empty on this rig today; if an axis ever graduates to read-back, a
        park that verified something certainly commanded something."""
        assert SafeParkResult(verified=["temperature at setpoint"]) \
            .commanded_anything is True

    def test_result_unverifiable_head_alone_commanded_anything_false(self):
        """The head note is on *every* result, including the empty park. If it
        counted, no park could ever report that it commanded nothing."""
        assert SafeParkResult(unverifiable=["head"]).commanded_anything is False

    def test_park_of_absent_manager_commanded_anything_false(self):
        """The live defect, end to end: nothing raised, nothing sent."""
        mgr = MagicMock()
        mgr.get.side_effect = KeyError("nope")

        result = safe_park(mgr)

        assert result.ok is True                    # unchanged, and correct
        assert result.commanded_anything is False

    def test_park_of_disconnected_manager_commanded_anything_false(self):
        mgr = _manager()
        for inst in mgr._insts.values():
            inst.is_connected = False

        result = safe_park(mgr)

        assert result.ok is True
        assert result.commanded_anything is False
        assert len(result.skipped) == 4

    def test_park_of_connected_manager_commanded_anything_true(self):
        assert safe_park(_manager()).commanded_anything is True

    def test_park_with_one_connected_instrument_commanded_anything_true(self):
        """A partial rig still reached something — this is not the empty case."""
        mgr = _manager()
        mgr._insts["syringe"].is_connected = False
        mgr._insts["temp_controller"].is_connected = False

        assert safe_park(mgr).commanded_anything is True


class TestHeadline:
    """One place decides the three-way, so no two dialogs can disagree."""

    def test_headline_commanded_is_the_reassuring_sentence_and_not_severe(self):
        assert safe_park(_manager()).headline() == (HEADLINE_COMMANDED, False)

    def test_headline_with_errors_is_partial_and_severe(self):
        result = SafeParkResult(commanded=["lamp off"], errors=["pump 0: dead"])
        assert result.headline() == (HEADLINE_PARTIAL, True)

    def test_headline_with_nothing_commanded_is_the_nothing_sentence_and_severe(self):
        mgr = MagicMock()
        mgr.get.side_effect = KeyError("nope")
        assert safe_park(mgr).headline() == (HEADLINE_NOTHING, True)

    def test_headline_with_errors_and_nothing_commanded_prefers_partial(self):
        """A rig that answered and refused *was* connected, so the "no
        instrument was connected" sentence would be a second false statement
        replacing the first."""
        result = SafeParkResult(errors=["pumps: driver exposes no halt_pump()"])
        assert result.commanded_anything is False
        assert result.headline() == (HEADLINE_PARTIAL, True)

    def test_headline_nothing_sentence_names_this_process(self):
        """It must not read as a claim about the rig — the instruments may be
        perfectly alive and owned by someone else."""
        assert "this process" in HEADLINE_NOTHING


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

    @pytest.mark.asyncio
    async def test_async_park_turns_the_humidifier_off(self):
        """The async wrapper is ``to_thread(safe_park, …)``; it inherits the step
        rather than restating it, and this is what says so."""
        mgr = _manager()
        result = await safe_park_async(mgr, reason="async test")
        mgr._insts["rh_controller"].safe_off.assert_called_once()
        assert any("humidifier off" in c for c in result.commanded)


class TestDryPurgeHumidifier:
    """Step 5 on its own, for the one exit that must leave the heater alone.

    ``--end-state hold`` means "the condition stands, a human is here". It
    cannot call :func:`safe_park` — that drives the heater to 10 °C, which is
    the opposite of holding — and it cannot leave the humidifier untouched
    either, because ``AsyncRHController.disconnect`` commands duty 0 on its way
    out and duty 0 is the firmware's valve shutoff.
    """

    def _rh(self, **config):
        from softae.drivers.mock_rh_controller import MockRHController

        rh = MockRHController(config=config or None)
        rh._state = InstrumentState.CONNECTED
        return rh

    def test_dry_purge_touches_only_the_humidifier(self):
        """The heater, the pumps, the head and the lamp are all left alone."""
        mgr = _manager()
        result = dry_purge_humidifier(mgr, reason="unit test")

        mgr._insts["rh_controller"].safe_dry.assert_called_once()
        mgr._insts["rh_controller"].safe_off.assert_not_called()
        mgr._insts["temp_controller"].write_sp.assert_not_called()
        mgr._insts["syringe"].halt_pump.assert_not_called()
        mgr._insts["syringe"].head_retract.assert_not_called()
        mgr._insts["lamp"].off.assert_not_called()
        # And it says nothing about anti-clog purging, because it suspended
        # nothing: no park latch is set by this call.
        assert result.notes == []

    def test_dry_purge_leaves_out_min_on_the_wire_not_zero(self):
        """The duty itself, off a driver with nothing patched over it.

        A spy on ``safe_dry`` would pass against a no-op; the number is what
        the Trinket would be sitting at, and ``0.0`` is the one value that
        means *both valves shut*.
        """
        rh = self._rh()
        result = dry_purge_humidifier(_manager(rh_controller=rh))

        assert rh._duty == pytest.approx(rh._out_min)
        assert rh._duty > 0.0
        assert rh._running is False
        assert rh._setpoint == 0.0
        assert result.ok and result.commanded

    def test_dry_purge_names_the_duty_and_what_closes_the_valves(self):
        """A standing command reported as a success has to say why it stands."""
        result = dry_purge_humidifier(_manager(rh_controller=self._rh()))
        text = " ".join(result.commanded)
        assert "DRY-PURGED" in text
        assert f"~{RH_DEADMAN_S:g} s" in text
        assert not any("DRY-PURGED" in e for e in result.errors)

    def test_dry_purge_reports_a_degenerate_out_min_as_an_error(self):
        """A config typo must not disable the purge silently — and the hardware
        still ends up safe, zeroed by the driver's own fallback."""
        rh = self._rh(out_min=0.0)
        result = dry_purge_humidifier(_manager(rh_controller=rh))

        assert not result.ok
        assert any("out_min" in e for e in result.errors)
        assert rh._duty == 0.0

    def test_dry_purge_skips_a_humidifier_this_process_never_opened(self):
        """Called after the port closed, or on a refusal that never connected."""
        rh = self._rh()
        rh._state = InstrumentState.DISCONNECTED
        result = dry_purge_humidifier(_manager(rh_controller=rh))

        assert result.skipped and not result.errors
        assert not result.commanded_anything
