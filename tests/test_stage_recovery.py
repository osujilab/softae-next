"""Graceful stage-timeout recovery: driver self-heal, dispense-committed
gating, and executor channel replay/skip.

Covers the cascade added to handle intermittent ``VI_ERROR_TMO`` on the Newport
ESP301 stage without aborting an HT/AE campaign:

1. :class:`AsyncStage` self-heals a VISA timeout (clear+retry → session reset)
   for idempotent moves;
2. :meth:`AsyncLiquidHandler.single_drop_simul` tags failures with whether
   elution was already committed (the double-dispense point of no return);
3. :class:`WorkflowExecutor` (``continue_on_error=True``) replays a channel from
   its precondition step when safe, else skips it and continues the campaign.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.errors import (
    AbortedError,
    CommunicationError,
    SafetyError,
    WorkflowError,
)
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep


def run(coro):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


_TMO = "VI_ERROR_TMO (-1073807339): Timeout expired before operation completed."


# ═══════════════════════════════════════════════════════════════════════
# 1. AsyncStage self-heal cascade (fake PyVISA)
# ═══════════════════════════════════════════════════════════════════════


class TestStageSelfHeal:
    """The stage transparently recovers idempotent moves from VISA timeouts."""

    def _make_stage(self, *, soft=2, resets=1):
        pv = types.ModuleType("pyvisa")
        mock_inst = MagicMock()
        mock_rm = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        pv.ResourceManager = MagicMock(return_value=mock_rm)
        with patch.dict("sys.modules", {"pyvisa": pv}):
            from softae.drivers.async_stage import AsyncStage

            s = AsyncStage(
                name="stage",
                config={
                    "port": "ASRL99::INSTR",
                    "tmo_soft_retries": soft,
                    "tmo_session_resets": resets,
                    "tmo_backoff_s": 0.0,  # no sleeps in tests
                },
            )
            run(s.connect())
        return s, pv, mock_rm, mock_inst

    def test_connect_sets_timeout_and_termination(self):
        s, _, _, mock_inst = self._make_stage()
        assert mock_inst.timeout == 8000
        assert mock_inst.write_termination == "\r"
        assert mock_inst.read_termination == "\r"

    def test_soft_retry_recovers(self):
        """A single transient timeout on the move command is re-issued and clears."""
        s, _, _, mock_inst = self._make_stage()
        mock_inst.query.return_value = "1\r\n"        # MD? → done
        mock_inst.query_ascii_values.return_value = [10.0]
        # First write raises a timeout, subsequent writes succeed.
        calls = {"n": 0}

        def _write(cmd):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception(_TMO)

        mock_inst.write.side_effect = _write
        s.move_to(10.0, 10.0)                          # must NOT raise
        mock_inst.clear.assert_called()                # soft tier issued a clear

    def test_session_reset_recovers(self):
        """When soft retries are exhausted, the session is reset then the op retries."""
        s, pv, _, mock_inst = self._make_stage(soft=1, resets=1)
        mock_inst.query.return_value = "1\r\n"
        mock_inst.query_ascii_values.return_value = [0.0]
        # Fail the first 2 attempts (initial + 1 soft retry), then succeed.
        calls = {"n": 0}

        def _write(cmd):
            calls["n"] += 1
            if calls["n"] <= 2 and cmd.startswith("1PA"):
                raise Exception(_TMO)

        mock_inst.write.side_effect = _write
        # The session-reset path re-imports pyvisa, so the fake module must stay
        # installed for the duration of the move (connect() exited the patch).
        with patch.dict("sys.modules", {"pyvisa": pv}):
            s.move_to(5.0, 5.0)
        # A reset re-opened the resource (ResourceManager built a second time).
        assert pv.ResourceManager.call_count >= 2

    def test_exhausted_cascade_raises_communication_error(self):
        s, _, _, mock_inst = self._make_stage(soft=0, resets=0)
        mock_inst.write.side_effect = Exception(_TMO)
        with pytest.raises(CommunicationError):
            s.move_to(1.0, 1.0)

    def test_non_recoverable_error_not_retried(self):
        """A non-timeout error propagates immediately without retrying."""
        s, _, _, mock_inst = self._make_stage()
        mock_inst.write.side_effect = ValueError("bad command")
        with pytest.raises(ValueError):
            s.move_to(1.0, 1.0)
        assert mock_inst.write.call_count == 1  # no retry

    def test_bounds_violation_fails_fast(self):
        s, _, _, _ = self._make_stage()
        with pytest.raises(SafetyError):
            s.move_to(9999.0, 9999.0)  # outside default travel limits


# ═══════════════════════════════════════════════════════════════════════
# 2. dispense_committed annotation on the composite deposit routine
# ═══════════════════════════════════════════════════════════════════════


class TestDispenseCommitted:
    def _mgr(self, *, stage_cfg=None, syr_cfg=None):
        m = create_mock_manager(
            config={"stage": stage_cfg or {}, "syringe": syr_cfg or {}}
        )
        run(m.connect_all())
        return m

    def _deposit(self, mgr):
        return mgr.get("liquid_handler").execute(
            "single_drop_simul",
            x=2.0, y=2.0, wick_x=1.0, wick_y=1.0,
            ids=[0, 1], disp_rate=50.0, vols=[3.0, 3.0], time_scale=0.0,
        )

    def test_stage_timeout_before_dispense_not_committed(self):
        """A failure on the initial electrode move must be flagged not-committed."""
        mgr = self._mgr(stage_cfg={"fail_at_xy": [2.0, 2.0], "fail_at_xy_times": 1})
        with pytest.raises(CommunicationError) as ei:
            run(self._deposit(mgr))
        assert getattr(ei.value, "dispense_committed", None) is False

    def test_syringe_timeout_marks_committed(self):
        """A failure once a pump is commanded must be flagged committed."""
        mgr = self._mgr(syr_cfg={"fail_next_n": 1})
        with pytest.raises(CommunicationError) as ei:
            run(self._deposit(mgr))
        assert getattr(ei.value, "dispense_committed", None) is True


# ═══════════════════════════════════════════════════════════════════════
# 3. Executor channel replay / skip
# ═══════════════════════════════════════════════════════════════════════


def _precondition(ch: str) -> WorkflowStep:
    return WorkflowStep(
        name=f"precondition_ch{ch}",
        instrument="liquid_handler",
        method="precondition_flush",
        params=dict(
            flush_x=0.0, flush_y=0.0, wick_x=1.0, wick_y=1.0,
            ids=[0, 1], vol_list=[5.0, 5.0], rate_total=100.0, time_scale=0.0,
        ),
        tags={"channel": ch, "phase": "precondition"},
    )


def _deposit(ch: str) -> WorkflowStep:
    return WorkflowStep(
        name=f"deposit_ch{ch}",
        instrument="liquid_handler",
        method="single_drop_simul",
        params=dict(
            x=2.0, y=2.0, wick_x=1.0, wick_y=1.0,
            ids=[0, 1], disp_rate=50.0, vols=[3.0, 3.0], time_scale=0.0,
        ),
        tags={"channel": ch, "phase": "deposit"},
    )


def _measure(ch: str) -> WorkflowStep:
    # Stand-in for a per-channel EIS step: instrument-free, always succeeds.
    return WorkflowStep(
        name=f"measure_ch{ch}", instrument="control", method="wait",
        params={"seconds": 0.0}, tags={"channel": ch, "phase": "eis"},
    )


class TestExecutorRecovery:
    def _mgr(self, config=None):
        m = create_mock_manager(config=config or {})
        run(m.connect_all())
        return m

    def test_channel_replayed_from_precondition_on_stage_timeout(self):
        """An uncommitted deposit timeout replays the channel from precondition."""
        mgr = self._mgr({"stage": {"fail_at_xy": [2.0, 2.0], "fail_at_xy_times": 1}})
        wf = Workflow(
            name="replay",
            setup=[_precondition("5"), _deposit("5"), _precondition("6"), _deposit("6")],
        )
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1)
        completed: list[str] = []
        recovered: list[str] = []
        ex.on_step_complete = lambda s, *a: completed.append(s.name)
        ex.on_step_recover = lambda step, err, attempt: recovered.append(step.name)

        run(ex.run(wf))

        assert ex.state is ExecutorState.COMPLETED
        # Channel 5 precondition ran twice (original + replay); its deposit
        # completed on the replay. Channel 6 ran once, cleanly.
        assert completed.count("precondition_ch5") == 2
        assert completed.count("deposit_ch5") == 1
        assert completed.count("deposit_ch6") == 1
        assert recovered == ["deposit_ch5"]

    def test_channel_skipped_when_dispense_committed(self):
        """A committed deposit timeout skips the channel's remaining steps."""
        mgr = self._mgr({"syringe": {"fail_next_n": 1}})
        wf = Workflow(
            name="skip",
            setup=[_deposit("5"), _measure("5"), _deposit("6"), _measure("6")],
        )
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1)
        completed: list[str] = []
        errored: list[str] = []
        skipped: list[str] = []
        ex.on_step_complete = lambda s, *a: completed.append(s.name)
        ex.on_step_error = lambda s, i, t, e: errored.append(s.name)
        ex.on_step_skipped = lambda s, i, t, r: skipped.append(s.name)

        run(ex.run(wf))

        assert ex.state is ExecutorState.COMPLETED
        assert errored == ["deposit_ch5"]           # committed → not replayed
        assert "measure_ch5" in skipped              # trailing channel step skipped
        assert completed.count("deposit_ch5") == 0   # never succeeded
        assert "deposit_ch6" in completed            # campaign continued
        assert "measure_ch6" in completed

    def test_deferred_measure_failure_replays_locally_not_the_deposit(self):
        """Batch layout: a recoverable EIS failure retries in place, not the cast.

        With measurement deferred into a per-batch block
        (deposit-all → measure-all), a channel's deposit and measure are NOT
        adjacent.  A recoverable measure failure must replay only the measure —
        replaying back to the (far earlier) deposit would re-dispense the drop.
        """
        # A stand-in EIS step: a stage move to a unique xy that fails once.
        def _measure_move(ch: str) -> WorkflowStep:
            return WorkflowStep(
                name=f"measure_ch{ch}", instrument="stage", method="move_to",
                params={"x": 9.0, "y": 9.0}, tags={"channel": ch, "phase": "eis"},
            )

        mgr = self._mgr({"stage": {"fail_at_xy": [9.0, 9.0], "fail_at_xy_times": 1}})
        wf = Workflow(
            name="deferred",
            setup=[_deposit("5"), _deposit("6"), _measure_move("5"), _measure_move("6")],
        )
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1)
        completed: list[str] = []
        recovered: list[str] = []
        ex.on_step_complete = lambda s, *a: completed.append(s.name)
        ex.on_step_recover = lambda step, err, attempt: recovered.append(step.name)

        run(ex.run(wf))

        assert ex.state is ExecutorState.COMPLETED
        assert recovered == ["measure_ch5"]              # only the measure replayed
        assert completed.count("deposit_ch5") == 1       # cast NOT re-run (no double-dispense)
        assert completed.count("measure_ch5") == 1       # measure succeeded on replay
        assert "measure_ch6" in completed

    def test_fail_fast_when_continue_on_error_disabled(self):
        mgr = self._mgr({"stage": {"fail_at_xy": [2.0, 2.0], "fail_at_xy_times": 99}})
        wf = Workflow(name="ff", setup=[_deposit("5")])
        ex = WorkflowExecutor(mgr, continue_on_error=False)
        with pytest.raises(WorkflowError):
            run(ex.run(wf))
        assert ex.state is ExecutorState.ERROR

    def test_campaign_level_step_hard_fails_even_with_recovery(self):
        """A step with no channel tag (e.g. startup flush) is never skipped."""
        mgr = self._mgr({"stage": {"fail_next_n": 99}})
        campaign_step = WorkflowStep(
            name="startup_flush", instrument="stage", method="move_to",
            params={"x": 0.0, "y": 0.0},  # no channel tag
        )
        wf = Workflow(name="camp", setup=[campaign_step])
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1)
        with pytest.raises(WorkflowError):
            run(ex.run(wf))
        assert ex.state is ExecutorState.ERROR


# ═══════════════════════════════════════════════════════════════════════
# 4. The mechanical ceiling: skipped-channel record + prompt-and-hold
# ═══════════════════════════════════════════════════════════════════════


def _move(ch: str, xy: float) -> WorkflowStep:
    """One channel-tagged stage move — the smallest wedgeable channel."""
    return WorkflowStep(
        name=f"move_ch{ch}", instrument="stage", method="move_to",
        params={"x": xy, "y": xy}, tags={"channel": ch, "phase": "eis"},
    )


class TestSkippedChannelRecord:
    """The executor counts what it abandons, so the run row can say so."""

    def _mgr(self, config=None):
        m = create_mock_manager(config=config or {})
        run(m.connect_all())
        return m

    def test_every_abandoned_channel_is_recorded_once(self):
        mgr = self._mgr({"stage": {"fail_next_n": 99}})
        wf = Workflow(name="wedged",
                      setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)])
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1)
        run(ex.run(wf))
        assert ex.state is ExecutorState.COMPLETED
        assert ex.skipped_channels == ["5", "6", "7"]

    def test_a_clean_run_records_an_empty_list_not_a_missing_one(self):
        mgr = self._mgr()
        wf = Workflow(name="clean", setup=[_move("5", 5.0)])
        ex = WorkflowExecutor(mgr, continue_on_error=True)
        run(ex.run(wf))
        assert ex.skipped_channels == []

    def test_the_record_describes_one_run_not_the_executor_lifetime(self):
        mgr = self._mgr({"stage": {"fail_next_n": 99}})
        ex = WorkflowExecutor(mgr, continue_on_error=True)
        run(ex.run(Workflow(name="first", setup=[_move("5", 5.0)])))
        assert ex.skipped_channels == ["5"]
        mgr.get("stage").reset_session()          # the wedge clears
        run(ex.run(Workflow(name="second", setup=[_move("6", 6.0)])))
        assert ex.skipped_channels == []


class TestConsecutiveFailureCeiling:
    """Prompt-and-hold, never an automatic park while someone may be watching."""

    def _mgr(self, config=None):
        m = create_mock_manager(config=config or {})
        run(m.connect_all())
        return m

    def _wedged(self, **kwargs):
        mgr = self._mgr({"stage": {"fail_next_n": 99}})
        wf = Workflow(name="wedged",
                      setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)])
        ex = WorkflowExecutor(
            mgr, continue_on_error=True, max_channel_retries=1, **kwargs
        )
        return mgr, wf, ex

    def test_failures_below_the_ceiling_never_interrupt_the_plate(self):
        """One bad well must still not cost a plate."""
        _, wf, ex = self._wedged(max_consecutive_channel_failures=4)
        holds: list[tuple] = []
        ex.on_channel_failure_hold = lambda *a: holds.append(a)
        run(ex.run(wf))
        assert holds == []
        assert ex.state is ExecutorState.COMPLETED

    def test_the_ceiling_pauses_and_asks_rather_than_parking(self):
        _, wf, ex = self._wedged(max_consecutive_channel_failures=2)
        holds: list[tuple] = []

        def _answer(channel, consecutive, timeout_s):
            holds.append((channel, consecutive, ex.state))
            ex.resume()                      # the operator says "carry on"

        ex.on_channel_failure_hold = _answer
        run(ex.run(wf))

        # Asked on the 2nd consecutive failure, with the run ALREADY paused — so a
        # synchronous answer is never dropped — and the plate continued after it.
        assert holds == [("6", 2, ExecutorState.PAUSED)]
        assert ex.state is ExecutorState.COMPLETED
        assert ex.skipped_channels == ["5", "6", "7"]

    def test_a_completed_channel_resets_the_streak(self):
        """Consecutive means consecutive: a good well breaks the chain."""
        mgr = self._mgr({"stage": {"fail_next_n": 2}})   # only channel 5 wedges
        stage = mgr.get("stage")
        wf = Workflow(name="intermittent",
                      setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)])
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1,
                              max_consecutive_channel_failures=2)
        holds: list[tuple] = []
        ex.on_channel_failure_hold = lambda *a: holds.append(a)
        # Re-arm the wedge once channel 6 is through, so 5 and 7 fail and 6 does
        # not: without the reset that is a streak of 2 and a spurious hold.
        ex.on_step_complete = lambda s, *a: (
            setattr(stage, "_fail_next_n", 2) if s.name == "move_ch6" else None
        )

        run(ex.run(wf))

        assert ex.skipped_channels == ["5", "7"]
        assert holds == []

    def test_an_operator_abort_from_the_hold_stops_the_plate(self):
        """The prompt is never a lockout — abort is honoured immediately."""
        _, wf, ex = self._wedged(max_consecutive_channel_failures=2)
        ex.on_channel_failure_hold = lambda *a: ex.abort()

        with pytest.raises(AbortedError):
            run(ex.run(wf))

        assert ex.state is ExecutorState.ABORTED
        assert "7" not in ex.skipped_channels     # channel 7 was never attempted

    def test_an_unanswered_hold_parks_and_stops_the_run(self, monkeypatch):
        """Nobody answered, so the run is unattended whatever the design assumed."""
        import softae.core.safe_park as safe_park_mod

        parked: list[str] = []

        async def _fake_park(manager, *, reason="", **kwargs):
            parked.append(reason)
            return safe_park_mod.SafeParkResult(commanded=["pump 0 halted"])

        monkeypatch.setattr(safe_park_mod, "safe_park_async", _fake_park)

        _, wf, ex = self._wedged(
            max_consecutive_channel_failures=2, channel_hold_timeout_s=0.05,
        )
        prompted: list[tuple] = []
        ex.on_channel_failure_hold = lambda *a: prompted.append(a)

        with pytest.raises(AbortedError):
            run(ex.run(wf))

        assert len(prompted) == 1                 # it asked first
        assert len(parked) == 1                   # …then parked on silence
        assert "no operator answered" in parked[0]
        assert ex.state is ExecutorState.ABORTED

    def test_a_park_failure_still_stops_the_run(self, monkeypatch):
        """A stop that fails must not leave the plate running on."""
        import softae.core.safe_park as safe_park_mod

        async def _boom(manager, **kwargs):
            raise RuntimeError("serial port gone")

        monkeypatch.setattr(safe_park_mod, "safe_park_async", _boom)

        _, wf, ex = self._wedged(
            max_consecutive_channel_failures=2, channel_hold_timeout_s=0.05,
        )
        with pytest.raises(AbortedError):
            run(ex.run(wf))
        assert ex.state is ExecutorState.ABORTED

    def test_the_ceiling_is_opt_in_not_on_by_default(self):
        """`AutonomousLoop._run_workflow` builds an executor with
        ``continue_on_error=True`` and wires no prompt. On by default, the ceiling
        would give an unattended campaign a silent hour-long stall instead of a
        question nobody is present to answer — so the default is off and the HT
        tab, the caller this was specified for, asks for it explicitly.
        """
        mgr = self._mgr({"stage": {"fail_next_n": 99}})
        wf = Workflow(name="wedged",
                      setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)])
        ex = WorkflowExecutor(mgr, continue_on_error=True, max_channel_retries=1)
        assert ex.max_consecutive_channel_failures == 0
        holds: list[tuple] = []
        ex.on_channel_failure_hold = lambda *a: holds.append(a)

        run(ex.run(wf))

        assert holds == []                                  # never held
        assert ex.skipped_channels == ["5", "6", "7"]       # still recorded

    def test_the_ceiling_is_off_for_the_fail_fast_executor(self):
        """Without continue_on_error there is no skip path to count."""
        mgr = self._mgr({"stage": {"fail_next_n": 99}})
        wf = Workflow(name="ff", setup=[_move("5", 5.0)])
        ex = WorkflowExecutor(mgr, continue_on_error=False)
        holds: list[tuple] = []
        ex.on_channel_failure_hold = lambda *a: holds.append(a)
        with pytest.raises(WorkflowError):
            run(ex.run(wf))
        assert holds == []
        assert ex.skipped_channels == []
