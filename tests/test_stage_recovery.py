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
from softae.errors import CommunicationError, SafetyError, WorkflowError
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
