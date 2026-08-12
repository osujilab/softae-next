"""Stage-motion head guard and the rig-activity registry (P8).

The guard exists because head-down stopped being a brief interval inside a known
sequence and became the rig's **resting** state. Previously every stage move was
preceded by ``head_retract`` purely by convention; a path that forgets now finds
the head down by default rather than by accident, and drags the tip.
"""

from __future__ import annotations

import pytest

from softae.core.rig_activity import RigActivity
from softae.drivers.mock_stage import MockStage
from softae.drivers.mock_syringe import MockSyringe
from softae.errors import SafetyError


def _rig(*, head_up: bool):
    stage, syringe = MockStage(name="stage"), MockSyringe(name="syringe")
    syringe.set_head_state(head_up)
    stage.head_source = syringe
    return stage, syringe


class TestHeadGuard:
    def test_a_move_with_the_head_down_is_refused(self):
        stage, _ = _rig(head_up=False)
        with pytest.raises(SafetyError, match="head is lowered"):
            stage.move_to(10.0, 10.0)

    def test_a_move_with_the_head_up_is_allowed(self):
        stage, _ = _rig(head_up=True)
        stage.move_to(10.0, 10.0)
        assert stage.live_position() == pytest.approx((10.0, 10.0), abs=0.01)

    def test_a_refused_move_does_not_move_the_stage(self):
        stage, _ = _rig(head_up=False)
        before = stage.live_position()
        with pytest.raises(SafetyError):
            stage.move_to(10.0, 10.0)
        assert stage.live_position() == before

    def test_an_explicit_opt_out_is_honoured(self):
        """In-drop patterning (star_mix) legitimately traces with the tip down."""
        stage, _ = _rig(head_up=False)
        stage.move_to(10.0, 10.0, head_may_be_down=True)
        assert stage.live_position() == pytest.approx((10.0, 10.0), abs=0.01)

    def test_move_by_is_guarded_too(self):
        stage, _ = _rig(head_up=False)
        with pytest.raises(SafetyError):
            stage.move_by(1.0, 1.0)

    def test_an_unwired_stage_still_moves(self):
        """A rig with no syringe configured has nothing to protect."""
        MockStage(name="stage").move_to(10.0, 10.0)

    def test_retracting_clears_the_refusal(self):
        stage, syringe = _rig(head_up=False)
        with pytest.raises(SafetyError):
            stage.move_to(10.0, 10.0)
        syringe.head_retract()
        stage.move_to(10.0, 10.0)          # now fine

    def test_the_guard_reaches_the_real_driver_too(self):
        """Mock and real share the check, so they cannot drift on what is legal."""
        from softae.drivers.async_stage import AsyncStage
        from softae.drivers.contracts import check_head_clear_to_move

        assert hasattr(AsyncStage, "_check_head_clear")
        syringe = MockSyringe(name="syringe")
        syringe.set_head_state(False)
        with pytest.raises(SafetyError):
            check_head_clear_to_move(syringe, instrument="stage")


class TestAttachHeadGuard:
    def test_attaching_wires_the_stage_to_the_syringe(self):
        from softae.core.hardware_safety import attach_head_guard
        from softae.drivers.mock_factory import create_mock_manager

        manager = create_mock_manager(config={})
        assert attach_head_guard(manager) is True

        syringe = manager.get("syringe")
        syringe.set_head_state(False)
        with pytest.raises(SafetyError):
            manager.get("stage").move_to(10.0, 10.0)

    def test_a_rig_missing_an_instrument_is_not_an_error(self):
        from softae.core.hardware_safety import attach_head_guard

        class _Empty:
            def get(self, name):
                raise KeyError(name)

        assert attach_head_guard(_Empty()) is False


class TestRigActivity:
    def test_a_fresh_registry_is_idle(self):
        activity = RigActivity()
        assert not activity.busy
        assert activity.describe() == "idle"

    def test_a_claim_makes_it_busy(self):
        activity = RigActivity()
        activity.acquire("campaign:demo")
        assert activity.busy
        assert "campaign:demo" in activity.describe()

    def test_releasing_returns_it_to_idle(self):
        activity = RigActivity()
        activity.acquire("ht")
        activity.release("ht")
        assert not activity.busy

    def test_claims_are_reentrant(self):
        """A nested claim must not be released early by the inner exit."""
        activity = RigActivity()
        activity.acquire("ht")
        activity.acquire("ht")
        activity.release("ht")
        assert activity.busy
        activity.release("ht")
        assert not activity.busy

    def test_releasing_an_unheld_claim_is_tolerated(self):
        """It runs on a cleanup path; raising there would mask the real fault."""
        RigActivity().release("never-acquired")

    def test_the_context_manager_releases_on_an_exception(self):
        """A leaked claim silently disables purging for the rest of the session."""
        activity = RigActivity()
        with pytest.raises(RuntimeError):
            with activity.claimed("ht"):
                raise RuntimeError("run blew up")
        assert not activity.busy

    def test_independent_owners_are_tracked_separately(self):
        activity = RigActivity()
        activity.acquire("ht")
        activity.acquire("campaign")
        activity.release("ht")
        assert activity.busy
        assert activity.owners() == ("campaign",)


class _OkInstrument:
    """Minimal connected instrument — the executor only needs execute()."""

    async def execute(self, method_name, **kwargs):
        return {"ok": True}


class _OkManager:
    def get(self, name):
        return _OkInstrument()


class TestConcurrentPurgeWindow:
    """The second in-run mechanism: purge *beside* an opaque blocking step.

    An EIS sweep is a single blocking read with no interior yield point, so the
    poll-hook pattern that serves a long anneal cannot serve it.
    """

    def _executor(self, manager=None):
        from softae.workflows.workflow_executor import WorkflowExecutor

        return WorkflowExecutor(manager if manager is not None else _OkManager())

    def _step(self, *, window: bool):
        from softae.workflows.workflow_model import WorkflowStep

        step = WorkflowStep(name="measure_eis_ch1", instrument="espico",
                            method="measure", params={})
        return step.with_tags(purge_window="concurrent") if window else step

    def test_a_tagged_step_opens_a_window(self):
        import asyncio

        ex = self._executor()
        seen: list = []
        ex.on_purge_window = lambda step: seen.append(step.name)

        asyncio.run(ex._run_step(self._step(window=True), 0, 1))
        assert seen == ["measure_eis_ch1"]

    def test_an_untagged_step_opens_no_window(self):
        import asyncio

        ex = self._executor()
        seen: list = []
        ex.on_purge_window = lambda step: seen.append(step.name)

        asyncio.run(ex._run_step(self._step(window=False), 0, 1))
        assert seen == []

    def test_the_purge_is_joined_before_the_step_returns(self):
        """THE safety property: the next step must never start mid-purge."""
        import asyncio
        import time

        ex = self._executor()
        finished: list = []

        def _slow(step):
            time.sleep(0.15)
            finished.append("purge")

        ex.on_purge_window = _slow
        asyncio.run(ex._run_step(self._step(window=True), 0, 1))

        # Already recorded by the time _run_step returned — not abandoned.
        assert finished == ["purge"]

    def test_a_purge_failure_does_not_fail_the_step(self):
        import asyncio

        ex = self._executor()
        ex.on_purge_window = lambda step: (_ for _ in ()).throw(
            RuntimeError("purge blew up"))

        asyncio.run(ex._run_step(self._step(window=True), 0, 1))   # must not raise

    def test_a_failing_step_still_joins_its_purge(self):
        """The join is in a finally — a step error must not abandon fluid."""
        import asyncio

        from softae.workflows.workflow_model import WorkflowStep

        class _Boom:
            def get(self, name):
                raise KeyError(name)

        ex = self._executor(_Boom())
        joined: list = []
        ex.on_purge_window = lambda step: joined.append("done")

        step = WorkflowStep(name="measure", instrument="espico", method="x",
                            params={}).with_tags(purge_window="concurrent")
        with pytest.raises(Exception):
            asyncio.run(ex._run_step(step, 0, 1))
        assert joined == ["done"]

    def test_a_retry_does_not_stack_another_purge(self):
        """A retrying step already failed once; do not pile actuation onto it."""
        import asyncio

        from softae.workflows.workflow_model import WorkflowStep

        class _Boom:
            def get(self, name):
                raise KeyError(name)

        ex = self._executor(_Boom())
        opened: list = []
        ex.on_purge_window = lambda step: opened.append(1)

        step = WorkflowStep(name="measure", instrument="espico", method="x",
                            params={}, retry=2).with_tags(
                                purge_window="concurrent")
        with pytest.raises(Exception):
            asyncio.run(ex._run_step(step, 0, 1))
        assert len(opened) == 1        # three attempts, one window
