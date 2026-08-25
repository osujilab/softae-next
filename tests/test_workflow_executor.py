"""``WorkflowExecutor.on_pause_hold`` — a run hands the rig back while it is *held*.

The operator's ruling is that pausing a run re-enables Manual Control. The claim
is what refuses Manual Control, so a paused run's claim must be **suspended** —
and a suspended owner is the one :meth:`RigActivity.conflicts` skips, which is
how manual control comes back.

The whole difficulty is *when*. ``pause()`` sets a flag; the executor keeps
driving until the top of the next tier or step. Suspending on the state change —
which is the obvious hook, and passes every test that does not model an
in-flight step — hands the syringe to Manual Control in the middle of a
dispense. So the callback fires at the **hold**, never at the request, and
:class:`TestPauseHoldFiresAtTheHold` is the test that fails the obvious
implementation.

The second difficulty is nesting. ``RigActivity.unsuspend`` is *membership, not
a counter* (its own docstring says so): an unsuspend from an inner hold clears an
outer pause's suspension. Since a suspended owner is the one that **permits**
manual control, an inner hold leaving would take manual control away in the
middle of the operator's own pause, silently. :class:`TestNestedHolds` pins that
it cannot.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from softae.core.rig_activity import PURGE_INSTRUMENTS, RigActivity
from softae.drivers.mock_factory import create_mock_manager
from softae.errors import AbortedError
from softae.gui.rig_claim import NULL_RIG_CLAIM, RigRunClaim
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

OWNER = "ht:cast_series"


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _until(predicate, *, timeout: float = 5.0, what: str = "condition"):
    """Await *predicate* becoming true, rather than sleeping a guessed interval."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f"{what} never became true within {timeout}s")
        await asyncio.sleep(0.01)


# ── A workflow with one step the test can hold open ─────────────────────────

class _Rig:
    """A mock rig whose ``hold`` step blocks until the test lets it go.

    Built on the real mock manager rather than a stand-in so the step really
    travels ``_run_step`` → ``_dispatch`` → ``BaseInstrument.execute``: the
    property under test is "the pause wait is not reached until the step
    returns", and a fake dispatch would not have a step to be in the middle of.
    """

    def __init__(self, *, continue_on_error: bool = False, **kwargs):
        self.manager = create_mock_manager(config={})
        run(self.manager.connect_all())
        self.entered = asyncio.Event()      # the run is inside the held step
        self.release = asyncio.Event()      # …and may now leave it

        async def _hold(**_params):
            self.entered.set()
            await self.release.wait()

        self.manager.get("stage").hold = _hold
        self.executor = WorkflowExecutor(
            self.manager, continue_on_error=continue_on_error, **kwargs
        )
        self.fired: list[bool] = []
        self.executor.on_pause_hold = self.fired.append
        self.completed: list[str] = []
        self.executor.on_step_complete = (
            lambda step, *a, **kw: self.completed.append(step.name)
        )

    async def pause_into_the_hold(self) -> None:
        """Request the pause *mid-step*, then let the step finish.

        The only way to reach a wait is through a step boundary, and the only
        honest way to reach one is to ask for the pause while a step is still in
        flight — which is also the case that matters.
        """
        await self.entered.wait()
        self.executor.pause()
        self.release.set()
        await _until(lambda: self.completed == ["hold"],
                     what="the held step to finish")

    @property
    def workflow(self) -> Workflow:
        return Workflow(name="held", setup=[
            WorkflowStep(name="hold", instrument="stage", method="hold"),
            WorkflowStep(name="after", instrument="stage", method="move_to",
                         params={"x": 0.0, "y": 0.0}),
        ])


def _claimed(activity: RigActivity, executor: WorkflowExecutor) -> None:
    """Wire *executor*'s hold to a real claim, the way a tab's ``rig_run`` does."""
    activity.acquire(OWNER, None)
    executor.on_pause_hold = RigRunClaim(activity, OWNER).set_held


# ═══════════════════════════════════════════════════════════════════════
# The X4 property: the hold, not the request
# ═══════════════════════════════════════════════════════════════════════

class TestPauseHoldFiresAtTheHold:
    def test_a_pause_requested_mid_step_does_not_fire_until_the_step_ends(self):
        """The one that fails the obvious ``on_state_change`` implementation."""
        rig = _Rig()

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.entered.wait()
            rig.executor.pause()
            assert rig.executor.state is ExecutorState.PAUSED   # requested…
            await asyncio.sleep(0.15)
            assert rig.fired == [], "handed the rig back mid-step"
            assert rig.completed == []                          # …still in flight

            rig.release.set()
            await _until(lambda: rig.fired == [True], what="the hold")
            assert rig.completed == ["hold"]                    # the step finished

            rig.executor.resume()
            await task
            assert rig.fired == [True, False]

        run(_drive())

    def test_leaving_the_hold_fires_false_before_the_next_step_runs(self):
        rig = _Rig()

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await _until(lambda: rig.fired == [True], what="the hold")

            seen: list[list[bool]] = []
            rig.executor.on_step_complete = lambda step, *a, **kw: seen.append(
                list(rig.fired)
            ) if step.name == "after" else None
            rig.executor.resume()
            await task

            assert seen == [[True, False]], "the next step ran still held"

        run(_drive())

    def test_a_run_that_is_never_paused_never_fires(self):
        """The guard, asserted directly.

        Wrapping the waits unconditionally would announce a hold at the top of
        every tier and every step of every run — a window, however brief, in
        which the instruments are handed back while the run is driving.
        """
        rig = _Rig()
        rig.release.set()
        run(rig.executor.run(rig.workflow))
        assert rig.executor.state is ExecutorState.COMPLETED
        assert rig.fired == []

    def test_nothing_changes_when_the_callback_is_unset(self):
        rig = _Rig()
        rig.executor.on_pause_hold = None

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await asyncio.sleep(0.1)
            assert rig.executor.state is ExecutorState.PAUSED   # really held
            rig.executor.resume()
            await task

        run(_drive())
        assert rig.executor.state is ExecutorState.COMPLETED

    def test_a_raising_callback_does_not_break_the_run(self):
        """A host that cannot draw its indicator must not stop a held run."""
        rig = _Rig()
        calls: list[bool] = []

        def _boom(held: bool) -> None:
            calls.append(held)
            raise RuntimeError("the sidebar exploded")

        rig.executor.on_pause_hold = _boom

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await _until(lambda: calls == [True], what="the hold")
            rig.executor.resume()
            await task

        run(_drive())
        assert calls == [True, False]
        assert rig.executor.state is ExecutorState.COMPLETED


# ═══════════════════════════════════════════════════════════════════════
# The claim actually moves — this is the operator-visible property
# ═══════════════════════════════════════════════════════════════════════

class TestTheClaimIsSuspendedWhileHeld:
    def test_manual_control_is_refused_while_driving_and_permitted_while_held(self):
        rig = _Rig()
        activity = RigActivity()
        _claimed(activity, rig.executor)

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.entered.wait()
            # Driving: the claim conflicts, so Manual Control is refused.
            assert activity.conflicts(PURGE_INSTRUMENTS) == OWNER
            rig.executor.pause()
            assert activity.conflicts(PURGE_INSTRUMENTS) == OWNER   # still driving

            rig.release.set()
            await _until(lambda: activity.conflicts(PURGE_INSTRUMENTS) is None,
                         what="the suspension")
            # Held: manual permitted, and the rig is still distinguishable from
            # an idle one — which is what lets the purge ask rather than assume.
            assert activity.suspended_conflict(PURGE_INSTRUMENTS) == OWNER
            assert activity.busy is True

            rig.executor.resume()
            await task
            assert activity.conflicts(PURGE_INSTRUMENTS) == OWNER   # driving again

        run(_drive())
        assert activity.owners() == (OWNER,), "a second owner entry was created"

    def test_a_pause_resume_round_trip_preserves_claim_depth(self):
        rig = _Rig()
        activity = RigActivity()
        _claimed(activity, rig.executor)

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await _until(lambda: activity.conflicts(PURGE_INSTRUMENTS) is None,
                         what="the suspension")
            rig.executor.resume()
            await task

        run(_drive())
        activity.release(OWNER)
        assert activity.busy is False, "the round trip unbalanced the stack"


# ═══════════════════════════════════════════════════════════════════════
# THE TRAP: an inner hold must not unsuspend an outer pause
# ═══════════════════════════════════════════════════════════════════════

class TestNestedHolds:
    """``unsuspend`` is membership, not a counter — so the executor counts.

    No production path nests these today: ``_hold_for_operator`` runs from the
    *body* of the linear loop, after that loop's own wait has already exited, and
    the tier and linear strategies are alternatives chosen once per run
    (``run()`` picks one). That argument is why the nesting is unreachable; the
    depth counter is why it stays unreachable if the argument ever stops holding
    — which is the difference between a proof and a hope, given that the failure
    is silent and its consequence is the operator losing manual control in the
    middle of their own pause.
    """

    def test_an_inner_hold_leaving_does_not_unsuspend_the_outer_pause(self):
        activity = RigActivity()
        executor = WorkflowExecutor(create_mock_manager(config={}))
        _claimed(activity, executor)

        async def _nest():
            async with executor._pause_hold():
                assert activity.conflicts(PURGE_INSTRUMENTS) is None
                async with executor._pause_hold():
                    assert activity.conflicts(PURGE_INSTRUMENTS) is None
                # The inner hold has LEFT and the outer pause has NOT ended.
                # Manual control must still be permitted; if `unsuspend` reached
                # the registry here the operator would silently lose it.
                assert activity.conflicts(PURGE_INSTRUMENTS) is None
                assert activity.suspended_conflict(PURGE_INSTRUMENTS) == OWNER
            assert activity.conflicts(PURGE_INSTRUMENTS) == OWNER

        run(_nest())

    def test_nested_holds_report_one_true_and_one_false(self):
        executor = WorkflowExecutor(create_mock_manager(config={}))
        fired: list[bool] = []
        executor.on_pause_hold = fired.append

        async def _nest():
            async with executor._pause_hold():
                async with executor._pause_hold():
                    assert fired == [True]
                assert fired == [True]
            assert fired == [True, False]

        run(_nest())
        assert executor._pause_hold_depth == 0

    def test_a_hold_left_by_an_exception_still_reports_the_release(self):
        executor = WorkflowExecutor(create_mock_manager(config={}))
        fired: list[bool] = []
        executor.on_pause_hold = fired.append

        async def _boom():
            async with executor._pause_hold():
                raise RuntimeError("the step exploded out of the hold")

        with pytest.raises(RuntimeError):
            run(_boom())
        assert fired == [True, False]
        assert executor._pause_hold_depth == 0


# ═══════════════════════════════════════════════════════════════════════
# All three waits, and the behaviours around them that must not move
# ═══════════════════════════════════════════════════════════════════════

def _move(ch: str, xy: float) -> WorkflowStep:
    return WorkflowStep(
        name=f"move_ch{ch}", instrument="stage", method="move_to",
        params={"x": xy, "y": xy}, tags={"channel": ch, "phase": "eis"},
    )


class TestEveryWaitAnnouncesItself:
    def test_the_tier_executor_holds(self):
        """``_run_tiers`` — the fail-fast path, ``continue_on_error=False``."""
        rig = _Rig(continue_on_error=False)

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await _until(lambda: rig.fired == [True], what="the tier hold")
            rig.executor.resume()
            await task

        run(_drive())
        assert rig.fired == [True, False]

    def test_the_linear_recovery_executor_holds(self):
        """``_run_linear_with_recovery`` — the HT/AE path."""
        rig = _Rig(continue_on_error=True)

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await _until(lambda: rig.fired == [True], what="the linear hold")
            rig.executor.resume()
            await task

        run(_drive())
        assert rig.fired == [True, False]

    def test_the_consecutive_failure_hold_announces_itself(self):
        """The executor's own pause at the ceiling is a hold like any other.

        The operator should have the rig during it — that is exactly the moment
        someone is standing at a wedged stage — and hooking the hold rather than
        the Pause button gives it to them with no second code path.
        """
        manager = create_mock_manager(config={"stage": {"fail_next_n": 99}})
        run(manager.connect_all())
        executor = WorkflowExecutor(
            manager, continue_on_error=True, max_channel_retries=1,
            max_consecutive_channel_failures=2,
        )
        activity = RigActivity()
        _claimed(activity, executor)
        prompted: list[tuple] = []
        executor.on_channel_failure_hold = lambda *a: prompted.append(a)
        wf = Workflow(name="wedged",
                      setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)])

        async def _drive():
            task = asyncio.create_task(executor.run(wf))
            await _until(lambda: activity.conflicts(PURGE_INSTRUMENTS) is None,
                         what="the ceiling hold")
            assert prompted, "held without asking"
            assert activity.suspended_conflict(PURGE_INSTRUMENTS) is not None
            executor.resume()
            await task

        run(_drive())
        assert activity.conflicts(PURGE_INSTRUMENTS) is not None  # driving again

    def test_a_synchronously_answered_ceiling_hold_never_hands_the_rig_back(self):
        """The run never came to rest, so nothing was handed back.

        ``_hold_for_operator`` pauses *before* it prompts, so a host that answers
        synchronously resumes the run before it reaches the wait. That is the
        correct outcome and it is asserted rather than assumed, because it is the
        difference between "the guard works" and "the guard is never exercised".
        """
        manager = create_mock_manager(config={"stage": {"fail_next_n": 99}})
        run(manager.connect_all())
        executor = WorkflowExecutor(
            manager, continue_on_error=True, max_channel_retries=1,
            max_consecutive_channel_failures=2,
        )
        fired: list[bool] = []
        executor.on_pause_hold = fired.append
        executor.on_channel_failure_hold = lambda *a: executor.resume()
        run(executor.run(Workflow(
            name="wedged", setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)]
        )))
        assert executor.state is ExecutorState.COMPLETED
        assert fired == []

    def test_an_unanswered_hold_re_guards_the_rig_before_it_parks(self, monkeypatch):
        """The park drives the hardware, so it must happen outside the hold.

        A hold hands the instruments back. Parking while still held would move
        the head and halt the pumps at the one moment a manual jog was permitted.
        """
        import softae.core.safe_park as safe_park_mod

        order: list[str] = []

        async def _fake_park(manager, *, reason="", **kwargs):
            order.append("park")
            return safe_park_mod.SafeParkResult(commanded=["pump 0 halted"])

        monkeypatch.setattr(safe_park_mod, "safe_park_async", _fake_park)

        manager = create_mock_manager(config={"stage": {"fail_next_n": 99}})
        run(manager.connect_all())
        executor = WorkflowExecutor(
            manager, continue_on_error=True, max_channel_retries=1,
            max_consecutive_channel_failures=2, channel_hold_timeout_s=0.05,
        )
        executor.on_pause_hold = lambda held: order.append(f"hold={held}")
        executor.on_channel_failure_hold = lambda *a: None

        with pytest.raises(AbortedError):
            run(executor.run(Workflow(
                name="wedged",
                setup=[_move("5", 5.0), _move("6", 6.0), _move("7", 7.0)],
            )))

        assert order == ["hold=True", "hold=False", "park"]
        assert executor.state is ExecutorState.ABORTED

    def test_an_abort_issued_while_paused_still_releases_no_step(self):
        """The invariant the pause-before-abort ordering exists to protect.

        ``abort()`` accepts PAUSED, so a run held in the wait is released by the
        very state change that must stop it. Wrapping the wait must not reorder
        that: the abort check still follows the wait, so the released iteration
        raises instead of running one more step.
        """
        rig = _Rig()

        async def _drive():
            task = asyncio.create_task(rig.executor.run(rig.workflow))
            await rig.pause_into_the_hold()
            await _until(lambda: rig.fired == [True], what="the hold")
            rig.executor.abort()
            with pytest.raises(AbortedError):
                await task

        run(_drive())
        assert rig.completed == ["hold"], "one more step ran after the abort"
        assert rig.fired == [True, False]
        assert rig.executor.state is ExecutorState.ABORTED


# ═══════════════════════════════════════════════════════════════════════
# The hosts: HT and Sandbox wire it, Arrhenius must not
# ═══════════════════════════════════════════════════════════════════════

class _CapturingExecutor:
    """Stands in for the executor so the *wiring* is what is observed."""

    def __init__(self) -> None:
        self.on_pause_hold = None
        self.seen: list = []

    async def run(self, wf) -> None:
        self.seen.append(self.on_pause_hold)


class TestTheTabsWireIt:
    """Every tab that owns a ``WorkflowExecutor`` hands its claim to it.

    Constructed windowless — which is how most of the suite builds these tabs,
    and the reason the no-window path has to yield a handle rather than ``None``.
    That the handle a windowless tab receives is :data:`NULL_RIG_CLAIM` is the
    assertion: it proves the wiring reached the executor *and* that a tab used
    outside the shell stays fully usable.
    """

    @pytest.fixture
    def qt(self):
        pytest.importorskip("PySide6")

    def _drive(self, tab, method_name: str, wf):
        import threading

        executor = _CapturingExecutor()
        tab._executor = executor
        thread = threading.Thread(target=getattr(tab, method_name), args=(wf,),
                                  daemon=True)
        thread.start()
        thread.join(timeout=20.0)
        assert not thread.is_alive(), "run thread did not finish"
        return executor

    def test_the_ht_tab_hands_its_claim_to_the_executor(self, qapp, qt):
        from softae.drivers.mock_factory import create_mock_manager as mk
        from softae.gui.tabs.tab_experiment import ExperimentBuilderTab

        tab = ExperimentBuilderTab(mk(config={}))
        try:
            tab._exp_logger = None
            for sig in (tab._sig_workflow_done, tab._sig_pause_hold):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            executor = self._drive(tab, "_run_workflow_thread",
                                   Workflow(name="cast_series"))
            # HT *wraps* the handle rather than assigning it bare, because the
            # same hold now also decides what its Pause button may say while the
            # run is held at the consecutive-failure ceiling — and that crossing
            # has to leave the executor's asyncio thread by signal. So the
            # identity check lives on the sandbox twin below, which still
            # assigns bare, and what is asserted here is that a handle arrived,
            # that firing it reaches the claim rather than raising, and that it
            # did not outlive the claim it points at. Owner-correctness for HT
            # is pinned against a real registry by `tests/test_tab_experiment.py
            # ::TestCeilingHoldLegibility::test_the_hold_still_suspends_the_runs_claim`.
            assert len(executor.seen) == 1 and callable(executor.seen[0])
            executor.seen[0](True)
            executor.seen[0](False)
        finally:
            tab.close()

        assert executor.on_pause_hold is None, "the handle outlived its claim"

    def test_the_sandbox_tab_is_wired_identically(self, qapp, qt):
        from softae.drivers.mock_factory import create_mock_manager as mk
        from softae.gui.tabs.tab_sandbox import SandboxTab

        tab = SandboxTab(mk(config={}))
        try:
            try:
                tab._sig_done.disconnect()
            except (RuntimeError, TypeError):
                pass
            executor = self._drive(tab, "_run_thread_fn", Workflow(name="bench"))
        finally:
            tab.close()

        assert executor.seen == [NULL_RIG_CLAIM.set_held]
        assert executor.on_pause_hold is None

    def test_the_arrhenius_tab_is_not_wired(self, qapp, qt):
        """An Arrhenius sweep has no pause, so its claim is never suspended.

        ``TempEISSweep`` has no hold loop and no pause concept, so there is
        nothing to key a suspension on. Wiring it anyway would be a suspension
        that is never lifted or never taken — this asserts neither happens.
        """
        import inspect
        import threading
        from types import SimpleNamespace

        from softae.drivers.mock_factory import create_mock_manager as mk
        from softae.gui.tabs.tab_arrhenius import ArrheniusTab

        source = inspect.getsource(ArrheniusTab._run_sweep_thread)
        assert "on_pause_hold" not in source
        assert "set_held" not in source

        tab = ArrheniusTab(mk(config={}))
        try:
            tab._run_id = None
            try:
                tab._sig_sweep_done.disconnect()
            except (RuntimeError, TypeError):
                pass

            class _Sweep:
                run_id = "20260821T000000Z_arr"
                temp_instrument = "temp_controller"
                eis_instrument = "pico1"
                config = SimpleNamespace(rh_instrument="rh_controller",
                                         thermal_model="arrhenius")

                async def run(self):
                    return []

                def abort(self):
                    pass

            sweep = _Sweep()
            thread = threading.Thread(target=tab._run_sweep_thread, args=(sweep,),
                                      daemon=True)
            thread.start()
            thread.join(timeout=20.0)
            assert not thread.is_alive()
        finally:
            tab.close()

        assert not hasattr(sweep, "on_pause_hold")


class TestTheNullClaim:
    def test_the_null_claim_answers_set_held_in_both_directions(self):
        """The no-window path is the *common* path in tests, not an edge case."""
        NULL_RIG_CLAIM.set_held(True)
        NULL_RIG_CLAIM.set_held(False)

    def test_a_real_claim_never_names_an_owner_the_caller_supplied(self):
        """The owner string never crosses the thread boundary a second time."""
        activity = RigActivity()
        activity.acquire(OWNER, {"stage"})
        claim = RigRunClaim(activity, OWNER, reason="paused by the operator")
        claim.set_held(True)
        assert activity.describe() == f"{OWNER} (paused by the operator)"
        claim.set_held(False)
        assert activity.describe() == OWNER
