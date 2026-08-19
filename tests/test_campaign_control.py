"""``control.json`` — reaching a campaign that is running in another process.

Stage 4 of ``docs/SubAgent docs/campaign_attach_architecture.md``. Three stop
scopes exist and two of them are here. **E-Stop is rig-scale and is not a
campaign control**, so nothing in this file touches it.

The claims these pin, and why each is worth a test rather than a comment:

**Pause is not Abort, and no future refactor may quietly make it one.** A Pause
during an eight-hour anneal takes effect when the anneal ends — that is the
specification, not a shortfall — and a paused hold leaves the setpoint, the lamp
and the head exactly as it found them. ``safe_park`` drops the setpoint to
10 °C, so a Pause routed through it would destroy the anneal the Pause exists to
preserve. Both directions are asserted.

**Abort is the one that cuts in**, through the ``_stop_wait`` hook the tree has
had all along and the campaign path never picked up — and it parks.

**The control file's failure direction is "not a request".** Absent, empty,
torn, unknown action, stale seq: eight ways to fail and all of them leave the
campaign running. A spurious Abort costs a board and a night; a missed one costs
a poll, because the operator is standing there and presses it again.

No rig, no waiting. The mock manager throughout, an injected clock for the
eight-hour hold, and a **step gate** rather than a sleep wherever a control has
to be issued "while a step is in flight" — a timing-based version of these tests
would be a coin flip about the very ordering they exist to pin.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from softae.core import purge_runner, rig_pose
from softae.core.autonomous_loop import (
    CONTROL_ALREADY_PAUSED,
    CONTROL_APPLIED,
    CONTROL_ENDED,
    CONTROL_NOT_PAUSED,
    AutonomousLoop,
    LoopState,
)
from softae.core.campaign_events import (
    CONTROL_FILENAME,
    CONTROL_PRE_EXISTING,
    CONTROL_STALE,
    CONTROL_UNREADABLE,
    ControlWatcher,
    read_control_request,
    write_control_request,
)
from softae.core.data_store import DataStore
from softae.core.rig_pose import RigPose, safe_to_interrupt
from softae.optimizers import GridSearchOptimizer
from softae.workflows.workflow_executor import ExecutorState
from softae.workflows.workflow_model import Workflow, WorkflowStep

SPACE = {"x": {"type": "float", "low": 0.0, "high": 10.0}}


# ── Fixtures and helpers ─────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path: Path):
    with DataStore(tmp_path / "project") as ds:
        yield ds


@pytest.fixture()
async def manager():
    from softae.drivers.factory import create_manager

    mgr = create_manager(mock=True)
    await mgr.connect_all()
    # Head raised: a quiescent pose, so a pause may hold at a step boundary.
    mgr.get("syringe").set_head_state(True)
    yield mgr
    await mgr.disconnect_all()


async def _until(predicate, *, timeout: float = 10.0) -> bool:
    """Wait for *predicate*, bounded. Returns whether it came true.

    Bounded rather than a bare spin because every stage-4 defect looks exactly
    like a hang — a pause that never lands, an abort that never lands — and a
    test that hangs reports nothing.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.002)
    return predicate()


def _counting_steps(
    instrument: Any, n: int, *, gate_at: int | None = None, on_step=None
) -> list[WorkflowStep]:
    """*n* harmless steps against one instrument, each recording that it ran.

    ``get_sp`` is a pure read on the mock and drives nothing, so a workflow of
    these exercises the executor's real linear step loop — the loop the pause
    and abort checks actually live in — with no hardware semantics to reason
    about.

    ``gate_at`` makes step *n* block until ``instrument.step_gate`` is set.
    Sync instrument methods are dispatched onto the shared I/O pool, so the step
    holds on a worker thread while the event loop stays free — which is what
    lets a test issue a control *from the loop thread, during a step*, exactly
    as the watcher task does in production, and know that it did.
    """
    ran: list[int] = []
    gate = threading.Event()
    original = instrument.get_sp

    def get_sp(*args, **kwargs):
        ran.append(len(ran))
        if on_step is not None:
            on_step(len(ran))
        if gate_at is not None and len(ran) == gate_at:
            gate.wait(timeout=30.0)
        return original(*args, **kwargs)

    instrument.get_sp = get_sp
    instrument.steps_ran = ran
    instrument.step_gate = gate
    return [WorkflowStep(f"s{i}", "temp_controller", "get_sp") for i in range(n)]


def _loop(store: DataStore, manager, steps, **kw) -> AutonomousLoop:
    run_id = store.start_run("control_test", "{}")
    return AutonomousLoop(
        optimizer=GridSearchOptimizer(SPACE, n_points=kw.pop("n_points", 5)),
        workflow_template=None,
        workflow_builder=lambda params: Workflow(name="trial", setup=list(steps)),
        manager=manager,
        data_store=store,
        run_id=run_id,
        objective_extractor=lambda results: 1.0,
        auto_approve=kw.pop("auto_approve", True),
        **kw,
    )


class _Clock:
    """Virtual clock: ``sleep`` advances time instead of blocking."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


# ═══════════════════════════════════════════════════════════════════════════
# The shared quiescence predicate — one definition, not two
# ═══════════════════════════════════════════════════════════════════════════

class TestSharedPosePredicate:
    def test_the_purge_runner_and_the_pause_read_the_same_classification(self):
        """Not "equivalent" — *identical*. Two definitions of "safe to stop
        here" is how a Pause comes to hold somewhere a purge would refuse."""
        assert purge_runner.classify_pose is rig_pose.classify_pose
        assert purge_runner.RigPose is rig_pose.RigPose
        assert purge_runner.FLUSH_TOLERANCE_MM == rig_pose.FLUSH_TOLERANCE_MM

    @pytest.mark.asyncio
    async def test_a_raised_head_is_a_pose_a_run_may_be_held_at(self, manager):
        manager.get("syringe").set_head_state(True)
        assert rig_pose.classify_pose(manager) is RigPose.HEAD_UP
        assert safe_to_interrupt(manager) is True

    @pytest.mark.asyncio
    async def test_a_head_down_over_a_well_is_not(self, manager):
        """``HEAD_DOWN_ELSEWHERE`` is a tip sitting in a drop."""
        manager.get("syringe").set_head_state(False)
        manager.get("stage").live_position = lambda *a, **k: (999.0, 999.0)
        assert rig_pose.classify_pose(manager) is RigPose.HEAD_DOWN_ELSEWHERE
        assert safe_to_interrupt(manager) is False

    def test_an_unreadable_rig_is_refused_exactly_like_a_head_down_one(self):
        """"I could not tell" and "it is unsafe" lead to the same action."""

        class _Blind:
            def get(self, name):
                raise RuntimeError("no such instrument")

        assert rig_pose.classify_pose(_Blind()) is RigPose.UNKNOWN
        assert safe_to_interrupt(_Blind()) is False


# ═══════════════════════════════════════════════════════════════════════════
# The control file — every ambiguity resolves to "not a request" (R6)
# ═══════════════════════════════════════════════════════════════════════════

class TestControlFile:
    @pytest.mark.parametrize("body", [
        None,                                     # absent
        "",                                       # empty
        '{"seq": 1, "action": "ab',               # torn mid-write
        "not json at all",
        '{"seq": 1}',                             # no action
        '{"seq": 1, "action": "self_destruct"}',  # outside the vocabulary
        '{"seq": "one", "action": "abort"}',      # seq that is not a number
        '["abort"]',                              # not an object
    ])
    def test_a_file_that_is_not_a_request_is_not_a_halt(self, tmp_path, body):
        if body is not None:
            (tmp_path / CONTROL_FILENAME).write_text(body, encoding="utf-8")
        assert read_control_request(tmp_path) is None

    def test_a_written_request_lands_whole_and_readable(self, tmp_path):
        request = write_control_request(
            tmp_path, "abort", reason="board looks wrong", requested_by="test")
        assert request.seq == 1
        assert read_control_request(tmp_path) == request
        # Written to a temp path and renamed on, so nothing is left behind and
        # a reader can never see a prefix of it.
        assert not list(tmp_path.glob("*.tmp"))

    def test_each_request_carries_a_higher_seq_than_the_one_it_replaces(
            self, tmp_path):
        write_control_request(tmp_path, "pause")
        write_control_request(tmp_path, "resume")
        third = write_control_request(tmp_path, "abort")
        assert third.seq == 3
        # A slot, not a queue: only the most recent intent survives on disk.
        assert read_control_request(tmp_path).action == "abort"


class TestControlWatcher:
    def _watcher(self, run_dir, seen: list, outcome: str = CONTROL_APPLIED):
        acks: list[dict] = []
        watcher = ControlWatcher(
            run_dir,
            handlers={
                a: (lambda req, _a=a: (seen.append(_a), outcome)[1])
                for a in ("pause", "resume", "abort")
            },
            on_ack=acks.append,
        )
        watcher.acks = acks
        return watcher

    def test_a_request_reaches_its_handler_and_is_acknowledged(self, tmp_path):
        seen: list[str] = []
        watcher = self._watcher(tmp_path, seen)
        write_control_request(tmp_path, "abort", reason="stop it")

        assert watcher.poll_once() is not None
        assert seen == ["abort"]
        ack = watcher.acks[-1]
        assert ack["action"] == "abort"
        assert ack["outcome"] == CONTROL_APPLIED
        assert ack["reason"] == "stop it"

    def test_an_unchanged_file_is_never_dispatched_twice(self, tmp_path):
        """Idempotence at the cheapest layer: repeated reads of one request."""
        seen: list[str] = []
        watcher = self._watcher(tmp_path, seen)
        write_control_request(tmp_path, "pause")

        watcher.poll_once()
        for _ in range(5):
            watcher.poll_once()
        assert seen == ["pause"]

    def test_a_request_at_or_below_the_last_acted_seq_is_ignored(self, tmp_path):
        seen: list[str] = []
        watcher = self._watcher(tmp_path, seen)
        write_control_request(tmp_path, "abort")
        watcher.poll_once()

        # Somebody rewinds the file by hand, or a resume restores an old copy.
        (tmp_path / CONTROL_FILENAME).write_text(
            json.dumps({"seq": 1, "action": "abort"}), encoding="utf-8")
        assert watcher.poll_once() is None
        assert seen == ["abort"]
        assert watcher.acks[-1]["outcome"] == CONTROL_STALE

    def test_whatever_was_already_in_the_file_is_never_obeyed(self, tmp_path):
        """A campaign resumed into the same run directory must not read
        yesterday's Abort and park itself on its first poll."""
        write_control_request(tmp_path, "abort", reason="yesterday")
        seen: list[str] = []
        watcher = self._watcher(tmp_path, seen)

        assert watcher.poll_once() is None
        assert seen == []
        assert watcher.acks[0]["outcome"] == CONTROL_PRE_EXISTING
        # …and a *new* request into the same file still works.
        write_control_request(tmp_path, "pause")
        assert watcher.poll_once() is not None
        assert seen == ["pause"]

    def test_a_torn_file_is_acknowledged_as_unreadable_rather_than_obeyed(
            self, tmp_path):
        seen: list[str] = []
        watcher = self._watcher(tmp_path, seen)
        (tmp_path / CONTROL_FILENAME).write_text('{"seq": 1, "act',
                                                 encoding="utf-8")
        assert watcher.poll_once() is None
        assert seen == []
        assert watcher.acks[-1]["outcome"] == CONTROL_UNREADABLE

    def test_no_control_file_at_all_is_a_silent_no_op(self, tmp_path):
        seen: list[str] = []
        watcher = self._watcher(tmp_path, seen)
        for _ in range(3):
            assert watcher.poll_once() is None
        assert seen == []
        assert watcher.acks == []

    def test_a_handler_that_raises_cannot_break_the_watcher(self, tmp_path):
        def _boom(request):
            raise RuntimeError("handler exploded")

        acks: list[dict] = []
        watcher = ControlWatcher(tmp_path, handlers={"abort": _boom},
                                 on_ack=acks.append)
        write_control_request(tmp_path, "abort")
        assert watcher.poll_once() is None
        assert acks[-1]["outcome"] == "handler_failed"

    @pytest.mark.asyncio
    async def test_the_polling_task_dispatches_and_stops_cleanly(self, tmp_path):
        seen: list[str] = []
        watcher = ControlWatcher(
            tmp_path,
            handlers={
                "pause": lambda req: (seen.append("pause"), CONTROL_APPLIED)[1]},
            poll_s=0.01,
        )
        watcher.start()
        write_control_request(tmp_path, "pause")
        assert await _until(lambda: seen == ["pause"])
        await watcher.aclose()


# ═══════════════════════════════════════════════════════════════════════════
# Pause — resumable, step-granular, and it touches nothing
# ═══════════════════════════════════════════════════════════════════════════

class TestPause:
    @pytest.mark.asyncio
    async def test_pause_takes_effect_at_the_step_boundary_and_not_mid_step(
            self, store, manager):
        """The heart of "stop issuing new steps, then hold".

        The pause is issued while step 1 is provably in flight — the step is
        holding a gate on a worker thread. Step 1 still finishes; step 2 does
        not start until the operator resumes.
        """
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 4, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=1)

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            assert loop.pause("operator") == CONTROL_APPLIED
            tc.step_gate.set()

            assert await _until(lambda: loop._paused)
            assert tc.steps_ran == [0], "a step ran after the pause was taken"
            assert loop._executor is not None
            assert loop._executor.state is ExecutorState.PAUSED

            # A pause is a hold, not a stop: it stays held.
            await asyncio.sleep(0.15)
            assert tc.steps_ran == [0]

            assert loop.resume() == CONTROL_APPLIED
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()
        assert tc.steps_ran == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_a_paused_campaign_never_parks_and_leaves_the_rig_alone(
            self, store, manager, monkeypatch):
        """``DEFAULT_SAFE_TEMP_C = 10.0`` is what makes this load-bearing.

        A Pause that reached ``safe_park`` would drop the setpoint by whatever
        an anneal is above 10 °C and kill the lamp — destroying the very hold
        the operator paused in order to keep.
        """
        import softae.core.safe_park as safe_park_mod

        monkeypatch.setattr(
            safe_park_mod, "safe_park",
            lambda *a, **k: pytest.fail("Pause must never reach safe_park"))

        tc = manager.get("temp_controller")
        tc.write_sp(140.0, print_flag=0)
        steps = _counting_steps(tc, 3, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=1)
        parks: list[str] = []
        loop.on_park = parks.append

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            loop.pause("look at a well")
            tc.step_gate.set()
            assert await _until(lambda: loop._paused)

            assert parks == []
            assert tc.get_sp() == 140.0            # the anneal is intact
            assert loop.state not in (
                LoopState.STOPPED, LoopState.CONVERGED, LoopState.ERROR)
            loop.resume()
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()
        assert parks == []
        assert loop.park_reason is None

    @pytest.mark.asyncio
    async def test_pause_holds_only_at_a_pose_the_purge_runner_would_accept(
            self, store, manager):
        """A head down over a well defers the hold to a later boundary.

        Not a refusal — the top-of-cycle gate always qualifies — so an
        unreadable or busy rig delays a pause and can never strand one.
        """
        tc = manager.get("temp_controller")
        syringe = manager.get("syringe")
        syringe.set_head_state(False)
        manager.get("stage").live_position = lambda *a, **k: (999.0, 999.0)

        phases: list[tuple[str, str]] = []

        def _on_phase(phase: str, detail: str) -> None:
            phases.append((phase, detail))
            if phase == "deferred":
                # The trial moves on and the rig comes back to a quiescent pose.
                syringe.set_head_state(True)

        steps = _counting_steps(tc, 5, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=1)
        loop.on_pause_change = _on_phase

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            loop.pause("operator")
            tc.step_gate.set()

            assert await _until(lambda: loop._paused)
            # Deferred past the head-down boundary; held at the first quiescent
            # one after it.
            assert tc.steps_ran == [0, 1]
            assert any(p == "deferred" for p, _ in phases)
            assert phases[-1][0] == "holding"
            loop.resume()
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()

    @pytest.mark.asyncio
    async def test_resume_continues_the_same_campaign_rather_than_restarting_it(
            self, store, manager):
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 3, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=2)
        checkpoints: list[int] = []
        loop.on_checkpoint = checkpoints.append

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            loop.pause("")
            tc.step_gate.set()
            assert await _until(lambda: loop._paused)
            iteration_when_paused = loop.iteration
            loop.resume()
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()

        assert iteration_when_paused == 0
        # Two trials, three steps each — nothing replayed, nothing skipped by
        # the round trip through pause.
        assert loop.iteration == 2
        assert checkpoints == [1, 2]
        assert tc.steps_ran == [0, 1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_a_doubled_pause_resolves_deterministically(
            self, store, manager):
        loop = _loop(store, manager, [])
        assert loop.pause("first") == CONTROL_APPLIED
        assert loop.pause("second") == CONTROL_ALREADY_PAUSED
        assert loop.resume() == CONTROL_APPLIED
        assert loop.resume() == CONTROL_NOT_PAUSED

    @pytest.mark.asyncio
    async def test_a_pause_between_cycles_holds_before_the_next_suggestion(
            self, store, manager):
        """The second half of the definition: "before next cycle/loop start"."""
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 1)
        loop = _loop(store, manager, steps, max_iterations=3)
        suggestions: list[int] = []
        loop.on_suggestion = lambda i, p: suggestions.append(i)
        once = {"done": False}

        def _after_first(i, params, objective):
            if not once["done"]:
                once["done"] = True
                loop.pause("after the first trial")

        loop.on_result = _after_first

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: loop._paused)
            await asyncio.sleep(0.15)
            assert suggestions == [0]        # no second suggestion was made
            loop.resume()
            await asyncio.wait_for(task, timeout=20)
        finally:
            task.cancel()
        assert suggestions == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════
# Abort — terminal, it cuts into a hold, and it parks
# ═══════════════════════════════════════════════════════════════════════════

class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_during_a_temperature_hold_lands_within_one_poll(
            self, store, manager):
        """The test that proves the eight-hour abort is gone.

        ``run_anneal_hold`` already derives ``monitored_hold``'s
        ``should_abort`` from the controller's ``_stop_wait``, and
        ``monitored_hold`` tests it at the top of every poll. The mechanism was
        built, tested and used by two other surfaces; the campaign path never
        picked it up. This asserts that ``AutonomousLoop.abort`` is what now
        sets it, and that a 28800 s hold ends one poll after the request rather
        than eight hours after it.
        """
        from softae.drivers.contracts import anneal_watchdog_config, run_anneal_hold

        poll = float(anneal_watchdog_config()["poll_interval_s"])
        loop = _loop(store, manager, [])
        loop._running = True
        tc = manager.get("temp_controller")
        # A healthy hold: the PV sits on target, so nothing but the abort can
        # end it. Otherwise the watchdog's own fault path would.
        tc.get_pv = lambda n_avg=1: 120.0
        clock = _Clock()
        requested_at: list[float] = []

        def _sleep(dt: float) -> None:
            clock.sleep(dt)
            if clock.t >= 120.0 and not loop._abort_requested:
                requested_at.append(clock.t)
                loop.abort("operator abort")

        report = run_anneal_hold(tc, 28800.0, 120.0, sleep=_sleep, now=clock.now)

        assert report.aborted is True
        assert requested_at, "the abort was never issued"
        assert clock.t - requested_at[0] <= 2 * poll, (
            f"the hold ran {clock.t - requested_at[0]}s past the request "
            f"(poll interval {poll}s)")
        assert clock.t < 28800.0

    @pytest.mark.asyncio
    async def test_the_stop_flag_is_clear_by_the_time_the_park_runs(
            self, store, manager):
        """A park issued with ``_stop_wait`` still set cannot drop the heater.

        The real controller's ``_with_retry`` raises ``CommunicationError`` on
        every command while the flag is set, and ``safe_park``'s whole thermal
        contribution is one ``write_sp(10 °C)``. Clearing the flag *before* the
        park is what keeps an aborted rig from being left hot behind a park that
        reports itself incomplete.
        """
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 4, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=1)
        flag_at_park: list[bool] = []
        loop.on_park = lambda reason: flag_at_park.append(tc._stop_wait.is_set())

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            loop.abort("operator")
            tc.step_gate.set()
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()

        assert flag_at_park == [False]
        assert tc._stop_wait.is_set() is False

    @pytest.mark.asyncio
    async def test_abort_parks_and_the_reason_reaches_the_park_verbatim(
            self, store, manager):
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 4, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=3)
        parks: list[str] = []
        loop.on_park = parks.append

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            loop.abort("operator: board looks wrong")
            tc.step_gate.set()
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()

        assert parks == ["operator: board looks wrong"]
        assert loop.park_reason == "operator: board looks wrong"
        assert loop.state is LoopState.STOPPED
        # `park_reason` is also the predicate the campaign wiring reads to
        # decide the checkpoint's fate: an operator abort keeps it, because the
        # run ended because somebody said so rather than because it finished.

    @pytest.mark.asyncio
    async def test_abort_refuses_the_next_step(self, store, manager):
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 6, gate_at=2)
        loop = _loop(store, manager, steps, max_iterations=1)

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0, 1])
            loop.abort("stop")
            tc.step_gate.set()
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()
        # The step that was running finished; nothing after it started.
        assert tc.steps_ran == [0, 1]

    @pytest.mark.asyncio
    async def test_pause_then_abort_executes_zero_further_steps(
            self, store, manager):
        """Abort beats Pause, in the ordering that is easiest to get wrong.

        The executor accepts ``abort()`` from ``PAUSED`` and checks its pause
        loop *before* its abort check, precisely so the state change that must
        stop it is also the one that releases it. This pins the same property a
        layer up, so the loop's new pause cannot reintroduce the off-by-one.
        """
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 6, gate_at=1)
        loop = _loop(store, manager, steps, max_iterations=1)
        parks: list[str] = []
        loop.on_park = parks.append

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(lambda: tc.steps_ran == [0])
            loop.pause("look")
            tc.step_gate.set()
            assert await _until(lambda: loop._paused)
            assert tc.steps_ran == [0]

            assert loop.abort("changed my mind") == CONTROL_APPLIED
            await asyncio.wait_for(task, timeout=20)
        finally:
            tc.step_gate.set()
            task.cancel()

        assert tc.steps_ran == [0]                # zero further steps
        assert parks == ["changed my mind"]       # and Abort's park still ran

    @pytest.mark.asyncio
    async def test_a_pause_arriving_after_an_abort_is_refused(
            self, store, manager):
        loop = _loop(store, manager, [])
        loop._running = True
        assert loop.abort("stop") == CONTROL_APPLIED
        assert loop.pause("wait") == CONTROL_ENDED
        assert loop.abort("stop again") == CONTROL_ENDED

    @pytest.mark.asyncio
    async def test_an_abort_after_the_run_ended_does_not_park(
            self, store, manager):
        """A converged run must not have its setpoint dropped by a late button.

        The rig has been left in a state somebody chose — possibly deliberately
        head-down with a wet tip — and a park arriving afterwards would be a
        control doing harm after the thing it controls has gone.
        """
        tc = manager.get("temp_controller")
        loop = _loop(store, manager, _counting_steps(tc, 1), max_iterations=1)
        parks: list[str] = []
        loop.on_park = parks.append

        await asyncio.wait_for(loop.run(), timeout=20)
        assert loop.abort("too late") == CONTROL_ENDED
        assert parks == []
        assert loop.park_reason is None

    @pytest.mark.asyncio
    async def test_abort_releases_an_approval_gate_without_approving_it(
            self, store, manager):
        """``abort()`` frees the gate by *setting* its event, which is exactly
        what an operator saying yes looks like. The trial must not run."""
        tc = manager.get("temp_controller")
        steps = _counting_steps(tc, 3)
        loop = _loop(store, manager, steps, max_iterations=2, auto_approve=False)
        parks: list[str] = []
        loop.on_park = parks.append

        task = asyncio.create_task(loop.run())
        try:
            assert await _until(
                lambda: loop.state is LoopState.AWAITING_APPROVAL)
            loop.abort("not this one")
            await asyncio.wait_for(task, timeout=20)
        finally:
            task.cancel()

        assert tc.steps_ran == []
        assert parks == ["not this one"]


# ═══════════════════════════════════════════════════════════════════════════
# The executor stash — the thing a control request actually reaches
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutorStash:
    @pytest.mark.asyncio
    async def test_the_trial_executor_is_reachable_while_it_runs_and_dropped_after(
            self, store, manager):
        """A stale handle would let a request aimed at one trial reach the next."""
        tc = manager.get("temp_controller")
        seen: list[Any] = []
        holder: dict[str, Any] = {}
        steps = _counting_steps(
            tc, 2, on_step=lambda n: seen.append(holder["loop"]._executor))
        loop = _loop(store, manager, steps, max_iterations=2)
        holder["loop"] = loop

        await asyncio.wait_for(loop.run(), timeout=20)

        assert seen and all(e is not None for e in seen)
        # Two trials, two distinct executors — the stash is per-trial.
        assert len({id(e) for e in seen}) == 2
        assert loop._executor is None


# ═══════════════════════════════════════════════════════════════════════════
# A campaign nobody is controlling is unaffected
# ═══════════════════════════════════════════════════════════════════════════

def _campaign_spec(name: str):
    from softae.core.autonomous_wiring import CampaignSpec

    return CampaignSpec(
        name=name,
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space={
            "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
            "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
        },
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=2,
        seed=7,
    )


class TestUncontrolledCampaign:
    @pytest.mark.asyncio
    async def test_a_campaign_with_no_control_file_behaves_exactly_as_before(
            self, tmp_path):
        """The invariant the whole channel rests on.

        The same spec run twice, with the watcher present and nothing ever
        writing to it, must give the same answer both times — and must not
        create the control file merely by watching for one.
        """
        from softae.core.autonomous_wiring import run_autonomous_campaign
        from softae.drivers.mock_factory import create_mock_manager

        results = []
        for name in ("a", "b"):
            mgr = create_mock_manager(config={})
            await mgr.connect_all()
            with DataStore(tmp_path / name) as ds:
                result = await run_autonomous_campaign(
                    _campaign_spec("uncontrolled"), manager=mgr, data_store=ds)
                results.append(result)
                assert not (ds.run_dir(result.run_id) / CONTROL_FILENAME).exists()
            await mgr.disconnect_all()

        first, second = results
        assert first.final_state == second.final_state
        assert first.n_trials == second.n_trials
        assert first.park_reason is None and second.park_reason is None
        assert [o for _, o in first.history] == [o for _, o in second.history]

    @pytest.mark.asyncio
    async def test_a_control_request_is_acknowledged_on_the_narration_stream(
            self, tmp_path):
        """Receipt and outcome are durable, on the stream everything else is on.

        A control an operator pressed and heard nothing back from is worse than
        no control, so the acknowledgement must outlive the process — and it
        goes where the narration goes rather than to a second channel with its
        own failure modes.

        The request is placed at ``run_started``, which is emitted before the
        watcher is built, so this also pins the refusal that matters most: a
        request already in the file when the campaign starts is **acknowledged
        and never obeyed**. That is what stops a resumed campaign reading
        yesterday's Abort and parking itself on its first poll.
        """
        from softae.core.autonomous_wiring import run_autonomous_campaign
        from softae.core.campaign_events import EVENTS_FILENAME
        from softae.drivers.mock_factory import create_mock_manager

        mgr = create_mock_manager(config={})
        await mgr.connect_all()

        with DataStore(tmp_path / "acked") as store:
            def _on_event(event: dict) -> None:
                if event.get("type") == "run_started":
                    write_control_request(
                        store.run_dir(event["run_id"]), "abort",
                        reason="left over from yesterday")

            result = await run_autonomous_campaign(
                _campaign_spec("acked"), manager=mgr, data_store=store,
                on_event=_on_event)
            lines = [
                json.loads(line)
                for line in (store.run_dir(result.run_id) / EVENTS_FILENAME)
                .read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        await mgr.disconnect_all()

        acks = [r for r in lines if r.get("type") == "control_ack"]
        assert acks, "the request left no durable acknowledgement"
        assert acks[0]["action"] == "abort"
        assert acks[0]["outcome"] == CONTROL_PRE_EXISTING
        # And it was refused, not obeyed: the campaign ran to completion.
        assert result.park_reason is None
        assert result.n_trials > 0
