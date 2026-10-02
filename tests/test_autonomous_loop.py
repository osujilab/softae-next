"""Tests for autonomous loop (C2) and DataStore DOE API (B4)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from softae.core.autonomous_loop import (
    AutonomousLoop,
    LoopState,
    default_convergence_check,
)
from softae.core.data_store import DataStore
from softae.optimizers import GridSearchOptimizer, RandomSearchOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_SPACE = {
    "x": {"type": "float", "low": 0.0, "high": 10.0},
}


@pytest.fixture()
def store(tmp_path: Path) -> DataStore:
    with DataStore(tmp_path / "test_project") as ds:
        yield ds


@pytest.fixture()
def store_with_run(store: DataStore) -> tuple[DataStore, str]:
    run_id = store.start_run("auto_test", "{}")
    return store, run_id


# ---------------------------------------------------------------------------
# B4 — DOE DataStore API
# ---------------------------------------------------------------------------


class TestDOEDataStore:
    def test_record_and_query_doe_parameter(self, store_with_run):
        store, run_id = store_with_run
        doe_id = store.record_doe_parameter(
            run_id, channel=0, iteration=0,
            parameters={"x": 5.0},
            objective_value=42.0,
        )
        assert doe_id >= 1

        rows = store.query_doe_parameters(run_id=run_id)
        assert len(rows) == 1
        assert rows[0]["objective_value"] == 42.0
        assert '"x": 5.0' in rows[0]["parameters_json"]

    def test_update_doe_objective(self, store_with_run):
        store, run_id = store_with_run
        doe_id = store.record_doe_parameter(
            run_id, channel=0, iteration=0,
            parameters={"x": 1.0},
        )
        # Initially None
        rows = store.query_doe_parameters(run_id=run_id)
        assert rows[0]["objective_value"] is None

        store.update_doe_objective(doe_id, 99.9)
        rows = store.query_doe_parameters(run_id=run_id)
        assert rows[0]["objective_value"] == pytest.approx(99.9)

    def test_query_doe_filter_by_channel(self, store_with_run):
        store, run_id = store_with_run
        store.record_doe_parameter(run_id, channel=0, iteration=0, parameters={"x": 1})
        store.record_doe_parameter(run_id, channel=1, iteration=0, parameters={"x": 2})
        store.record_doe_parameter(run_id, channel=0, iteration=1, parameters={"x": 3})

        ch0 = store.query_doe_parameters(run_id=run_id, channel=0)
        assert len(ch0) == 2
        ch1 = store.query_doe_parameters(run_id=run_id, channel=1)
        assert len(ch1) == 1

    def test_doe_iteration_ordering(self, store_with_run):
        store, run_id = store_with_run
        for i in [2, 0, 1]:
            store.record_doe_parameter(run_id, channel=0, iteration=i, parameters={"i": i})
        rows = store.query_doe_parameters(run_id=run_id)
        assert [r["iteration"] for r in rows] == [0, 1, 2]


# ---------------------------------------------------------------------------
# C2 — Convergence check
# ---------------------------------------------------------------------------


class TestConvergenceCheck:
    def test_not_enough_history(self):
        history = [({"x": 1}, 1.0), ({"x": 2}, 2.0)]
        assert default_convergence_check(history, patience=5) is False

    def test_flat_history_converges(self):
        history = [({"x": i}, 10.0) for i in range(10)]
        assert default_convergence_check(history, patience=5) is True

    def test_improving_history_not_converged(self):
        history = [({"x": i}, float(i)) for i in range(10)]
        assert default_convergence_check(history, patience=5) is False

    def test_custom_patience(self):
        # Flat for last 3 but not last 5
        history = [({"x": i}, float(i)) for i in range(5)]
        history += [({"x": i}, 5.0) for i in range(5, 9)]
        assert default_convergence_check(history, patience=3) is True
        assert default_convergence_check(history, patience=7) is False


# ---------------------------------------------------------------------------
# C2 — Autonomous loop (mock execution)
# ---------------------------------------------------------------------------


def _make_mock_manager():
    """Create a minimal mock InstrumentManager."""
    from softae.drivers.factory import create_manager
    return create_manager(mock=True)


def _simple_objective(step_results: dict[str, Any]) -> float:
    """Trivial objective extractor — returns a fixed value or sum."""
    return sum(v for v in step_results.values() if isinstance(v, (int, float)))


class TestAutonomousLoop:
    @pytest.mark.asyncio
    async def test_loop_runs_to_budget(self, store_with_run):
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        opt = GridSearchOptimizer(SIMPLE_SPACE, n_points=3)

        from softae.workflows.workflow_model import Workflow

        template = Workflow(name="empty_trial")

        # Objective extractor returns iteration index as value
        counter = {"n": 0}
        def extractor(results):
            counter["n"] += 1
            return float(counter["n"])

        loop = AutonomousLoop(
            optimizer=opt,
            workflow_template=template,
            manager=manager,
            data_store=store,
            run_id=run_id,
            objective_extractor=extractor,
            auto_approve=True,
        )

        best = await loop.run()
        assert best is not None
        assert loop.iteration == 3  # grid has 3 points
        assert opt.n_trials == 3

        # DOE rows should be recorded
        doe_rows = store.query_doe_parameters(run_id=run_id)
        assert len(doe_rows) == 3
        assert all(r["objective_value"] is not None for r in doe_rows)

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_unmeasured_trials_are_skipped_not_told(self, store_with_run):
        """P0.1: a None objective must not become an observation.

        Previously the extractor returned 0.0 for an unusable measurement and the
        loop told it to the optimizer, so the surrogate became confident about a
        point that was never measured.  Now the trial is skipped: the optimizer
        sees only real data and the DOE row keeps a NULL objective.
        """
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        opt = GridSearchOptimizer(SIMPLE_SPACE, n_points=3)

        from softae.workflows.workflow_model import Workflow

        # Middle trial yields no usable measurement.
        seq = [1.0, None, 3.0]
        calls = {"n": 0}

        def extractor(results):
            v = seq[calls["n"]]
            calls["n"] += 1
            return v

        loop = AutonomousLoop(
            optimizer=opt,
            workflow_template=Workflow(name="empty_trial"),
            manager=manager,
            data_store=store,
            run_id=run_id,
            objective_extractor=extractor,
            auto_approve=True,
        )

        await loop.run()

        # All three suggestions ran, but only the two measured ones were told.
        assert calls["n"] == 3
        assert opt.n_trials == 2
        assert [v for _, v in opt.history] == [1.0, 3.0]
        assert 0.0 not in [v for _, v in opt.history]  # no fabricated observation

        # The unmeasured trial is still recorded, with a NULL objective.
        rows = store.query_doe_parameters(run_id=run_id)
        assert len(rows) == 3
        assert sum(1 for r in rows if r["objective_value"] is None) == 1

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_transient_failures_do_not_end_the_campaign(self, store_with_run):
        """P1.1/1.2: an isolated failure is survivable, not terminal.

        Before recovery was wired in, the first exception from a trial set
        ``ERROR`` and ended the run.  Now a one-off failure costs that trial only.
        """
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        calls = {"n": 0}

        def extractor(results):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("flaky analyzer")
            return float(calls["n"])

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=3),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=extractor, auto_approve=True,
            park_after_failed_trials=3,
        )
        await loop.run()

        assert loop.state is not LoopState.ERROR
        assert loop.park_reason is None          # one failure must not park
        assert loop._optimizer.n_trials == 2     # trials 2 and 3 were told
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_systematic_failures_park_the_loop(self, store_with_run):
        """P1.2: repeated post-retry failures stop burning wells."""
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        parked: list[str] = []

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=20),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: None,   # nothing ever measures
            auto_approve=True,
            park_after_failed_trials=3,
        )
        loop.on_park = parked.append
        await loop.run()

        assert loop.state is LoopState.STOPPED
        assert loop.park_reason is not None
        assert "consecutive" in loop.park_reason
        assert len(parked) == 1                  # on_park fired exactly once
        assert loop.iteration == 3               # parked after 3, not 20
        assert loop._optimizer.n_trials == 0     # nothing fabricated
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_success_resets_the_failure_run_length(self, store_with_run):
        """Alternating failures must not accumulate toward a park."""
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        seq = [None, 1.0, None, 2.0, None, 3.0]
        calls = {"n": 0}

        def extractor(results):
            v = seq[calls["n"] % len(seq)]
            calls["n"] += 1
            return v

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=6),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=extractor, auto_approve=True,
            park_after_failed_trials=3,
        )
        await loop.run()

        assert loop.park_reason is None          # never 3 in a row
        assert loop._optimizer.n_trials == 3
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_hard_fault_parks_immediately_without_retries(self, store_with_run):
        """SafetyError is a refusal, not a glitch — park on the first one."""
        from softae.errors import SafetyError

        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        def extractor(results):
            raise AssertionError("must not reach analyze")

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=20),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=extractor, auto_approve=True,
            park_after_failed_trials=3,
        )

        async def boom(_wf):
            raise SafetyError("reservoir hard-stop", instrument="syringe")

        loop._run_workflow = boom
        await loop.run()

        assert loop.state is LoopState.STOPPED
        assert "hard fault" in (loop.park_reason or "")
        assert loop.iteration == 0        # parked before consuming the budget
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_uncompilable_plan_parks_on_the_first_trial(self, store_with_run):
        """A plan that cannot compile is a refusal — retrying rebuilds the same one.

        Left as a soft failure it would burn wells and suggestions up to the
        park limit for a workflow that could never have been built.
        """
        from softae.core.deposition_recipe import PlanCompileError

        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        builds = {"n": 0}

        def refusing_builder(params):
            builds["n"] += 1
            raise PlanCompileError(
                "the plan names task 'startup_flush_fll' for the campaign-start "
                "flush, and the task catalog does not hold it"
            )

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=20),
            workflow_template=None,
            workflow_builder=refusing_builder,
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0, auto_approve=True,
            park_after_failed_trials=3,
        )
        await loop.run()

        assert loop.state is LoopState.STOPPED
        assert builds["n"] == 1                   # not retried
        assert loop.consecutive_failures == 0     # retry budget untouched
        assert loop.iteration == 0                # no well counted as consumed
        assert "PlanCompileError" in (loop.park_reason or "")
        # The operator reads the park reason in the terminal, so it has to name
        # the task the catalog could not resolve.
        assert "startup_flush_fll" in (loop.park_reason or "")
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_unanswered_approval_gate_parks_instead_of_hanging(self, store_with_run):
        """P1.4: the gate that could hang an overnight run must self-bound.

        With ``auto_approve=False`` and nobody ever calling ``approve()``, the
        loop used to wait forever — indistinguishable from healthy work.
        """
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=3),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=False,             # nobody will ever approve
            gate_timeout_s=0.05,            # tiny, so the test is fast
        )
        parked: list[str] = []
        loop.on_park = parked.append

        await asyncio.wait_for(loop.run(), timeout=10)   # must return on its own

        assert loop.state is LoopState.STOPPED
        assert "approval" in (loop.park_reason or "")
        assert "timed out" in (loop.park_reason or "")
        assert len(parked) == 1
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_approval_before_timeout_proceeds_normally(self, store_with_run):
        """A gate that *is* answered must behave exactly as before."""
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=2),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=False,
            gate_timeout_s=30.0,
        )

        async def _approver():
            while loop.state is not LoopState.STOPPED:
                if loop.state is LoopState.AWAITING_APPROVAL:
                    loop.approve()
                await asyncio.sleep(0.01)

        task = asyncio.create_task(_approver())
        try:
            await asyncio.wait_for(loop.run(), timeout=15)
        finally:
            task.cancel()

        assert loop.park_reason is None
        assert loop._optimizer.n_trials == 2
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_gate_timeout_none_waits_forever(self, store_with_run):
        """Opting out must genuinely restore the unbounded wait."""
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=1),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=False,
            gate_timeout_s=None,
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(loop.run(), timeout=0.4)
        assert loop.park_reason is None
        await manager.disconnect_all()

    def test_gate_timeout_defaults_to_bounded(self):
        """The *default* must be safe — an unbounded gate is the hazard."""
        from softae.core.autonomous_loop import DEFAULT_GATE_TIMEOUT_S
        from softae.workflows.workflow_model import Workflow
        from unittest.mock import MagicMock

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=1),
            workflow_template=Workflow(name="w"),
            manager=MagicMock(), data_store=MagicMock(), run_id="r",
            objective_extractor=lambda r: 1.0,
        )
        assert loop._gate_timeout_s == DEFAULT_GATE_TIMEOUT_S
        assert DEFAULT_GATE_TIMEOUT_S is not None

    def test_recovery_is_enabled_by_default(self):
        """The executor's replay/skip machinery must actually be turned on."""
        from softae.workflows.workflow_model import Workflow
        from unittest.mock import MagicMock

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=1),
            workflow_template=Workflow(name="w"),
            manager=MagicMock(), data_store=MagicMock(), run_id="r",
            objective_extractor=lambda r: 1.0,
        )
        assert loop._continue_on_error is True
        assert loop._max_channel_retries >= 1

    @pytest.mark.asyncio
    async def test_loop_stops_on_stop(self, store_with_run):
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        opt = RandomSearchOptimizer(SIMPLE_SPACE, n_trials=100, seed=42)
        from softae.workflows.workflow_model import Workflow
        template = Workflow(name="trial")

        loop = AutonomousLoop(
            optimizer=opt,
            workflow_template=template,
            manager=manager,
            data_store=store,
            run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=True,
        )

        # Stop after 2 iterations via callback
        def on_result(iteration, params, objective):
            if iteration >= 1:
                loop.stop()

        loop.on_result = on_result

        best = await loop.run()
        assert loop.iteration <= 3  # should stop early
        assert loop.state == LoopState.STOPPED

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_loop_converges(self, store_with_run):
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        opt = RandomSearchOptimizer(SIMPLE_SPACE, n_trials=100, seed=42)
        from softae.workflows.workflow_model import Workflow
        template = Workflow(name="trial")

        # Always return the same value → convergence
        loop = AutonomousLoop(
            optimizer=opt,
            workflow_template=template,
            manager=manager,
            data_store=store,
            run_id=run_id,
            objective_extractor=lambda r: 42.0,
            auto_approve=True,
            convergence_fn=lambda h: len(h) >= 6,  # converge after 6
        )

        converge_events = []
        loop.on_converged = lambda it, best: converge_events.append((it, best))

        best = await loop.run()
        assert loop.state == LoopState.CONVERGED
        assert len(converge_events) == 1

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_loop_callbacks_fire(self, store_with_run):
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        opt = GridSearchOptimizer(SIMPLE_SPACE, n_points=2)
        from softae.workflows.workflow_model import Workflow
        template = Workflow(name="trial")

        suggestions = []
        results = []
        state_changes = []

        loop = AutonomousLoop(
            optimizer=opt,
            workflow_template=template,
            manager=manager,
            data_store=store,
            run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=True,
        )
        loop.on_suggestion = lambda it, p: suggestions.append((it, p))
        loop.on_result = lambda it, p, v: results.append((it, p, v))
        loop.on_state_change = lambda old, new: state_changes.append((old, new))

        await loop.run()

        assert len(suggestions) == 2
        assert len(results) == 2
        assert len(state_changes) > 0  # multiple state transitions

        await manager.disconnect_all()

    def test_loop_state_enum(self):
        assert LoopState.IDLE.name == "IDLE"
        assert LoopState.CONVERGED.name == "CONVERGED"


# ── Terminal-state guard ────────────────────────────────────────────────────


class TestTerminalStateGuard:
    """A stop that lands mid-trial must stick.

    ``stop()`` sets ``STOPPED``; before the guard, the next statement on every
    round path was ``_set_state(ANALYZING)``, which overwrote it. The request was
    discarded with no error and nothing in the log but an ordinary state-change
    line — the loop simply ran on.
    """

    def _idle_loop(self, store_with_run, manager):
        from softae.workflows.workflow_model import Workflow

        store, run_id = store_with_run
        return AutonomousLoop(
            optimizer=RandomSearchOptimizer(SIMPLE_SPACE, n_trials=100, seed=7),
            workflow_template=Workflow(name="trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0, auto_approve=True,
        )

    @pytest.mark.asyncio
    async def test_stop_during_executing_is_not_overwritten_by_analyzing(
        self, store_with_run
    ):
        """Asserted through ``stop()``, not ``_set_state`` — the behaviour, not the guard."""
        manager = _make_mock_manager()
        await manager.connect_all()

        loop = self._idle_loop(store_with_run, manager)
        seen: list[LoopState] = []

        def on_state_change(old, new):
            seen.append(new)
            if new is LoopState.EXECUTING and not any(
                s is LoopState.STOPPED for s in seen
            ):
                loop.stop()

        loop.on_state_change = on_state_change
        await loop.run()

        assert loop.state is LoopState.STOPPED
        # The trial in flight finishes analysing what it already cast, and no
        # second suggestion is made. Without the guard the optimizer would run
        # to exhaustion, since every round re-set ANALYZING over the stop.
        assert loop.iteration == 1
        assert LoopState.ANALYZING not in seen

        await manager.disconnect_all()

    def test_set_state_logs_a_refused_transition(self, store_with_run):
        """The refusal is never silent — a swallowed transition is undebuggable."""
        import structlog

        manager = _make_mock_manager()
        loop = self._idle_loop(store_with_run, manager)

        loop.stop()
        with structlog.testing.capture_logs() as logs:
            loop._set_state(LoopState.ANALYZING)

        refusals = [e for e in logs if e["event"] == "loop_state_change_refused"]
        assert len(refusals) == 1
        assert refusals[0]["log_level"] == "warning"
        assert refusals[0]["current"] == "STOPPED"
        assert refusals[0]["attempted"] == "ANALYZING"
        assert loop.state is LoopState.STOPPED

    def test_park_from_a_hook_needs_no_break_to_hold(self, store_with_run):
        """The guard is what makes a park structural rather than per-call-site.

        ``CONVERGED`` and ``ERROR`` latch for the same reason ``STOPPED`` does:
        all three are how the loop's own ``while`` spells "the run is over".
        """
        manager = _make_mock_manager()
        loop = self._idle_loop(store_with_run, manager)

        for terminal in (LoopState.STOPPED, LoopState.CONVERGED, LoopState.ERROR):
            loop._state = terminal
            loop._set_state(LoopState.ANALYZING)
            assert loop.state is terminal

    def test_clear_terminal_reopens_the_loop(self, store_with_run):
        """A latch with no release is a trap for the next restartable loop."""
        manager = _make_mock_manager()
        loop = self._idle_loop(store_with_run, manager)

        loop._park("bench test")
        assert loop.state is LoopState.STOPPED
        assert loop.park_reason == "bench test"

        loop._clear_terminal()
        assert loop.state is LoopState.IDLE
        assert loop.park_reason is None      # else the next run reads as parked

        loop._set_state(LoopState.SUGGESTING)
        assert loop.state is LoopState.SUGGESTING


# ── Round width: budget and board, not a fixed q ────────────────────────────


class TestRoundWidth:
    """A round is sized to what is actually available, and never overruns.

    Two behaviours were changed together because they are the same mistake seen
    twice — treating q as fixed and letting the *world* absorb the mismatch:

    - the budget rounded **up** to the next multiple of q, spending as many as q-1
      electrodes and their anneal hours beyond what the operator asked for;
    - a round wider than the board's free wells was **split** across a plate
      exchange, holding half a cast batch through an operator prompt of unbounded
      duration and telling its members either side of an arbitrary gap.

    Nothing in q-BO requires every round to be the same width — ``suggest_batch(q)``
    takes any q and may return fewer — so both now narrow the round instead.
    """

    def _loop(self, *, iteration: int, budget: int | None):
        from softae.core.autonomous_loop import AutonomousLoop

        loop = AutonomousLoop.__new__(AutonomousLoop)   # no hardware needed
        loop._iteration = iteration
        loop._max_iterations = budget
        return loop

    def test_a_full_round_fits_inside_the_budget(self):
        assert self._loop(iteration=0, budget=8)._round_q(4) == 4

    def test_the_final_round_narrows_to_what_is_left(self):
        # budget 5, q 4 → 4 then 1, not 4 then 4.
        assert self._loop(iteration=4, budget=5)._round_q(4) == 1

    def test_a_spent_budget_yields_no_round_at_all(self):
        assert self._loop(iteration=5, budget=5)._round_q(4) == 0
        assert self._loop(iteration=9, budget=5)._round_q(4) == 0

    def test_an_unbounded_campaign_always_gets_the_full_width(self):
        assert self._loop(iteration=99, budget=None)._round_q(4) == 4


# ── Checkpoint invariant: every consumed well moves the resume point ─────────


class TestRoundCheckpointInvariant:
    """Every path that finishes a trial must advance through ``_advance_iteration``.

    The invariant exists because a well is consumed whether or not the trial
    measured anything: a resume point that lags the iteration counter re-casts
    used wells. The batch round's execute-failure path used to bump
    ``_iteration`` directly (``+= len(batch)``), and the board-aware round's
    execute-failure path did not advance at all — both left the checkpoint
    stale while electrodes had been spent.
    """

    @pytest.mark.asyncio
    async def test_failed_batch_round_checkpoints_every_consumed_well(self, store_with_run):
        """A round that dies in execute still spent q wells — checkpoint each."""
        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        def exploding_builder(batch):
            raise RuntimeError("clogged tip mid-round")

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=3),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=True,
            batch_size=3,
            batch_workflow_builder=exploding_builder,
            batch_objective_extractor=lambda res, k, p: 1.0,
            max_iterations=3,
            park_after_failed_trials=5,     # one round failure must not park
        )
        checkpoints: list[int] = []
        loop.on_checkpoint = checkpoints.append

        await loop.run()

        # The counter and the resume point must move together: iteration 3 with
        # no checkpoint is exactly the stale-resume hazard.
        assert loop.iteration == 3
        assert checkpoints == [1, 2, 3]
        assert loop.park_reason is None      # still one failure, not a park
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_failed_board_aware_round_checkpoints_every_allocated_well(self, store_with_run):
        """Allocator wells are spent before the cast — a failed cast still counts."""
        from softae.core.electrode_allocator import ElectrodeAllocator

        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        from softae.workflows.workflow_model import Workflow

        def exploding_builder(batch, channels):
            raise RuntimeError("stage fault mid-round")

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=4),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager, data_store=store, run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=True,
            batch_size=2,
            electrode_allocator=ElectrodeAllocator(capacity=4),
            placement_workflow_builder=exploding_builder,
            placement_objective_extractor=lambda res, ch, p: 1.0,
            max_iterations=2,
            park_after_failed_trials=5,
        )
        checkpoints: list[int] = []
        loop.on_checkpoint = checkpoints.append

        await loop.run()

        # Two wells were allocated and (attempted to be) cast, so the budget is
        # spent and each consumed well moved the resume point.
        assert loop.iteration == 2
        assert checkpoints == [1, 2]
        assert loop.park_reason is None
        await manager.disconnect_all()


# ---------------------------------------------------------------------------
# `_post_measure`: a failing observation seam is itself reportable
# ---------------------------------------------------------------------------
#
# `_post_measure` catches everything `on_trial_measured` raises — that contract
# is deliberate and is pinned in `tests/test_feasibility_runtime_seam.py`. What
# was missing is that the catch was *silent*: only a `logger.warning`, no
# campaign event. On the campaign path that hook is the entire settle →
# production-read → gate sequence, so a raise there silently keeps the trial's
# stale pre-settle sweep and the run looks identical to one that agreed.
#
# Loops here are built with `__new__` deliberately, matching the existing
# `_post_measure` tests: no hardware, and it is also the shape that proves the
# new attribute is read defensively rather than assumed onto every instance.

class TestPostMeasureReportsItsOwnHookFailing:
    @pytest.mark.asyncio
    async def test_a_raising_hook_fires_the_failure_alert_with_the_exception_text(self):
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 3
        seen: list[str] = []
        loop.on_trial_measured_hook_failed = seen.append

        async def broken(results):
            raise RuntimeError("settle exploded")

        loop.on_trial_measured = broken

        # The original results survive untouched — the pre-existing contract.
        assert await loop._post_measure({"a": 1}) == {"a": 1}

        assert len(seen) == 1
        assert "RuntimeError" in seen[0]
        assert "settle exploded" in seen[0]
        # A string, never the exception object: every other hook here passes
        # plain strings/numbers.
        assert isinstance(seen[0], str)

    @pytest.mark.asyncio
    async def test_a_broken_alert_cannot_break_the_never_fatal_contract(self):
        """The alert about the failure must not become the failure."""
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0

        def exploding_alert(error):
            raise ValueError("the alerter is broken too")

        loop.on_trial_measured_hook_failed = exploding_alert

        async def broken(results):
            raise RuntimeError("handler bug")

        loop.on_trial_measured = broken

        # Both the hook and its alert raised; neither escapes.
        assert await loop._post_measure({"a": 1}) == {"a": 1}

    @pytest.mark.asyncio
    async def test_no_alert_configured_is_the_default_and_stays_non_fatal(self):
        """A loop built without the attribute at all must not raise here.

        `_post_measure` is exercised on `__new__`-built loops, so reading the new
        attribute bare would raise `AttributeError` from inside the very `except`
        that exists to make this seam non-fatal.
        """
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0
        assert not hasattr(loop, "on_trial_measured_hook_failed")

        async def broken(results):
            raise RuntimeError("handler bug")

        loop.on_trial_measured = broken
        assert await loop._post_measure({"a": 1}) == {"a": 1}

    @pytest.mark.asyncio
    async def test_a_hook_that_succeeds_never_fires_the_alert(self):
        """Positive control: the alert is not simply always called."""
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0
        seen: list[str] = []
        loop.on_trial_measured_hook_failed = seen.append
        loop.on_trial_measured = lambda results: {**results, "ok": True}

        assert (await loop._post_measure({"a": 1}))["ok"] is True
        assert seen == []

    @pytest.mark.asyncio
    async def test_the_new_hook_defaults_to_none_on_a_real_loop(self, store_with_run):
        """A properly constructed loop has the attribute, and it is unset.

        The `__new__`-built cases above deliberately lack it; this is the other
        half — `__init__` really does define it, so the wiring layer has
        something to bind to.
        """
        from softae.workflows.workflow_model import Workflow

        store, run_id = store_with_run
        manager = _make_mock_manager()
        await manager.connect_all()

        loop = AutonomousLoop(
            optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=2),
            workflow_template=Workflow(name="empty_trial"),
            manager=manager,
            data_store=store,
            run_id=run_id,
            objective_extractor=lambda r: 1.0,
            auto_approve=True,
        )

        assert loop.on_trial_measured_hook_failed is None
        await manager.disconnect_all()


# ---------------------------------------------------------------------------
# R1 / R2 / R8 / O1 — a measured well with no told value, and whole rounds
# ---------------------------------------------------------------------------
#
# Operator rulings 2026-10-01 (`censored_tell_path_recommendations.md` R1–R8,
# `censored_observation_tell_path.md` O1/O4): a bound is a successful
# measurement. The park counter counts only wells that were not measured; a
# classified spectrum or a bound resets it, is recorded, is never told as a
# number, and a run of them alerts without parking. And a round closes every
# well it cast before it parks or converges.


def _obs(kind, *, regime="B", admitted=False, reported=1e-6, lower=None):
    """A lane-A ``SigmaObservation``, the type the σ extractors hand the loop."""
    from softae.analysis.eis.observation import CLASSIFIED_REGIMES, SigmaObservation

    return SigmaObservation(
        kind=kind, reported=reported if kind else None, lower=lower,
        regime=regime, basis="test", R_film_ohm=None, admitted=admitted,
        classified=regime in CLASSIFIED_REGIMES)


#: Rung 3b / 3c-PAA shape: an old-route upper bound on a regime-B film.
REGIME_B_BOUND = _obs("upper_bound", regime="B")


def _unmeasured_u():
    from softae.analysis.eis.observation import unmeasured

    return unmeasured("U", regime_aware=True)


def _scripted(outcomes):
    """An extractor returning *outcomes* in order (an Exception is raised)."""
    it = iter(outcomes)

    def extract(*_args):
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v
    return extract


def _placement_loop(store, run_id, manager, outcomes, *, q=None, park_after=3,
                    convergence_fn=None):
    """Board-aware rounds over len(outcomes) wells; nothing is actually cast."""
    from softae.core.electrode_allocator import ElectrodeAllocator
    from softae.workflows.workflow_model import Workflow

    n = len(outcomes)
    return AutonomousLoop(
        optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=max(n, 2)),
        workflow_template=Workflow(name="empty_trial"),
        manager=manager, data_store=store, run_id=run_id,
        objective_extractor=lambda r: 1.0,
        auto_approve=True,
        batch_size=q or n,
        electrode_allocator=ElectrodeAllocator(capacity=max(n, 2)),
        placement_workflow_builder=lambda batch, chs: Workflow(name="empty_trial"),
        placement_objective_extractor=_scripted(outcomes),
        max_iterations=n,
        park_after_failed_trials=park_after,
        convergence_fn=convergence_fn or (lambda h: False),
    )


def _three_path_loop(path, store, run_id, manager, outcomes, *, park_after=3):
    """The same scripted wells through the single-point, batch or placement path."""
    from softae.workflows.workflow_model import Workflow

    n = len(outcomes)
    if path == "placement":
        return _placement_loop(store, run_id, manager, outcomes,
                               park_after=park_after)
    common = dict(
        optimizer=GridSearchOptimizer(SIMPLE_SPACE, n_points=n),
        workflow_template=Workflow(name="empty_trial"),
        manager=manager, data_store=store, run_id=run_id,
        auto_approve=True, max_iterations=n,
        park_after_failed_trials=park_after,
        convergence_fn=lambda h: False,
    )
    if path == "single":
        return AutonomousLoop(objective_extractor=_scripted(outcomes), **common)
    return AutonomousLoop(
        objective_extractor=lambda r: 1.0, batch_size=n,
        batch_workflow_builder=lambda batch: Workflow(name="empty_trial"),
        batch_objective_extractor=_scripted(outcomes), **common)


@pytest.fixture()
async def mock_manager():
    manager = _make_mock_manager()
    await manager.connect_all()
    yield manager
    await manager.disconnect_all()


class TestMeasuredButNotTold:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["single", "batch", "placement"])
    async def test_close_well_three_regime_b_bounds_does_not_park(
            self, store_with_run, mock_manager, path):
        """R1: the rung-3b / 3c-PAA shape — every well a regime-B bound — measured.

        Under the admission gate a B bound is not admitted, and before R1 that
        made it *unmeasured*: three in a row parked the run, which is exactly
        what O1 says a bound streak must never do. Parametrised over all three
        tell paths, because a fix landing in one branch of three is this
        codebase's recurring failure.
        """
        store, run_id = store_with_run
        loop = _three_path_loop(path, store, run_id, mock_manager,
                                [REGIME_B_BOUND] * 3)

        await loop.run()

        assert loop.park_reason is None
        assert loop.consecutive_failures == 0
        assert loop.iteration == 3
        # Never told as a number (O5 deferred), so nothing reaches `best()`.
        assert loop._optimizer.history == []
        rows = store.query_doe_parameters(run_id=run_id)
        assert len(rows) == 3
        assert all(r["objective_value"] is None for r in rows)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["single", "batch", "placement"])
    async def test_close_well_three_unmeasured_still_parks(
            self, store_with_run, mock_manager, path):
        """Positive control for the test above: unmeasured wells DO still park.

        ``None`` (acquisition failure) and an observation that does not count as
        measured (regime U, the empty well) are the systematic-fault shapes the
        park counter exists for.
        """
        store, run_id = store_with_run
        loop = _three_path_loop(path, store, run_id, mock_manager,
                                [None, _unmeasured_u(), None])

        await loop.run()

        assert loop.park_reason is not None
        assert "3 consecutive trial failures" in loop.park_reason
        assert loop._optimizer.history == []

    @pytest.mark.asyncio
    async def test_close_well_measured_untold_is_reported_not_told(
            self, store_with_run, mock_manager):
        """R2/O4: the caller hears about the measurement (and labels it feasible)."""
        store, run_id = store_with_run
        loop = _placement_loop(store, run_id, mock_manager, [REGIME_B_BOUND])
        untold: list[tuple] = []
        results: list[tuple] = []
        loop.on_measured_untold = lambda i, p, o: untold.append((i, o))
        loop.on_result = lambda i, p, o: results.append((i, o))

        await loop.run()

        assert untold == [(0, REGIME_B_BOUND)]
        assert results == []

    @pytest.mark.asyncio
    async def test_close_well_admitted_bound_writes_kind_and_is_never_told(
            self, store_with_run, mock_manager):
        """O2/R3: an admitted bound's row carries the number, its kind and the
        interval's other side; ``objective_value`` stays value-only (NULL)."""
        store, run_id = store_with_run
        bound = _obs("upper_bound", regime="A", admitted=True,
                     reported=0.15, lower=0.031)
        loop = _placement_loop(store, run_id, mock_manager, [bound, 2.0])

        await loop.run()

        rows = store.query_doe_parameters(run_id=run_id)
        assert [r["objective_value"] for r in rows] == [None, 2.0]
        assert rows[0]["objective_kind"] == "upper_bound"
        assert rows[0]["objective_reported"] == pytest.approx(0.15)
        assert rows[0]["objective_lower"] == pytest.approx(0.031)
        # Only the value reached the optimizer.
        assert [v for _, v in loop._optimizer.history] == [2.0]

    @pytest.mark.asyncio
    async def test_close_well_unadmitted_bound_writes_no_kind(
            self, store_with_run, mock_manager):
        """R3a: a kind is written only for an ADMITTED observation."""
        store, run_id = store_with_run
        loop = _placement_loop(store, run_id, mock_manager, [REGIME_B_BOUND])

        await loop.run()

        row, = store.query_doe_parameters(run_id=run_id)
        assert row["objective_value"] is None
        assert row.get("objective_kind") is None


class TestBoundStreakAlert:
    @pytest.mark.asyncio
    async def test_close_well_bound_streak_alerts_once_and_never_parks(
            self, store_with_run, mock_manager):
        """O1: a run of measured-but-untold wells alerts, once, and never parks."""
        store, run_id = store_with_run
        loop = _placement_loop(store, run_id, mock_manager,
                               [REGIME_B_BOUND] * 5, q=1)
        alerts: list[tuple[int, int]] = []
        loop.on_bound_streak = lambda streak, limit: alerts.append((streak, limit))

        await loop.run()

        assert alerts == [(3, 3)]
        assert loop.park_reason is None
        assert loop.iteration == 5

    @pytest.mark.asyncio
    async def test_close_well_told_value_resets_bound_streak(
            self, store_with_run, mock_manager):
        """Only a told value ends a streak; two short streaks never alert."""
        store, run_id = store_with_run
        loop = _placement_loop(
            store, run_id, mock_manager,
            [REGIME_B_BOUND, REGIME_B_BOUND, 1.0, REGIME_B_BOUND, REGIME_B_BOUND],
            q=1)
        alerts: list[tuple[int, int]] = []
        loop.on_bound_streak = lambda streak, limit: alerts.append((streak, limit))

        await loop.run()

        assert alerts == []


class TestRoundClosesEveryWell:
    @pytest.mark.asyncio
    async def test_tell_placed_park_on_well_two_still_records_wells_three_and_four(
            self, store_with_run, mock_manager):
        """R8: the streak hits its limit at well 2 of 4; wells 3–4 still land.

        The old `_tell_placed` returned at the park, so the round's remaining
        cast-and-measured wells got no DOE row and no tell. Now every row is
        written and every measured well told, THEN the run parks — with the
        resume point already on disk (count, checkpoint, park: preserved).
        """
        store, run_id = store_with_run
        loop = _placement_loop(store, run_id, mock_manager,
                               [None, None, 5.0, 6.0], park_after=2)
        order: list[str] = []
        loop.on_checkpoint = lambda i: order.append(f"ckpt{i}")
        loop.on_park = lambda reason: order.append("park")

        await loop.run()

        assert loop.park_reason is not None
        assert "2 consecutive trial failures" in loop.park_reason
        rows = store.query_doe_parameters(run_id=run_id)
        assert [r["objective_value"] for r in rows] == [None, None, 5.0, 6.0]
        assert [v for _, v in loop._optimizer.history] == [5.0, 6.0]
        assert order == ["ckpt1", "ckpt2", "ckpt3", "ckpt4", "park"]

    @pytest.mark.asyncio
    async def test_tell_placed_extraction_error_still_writes_the_row(
            self, store_with_run, mock_manager):
        """A cast well whose extractor raised still gets its DOE row (R8)."""
        store, run_id = store_with_run
        loop = _placement_loop(store, run_id, mock_manager,
                               [RuntimeError("bad spectrum"), 7.0])

        await loop.run()

        rows = store.query_doe_parameters(run_id=run_id)
        assert [r["objective_value"] for r in rows] == [None, 7.0]

    @pytest.mark.asyncio
    async def test_tell_placed_convergence_mid_round_still_tells_remaining_wells(
            self, store_with_run, mock_manager):
        """R8: a round that converges at well 2 still tells wells 3 and 4."""
        store, run_id = store_with_run
        loop = _placement_loop(store, run_id, mock_manager,
                               [1.0, 2.0, 3.0, 4.0],
                               convergence_fn=lambda h: len(h) >= 2)
        converged: list[int] = []
        loop.on_converged = lambda i, best: converged.append(i)

        await loop.run()

        assert loop.state is LoopState.CONVERGED
        assert [v for _, v in loop._optimizer.history] == [1.0, 2.0, 3.0, 4.0]
        rows = store.query_doe_parameters(run_id=run_id)
        assert [r["objective_value"] for r in rows] == [1.0, 2.0, 3.0, 4.0]
        assert converged == [4]

    @pytest.mark.asyncio
    async def test_batch_round_park_mid_round_still_tells_the_rest(
            self, store_with_run, mock_manager):
        """The batch round had the same early return; same rule, same result."""
        store, run_id = store_with_run
        loop = _three_path_loop("batch", store, run_id, mock_manager,
                                [None, None, 5.0], park_after=2)

        await loop.run()

        assert loop.park_reason is not None
        assert [v for _, v in loop._optimizer.history] == [5.0]
        rows = store.query_doe_parameters(run_id=run_id)
        assert [r["objective_value"] for r in rows] == [None, None, 5.0]
