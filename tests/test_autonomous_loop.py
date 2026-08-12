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
            max_consecutive_failures=3,
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
            max_consecutive_failures=3,
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
            max_consecutive_failures=3,
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
            max_consecutive_failures=3,
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
            max_consecutive_failures=5,     # one round failure must not park
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
            max_consecutive_failures=5,
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
