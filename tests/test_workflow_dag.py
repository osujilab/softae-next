"""Tests for DAG-based dependency execution in WorkflowExecutor."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.errors import AbortedError, ValidationError_
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep
from softae.workflows.workflow_parser import parse_dict


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
async def mgr() -> InstrumentManager:
    m = create_mock_manager(config={})
    await m.connect_all()
    return m


# ═══════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════


def _make_step(name: str, depends_on: list[str] | None = None) -> WorkflowStep:
    """Create a WorkflowStep targeting the mock stage (move_to is always valid)."""
    return WorkflowStep(
        name=name,
        instrument="stage",
        method="move_to",
        params={"x": 0, "y": 0},
        depends_on=depends_on or [],
    )


# ═══════════════════════════════════════════════════════════════════════
# Test Class
# ═══════════════════════════════════════════════════════════════════════


class TestDAGExecution:

    # ── 1. Linear chain: A → B → C  (sequential preserved) ─────────

    async def test_linear_chain_sequential(self, mgr):
        """A→B→C with explicit depends_on executes in order."""
        wf = Workflow(
            name="linear_chain",
            setup=[
                _make_step("A"),
                _make_step("B", depends_on=["A"]),
                _make_step("C", depends_on=["B"]),
            ],
        )
        order: list[str] = []

        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, *_: order.append(s.name)
        await executor.run(wf)

        assert order == ["A", "B", "C"]
        assert executor.state is ExecutorState.COMPLETED

    # ── 1b. Duplicate step names — clear error, not a false cycle ───

    async def test_duplicate_step_names_raise_clear_error(self, mgr):
        """Two steps sharing a name is rejected up front (not as a cryptic cycle).

        The name-keyed DAG collapses duplicates, which previously surfaced as a
        confusing 'Dependency cycle detected' — now it's a direct message.
        """
        wf = Workflow(name="dup", setup=[_make_step("a"), _make_step("a")])
        executor = WorkflowExecutor(mgr)
        with pytest.raises(ValidationError_, match="Duplicate step name"):
            await executor.run(wf)
        assert executor.state is ExecutorState.ERROR

    # ── 1c. Built-in 'control' wait — instrument-free general-purpose step ──

    async def test_control_wait_step_completes(self, mgr):
        wf = Workflow(
            name="w",
            setup=[WorkflowStep(name="pause", instrument="control", method="wait",
                                params={"seconds": 0.0})],
        )
        results: dict[str, Any] = {}
        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, i, t, r, e: results.update({s.name: r})
        await executor.run(wf)
        assert executor.state is ExecutorState.COMPLETED
        assert results["pause"] == {"waited_s": 0.0}

    async def test_control_wait_interleaves_with_hardware_steps(self, mgr):
        wf = Workflow(
            name="mix",
            setup=[
                _make_step("a"),
                WorkflowStep(name="pause", instrument="control", method="wait",
                             params={"seconds": 0.0}),
                _make_step("b", depends_on=["pause"]),
            ],
        )
        order: list[str] = []
        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, *_: order.append(s.name)
        await executor.run(wf)
        assert order == ["a", "pause", "b"]

    async def test_control_wait_negative_raises(self, mgr):
        # Raised during step execution → wrapped by the retry layer, but the
        # original message is preserved.
        wf = Workflow(name="wn", setup=[
            WorkflowStep(name="pause", instrument="control", method="wait",
                         params={"seconds": -1.0})])
        with pytest.raises(Exception, match="non-negative"):
            await WorkflowExecutor(mgr).run(wf)

    async def test_unknown_control_method_raises(self, mgr):
        wf = Workflow(name="uc", setup=[
            WorkflowStep(name="x", instrument="control", method="nope", params={})])
        with pytest.raises(Exception, match="Unknown control method"):
            await WorkflowExecutor(mgr).run(wf)

    def test_wait_task_in_catalog(self):
        from softae.config import loader
        from softae.core.task_catalog import TaskCatalog

        cat = TaskCatalog.load_toml(loader.tasks_toml_path())
        assert "wait" in cat
        t = cat.get("wait")
        assert t.instrument == "control" and t.method == "wait"
        assert t.category == "general"
        assert "seconds" in t.params

    # ── 2. Two independent steps — parallel execution ───────────────

    async def test_independent_steps_parallel(self, mgr):
        """Two steps with depends_on=[] and a common predecessor run concurrently."""
        wf = Workflow(
            name="parallel_pair",
            setup=[
                _make_step("root"),
                _make_step("left", depends_on=["root"]),
                _make_step("right", depends_on=["root"]),
            ],
        )
        timestamps: dict[str, float] = {}

        original_run_step = WorkflowExecutor._run_step

        async def timed_run_step(self_exec, step, index, total):
            timestamps[f"{step.name}_start"] = time.monotonic()
            await original_run_step(self_exec, step, index, total)
            timestamps[f"{step.name}_end"] = time.monotonic()

        executor = WorkflowExecutor(mgr)
        executor._run_step = lambda s, i, t: timed_run_step(executor, s, i, t)
        await executor.run(wf)

        # "left" and "right" should have overlapping time windows
        # (both start after "root" completes)
        assert timestamps["left_start"] >= timestamps["root_end"] - 0.01
        assert timestamps["right_start"] >= timestamps["root_end"] - 0.01
        assert executor.state is ExecutorState.COMPLETED

    # ── 3. Diamond pattern ──────────────────────────────────────────

    async def test_diamond_pattern(self, mgr):
        """A→{B,C}→D: B and C are parallel, D waits for both."""
        wf = Workflow(
            name="diamond",
            setup=[
                _make_step("A"),
                _make_step("B", depends_on=["A"]),
                _make_step("C", depends_on=["A"]),
                _make_step("D", depends_on=["B", "C"]),
            ],
        )
        order: list[str] = []

        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, *_: order.append(s.name)
        await executor.run(wf)

        assert order[0] == "A"
        assert set(order[1:3]) == {"B", "C"}
        assert order[3] == "D"

    # ── 4. Cycle detection ──────────────────────────────────────────

    async def test_cycle_detection_raises(self, mgr):
        """A→B→A cycle raises ValidationError_ during execution."""
        wf = Workflow(
            name="cycle",
            setup=[
                _make_step("A", depends_on=["B"]),
                _make_step("B", depends_on=["A"]),
            ],
        )
        executor = WorkflowExecutor(mgr)
        with pytest.raises(ValidationError_, match="cycle"):
            await executor.run(wf)

    def test_cycle_detection_at_parse_time(self):
        """Parser rejects cyclic deps at parse time."""
        data = {
            "name": "cyclic",
            "setup": [
                {"name": "A", "instrument": "stage", "method": "move_to",
                 "depends_on": ["B"]},
                {"name": "B", "instrument": "stage", "method": "move_to",
                 "depends_on": ["A"]},
            ],
        }
        with pytest.raises(ValidationError_, match="cycle"):
            parse_dict(data)

    # ── 5. Missing dependency name ──────────────────────────────────

    async def test_missing_dependency_raises(self, mgr):
        """Referencing a non-existent step name raises ValidationError_."""
        wf = Workflow(
            name="bad_dep",
            setup=[
                _make_step("A", depends_on=["nonexistent"]),
            ],
        )
        executor = WorkflowExecutor(mgr)
        with pytest.raises(ValidationError_, match="does not exist"):
            await executor.run(wf)

    def test_missing_dependency_at_parse_time(self):
        """Parser catches missing deps at parse time."""
        data = {
            "name": "bad_dep",
            "setup": [
                {"name": "A", "instrument": "stage", "method": "move_to",
                 "depends_on": ["ghost"]},
            ],
        }
        with pytest.raises(ValidationError_, match="not a setup step"):
            parse_dict(data)

    # ── 6. No depends_on — identical to current sequential ──────────

    async def test_no_depends_on_sequential(self, mgr):
        """Without any depends_on, steps run in definition order."""
        wf = Workflow(
            name="sequential",
            setup=[
                _make_step("A"),
                _make_step("B"),
                _make_step("C"),
            ],
        )
        order: list[str] = []

        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, *_: order.append(s.name)
        await executor.run(wf)

        assert order == ["A", "B", "C"]

    # ── 7. Mixed: some steps have deps, some don't ──────────────────

    async def test_mixed_deps_and_no_deps(self, mgr):
        """Steps without deps chain sequentially; steps with deps honor DAG."""
        wf = Workflow(
            name="mixed",
            setup=[
                _make_step("A"),                          # no deps → first step
                _make_step("B"),                          # no deps → implicit dep on A
                _make_step("C", depends_on=["A"]),        # explicit dep on A
                _make_step("D", depends_on=["B", "C"]),   # waits for both
            ],
        )
        order: list[str] = []

        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, *_: order.append(s.name)
        await executor.run(wf)

        # A must be first
        assert order[0] == "A"
        # B depends implicitly on A; C depends explicitly on A
        # Both can appear in any order, but both before D
        assert "B" in order[1:3]
        assert "C" in order[1:3]
        assert order[3] == "D"

    # ── 8. Abort during parallel tier ───────────────────────────────

    async def test_abort_during_parallel_tier(self, mgr):
        """Aborting during a parallel tier raises AbortedError."""
        wf = Workflow(
            name="abort_parallel",
            setup=[
                _make_step("root"),
                _make_step("left", depends_on=["root"]),
                _make_step("right", depends_on=["root"]),
                _make_step("final", depends_on=["left", "right"]),
            ],
        )
        executor = WorkflowExecutor(mgr)

        def on_start(step, idx, total):
            if step.name == "left":
                executor.abort()

        executor.on_step_start = on_start

        with pytest.raises((AbortedError, Exception)):
            await executor.run(wf)

        assert executor.state is ExecutorState.ABORTED

    # ── 9. Loop step intra-iteration dependency ─────────────────────

    async def test_loop_step_intra_iteration_dep(self, mgr):
        """Loop step 'run_eis' depending on 'deposit' within each iteration."""
        wf = Workflow(
            name="loop_dep",
            loop_steps=[
                WorkflowStep("deposit", "stage", "move_to",
                             params={"x": 0, "y": 0}),
                WorkflowStep("run_eis", "stage", "move_to",
                             params={"x": 1, "y": 1},
                             depends_on=["deposit"]),
            ],
            iterations=2,
        )
        order: list[str] = []

        executor = WorkflowExecutor(mgr)
        executor.on_step_complete = lambda s, *_: order.append(s.name)
        await executor.run(wf)

        # deposit__iter0 before run_eis__iter0, deposit__iter1 before run_eis__iter1
        assert order.index("deposit__iter0") < order.index("run_eis__iter0")
        assert order.index("deposit__iter1") < order.index("run_eis__iter1")

    # ── 10. Self-dependency ─────────────────────────────────────────

    async def test_self_dependency_raises(self, mgr):
        """A step depending on itself is a cycle."""
        wf = Workflow(
            name="self_dep",
            setup=[
                _make_step("A", depends_on=["A"]),
            ],
        )
        executor = WorkflowExecutor(mgr)
        with pytest.raises(ValidationError_, match="cycle"):
            await executor.run(wf)
