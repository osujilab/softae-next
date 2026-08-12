"""Tests for workflow model, parser, executor, and experiment logger."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.errors import AbortedError, ValidationError_, WorkflowError
from softae.server.manager import InstrumentManager
from softae.workflows.experiment_logger import ExperimentLogger
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep
from softae.workflows.workflow_parser import parse_dict, parse_file


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def manager() -> InstrumentManager:
    return create_mock_manager(config={})


@pytest.fixture
async def connected_manager(manager: InstrumentManager):
    """Return manager with all instruments connected."""
    await manager.connect_all()
    return manager


@pytest.fixture
def simple_workflow_dict() -> dict[str, Any]:
    """Minimal valid workflow definition as a dict."""
    return {
        "name": "test_workflow",
        "description": "Unit test workflow",
        "variables": {
            "temp": 40,
            "preset": "Quick",
        },
        "setup": [
            {
                "name": "set_temp",
                "instrument": "temp_controller",
                "method": "write_sp",
                "params": {"T_SP": "$temp", "print_flag": 0},
            },
        ],
        "loop": {
            "iterate_over": "channels",
            "steps": [
                {
                    "name": "move_stage",
                    "instrument": "stage",
                    "method": "move_to",
                    "params": {"x": 10, "y": 20},
                    "timeout_s": 5.0,
                },
            ],
        },
        "teardown": [
            {
                "name": "cool_down",
                "instrument": "temp_controller",
                "method": "write_sp",
                "params": {"T_SP": 10, "print_flag": 0},
            },
        ],
        "metadata": {"operator": "test"},
    }


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ═══════════════════════════════════════════════════════════════════════
# Workflow Model Tests
# ═══════════════════════════════════════════════════════════════════════


class TestWorkflowStep:
    def test_creation(self):
        step = WorkflowStep(name="s1", instrument="stage", method="move_to", params={"x": 1})
        assert step.name == "s1"
        assert step.params == {"x": 1}
        assert step.retry == 0
        assert step.timeout_s is None

    def test_with_params_returns_copy(self):
        step = WorkflowStep(name="s1", instrument="stage", method="move_to", params={"x": 1})
        new = step.with_params(x=99, y=50)
        assert new.params == {"x": 99, "y": 50}
        assert step.params == {"x": 1}  # original unchanged

    def test_with_tags(self):
        step = WorkflowStep(name="s1", instrument="stage", method="move_to")
        tagged = step.with_tags(iteration="3", channel="7")
        assert tagged.tags == {"iteration": "3", "channel": "7"}
        assert step.tags == {}  # original unchanged


class TestWorkflow:
    def test_resolve_steps_no_loop(self):
        wf = Workflow(
            name="test",
            setup=[WorkflowStep("a", "stage", "move_to")],
            teardown=[WorkflowStep("z", "temp_controller", "write_sp")],
        )
        steps = wf.resolve_steps()
        assert len(steps) == 2
        assert steps[0].name == "a"
        assert steps[1].name == "z"

    def test_resolve_steps_with_loop(self):
        wf = Workflow(
            name="test",
            setup=[WorkflowStep("setup1", "stage", "move_to")],
            loop_steps=[WorkflowStep("loop1", "stage", "move_to")],
            iterations=3,
            teardown=[WorkflowStep("td1", "temp_controller", "write_sp")],
        )
        steps = wf.resolve_steps()
        # 1 setup + 3 loop + 1 teardown = 5
        assert len(steps) == 5
        # Loop steps get iteration-suffixed names
        assert steps[1].name == "loop1__iter0"
        assert steps[2].name == "loop1__iter1"
        assert steps[3].name == "loop1__iter2"
        # Each loop step has an iteration tag
        assert steps[1].tags["iteration"] == "0"
        assert steps[3].tags["iteration"] == "2"

    def test_total_steps(self):
        wf = Workflow(
            name="test",
            setup=[WorkflowStep("s", "x", "y")],
            loop_steps=[WorkflowStep("l1", "x", "y"), WorkflowStep("l2", "x", "y")],
            iterations=4,
            teardown=[WorkflowStep("t", "x", "y")],
        )
        assert wf.total_steps == 1 + 2 * 4 + 1  # 10


# ═══════════════════════════════════════════════════════════════════════
# Parser Tests
# ═══════════════════════════════════════════════════════════════════════


class TestParser:
    def test_variable_interpolation(self, simple_workflow_dict):
        wf = parse_dict(simple_workflow_dict)
        # $temp should be replaced with 40
        assert wf.setup[0].params["T_SP"] == 40

    def test_iteration_count_from_list_variable(self):
        data = {
            "name": "iter_test",
            "variables": {"channels": [0, 1, 2]},
            "loop": {
                "iterate_over": "channels",
                "steps": [{"name": "s", "instrument": "stage", "method": "m"}],
            },
        }
        wf = parse_dict(data)
        assert wf.iterations == 3

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError_):
            parse_dict({"setup": [{"name": "s", "instrument": "x", "method": "y"}]})

    def test_empty_workflow_raises(self):
        with pytest.raises(ValidationError_):
            parse_dict({"name": "empty"})

    def test_missing_step_field_raises(self):
        with pytest.raises(ValidationError_):
            parse_dict({
                "name": "bad",
                "setup": [{"name": "s"}],  # missing instrument & method
            })

    def test_parse_yaml_file(self, tmp_dir):
        yaml_content = """
name: file_test
setup:
  - name: step1
    instrument: stage
    method: move_to
    params:
      x: 5
      y: 10
"""
        yaml_path = tmp_dir / "test.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        wf = parse_file(yaml_path)
        assert wf.name == "file_test"
        assert wf.setup[0].params == {"x": 5, "y": 10}

    def test_parse_json_file(self, tmp_dir):
        data = {
            "name": "json_test",
            "setup": [{"name": "s", "instrument": "stage", "method": "move_to", "params": {"x": 1}}],
        }
        json_path = tmp_dir / "test.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        wf = parse_file(json_path)
        assert wf.name == "json_test"

    def test_partial_string_interpolation(self):
        data = {
            "name": "partial",
            "variables": {"prefix": "run42"},
            "setup": [
                {
                    "name": "s",
                    "instrument": "pico1",
                    "method": "sendscript_getdata",
                    "params": {"output_path": "data/$prefix/output"},
                }
            ],
        }
        wf = parse_dict(data)
        # $prefix should be interpolated (partial match — embedded in string)
        assert wf.setup[0].params["output_path"] == "data/run42/output"


# ═══════════════════════════════════════════════════════════════════════
# Executor Tests
# ═══════════════════════════════════════════════════════════════════════


class TestExecutor:
    @pytest.mark.asyncio
    async def test_simple_execution(self, connected_manager):
        """Run a workflow with a single setup step against mock instruments."""
        wf = Workflow(
            name="exec_test",
            setup=[
                WorkflowStep("set_temp", "temp_controller", "write_sp", params={"T_SP": 45, "print_flag": 0}),
            ],
        )
        executor = WorkflowExecutor(connected_manager)
        await executor.run(wf)
        assert executor.state is ExecutorState.COMPLETED

        # Verify side-effect
        tc = connected_manager.get("temp_controller")
        assert tc.get_sp() == 45.0

    @pytest.mark.asyncio
    async def test_teardown_runs_on_success(self, connected_manager):
        """Verify teardown runs even on normal completion."""
        wf = Workflow(
            name="td_test",
            setup=[
                WorkflowStep("s1", "temp_controller", "write_sp", params={"T_SP": 60, "print_flag": 0}),
            ],
            teardown=[
                WorkflowStep("cool", "temp_controller", "write_sp", params={"T_SP": 15, "print_flag": 0}),
            ],
        )
        executor = WorkflowExecutor(connected_manager)
        await executor.run(wf)
        tc = connected_manager.get("temp_controller")
        assert tc.get_sp() == 15.0  # teardown set it to 15

    @pytest.mark.asyncio
    async def test_multi_step_execution(self, connected_manager):
        """Run setup + loop + teardown against mocks."""
        wf = Workflow(
            name="multi_test",
            setup=[
                WorkflowStep("s1", "temp_controller", "write_sp", params={"T_SP": 50, "print_flag": 0}),
            ],
            loop_steps=[
                WorkflowStep("move", "stage", "move_to", params={"x": 10, "y": 20}),
            ],
            iterations=2,
            teardown=[
                WorkflowStep("cool", "temp_controller", "write_sp", params={"T_SP": 10, "print_flag": 0}),
            ],
        )
        completed_steps: list[str] = []

        def on_complete(step, idx, total, result, elapsed=0.0):
            completed_steps.append(step.name)

        executor = WorkflowExecutor(connected_manager)
        executor.on_step_complete = on_complete
        await executor.run(wf)

        assert executor.state is ExecutorState.COMPLETED
        # setup(1) + loop(2) = 3 tracked completions (teardown runs in finally)
        assert len(completed_steps) == 3
        assert completed_steps[0] == "s1"

    @pytest.mark.asyncio
    async def test_abort(self, connected_manager):
        """Abort during a multi-step workflow."""
        wf = Workflow(
            name="abort_test",
            setup=[
                WorkflowStep("s1", "stage", "move_to", params={"x": 0, "y": 0}),
                WorkflowStep("s2", "stage", "move_to", params={"x": 1, "y": 1}),
                WorkflowStep("s3", "stage", "move_to", params={"x": 2, "y": 2}),
            ],
            teardown=[
                WorkflowStep("td", "temp_controller", "write_sp", params={"T_SP": 10, "print_flag": 0}),
            ],
        )

        executor = WorkflowExecutor(connected_manager)

        # Abort after first step completes
        def on_complete(step, idx, total, result, elapsed=0.0):
            if idx == 0:
                executor.abort()

        executor.on_step_complete = on_complete

        with pytest.raises(AbortedError):
            await executor.run(wf)

        assert executor.state is ExecutorState.ABORTED

    @pytest.mark.asyncio
    async def test_pause_resume(self, connected_manager):
        """Pause and resume a workflow."""
        wf = Workflow(
            name="pause_test",
            setup=[
                WorkflowStep("s1", "stage", "move_to", params={"x": 0, "y": 0}),
                WorkflowStep("s2", "stage", "move_to", params={"x": 1, "y": 1}),
            ],
        )

        executor = WorkflowExecutor(connected_manager)
        completed = []

        def on_complete(step, idx, total, result, elapsed=0.0):
            completed.append(step.name)
            if idx == 0:
                executor.pause()

        executor.on_step_complete = on_complete

        async def resume_after_delay():
            await asyncio.sleep(0.15)
            executor.resume()

        # Run executor and resume concurrently
        await asyncio.gather(
            executor.run(wf),
            resume_after_delay(),
        )

        assert executor.state is ExecutorState.COMPLETED
        assert len(completed) == 2

    @pytest.mark.asyncio
    async def test_state_callbacks(self, connected_manager):
        """Verify state change callbacks fire."""
        wf = Workflow(
            name="cb_test",
            setup=[
                WorkflowStep("s1", "temp_controller", "write_sp", params={"T_SP": 30, "print_flag": 0}),
            ],
        )
        transitions: list[tuple] = []

        executor = WorkflowExecutor(connected_manager)
        executor.on_state_change = lambda old, new: transitions.append((old.name, new.name))

        await executor.run(wf)

        assert ("IDLE", "RUNNING") in transitions
        assert ("RUNNING", "COMPLETED") in transitions


# ═══════════════════════════════════════════════════════════════════════
# Experiment Logger Tests
# ═══════════════════════════════════════════════════════════════════════


class TestExperimentLogger:
    def test_log_step_creates_file(self, tmp_dir):
        with ExperimentLogger(tmp_dir, "test_wf") as el:
            step = WorkflowStep("s1", "stage", "move_to", params={"x": 1})
            el.log_step("test_wf", step, duration_s=0.5, result="ok")

        # Find the log file
        logs = list(tmp_dir.glob("test_wf_*.jsonl"))
        assert len(logs) == 1

        lines = logs[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["step"] == "s1"
        assert record["duration_s"] == 0.5
        assert record["result"] == "ok"
        assert "timestamp" in record

    def test_log_event(self, tmp_dir):
        with ExperimentLogger(tmp_dir, "events") as el:
            el.log_event("pause", reason="user_request")
            el.log_event("resume")

        logs = list(tmp_dir.glob("events_*.jsonl"))
        lines = logs[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        assert r1["event"] == "pause"
        assert r1["reason"] == "user_request"

    @pytest.mark.asyncio
    async def test_executor_with_logger(self, tmp_dir):
        """End-to-end: executor + logger produces a valid log file."""
        mgr = create_mock_manager(config={})
        await mgr.connect_all()

        wf = Workflow(
            name="logged_run",
            setup=[
                WorkflowStep("set_temp", "temp_controller", "write_sp", params={"T_SP": 35, "print_flag": 0}),
                WorkflowStep("move", "stage", "move_to", params={"x": 5, "y": 5}),
            ],
        )

        with ExperimentLogger(tmp_dir, "logged_run") as el:
            executor = WorkflowExecutor(mgr, experiment_logger=el)
            await executor.run(wf)

        logs = list(tmp_dir.glob("logged_run_*.jsonl"))
        assert len(logs) == 1

        lines = logs[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            record = json.loads(line)
            assert record["result"] == "ok"
            assert record["duration_s"] >= 0
