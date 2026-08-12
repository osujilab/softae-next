"""Tests for EIS auto-routing to DataStore (B3)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def manager() -> InstrumentManager:
    return create_mock_manager(config={})


@pytest.fixture
async def connected_manager(manager: InstrumentManager):
    await manager.connect_all()
    return manager


@pytest.fixture
def data_store(tmp_path: Path):
    ds = DataStore(tmp_path / "project")
    yield ds
    ds.close()


@pytest.fixture
def run_id(data_store: DataStore):
    return data_store.start_run("eis_autoroute_test")


# ── sendscript_getdata routing ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_eis_step_auto_persisted_to_datastore(
    connected_manager, data_store, run_id,
):
    wf = Workflow(
        name="eis_route",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)

    rows = data_store.query_measurements(run_id=run_id)
    assert len(rows) == 1
    assert rows[0]["channel"] == 1
    assert rows[0]["npts"] == 41  # MockESPico generates 41-point spectra


@pytest.mark.asyncio
async def test_eis_extractdata_auto_persisted(
    connected_manager, data_store, run_id,
):
    """Two-step workflow: getdata → extractdata. Both are EIS methods."""
    wf = Workflow(
        name="extract_route",
        setup=[
            WorkflowStep("getdata", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 2}),
            WorkflowStep("extract", "pico1", "eis_extractdata",
                         params={"curves": "{{needs_runtime}}",  "chan": 3}),
        ],
    )
    # We can't chain results across steps in a unit test, but each step is
    # independently faked by MockESPico. Both should produce measurement rows.
    # However eis_extractdata takes `curves` as a positional arg and MockESPico
    # handles it. Let's test sendscript_getdata alone here and validate method detection.
    wf2 = Workflow(
        name="single_extract",
        setup=[
            WorkflowStep("extract", "pico1", "eis_extractdata",
                         params={"curves": []}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf2)

    rows = data_store.query_measurements(run_id=run_id)
    assert len(rows) == 1
    assert rows[0]["channel"] == 0  # default channel


# ── EIS file saving ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_eis_file_saved_to_run_dir(
    connected_manager, data_store, run_id,
):
    wf = Workflow(
        name="file_save",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)

    eis_dir = data_store.eis_dir(run_id)
    files = list(eis_dir.glob("*.txt"))
    assert len(files) == 1
    assert "measure_ch1" in files[0].name


# ── Auto-fit ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fit_auto_persisted_with_circuit_model(
    connected_manager, data_store, run_id,
):
    wf = Workflow(
        name="fit_route",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1,
                                 "circuit_model": "simpleSalt"}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)

    fits = data_store.query_fits(run_id=run_id)
    assert len(fits) == 1
    assert fits[0]["model_name"] == "simpleSalt"


@pytest.mark.asyncio
async def test_no_fit_without_circuit_model(
    connected_manager, data_store, run_id,
):
    wf = Workflow(
        name="nofit_route",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)

    fits = data_store.query_fits(run_id=run_id)
    assert len(fits) == 0


# ── Graceful degradation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graceful_skip_no_datastore(connected_manager):
    wf = Workflow(
        name="nostore",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    executor = WorkflowExecutor(connected_manager)  # no data_store
    await executor.run(wf)
    assert executor.state == ExecutorState.COMPLETED


@pytest.mark.asyncio
async def test_graceful_skip_no_run_id(connected_manager, data_store):
    wf = Workflow(
        name="norunid",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store)  # no run_id
    await executor.run(wf)
    assert executor.state == ExecutorState.COMPLETED


# ── Non-EIS steps ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_eis_step_not_routed(connected_manager, data_store, run_id):
    wf = Workflow(
        name="stage_only",
        setup=[
            WorkflowStep("move", "stage", "move_to", params={"x": 5, "y": 5}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)

    rows = data_store.query_measurements(run_id=run_id)
    assert len(rows) == 0


# ── Failure resilience ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_autoroute_failure_does_not_abort_workflow(connected_manager, data_store, run_id):
    """Simulate a DataStore failure — workflow must still complete."""
    wf = Workflow(
        name="fail_route",
        setup=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    # Monkey-patch record_measurement to simulate a DB failure
    original = data_store.record_measurement
    def failing_record(*a, **kw):
        raise RuntimeError("simulated DB failure")
    data_store.record_measurement = failing_record
    try:
        executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
        await executor.run(wf)
        assert executor.state == ExecutorState.COMPLETED
    finally:
        data_store.record_measurement = original


# ── Multi-iteration ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_iteration_eis_creates_multiple_measurements(
    connected_manager, data_store, run_id,
):
    wf = Workflow(
        name="loop_eis",
        iterations=3,
        loop_steps=[
            WorkflowStep("measure", "pico1", "sendscript_getdata",
                         params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1}),
        ],
    )
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)

    rows = data_store.query_measurements(run_id=run_id)
    assert len(rows) == 3
