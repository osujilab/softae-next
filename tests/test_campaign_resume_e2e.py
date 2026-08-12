"""End-to-end resume: interrupt a campaign, restart it, continue the search.

This is P3.3's acceptance criterion driven through the real
``run_autonomous_campaign`` rather than the resume helper alone — a checkpoint is
only worth having if the campaign entry point actually consumes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.autonomous_wiring import (
    CampaignSpec,
    composition_target_objective,
    run_autonomous_campaign,
)
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="resumable",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        time_scale=0.0,
        budget=3,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


@pytest.mark.asyncio
async def test_checkpoint_is_written_during_a_campaign(connected, tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    spec = _spec(budget=3)
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    await run_autonomous_campaign(
        spec, manager=connected, data_store=store, objective_extractor=obj)

    # Cleared on an intentional finish — the campaign ran to budget.
    assert store.campaign_checkpoint(spec.name) is None
    store.close()


@pytest.mark.asyncio
async def test_a_clean_finish_clears_the_checkpoint(connected, tmp_path: Path):
    """Only an *interrupted* run stays resumable.

    A campaign that reached its budget ended on purpose; leaving its checkpoint
    behind would let a later ``resume=True`` silently continue a finished
    experiment instead of repeating it.
    """
    store = DataStore(tmp_path / "proj")
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    await run_autonomous_campaign(
        _spec(budget=2), manager=connected, data_store=store,
        objective_extractor=obj)

    assert store.campaign_checkpoint("resumable") is None
    store.close()


@pytest.mark.asyncio
async def test_resume_continues_from_the_checkpoint(
    connected, tmp_path: Path, monkeypatch
):
    """The acceptance test: interrupt mid-campaign, resume, continue the search.

    The interruption is simulated by suppressing the clean-finish cleanup, which
    is exactly the state a crash or power cut leaves behind: iterations recorded
    and a checkpoint still present because no terminal path ran.
    """
    store = DataStore(tmp_path / "proj")
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    monkeypatch.setattr(store, "clear_campaign_checkpoint", lambda *_a, **_k: None)
    first = await run_autonomous_campaign(
        _spec(budget=2), manager=connected, data_store=store,
        objective_extractor=obj)
    assert first.n_trials == 2

    cp = store.campaign_checkpoint("resumable")
    assert cp is not None and cp["iteration"] == 2
    monkeypatch.undo()

    # Resume with a raised budget — the continuation runs the extra iterations
    # rather than restarting the search from zero.
    events: list[dict] = []
    resumed = await run_autonomous_campaign(
        _spec(budget=4), manager=connected, data_store=store,
        objective_extractor=obj, on_event=events.append, resume=True)

    resumed_evt = [e for e in events if e["type"] == "resumed"]
    assert resumed_evt, "campaign did not report resuming"
    assert resumed_evt[0]["n_observations"] == 2      # history carried over
    assert resumed_evt[0]["remaining_budget"] == 2

    # Iteration count continues (2 → 4) instead of rewinding to 0.
    assert resumed.n_trials == 4
    assert len(resumed.history) == 4
    # Only 2 *new* suggestions were needed to reach the raised budget.
    assert sum(e["type"] == "suggestion" for e in events) == 2
    store.close()


@pytest.mark.asyncio
async def test_resume_without_a_checkpoint_starts_normally(connected, tmp_path: Path):
    """A missing checkpoint must not be fatal — it is just a fresh start."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    result = await run_autonomous_campaign(
        _spec(budget=2), manager=connected, data_store=store,
        objective_extractor=obj, on_event=events.append, resume=True)

    assert result.n_trials == 2
    assert any(e["type"] == "resume_no_checkpoint" for e in events)
    store.close()


@pytest.mark.asyncio
async def test_a_resumed_run_does_not_recast_occupied_wells(
    connected, tmp_path: Path
):
    """Single-use wells: the durable occupancy record must gate the resume."""
    store = DataStore(tmp_path / "proj")
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    spec = _spec(budget=2, electrode_capacity=8, electrode_start=1, channels=(1,))
    await run_autonomous_campaign(
        spec, manager=connected, data_store=store, objective_extractor=obj)

    board = store.current_board_id()
    used = store.occupied_electrodes(board)
    assert used, "the run should have recorded occupancy"

    # A second run on the same board must allocate only unused wells.
    from softae.core.electrode_allocator import ElectrodeAllocator

    alloc = ElectrodeAllocator(capacity=8, start=1, occupied=frozenset(used))
    granted = alloc.allocate(3).channels
    assert not (set(granted) & set(used))
    store.close()
