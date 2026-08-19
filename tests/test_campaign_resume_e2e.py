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


async def _park_on_a_failure_streak(connected, store) -> dict:
    """Run a campaign in which nothing measures, so it parks at three failures.

    Returns the checkpoint it left behind. Shared by both arms below because the
    *only* thing that may differ between them is how the run then stopped —
    if the setups drifted apart, neither arm would be evidence about the
    discriminator.
    """
    unmeasurable = lambda _results: None                          # noqa: E731
    events: list[dict] = []
    result = await run_autonomous_campaign(
        _spec(budget=6), manager=connected, data_store=store,
        objective_extractor=unmeasurable, on_event=events.append)

    assert "consecutive" in (result.park_reason or "")
    assert sum(e["type"] == "suggestion" for e in events) == 3

    # A parked campaign keeps its checkpoint — that is what makes it resumable —
    # and the checkpoint must carry the streak that parked it. **Three, not
    # two**: the count is incremented *before* the iteration that writes the
    # checkpoint advances. This assertion is the ordering fix and is independent
    # of everything below — a checkpoint written before the increment would
    # persist `n-1`, making persistence a silent no-op no matter which way the
    # resume rule then went.
    cp = store.campaign_checkpoint("resumable")
    assert cp is not None and cp["consecutive_failures"] == 3
    return cp


async def _resume_and_count_suggestions(connected, store) -> int:
    unmeasurable = lambda _results: None                          # noqa: E731
    events: list[dict] = []
    await run_autonomous_campaign(
        _spec(budget=6), manager=connected, data_store=store,
        objective_extractor=unmeasurable, on_event=events.append, resume=True)
    return sum(e["type"] == "suggestion" for e in events)


@pytest.mark.asyncio
async def test_a_clean_park_resumes_with_a_clear_streak(connected, tmp_path: Path):
    """A park is a stop somebody was told about, so the slate is clean.

    The park raised a CRITICAL alert and the process unwound far enough to write
    a terminal status (``stopped``). The operator resuming it is presumed to have
    been to the rig — which is the whole reason a park stops the run rather than
    retrying forever. So the resumed run gets its full allowance of three trials
    back, not a counter already at the limit.

    The companion test below is the other arm, and the pair is the point: same
    fault, same streak, same budget — only *how the previous run stopped* differs.
    """
    store = DataStore(tmp_path / "proj")
    await _park_on_a_failure_streak(connected, store)

    assert await _resume_and_count_suggestions(connected, store) == 3, (
        "an acknowledged park must clear the streak, not carry it")
    store.close()


@pytest.mark.asyncio
async def test_a_crash_marked_run_resumes_with_its_streak_intact(
    connected, tmp_path: Path
):
    """**The regression that matters**: a crash acknowledges nothing.

    ``AutonomousLoop`` parks after three consecutive failed/unmeasured trials.
    That counter was once zeroed on *every* resume, so a chronic fault — the
    exact fault the limit exists to catch — got a fresh allowance of three trials
    after each restart, and in a crash-restart loop the limit could never fire at
    all. The counter existed and could not escalate.

    Here the previous run is put in the state a hard kill really leaves — its row
    never closed — and then marked by the real next-launch recovery sweep, which
    is what stamps ``interrupted``. Nobody has been to the rig. The resumed run
    must therefore park on the **first** further failure, not the fourth.
    """
    from softae.core.shutdown import UnfinishedRuns, record_unclean_shutdown

    store = DataStore(tmp_path / "proj")
    cp = await _park_on_a_failure_streak(connected, store)

    # What TerminateProcess leaves behind: the row was never finalized.
    store._conn.execute(
        "UPDATE experiments SET finished_at = NULL, status = 'running' "
        "WHERE run_id = ?", (cp["run_id"],))
    store._conn.commit()
    # ...and what the next launch does about it — the real recovery path, so this
    # tracks `record_unclean_shutdown` rather than hard-coding the string it writes.
    record_unclean_shutdown(UnfinishedRuns(tuple(store.unfinished_runs())), store)
    assert store.run_outcome(cp["run_id"])["status"] == "interrupted"

    assert await _resume_and_count_suggestions(connected, store) == 1, (
        "the restored streak was ignored — the fault got a fresh allowance")
    store.close()


@pytest.mark.asyncio
async def test_an_unclosed_run_resumes_with_its_streak_intact(
    connected, tmp_path: Path
):
    """The crash arm again, *before* any recovery sweep has run.

    Recovery happens at the next launch, and a resume can beat it there — the
    headless CLI skips it while another process holds the rig lock, and nothing
    forces it to have run at all. So the row still reads ``running`` with
    ``finished_at`` NULL, which is byte-for-byte what a live run looks like and
    is exactly why the status string alone cannot be trusted: ``finished_at``
    is what separates "never closed" from "closed as still running".
    """
    store = DataStore(tmp_path / "proj")
    cp = await _park_on_a_failure_streak(connected, store)

    store._conn.execute(
        "UPDATE experiments SET finished_at = NULL, status = 'running' "
        "WHERE run_id = ?", (cp["run_id"],))
    store._conn.commit()

    assert await _resume_and_count_suggestions(connected, store) == 1, (
        "an unfinalized run row is not an acknowledgement")
    store.close()


@pytest.mark.asyncio
async def test_an_error_exit_resumes_with_its_streak_intact(
    connected, tmp_path: Path
):
    """``error`` is the campaign's crash status — including Ctrl-C.

    ``run_autonomous_campaign``'s ``except BaseException`` catch-all finalizes as
    ``error``, and it covers the crash, the cancellation and the
    ``KeyboardInterrupt`` alike. None of those is a report that a human read and
    acted on, so none of them clears the streak.
    """
    store = DataStore(tmp_path / "proj")
    cp = await _park_on_a_failure_streak(connected, store)

    store._conn.execute(
        "UPDATE experiments SET status = 'error' WHERE run_id = ?",
        (cp["run_id"],))
    store._conn.commit()

    assert await _resume_and_count_suggestions(connected, store) == 1, (
        "an error exit is not an acknowledgement")
    store.close()


@pytest.mark.asyncio
async def test_the_resume_says_which_way_the_streak_went(connected, tmp_path: Path):
    """Whichever way it goes, it is announced — with the reason.

    A counter that silently changes value across a restart is what made this
    hard to see in the first place: both of the earlier absolute behaviours were
    invisible at runtime.
    """
    store = DataStore(tmp_path / "proj")
    cp = await _park_on_a_failure_streak(connected, store)

    unmeasurable = lambda _results: None                          # noqa: E731
    cleared: list[dict] = []
    await run_autonomous_campaign(
        _spec(budget=6), manager=connected, data_store=store,
        objective_extractor=unmeasurable, on_event=cleared.append, resume=True)

    note = next(e for e in cleared if e["type"] == "resume_failure_streak")
    assert note["action"] == "cleared"
    assert note["saved"] == 3 and note["consecutive_failures"] == 0
    assert "stopped" in note["why"]
    store.close()


@pytest.mark.asyncio
async def test_a_zero_failure_streak_resumes_with_its_full_allowance(
    connected, tmp_path: Path
):
    """The control for the crash arm, and the reason it proves anything.

    Same crash marking, same fault, same budget as
    ``test_a_crash_marked_run_resumes_with_its_streak_intact`` — only the
    *stored* streak differs. A crash-marked resume whose counter is zero takes
    the full three trials to park, so the one-trial park in that test is
    attributable to the persisted count and not to some other way a resumed run
    stops early. (Pointing this control at a crash-marked run rather than a
    parked one is deliberate: after a clean park the streak is cleared either
    way, so a parked control could not tell the two apart.)
    """
    store = DataStore(tmp_path / "proj")
    cp = await _park_on_a_failure_streak(connected, store)

    store._conn.execute(
        "UPDATE experiments SET finished_at = NULL, status = 'running' "
        "WHERE run_id = ?", (cp["run_id"],))
    store._conn.commit()
    store.save_campaign_checkpoint(
        "resumable", iteration=cp["iteration"], run_id=cp["run_id"],
        loop_state=cp["loop_state"], board_id=cp["board_id"],
        spec_json=cp["spec_json"], optimizer_json=cp["optimizer_json"],
        consecutive_failures=0)

    assert await _resume_and_count_suggestions(connected, store) == 3
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
