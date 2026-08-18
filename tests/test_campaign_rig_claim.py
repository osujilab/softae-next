"""The rig lock is held for a whole campaign, and says which campaign holds it.

Before this, ``WorkflowExecutor.run`` took the lock per *workflow* and dropped it
in its ``finally`` — and one trial is one ``executor.run``. So for the whole gap
between trials (the BO fit and suggest, the checkpoint write, the settle sidecar)
the lock file did not exist, and anything asking ``read_run_lock()`` was told the
rig was **free** while a campaign was mid-round. A lock that blinks is worse than
no lock, because something believes it.

Two defects had to be closed together and the ordering matters:

1. the campaign claims the rig once, for its whole length (this module's subject);
2. the executor stops releasing a claim it did not make. ``acquire_run_lock`` is
   re-entrant — asked by the process that already owns the lock it hands the
   existing claim back rather than raising — so without (2) the *first trial's*
   teardown would delete the campaign's claim and every trial after it would run
   on a rig the lock file called free. (1) alone would have been a lie.

**No rig.** The mock manager is exempt from the lock by construction
(``rig_is_simulated``), so these tests patch that predicate to make the mock read
as real, and redirect ``DEFAULT_SCOPE`` at ``tmp_path`` so nothing touches the
machine's real ``~/.softae/rig.lock``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core import autonomous_wiring as aw
from softae.core import run_lock as rl
from softae.core.autonomous_wiring import CampaignSpec, run_autonomous_campaign
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager
from softae.workflows.workflow_model import Workflow, WorkflowStep
from softae.workflows.workflow_executor import WorkflowExecutor

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="claim_campaign",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=2,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
def rig_scope(tmp_path: Path, monkeypatch) -> Path:
    """Point the lock at *tmp_path* and make the mock rig read as real.

    Both are necessary. Without the scope redirect a test would take (and break)
    the operator's actual rig lock; without the simulation override no lock is
    taken at all, since a mock run must never lock out a real one.
    """
    scope = tmp_path / "lockscope"
    scope.mkdir()
    monkeypatch.setattr(rl, "DEFAULT_SCOPE", scope)
    # Both bindings, deliberately: `autonomous_wiring` imports the predicate at
    # module scope while `workflow_executor` imports it inside `run()`, so one
    # patch reaches one of them and silently leaves the other simulated.
    monkeypatch.setattr(rl, "rig_is_simulated", lambda _manager: False)
    monkeypatch.setattr(aw, "rig_is_simulated", lambda _manager: False)
    return scope


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


# ── The executor must not free a claim it did not make ───────────────────────

def _trivial_workflow() -> Workflow:
    return Workflow(
        name="inner",
        setup=[WorkflowStep(name="pos", instrument="stage",
                            method="live_position", params={})],
    )


@pytest.mark.asyncio
async def test_a_workflow_does_not_release_a_lock_its_caller_already_held(
        rig_scope, connected):
    """The bug that would have made a campaign-length claim a lie.

    A campaign claims the rig, then runs trial after trial through the executor.
    Each ``executor.run`` re-acquires (re-entrantly) and releases — so the first
    trial's teardown used to delete the outer claim.
    """
    rl.acquire_run_lock(what="campaign:claim_campaign:run-1")

    await WorkflowExecutor(connected).run(_trivial_workflow())

    still = rl.read_run_lock()
    assert still is not None, "the workflow freed the rig its caller was using"
    assert still.what == "campaign:claim_campaign:run-1"


@pytest.mark.asyncio
async def test_a_workflow_with_no_outer_claim_still_takes_and_returns_the_rig(
        rig_scope, connected):
    """The fix must not turn the ordinary case into a leak."""
    seen: list[object] = []
    executor = WorkflowExecutor(connected)
    executor.on_step_complete = lambda *_: seen.append(rl.read_run_lock())

    await executor.run(_trivial_workflow())

    assert seen and seen[0] is not None, "no lock was held while stepping"
    assert rl.read_run_lock() is None, "the workflow kept the rig after finishing"


# ── The campaign-length claim ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_rig_lock_is_held_for_every_moment_of_a_campaign(
        rig_scope, connected, tmp_path: Path):
    """Including the gaps between trials, which is where it used to vanish."""
    store = DataStore(tmp_path / "proj")
    free_at: list[str] = []

    def watch(event: dict) -> None:
        if rl.read_run_lock() is None:
            free_at.append(event.get("type", "?"))

    try:
        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store, on_event=watch,
        )
    finally:
        store.close()

    assert result.n_trials == 2
    assert free_at == [], f"the rig read as free during {free_at}"


@pytest.mark.asyncio
async def test_the_claim_names_the_campaign_and_the_run_it_belongs_to(
        rig_scope, connected, tmp_path: Path):
    """``what`` is the discovery channel: ``campaign:<name>:<run_id>``.

    A reader that finds this lock can name the owner and open its run directory
    from one file read — no registry, no scan, no second ownership file that
    could disagree with this one.
    """
    store = DataStore(tmp_path / "proj")
    seen: list[object] = []

    def watch(event: dict) -> None:
        if event.get("type") == "suggestion" and not seen:
            seen.append(rl.read_run_lock())

    try:
        result = await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store, on_event=watch,
        )
    finally:
        store.close()

    lock = seen[0]
    assert lock is not None
    assert lock.what == f"campaign:claim_campaign:{result.run_id}"
    assert lock.log_path.endswith(result.run_id)
    # And it renders through the ordinary owner vocabulary, not a second one.
    from softae.gui.widgets.rig_owner import campaign_identity

    assert campaign_identity(lock) == ("claim_campaign", result.run_id)


@pytest.mark.asyncio
async def test_the_rig_is_given_back_when_the_campaign_ends(
        rig_scope, connected, tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    try:
        await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store)
    finally:
        store.close()

    assert rl.read_run_lock() is None


@pytest.mark.asyncio
async def test_a_campaign_that_dies_mid_run_still_gives_the_rig_back(
        rig_scope, connected, tmp_path: Path):
    """A crash must not leave the rig claimed until someone deletes a file.

    ``read_run_lock``'s liveness check is the backstop for a process that dies
    outright; it must not become the mechanism for a campaign that merely raised.
    """
    store = DataStore(tmp_path / "proj")

    def explode(event: dict) -> None:
        if event.get("type") == "suggestion":
            raise RuntimeError("trial blew up")

    try:
        with pytest.raises(RuntimeError):
            await run_autonomous_campaign(
                _spec(), manager=connected, data_store=store, on_event=explode)
    finally:
        store.close()

    assert rl.read_run_lock() is None


@pytest.mark.asyncio
async def test_a_second_campaign_is_refused_while_one_holds_the_rig(
        rig_scope, connected, tmp_path: Path):
    """The refusal names the holder — a PID alone cannot be acted on."""
    import os

    other = rl.RunLock(pid=os.getpid(), what="campaign:other:run-9",
                       started_at="2026-08-17T09:00:00+00:00", host="somewhere-else")
    rl.lock_path().write_text(
        __import__("json").dumps(other.to_dict()), encoding="utf-8")

    store = DataStore(tmp_path / "proj")
    try:
        with pytest.raises(rl.RunLockHeld) as excinfo:
            await run_autonomous_campaign(
                _spec(budget=1), manager=connected, data_store=store)
    finally:
        store.close()

    message = str(excinfo.value)
    assert "campaign:other:run-9" in message
    assert "2026-08-17T09:00:00+00:00" in message


@pytest.mark.asyncio
async def test_a_simulated_campaign_takes_no_lock_at_all(
        tmp_path: Path, monkeypatch, connected):
    """The one legitimate exemption, kept.

    A mock run holding the rig turns a dry run into an outage for a real one, so
    the campaign claim inherits the executor's exemption rather than inventing a
    stricter rule of its own.
    """
    scope = tmp_path / "lockscope"
    scope.mkdir()
    monkeypatch.setattr(rl, "DEFAULT_SCOPE", scope)

    store = DataStore(tmp_path / "proj")
    seen: list[object] = []
    try:
        await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store,
            on_event=lambda e: seen.append(rl.read_run_lock()),
        )
    finally:
        store.close()

    assert seen and all(s is None for s in seen)
