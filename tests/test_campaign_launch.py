"""Handing the rig to a child that outlives the window (S5.J).

These cover the launch *mechanics* with no Qt at all: what the child is told,
where its files go, and — the load-bearing one — that it genuinely survives the
process that started it. The tab-level contract (refuse while connected, attach
by discovery) lives in ``test_tab_bo_live.py`` and
``test_autonomous_run_mixin.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from softae.core.autonomous_wiring import CampaignSpec
from softae.gui import campaign_launch as launch


def _spec(**over):
    base = dict(
        name="phase map",
        channels=(1, 2),
        parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
        vol_params=("vol_p0",),
        pump_ids=(0,),
        budget=4,
    )
    base.update(over)
    return CampaignSpec(**base)


class TestArgv:
    """The child runs the CLI, not a GUI-only path that could drift from it."""

    def test_argv_names_the_campaign_cli_with_the_spec_positionally(self, tmp_path):
        argv = launch.campaign_run_argv(tmp_path / "s.toml", tmp_path,
                                        head_up=True)
        assert argv[:3] == ["-m", "softae.tools.campaign", "run"]
        assert argv[3] == str(tmp_path / "s.toml")     # positional, no --spec
        assert argv[argv.index("--project") + 1] == str(tmp_path)

    def test_argv_pre_approves_prompts_because_the_child_has_no_stdin(self, tmp_path):
        """``spawn_detached`` gives the child ``DEVNULL``; every prompt is a refusal."""
        assert "--yes" in launch.campaign_run_argv(tmp_path / "s.toml", tmp_path,
                                                   head_up=True)

    def test_argv_states_the_head_position_rather_than_letting_it_be_asked(
        self, tmp_path
    ):
        up = launch.campaign_run_argv(tmp_path / "s.toml", tmp_path, head_up=True)
        down = launch.campaign_run_argv(tmp_path / "s.toml", tmp_path, head_up=False)
        assert "--head-up" in up and "--head-down" not in up
        assert "--head-down" in down and "--head-up" not in down

    def test_argv_with_an_unknown_head_position_emits_no_flag(self, tmp_path):
        """Which the child refuses to start on — loudly, rather than guessing."""
        argv = launch.campaign_run_argv(tmp_path / "s.toml", tmp_path, head_up=None)
        assert "--head-up" not in argv and "--head-down" not in argv

    def test_argv_passes_mock_through_so_a_simulated_rig_stays_simulated(
        self, tmp_path
    ):
        assert "--mock" in launch.campaign_run_argv(
            tmp_path / "s.toml", tmp_path, mock=True, head_up=True)
        assert "--mock" not in launch.campaign_run_argv(
            tmp_path / "s.toml", tmp_path, mock=False, head_up=True)


class TestSpecFile:
    def test_the_launched_spec_is_written_beside_the_project(self, tmp_path):
        path = launch.write_launch_spec(_spec(), tmp_path)
        assert path.parent == tmp_path / launch.LAUNCHED_DIRNAME
        assert path.suffix == ".toml"

    def test_the_written_spec_reloads_as_the_campaign_that_was_launched(self, tmp_path):
        from softae.core.campaign_spec_io import load_campaign_spec

        path = launch.write_launch_spec(_spec(budget=9), tmp_path)
        reloaded = load_campaign_spec(path)
        assert reloaded.name == "phase map"
        assert reloaded.budget == 9
        assert reloaded.channels == (1, 2)
        assert reloaded.vol_params == ("vol_p0",)

    def test_a_campaign_name_cannot_escape_the_directory_it_is_written_into(
        self, tmp_path
    ):
        path = launch.write_launch_spec(_spec(name="../../etc/passwd"), tmp_path)
        assert path.parent == tmp_path / launch.LAUNCHED_DIRNAME


class TestConnectedInstruments:
    def test_an_enumeration_that_fails_answers_something_is_open(self):
        """"I could not tell" must not be spelled the same way as "nothing"."""
        class _Broken:
            def list_instruments(self):
                raise RuntimeError("manager is gone")

        assert launch.connected_instruments(_Broken()) != []

    def test_only_connected_instruments_are_reported(self):
        class _M:
            def list_instruments(self):
                return [{"name": "stage", "connected": False},
                        {"name": "syringe", "connected": True}]

        assert launch.connected_instruments(_M()) == ["syringe"]


class TestDetachment:
    """The one property the whole step rests on: closing the GUI is free."""

    def test_spawn_detached_starts_the_child_outside_this_process_group(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        from softae.gui.widgets import calibration_launcher

        seen: dict = {}

        class _Proc:
            pid = 4321

        def fake_popen(argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs
            return _Proc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        pid = calibration_launcher.spawn_detached(
            ["-m", "softae.tools.campaign", "run"], log_file=tmp_path / "c.log")

        assert pid == 4321
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP. Without them the child
            # joins this console's group and dies with the GUI — which is the one
            # thing a multi-day campaign may not do.
            assert seen["kwargs"]["creationflags"] & 0x00000008
            assert seen["kwargs"]["creationflags"] & 0x00000200
        else:
            assert seen["kwargs"]["start_new_session"] is True
        # A pipe would need a reader for the child's whole life, reintroducing the
        # dependency on the GUI staying open, and a full buffer would block the run.
        assert seen["kwargs"].get("stdin") is subprocess.DEVNULL

    def test_spawn_campaign_reuses_the_shipped_detach_helper(self, tmp_path,
                                                             monkeypatch):
        """One statement of what "detached" means, not two that can drift."""
        from softae.gui.widgets import calibration_launcher

        called: list = []
        monkeypatch.setattr(
            calibration_launcher, "spawn_detached",
            lambda argv, *, log_file, cwd=None: called.append(argv) or 99)

        assert launch.spawn_campaign(["-m", "x"], log_file=tmp_path / "l.log") == 99
        assert called == [["-m", "x"]]


class TestWhyTheChildIsADifferentProcess:
    """The regression the handover closes, demonstrated on a temp lock scope.

    :func:`~softae.core.run_lock.acquire_run_lock` is re-entrant for the process
    that already holds the lock — deliberately, so a workflow that acquires and
    then calls a helper that also acquires does not deadlock on itself. But the
    GUI now claims the rig for its instrument *session*
    (:func:`softae.core.rig_session.claim_rig_session`), so a campaign running
    **inside** the GUI got that claim handed back unchanged: the lock kept saying
    ``gui:desktop`` with no run directory, and every other surface was told there
    was no campaign to attach to while one was mid-anneal.

    A child has its own PID, so its acquire is not re-entrant — provided the GUI
    released first, which is why the handover refuses to spawn while the claim
    survives.
    """

    def test_an_in_process_campaign_cannot_publish_its_own_identity(self, tmp_path):
        from softae.core.campaign_discovery import find_running_campaign
        from softae.core.run_lock import (
            acquire_run_lock,
            read_run_lock,
            release_run_lock,
        )

        acquire_run_lock(tmp_path, "gui:desktop", log_path="")
        try:
            acquire_run_lock(tmp_path, "campaign:phase_map:20260819T1200Z",
                             log_path=str(tmp_path / "runs" / "x"))
            lock = read_run_lock(tmp_path)
            assert lock.what == "gui:desktop"       # the campaign's `what` is lost
            assert lock.log_path == ""              # and so is its run directory
            assert not find_running_campaign(lock_reader=lambda: lock).controllable
        finally:
            release_run_lock(tmp_path)

    def test_a_campaign_that_owns_the_rig_alone_is_attachable(self, tmp_path):
        """What the detached child gets instead, once the GUI has let go."""
        from softae.core.campaign_discovery import find_running_campaign
        from softae.core.run_lock import acquire_run_lock, release_run_lock

        run_dir = tmp_path / "runs" / "20260819T1200Z_phase_map"
        lock = acquire_run_lock(tmp_path, "campaign:phase_map:20260819T1200Z",
                                log_path=str(run_dir))
        try:
            target = find_running_campaign(lock_reader=lambda: lock)
            assert target.controllable
            assert target.run_dir == str(run_dir)
            assert target.detail == "campaign:phase_map:20260819T1200Z"
        finally:
            release_run_lock(tmp_path)


class TestAHeadlessCampaignIsDiscoverableDespiteTheCLIsEarlyClaim:
    """The same defect as the class above, arriving from the other direction.

    ``TestWhyTheChildIsADifferentProcess`` covers the *unfixable* case: the GUI's
    ``gui:desktop`` session claim is a genuinely different owner, so a campaign
    running inside it cannot publish its own identity and the answer is a child
    process. This is the case that is not that. ``tools/campaign.py`` now claims
    the rig itself before it builds a DataStore — closing the unclaimed window
    around ``connect_all`` — and that claim is the **same** campaign, taken a few
    hundred milliseconds early, when the run id does not exist yet. Handing it
    back unchanged is what ``acquire_run_lock``'s re-entrancy does, and it would
    have made every headless run undiscoverable to ``softae-campaign control``
    for its whole length.

    So this drives the real machinery end to end — placeholder claim, then
    ``run_autonomous_campaign`` — and asks the production discovery path, with no
    injected ``lock_reader``, what a controller would find.
    """

    @staticmethod
    def _campaign_spec():
        # Board-shaped, unlike the launch-mechanics `_spec` above: this one is
        # actually run, so it needs channels a real PCB has.
        return CampaignSpec(
            name="early_claim",
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
            budget=1,
            seed=7,
        )

    @pytest.fixture
    def _real_rig_on_a_temp_scope(self, tmp_path, monkeypatch):
        """A mock rig that reads as real, on a lock nobody else can see.

        Both halves are needed and for opposite reasons: without the scope
        redirect this test would take the operator's actual ``~/.softae/rig.lock``
        mid-experiment, and without the simulation override no claim is taken at
        all, since a mock run must never lock out a real one. Patched on **both**
        modules — ``autonomous_wiring`` binds the predicate at import time — or
        the campaign's own claim silently never happens and the test passes by
        never reaching the code it is about.
        """
        from softae.core import autonomous_wiring as aw
        from softae.core import run_lock as rl

        scope = tmp_path / "lockscope"
        scope.mkdir()
        monkeypatch.setattr(rl, "DEFAULT_SCOPE", scope)
        monkeypatch.setattr(rl, "rig_is_simulated", lambda _m: False)
        monkeypatch.setattr(aw, "rig_is_simulated", lambda _m: False)
        return scope

    async def test_a_run_under_the_clis_early_claim_still_publishes_its_run_dir(
        self, tmp_path, _real_rig_on_a_temp_scope
    ):
        """What ``softae-campaign control`` finds while the campaign is running."""
        from softae.core.autonomous_wiring import run_autonomous_campaign
        from softae.core.campaign_discovery import find_running_campaign
        from softae.core.data_store import DataStore
        from softae.core.run_lock import held_run_lock
        from softae.drivers.mock_factory import create_mock_manager

        spec = self._campaign_spec()
        seen: list = []

        def watch(event: dict) -> None:
            if event.get("type") == "suggestion" and not seen:
                # No `lock_reader`: the production path, reading the same file a
                # separate controller process would open.
                seen.append(find_running_campaign())

        manager = create_mock_manager(config={})
        await manager.connect_all()
        store = DataStore(tmp_path / "proj")
        try:
            # Exactly what `tools/campaign.py` takes before it has a run id: the
            # campaign's name, and nothing else it does not know yet.
            with held_run_lock(what=f"campaign:{spec.name}"):
                result = await run_autonomous_campaign(
                    spec, manager=manager, data_store=store, on_event=watch)
        finally:
            store.close()
            await manager.disconnect_all()

        target = seen[0]
        assert target.controllable, (
            "a controller would have refused a live campaign: "
            f"{target.refusal}")
        assert target.run_dir == str(store.run_dir(result.run_id))
        assert target.detail == f"campaign:{spec.name}:{result.run_id}"

    async def test_the_early_claim_is_still_the_same_claim_afterwards(
        self, tmp_path, _real_rig_on_a_temp_scope
    ):
        """Renaming a claim must not silently release or re-take it.

        The outer ``with`` is what owns the rig for the CLI's whole window, so if
        refining the label handed the lock away — or replaced it with a claim the
        outer block does not think it made — the window this all exists to close
        would reopen at the far end.
        """
        from softae.core.autonomous_wiring import run_autonomous_campaign
        from softae.core.data_store import DataStore
        from softae.core.run_lock import held_run_lock, read_run_lock
        from softae.drivers.mock_factory import create_mock_manager

        spec = self._campaign_spec()
        manager = create_mock_manager(config={})
        await manager.connect_all()
        store = DataStore(tmp_path / "proj")
        try:
            with held_run_lock(what=f"campaign:{spec.name}") as outer:
                await run_autonomous_campaign(
                    spec, manager=manager, data_store=store)
                still = read_run_lock()
                assert still is not None, "the campaign gave the rig away on exit"
                assert (still.pid, still.started_at) == (
                    outer.pid, outer.started_at), (
                    "the claim was re-taken rather than renamed")
        finally:
            store.close()
            await manager.disconnect_all()

        assert read_run_lock() is None, "the CLI's own claim outlived its block"


class TestDetachedCampaignHandle:
    def test_the_handle_offers_no_abort_so_closing_the_window_cannot_stop_the_run(
        self, tmp_path
    ):
        """The filed ``self._runner`` defect, fixed by giving it the right shape.

        ``BOTabBase._abort_run_impl`` calls ``abort()`` on whatever ``_runner``
        holds, and ``DaemonRunnerMixin.cleanup`` calls that from the window's
        ``closeEvent``. A handle that could stop the campaign would stop it every
        time the operator closed the window.
        """
        child = launch.DetachedCampaign(
            pid=11, name="c", spec_path=Path("s.toml"), log_path=Path("l.log"),
            project_dir=tmp_path, started_at="2026-08-19T00:00:00+00:00")
        assert not hasattr(child, "abort")
        assert not hasattr(child, "terminate")
        assert not hasattr(child, "kill")

    def test_the_handle_says_the_run_outlives_this_window(self, tmp_path):
        child = launch.DetachedCampaign(
            pid=11, name="phase_map", spec_path=Path("s.toml"),
            log_path=Path("l.log"), project_dir=tmp_path,
            started_at="2026-08-19T00:00:00+00:00")
        text = child.describe()
        assert "PID 11" in text and "close" in text
