"""The rig claim follows the instrument session, not the run that uses it.

``gui/app.py`` in owner mode armed the interlock and opened every serial port and
took **no lock at all** — the file contained zero ``run_lock`` references. So for
the whole idle life of an open window the rig read as *free* while the GUI held
every port, and a headless ``softae-campaign run`` started in that window passed
its own guard and connected on top. These tests pin the rule that closes it:
*acquire when the ports open, release when they close.*

Two of them use a **real second process**, because that is the only honest way to
ask the question. ``foreign_run_lock`` deliberately does not report this process's
own lock as foreign, so a same-process assertion could only be made true by
patching the predicate under test. A child interpreter costs ~0.3 s and asks the
real question.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from softae.core import rig_session as rs
from softae.core import run_lock as rl
from softae.core.rig_session import (
    DESKTOP_SESSION,
    claim_rig_session,
    held_rig_session,
    release_rig_session,
    session_is_simulated,
)
from softae.core.run_lock import RunLockHeld
from softae.drivers.mock_factory import create_mock_manager
from softae.drivers.mock_keithley import MockKeithley
from softae.drivers.mock_temp_controller import MockTempController


# ── Fixtures and stand-ins ───────────────────────────────────────────────────


@pytest.fixture
def rig_scope(tmp_path: Path, monkeypatch) -> Path:
    """A private lock scope, so nothing here can see the operator's rig.lock."""
    scope = tmp_path / "lockscope"
    scope.mkdir()
    monkeypatch.setattr(rl, "DEFAULT_SCOPE", scope)
    return scope


class StandInTempController(MockTempController):
    """A driver that is *not* a mock as far as either predicate can tell.

    Subclassed rather than stubbed so it behaves like an instrument in a running
    widget; only the class name matters to the detection, which keys on the
    ``Mock`` prefix exactly as :func:`softae.core.hardware_safety.probe_motion`
    does.
    """


class StandInKeithley(MockKeithley):
    """As above, for a real potentiostat."""


class UnlistableManager:
    """A manager whose registry cannot be read — the "I could not look" case."""

    @property
    def names(self):
        raise OSError("HID bus gone")


def _real_heater_mock_stage_manager():
    """A real potentiostat and a real heater, with the stage still a mock.

    The configuration ``rig_is_simulated`` cannot see: it delegates to
    ``probe_motion``, defined only over ``("stage", "syringe", "piezo")``, so this
    rig reads as *simulated* to it while owning two real ports.
    """
    mgr = create_mock_manager(config={})
    mgr.register(StandInTempController("temp_controller", {}))
    mgr.register(StandInKeithley("keithley", {}))
    return mgr


def _write_foreign_lock(scope: Path, **fields) -> None:
    """A lock held by a live process that is not this one.

    Another *host* rather than another PID: a foreign host reads as alive by
    design, so the fixture never has to find a real running process.
    """
    data = dict(pid=os.getpid() + 1, what="campaign:phase_map:run_042",
                started_at="2026-08-19T14:02:00+00:00",
                host="some-other-machine", log_path="")
    data.update(fields)
    path = rl.lock_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ── What counts as a session worth claiming for ──────────────────────────────


class TestSessionIsSimulated:
    def test_session_is_simulated_all_mock_suite_returns_true(self):
        assert session_is_simulated(create_mock_manager(config={})) is True

    def test_session_is_simulated_real_heater_and_mock_stage_returns_false(self):
        """The hole this predicate exists to close, stated as a contrast.

        Both answers are asserted together on purpose: the motion-scoped one is
        not wrong, it is answering a different question (may this *move*), and
        the claim needs the port-scoped one (did this *open* anything real).
        """
        from softae.core.run_lock import rig_is_simulated

        mgr = _real_heater_mock_stage_manager()
        assert rig_is_simulated(mgr) is True, "motion-scoped: unchanged, and correct"
        assert session_is_simulated(mgr) is False, "port-scoped: two real sessions"

    def test_session_is_simulated_unlistable_manager_returns_false(self):
        """"I could not look" must not be spelled the same way as "only mocks"."""
        assert session_is_simulated(UnlistableManager()) is False

    def test_session_is_simulated_unreadable_driver_returns_false(self):
        mgr = create_mock_manager(config={})
        with patch.object(type(mgr), "get", side_effect=RuntimeError("driver gone")):
            assert session_is_simulated(mgr) is False

    def test_session_is_simulated_ignores_the_portless_coordinator(self):
        """``AsyncLiquidHandler`` has no ``Mock`` prefix and owns no port.

        Counting it would make every mock suite read as real, since the mock
        factory registers the same coordinator class the real one does.
        """
        mgr = create_mock_manager(config={})
        assert "liquid_handler" in mgr.names
        assert type(mgr.get("liquid_handler")).__name__ == "AsyncLiquidHandler"
        assert session_is_simulated(mgr) is True


# ── The ``what`` string ──────────────────────────────────────────────────────


class TestSessionWhatString:
    def test_desktop_session_what_omits_the_run_id_field_entirely(self):
        """``<kind>:<name>`` — no third field, and no empty third field.

        A trailing colon would assert "there is a run id and it is blank", which
        is what a campaign that failed to stamp its run produces. A GUI session
        has no run because it is not a run.
        """
        assert DESKTOP_SESSION == "gui:desktop"
        assert not DESKTOP_SESSION.endswith(":")
        assert DESKTOP_SESSION.count(":") == 1

    def test_desktop_session_what_renders_through_the_shared_owner_line(self):
        from softae.gui.widgets.rig_owner import owner_line

        lock = rl.RunLock(pid=8821, what=DESKTOP_SESSION,
                          started_at="2026-08-19T14:02:00+00:00")
        line = owner_line(lock)
        assert "gui:desktop" in line and "8821" in line and "14:02" in line

    def test_desktop_session_what_renders_through_busy_rig_message(self):
        lock = rl.RunLock(pid=8821, what=DESKTOP_SESSION, started_at="14:02")
        message = rl.busy_rig_message(lock, action="This campaign")
        assert "gui:desktop" in message and "8821" in message

    def test_desktop_session_what_is_not_mistaken_for_a_campaign(self):
        """A GUI publishes no event stream and offers no control channel."""
        from softae.gui.widgets.rig_owner import campaign_identity

        assert campaign_identity(rl.RunLock(what=DESKTOP_SESSION)) is None


# ── Claim and release ────────────────────────────────────────────────────────


class TestClaimRigSession:
    def test_claim_rig_session_simulated_rig_claims_nothing(self, rig_scope):
        """Two mock suites collide over nothing; one holding the rig is an outage."""
        assert claim_rig_session(create_mock_manager(config={})) is None
        assert rl.read_run_lock() is None

    def test_claim_rig_session_real_session_writes_the_desktop_claim(self, rig_scope):
        claim_rig_session(_real_heater_mock_stage_manager())

        lock = rl.read_run_lock()
        assert lock is not None
        assert lock.what == DESKTOP_SESSION
        assert lock.pid == os.getpid()

    def test_claim_rig_session_publishes_no_run_directory(self, rig_scope):
        """``log_path`` is a campaign's run directory. A GUI has no run to offer,
        and offering its project directory would hand over an *earlier* run's
        ``events.jsonl`` as though it were the live holder's."""
        claim_rig_session(_real_heater_mock_stage_manager())
        assert rl.read_run_lock().log_path == ""

    def test_claim_rig_session_while_another_process_holds_the_rig_raises(self, rig_scope):
        _write_foreign_lock(rig_scope)
        with pytest.raises(RunLockHeld):
            claim_rig_session(_real_heater_mock_stage_manager())

    def test_claim_rig_session_twice_in_one_process_is_reentrant(self, rig_scope):
        mgr = _real_heater_mock_stage_manager()
        first = claim_rig_session(mgr)
        second = claim_rig_session(mgr, what="gui:desktop")
        assert first is not None and second is not None
        assert second.pid == first.pid

    def test_release_rig_session_removes_this_processs_claim(self, rig_scope):
        claim_rig_session(_real_heater_mock_stage_manager())
        assert release_rig_session() is True
        assert rl.read_run_lock() is None

    def test_release_rig_session_refuses_to_free_another_processs_claim(self, rig_scope):
        """An attached window's Disconnect All must not free the campaign's lock."""
        _write_foreign_lock(rig_scope)
        assert release_rig_session() is False
        assert rl.read_run_lock() is not None

    def test_held_rig_session_releases_on_the_way_out_even_after_an_exception(
            self, rig_scope):
        with pytest.raises(ValueError):
            with held_rig_session(_real_heater_mock_stage_manager()):
                assert rl.read_run_lock() is not None
                raise ValueError("boom")
        assert rl.read_run_lock() is None

    def test_held_rig_session_nested_does_not_free_the_outer_claim(self, rig_scope):
        mgr = _real_heater_mock_stage_manager()
        with held_rig_session(mgr, what="gui:outer"):
            with held_rig_session(mgr, what="gui:inner"):
                pass
            assert rl.read_run_lock() is not None, "the inner block gave the rig away"
        assert rl.read_run_lock() is None


# ── The GUI's own acquire/release, at both seams ─────────────────────────────


def _stub_owner_session(monkeypatch):
    """Replace everything ``_begin_owner_session`` would open or pop up."""
    from softae.gui import app as app_mod
    from softae.gui.widgets import head_check_dialog

    stubs = MagicMock()
    monkeypatch.setattr(app_mod, "asyncio", stubs.asyncio)
    monkeypatch.setattr(head_check_dialog, "ask_head_state", stubs.ask_head_state)
    monkeypatch.setattr(head_check_dialog, "register_head_state",
                        stubs.register_head_state)
    return stubs


def _close_scheduled(stubs) -> None:
    """A coroutine handed to a mocked ``ensure_future`` is never awaited.

    Left open it outlives this test and is reported as a warning against whatever
    unrelated test happens to trigger the GC.
    """
    for call in stubs.asyncio.ensure_future.call_args_list:
        for arg in call.args:
            if hasattr(arg, "close"):
                arg.close()


class TestGuiOwnerSessionClaim:
    def test_begin_owner_session_with_a_real_manager_claims_the_rig(
            self, rig_scope, monkeypatch):
        """The defect, inverted: an open GUI is now visible to everyone else."""
        from softae.gui.app import _begin_owner_session

        stubs = _stub_owner_session(monkeypatch)

        assert _begin_owner_session(_real_heater_mock_stage_manager(), MagicMock())

        lock = rl.read_run_lock()
        assert lock is not None and lock.what == DESKTOP_SESSION
        _close_scheduled(stubs)

    def test_begin_owner_session_claims_before_it_prompts_for_the_head(
            self, rig_scope, monkeypatch):
        """The head prompt is modal and an operator may leave it open for minutes.

        A claim taken after it would leave exactly that long a window in which the
        GUI is about to own the rig and the lock file says it is free.
        """
        from softae.gui.app import _begin_owner_session

        stubs = _stub_owner_session(monkeypatch)
        seen: list[object] = []
        stubs.ask_head_state.side_effect = lambda *_a, **_k: seen.append(
            rl.read_run_lock())

        _begin_owner_session(_real_heater_mock_stage_manager(), MagicMock())

        assert seen and seen[0] is not None, "prompted on an unclaimed rig"
        _close_scheduled(stubs)

    def test_begin_owner_session_with_a_mock_rig_claims_nothing(
            self, rig_scope, monkeypatch):
        from softae.gui.app import _begin_owner_session

        stubs = _stub_owner_session(monkeypatch)

        assert _begin_owner_session(create_mock_manager(config={}), MagicMock())
        assert rl.read_run_lock() is None
        _close_scheduled(stubs)

    def test_begin_owner_session_losing_the_race_opens_nothing(
            self, rig_scope, monkeypatch):
        """A headless run that started between the launch decision and here."""
        from softae.gui.app import _begin_owner_session

        stubs = _stub_owner_session(monkeypatch)
        _write_foreign_lock(rig_scope)

        assert _begin_owner_session(_real_heater_mock_stage_manager(),
                                    MagicMock()) is False

        assert stubs.asyncio.ensure_future.call_count == 0, "opened ports anyway"
        assert stubs.ask_head_state.call_count == 0
        assert rl.read_run_lock().host == "some-other-machine", "stole the claim"


def _stub_run_app(monkeypatch):
    """Everything ``run_app`` builds, so only the ownership wiring stays real."""
    from softae.gui import app as app_mod
    from softae.gui.widgets import unclean_shutdown

    stubs = MagicMock()
    for name in ("QApplication", "DataStore", "MainWindow", "qasync", "asyncio",
                 "structlog"):
        monkeypatch.setattr(app_mod, name, getattr(stubs, name))
    monkeypatch.setattr(app_mod, "create_manager", stubs.create_manager)
    monkeypatch.setattr(unclean_shutdown, "check_unclean_shutdown",
                        stubs.check_unclean_shutdown)
    monkeypatch.setattr(app_mod, "_begin_owner_session", stubs.begin_owner_session)
    return stubs


class TestGuiAttachedModeClaimsNothing:
    def test_run_app_in_attached_mode_takes_no_claim_of_its_own(
            self, rig_scope, monkeypatch):
        """Claiming in attached mode would violate the invariant attach protects:
        a process may command only the sessions it opened, and this one opened
        none."""
        from softae.gui.app import run_app

        _write_foreign_lock(rig_scope)
        stubs = _stub_run_app(monkeypatch)

        run_app(mock=None)

        assert stubs.begin_owner_session.call_count == 0
        lock = rl.read_run_lock()
        assert lock is not None and lock.host == "some-other-machine"

    def test_run_app_in_attached_mode_does_not_release_the_holders_claim_on_exit(
            self, rig_scope, monkeypatch):
        """Closing an attached window must not free the campaign it was watching."""
        from softae.gui.app import run_app

        _write_foreign_lock(rig_scope)
        _stub_run_app(monkeypatch)

        run_app(mock=None)

        assert rl.read_run_lock() is not None

    def test_run_app_releases_a_claim_taken_after_launch_out_of_attached_mode(
            self, rig_scope, monkeypatch):
        """Init tab → Connect All is the documented way out of attached mode.

        A release branched on the *launch* decision would leave that claim behind
        on exit — held by a PID that no longer exists, and cleared only when the
        next reader happens to notice it is stale. The release is therefore keyed
        on what this process actually holds.
        """
        from softae.gui.app import run_app

        _write_foreign_lock(rig_scope)
        stubs = _stub_run_app(monkeypatch)

        def _operator_takes_the_rig_while_the_window_is_up():
            rl.break_run_lock()          # the other run finished
            claim_rig_session(_real_heater_mock_stage_manager())
            return 0

        stubs.qasync.QEventLoop.return_value.run_forever.side_effect = (
            _operator_takes_the_rig_while_the_window_is_up)

        run_app(mock=None)

        assert rl.read_run_lock() is None, "the window kept the rig after closing"


# ── The hand-off buttons ─────────────────────────────────────────────────────


@pytest.fixture
def real_tab(qapp, rig_scope):
    from softae.gui.tabs.tab_init import InitCalibrationTab

    widget = InitCalibrationTab(_real_heater_mock_stage_manager())
    yield widget
    widget.cleanup()
    widget.close()


class TestInitTabHandOff:
    def test_connect_all_claims_the_rig_before_opening_ports(self, real_tab):
        with patch.object(real_tab, "_schedule_async",
                          side_effect=lambda coro: coro.close()) as sched:
            real_tab._on_connect_all()
        assert sched.call_count == 1
        lock = rl.read_run_lock()
        assert lock is not None and lock.what == DESKTOP_SESSION

    def test_connect_all_while_another_process_holds_the_rig_claims_nothing(
            self, real_tab, rig_scope):
        _write_foreign_lock(rig_scope)
        with patch("softae.gui.tabs.tab_init.QMessageBox.warning") as warn, \
             patch.object(real_tab, "_schedule_async") as sched:
            real_tab._on_connect_all()
        assert sched.call_count == 0 and warn.call_count == 1
        assert rl.read_run_lock().host == "some-other-machine"

    def test_connect_all_losing_the_race_after_the_guard_still_opens_nothing(
            self, real_tab, rig_scope):
        """The guard is check-then-act; the acquire is the atomic form.

        ``_refuse_if_rig_held`` sees a free rig, and the lock appears before the
        claim — the window a check alone cannot close.
        """
        def _steal(_action):
            _write_foreign_lock(rig_scope)
            return False

        with patch.object(real_tab, "_refuse_if_rig_held", side_effect=_steal), \
             patch("softae.gui.tabs.tab_init.QMessageBox.warning") as warn, \
             patch.object(real_tab, "_schedule_async") as sched:
            real_tab._on_connect_all()

        assert sched.call_count == 0, "opened ports on a rig taken mid-check"
        assert warn.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_all_releases_the_claim_after_the_ports_close(
            self, real_tab):
        claim_rig_session(real_tab._manager)
        order: list[str] = []
        scheduled: list[object] = []

        async def _closing():
            order.append("closed")

        with patch.object(real_tab, "_schedule_async", side_effect=scheduled.append), \
             patch.object(real_tab._manager, "disconnect_all", side_effect=_closing):
            real_tab._on_disconnect_all()
            assert rl.read_run_lock() is not None, "released before the ports closed"
            await scheduled[0]

        assert order == ["closed"]
        assert rl.read_run_lock() is None

    @pytest.mark.asyncio
    async def test_disconnect_all_releases_even_when_a_port_fails_to_close(
            self, real_tab):
        claim_rig_session(real_tab._manager)
        scheduled: list[object] = []

        async def _failing():
            raise OSError("port wedged")

        with patch.object(real_tab, "_schedule_async", side_effect=scheduled.append), \
             patch.object(real_tab._manager, "disconnect_all", side_effect=_failing):
            real_tab._on_disconnect_all()
            with pytest.raises(OSError):
                await scheduled[0]

        assert rl.read_run_lock() is None

    def test_disconnect_selected_keeps_the_claim_because_other_ports_stay_open(
            self, real_tab):
        """The claim ends when the *session* does, not when one port of it does."""
        claim_rig_session(real_tab._manager)
        with patch.object(real_tab, "_selected_instrument", return_value="stage"), \
             patch.object(real_tab, "_schedule_async",
                          side_effect=lambda coro: coro.close()) as sched:
            real_tab._on_disconnect_selected()
        assert sched.call_count == 1
        assert rl.read_run_lock() is not None


# ── A workflow started from a connected GUI ──────────────────────────────────


class TestWorkflowFromAConnectedGui:
    """The thing most likely to break, and the reason re-entrancy is verified.

    ``WorkflowExecutor.run`` acquires the rig lock itself. If the GUI already
    holds it, that acquire must hand back the existing claim rather than raise,
    and the executor's teardown must not release a claim it did not make.

    Both predicates are forced to "real": a mock suite cannot be made to open a
    real port, and the two questions being made to agree here are exactly the two
    that would otherwise leave the executor exempt while the GUI claimed.
    """

    @pytest.fixture
    def real_enough(self, rig_scope, monkeypatch):
        monkeypatch.setattr(rl, "rig_is_simulated", lambda _m: False)
        monkeypatch.setattr(rs, "session_is_simulated", lambda _m: False)

    @pytest.mark.asyncio
    async def test_a_workflow_started_from_a_connected_gui_does_not_raise(
            self, real_enough):
        from softae.workflows.workflow_executor import WorkflowExecutor
        from softae.workflows.workflow_model import Workflow, WorkflowStep

        mgr = create_mock_manager(config={})
        await mgr.connect_all()
        claim_rig_session(mgr)
        try:
            await WorkflowExecutor(mgr).run(Workflow(
                name="ht_run",
                setup=[WorkflowStep(name="pos", instrument="stage",
                                    method="live_position", params={})]))
        finally:
            await mgr.disconnect_all()

        lock = rl.read_run_lock()
        assert lock is not None, "the workflow freed the rig the GUI was holding"
        assert lock.what == DESKTOP_SESSION, "the GUI's own claim was replaced"


# The inverse — a workflow with no outer claim taking and returning the rig — is
# already pinned by ``test_campaign_rig_claim.py`` and is not repeated here.


# ── Across a real process boundary ───────────────────────────────────────────


_ASK = (
    "import sys, json;"
    "from softae.core.run_lock import foreign_run_lock;"
    "l = foreign_run_lock(sys.argv[1]);"
    "print(json.dumps(l.to_dict() if l else None))"
)


def _foreign_view_of(scope: Path):
    """What a *second process* sees: the holder, or ``None`` if the rig is free.

    This is the predicate ``softae-campaign run`` gates on
    (``tools/campaign._cmd_run``), asked from where it is actually asked. Bounded
    by a timeout, and ``subprocess.run`` reaps this child — one we launched — by
    handle.
    """
    done = subprocess.run([sys.executable, "-c", _ASK, str(scope)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip())


class TestForeignToolAcrossProcesses:
    def test_a_connected_gui_session_is_visible_to_a_foreign_tool(self, rig_scope):
        """The bug, in one assertion: this used to come back ``None``."""
        claim_rig_session(_real_heater_mock_stage_manager())

        holder = _foreign_view_of(rig_scope)

        assert holder is not None, "a headless run would have connected on top"
        assert holder["what"] == DESKTOP_SESSION
        assert holder["pid"] == os.getpid()

    def test_releasing_the_gui_session_frees_the_rig_for_a_foreign_tool(
            self, rig_scope):
        """And the refusal must not outlive the session, or the GUI is a hostage."""
        claim_rig_session(_real_heater_mock_stage_manager())
        assert _foreign_view_of(rig_scope) is not None

        release_rig_session()

        assert _foreign_view_of(rig_scope) is None


# ── The factory's required keyword ───────────────────────────────────────────


class TestCreateManagerRequiresMock:
    def test_create_manager_without_mock_raises(self):
        """Auto-detect was the silent default, and it is the one mode that builds
        a partly-real manager — the shape a motion-scoped check calls simulated."""
        from softae.drivers.factory import create_manager

        with pytest.raises(TypeError):
            create_manager()  # type: ignore[call-arg]

    # That an explicit ``mock=None`` still auto-detects is already pinned by
    # ``test_real_drivers.TestDriverFactory.test_auto_falls_back_to_mocks``.
