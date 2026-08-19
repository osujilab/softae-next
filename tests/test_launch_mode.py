"""Owner or attached — the one decision that stops the GUI opening a busy rig.

Two halves, and the split is the point. :func:`decide_launch_mode` is pure, so it
is tested directly and exhaustively; ``gui.app.run_app`` constructs a
``QApplication`` and has never carried a test, so it is tested for exactly one
thing — that a foreign lock reaches the three start-up acts and stops all three.

The conservative direction is asserted more than once on purpose. "I could not
read the lock" and "there is no lock" must not be the same answer: one of them
opens ports on a rig a campaign is driving.
"""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from softae.core import run_lock as run_lock_mod
from softae.core.run_lock import RunLock, lock_path
from softae.gui.launch_mode import LaunchMode, decide_launch_mode


# ── Fixtures and fakes ───────────────────────────────────────────────────────


def _foreign(**kw) -> RunLock:
    """A lock held by a live process that is not this one.

    Another *host* rather than another PID: a foreign host reads as alive by
    design (``run_lock.RunLock.is_alive``), so the fixture never depends on
    finding a real running PID.
    """
    defaults = dict(pid=os.getpid() + 1, what="campaign:phase_map:run_042",
                    started_at="2026-08-19T14:02:00+00:00",
                    host="some-other-machine", log_path=r"C:\proj\runs\run_042")
    defaults.update(kw)
    return RunLock(**defaults)


def _reader(value):
    return lambda: value


def _raising_reader(exc: Exception):
    def _read():
        raise exc
    return _read


@pytest.fixture
def rig_scope(tmp_path, monkeypatch):
    """A private lock scope, so the real chain can be exercised end to end.

    The session-wide ``rig_lock_scope`` fixture already keeps the suite away from
    the operator's ``~/.softae/rig.lock``; this narrows it per test so a lock
    written here cannot leak into another one.
    """
    monkeypatch.setattr(run_lock_mod, "DEFAULT_SCOPE", tmp_path)
    return tmp_path


def _write_lock(scope, **fields) -> None:
    data = dict(pid=os.getpid() + 1, what="bench sequence",
                started_at="2026-08-19T14:02:00+00:00",
                host="some-other-machine", log_path="")
    data.update(fields)
    path = lock_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ── The decision ─────────────────────────────────────────────────────────────


class TestDecideLaunchMode:
    def test_decide_launch_mode_no_lock_returns_owner(self):
        mode = decide_launch_mode(lock_reader=_reader(None))
        assert mode.attached is False
        assert mode.owner is True
        assert mode.campaign is None and mode.run_dir is None and mode.holder is None

    def test_decide_launch_mode_foreign_campaign_lock_returns_attached_with_identity(self):
        lock = _foreign()
        mode = decide_launch_mode(lock_reader=_reader(lock))
        assert mode.attached is True
        assert mode.campaign == ("phase_map", "run_042")
        assert mode.run_dir == r"C:\proj\runs\run_042"
        assert mode.holder is lock
        assert mode.attachable is True

    def test_decide_launch_mode_non_campaign_lock_returns_attached_with_no_campaign(self):
        """A bench sequence owns the rig just as hard as a campaign does.

        There is simply nothing to tail or ask: no event stream, no control
        channel. So the gating is identical and only the rendering differs —
        occupied, but not attachable.
        """
        mode = decide_launch_mode(lock_reader=_reader(_foreign(what="workflow 'ht_run'")))
        assert mode.attached is True, "a foreign owner is a foreign owner"
        assert mode.campaign is None
        assert mode.run_dir is None
        assert mode.attachable is False
        assert mode.holder is not None, "the holder is still rendered"

    def test_decide_launch_mode_campaign_without_a_run_dir_returns_attached_with_no_campaign(self):
        """Mirrors ``tools/campaign._running_campaign_run_dir``'s second refusal.

        A campaign that published no run directory cannot be followed and cannot
        be sent a control request, so offering either would be a lie.
        """
        mode = decide_launch_mode(lock_reader=_reader(_foreign(log_path="")))
        assert mode.attached is True
        assert mode.campaign is None
        assert mode.attachable is False
        assert "phase_map" in mode.reason, "the operator is still told who has the rig"

    def test_decide_launch_mode_unreadable_lock_returns_attached_not_free(self):
        """The rule from ``unclean_shutdown`` one level up.

        Deferring costs a launch; guessing wrong costs the run. This is the whole
        reason the raising reader is used rather than the never-raises wrapper —
        a wrapper answering ``None`` would land this case in owner mode.
        """
        mode = decide_launch_mode(lock_reader=_raising_reader(OSError("network share gone")))
        assert mode.attached is True
        assert mode.campaign is None
        assert mode.holder is None
        assert "could not be read" in mode.reason

    def test_decide_launch_mode_identifier_that_raises_returns_attached_not_free(self):
        """The conservative direction cannot be undone by the second collaborator."""
        def _boom(_lock):
            raise RuntimeError("malformed what field")

        mode = decide_launch_mode(lock_reader=_reader(_foreign()), identify=_boom)
        assert mode.attached is True
        assert mode.campaign is None

    def test_decide_launch_mode_reason_names_the_holder_rather_than_saying_busy(self):
        """An anonymous refusal leaves the operator deleting files to find out why."""
        reason = decide_launch_mode(lock_reader=_reader(_foreign(what="bench sequence"))).reason
        assert "bench sequence" in reason
        assert str(os.getpid() + 1) in reason

    def test_decide_launch_mode_never_raises_on_a_lock_object_it_cannot_read(self):
        mode = decide_launch_mode(lock_reader=_reader(object()))
        assert mode.attached is True


class TestLaunchModeShape:
    def test_launch_mode_owner_is_the_inverse_of_attached(self):
        assert LaunchMode(True, None, None, None, "").owner is False
        assert LaunchMode(False, None, None, None, "").owner is True

    def test_launch_mode_is_frozen_so_the_decision_cannot_drift_after_launch(self):
        """Ownership changes by an operator act, not by an assignment."""
        mode = decide_launch_mode(lock_reader=_reader(None))
        with pytest.raises(FrozenInstanceError):
            mode.attached = True  # type: ignore[misc]


# ── The real chain: an actual lock file, the shipped predicate ───────────────


class TestAgainstTheRealRunLock:
    def test_decide_launch_mode_with_a_real_foreign_lock_file_returns_attached(self, rig_scope):
        _write_lock(rig_scope, what="campaign:phase_map:run_007",
                    log_path=str(rig_scope / "runs" / "run_007"))
        mode = decide_launch_mode()
        assert mode.attached is True
        assert mode.campaign == ("phase_map", "run_007")

    def test_decide_launch_mode_with_my_own_lock_returns_owner(self, rig_scope):
        """Re-entrancy: a GUI holding its own claim must not lock itself out.

        ``foreign_run_lock`` already excludes this process's own lock, and that
        exclusion is deliberately not reimplemented here — this test pins that
        the shared predicate is the one being used.
        """
        import socket

        _write_lock(rig_scope, pid=os.getpid(), host=socket.gethostname(),
                    what="campaign:mine:run_001", log_path=str(rig_scope))
        mode = decide_launch_mode()
        assert mode.attached is False

    def test_decide_launch_mode_with_no_lock_file_returns_owner(self, rig_scope):
        assert decide_launch_mode().attached is False


# ── The three start-up acts, gated ───────────────────────────────────────────


def _stub_run_app(monkeypatch):
    """Replace everything ``run_app`` builds, leaving only the launch decision real.

    Every stub stands in for something that would open a window, a database or a
    serial port. The launch-mode decision itself is *not* stubbed: this test
    exists to prove the lock reaches the three acts, so a patched decision would
    assert nothing.
    """
    from softae.core import hardware_safety
    from softae.gui import app as app_mod
    from softae.gui.widgets import head_check_dialog, unclean_shutdown

    stubs = MagicMock()
    for name in ("QApplication", "DataStore", "MainWindow", "qasync",
                 "asyncio", "structlog"):
        monkeypatch.setattr(app_mod, name, getattr(stubs, name))
    monkeypatch.setattr(app_mod, "create_manager", stubs.create_manager)

    # A rig with real motion present — so an unarmed result is the gate working,
    # not the absence of hardware to arm.
    monkeypatch.setattr(hardware_safety, "real_motion_instruments",
                        stubs.real_motion_instruments)
    stubs.real_motion_instruments.return_value = True
    monkeypatch.setattr(hardware_safety, "arm_hardware", stubs.arm_hardware)
    monkeypatch.setattr(head_check_dialog, "ask_head_state", stubs.ask_head_state)
    monkeypatch.setattr(head_check_dialog, "register_head_state",
                        stubs.register_head_state)
    monkeypatch.setattr(unclean_shutdown, "check_unclean_shutdown",
                        stubs.check_unclean_shutdown)
    return stubs


def _close_scheduled_coroutines(stubs) -> None:
    """A coroutine handed to a mocked ``ensure_future`` is never awaited."""
    for call in stubs.asyncio.ensure_future.call_args_list:
        for arg in call.args:
            if hasattr(arg, "close"):
                arg.close()


class TestRunAppStartup:
    def test_run_app_with_a_foreign_campaign_lock_does_not_connect_arm_or_prompt(
            self, rig_scope, monkeypatch):
        """The hazard this step exists to close, all three parts of it.

        Connecting opens ports the campaign is driving; arming licenses a process
        that holds no sessions; and the head prompt asks the operator to eyeball
        a head the campaign may be flipping while they walk over to look.
        """
        from softae.gui.app import run_app

        _write_lock(rig_scope, what="campaign:phase_map:run_042",
                    log_path=str(rig_scope / "run_042"))
        stubs = _stub_run_app(monkeypatch)

        run_app()

        assert stubs.asyncio.ensure_future.call_count == 0, \
            "must not open ports the campaign is driving"
        assert stubs.arm_hardware.call_count == 0, \
            "a process with no sessions must not hold the interlock"
        assert stubs.ask_head_state.call_count == 0, \
            "the answer would be stale before the operator finished giving it"
        assert stubs.register_head_state.call_count == 0

    def test_run_app_with_a_non_campaign_foreign_lock_does_not_connect_arm_or_prompt(
            self, rig_scope, monkeypatch):
        """Nothing to attach to is not the same as nothing to collide with."""
        from softae.gui.app import run_app

        _write_lock(rig_scope, what="workflow 'ht_run'")
        stubs = _stub_run_app(monkeypatch)

        run_app()

        assert stubs.asyncio.ensure_future.call_count == 0
        assert stubs.arm_hardware.call_count == 0
        assert stubs.ask_head_state.call_count == 0

    def test_run_app_with_an_unreadable_lock_does_not_connect(
            self, rig_scope, monkeypatch):
        """The conservative branch, end to end rather than only in the unit."""
        from softae.gui import app as app_mod
        from softae.gui.app import run_app

        stubs = _stub_run_app(monkeypatch)
        undecidable = decide_launch_mode  # bound before the patch, not through it
        monkeypatch.setattr(
            app_mod, "decide_launch_mode",
            lambda: undecidable(lock_reader=_raising_reader(OSError("share gone"))))

        run_app()

        assert stubs.asyncio.ensure_future.call_count == 0
        assert stubs.arm_hardware.call_count == 0
        assert stubs.ask_head_state.call_count == 0

    def test_run_app_with_no_lock_connects_arms_and_prompts_for_the_head(
            self, rig_scope, monkeypatch):
        """The inverse, so the change cannot be written as "never start"."""
        from softae.gui.app import run_app

        stubs = _stub_run_app(monkeypatch)

        run_app()

        assert stubs.arm_hardware.call_count == 1
        assert stubs.ask_head_state.call_count == 1
        assert stubs.register_head_state.call_count == 1
        assert stubs.asyncio.ensure_future.call_count == 1
        _close_scheduled_coroutines(stubs)

    def test_run_app_passes_the_decision_into_the_window_constructor(
            self, rig_scope, monkeypatch):
        """One decision, made once — and handed over before the window is built.

        It was briefly an attribute set afterwards. It cannot be: the window is
        *constructed* differently by it (no exit park, no idle-purge timer, no
        Safe Exit in attached mode), and none of that can be undone by a later
        assignment. ``MainWindow.launch_mode`` is read-only for the same reason,
        so the assignment this replaced would now raise at launch.
        """
        from softae.gui.app import run_app

        _write_lock(rig_scope, what="campaign:phase_map:run_042",
                    log_path=str(rig_scope / "run_042"))
        stubs = _stub_run_app(monkeypatch)

        run_app()

        mode = stubs.MainWindow.call_args.kwargs["launch_mode"]
        assert mode.attached is True
        assert mode.campaign == ("phase_map", "run_042")
