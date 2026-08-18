"""One owner of the rig at a time, across processes.

Every prior lock in this codebase is in-process — `asyncio.Lock` per instrument, `QMutex`
in the GUI workers — and none of them sees a second *process*. The calibration launcher
starts one deliberately, so this is the guard that keeps the GUI and a headless sweep off
the same COM ports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from softae.core.run_lock import (
    DEFAULT_SCOPE,
    RunLock,
    RunLockHeld,
    acquire_run_lock,
    break_run_lock,
    held_run_lock,
    lock_path,
    read_run_lock,
    release_run_lock,
)


def _write_foreign(scope, *, what="another run", pid=1):
    """A lock owned by a live process that is not us.

    Another *host* rather than another PID: a foreign host always reads as alive by
    design, so the fixture does not depend on finding a real PID that happens to be
    running and inspectable.
    """
    lock_path(scope).parent.mkdir(parents=True, exist_ok=True)
    lock_path(scope).write_text(json.dumps(
        {"pid": pid, "what": what, "started_at": "2026-01-01T00:00:00+00:00",
         "host": "some-other-machine", "log_path": ""}), encoding="utf-8")


class TestReentrancy:
    """Acquiring twice in one process must not deadlock against itself.

    This is the most likely way a *new* caller meets this module: a workflow takes the
    lock, then calls a helper that also takes it. Refusing there would be a deadlock
    with no second party, and the error message would name the victim as the culprit.
    """

    def test_the_same_process_re_acquiring_is_idempotent(self, tmp_path):
        first = acquire_run_lock(tmp_path, "outer")
        second = acquire_run_lock(tmp_path, "inner helper")
        assert second.pid == first.pid
        assert second.what == "outer"          # the original claim is preserved

    def test_a_nested_block_does_not_free_the_outer_ones_lock(self, tmp_path):
        # The failure this prevents: an inner `with` exiting hands the rig away while
        # the outer block is still using it.
        with held_run_lock(tmp_path, "outer"):
            with held_run_lock(tmp_path, "inner"):
                pass
            assert read_run_lock(tmp_path) is not None, "inner block freed the rig"
        assert read_run_lock(tmp_path) is None

    def test_the_outermost_block_still_releases(self, tmp_path):
        with held_run_lock(tmp_path, "outer"):
            pass
        assert read_run_lock(tmp_path) is None


class TestScopeIsTheMachineNotTheProject:
    """The rig is physical and per-machine; a project directory is bookkeeping.

    Keying on the project (the first version of this module) meant two real runs
    started with different --project values took different locks and drove the same
    COM ports — with a lock file present, claiming all was well.
    """

    def test_the_default_scope_is_not_a_project_directory(self):
        assert DEFAULT_SCOPE == Path.home() / ".softae"

    def test_two_project_dirs_share_one_default_lock(self):
        # Not parameterised on any project argument at all: there is nothing to differ.
        assert lock_path() == lock_path()
        # Stated against the scope itself rather than by looking for "softae" in the
        # string: the claim is "the default path is the scope's lock and nothing
        # else", which holds wherever the scope points. That the scope *is*
        # ``~/.softae`` is the sibling test's job, and it reads the real constant.
        from softae.core import run_lock

        assert lock_path() == run_lock.DEFAULT_SCOPE / "rig.lock"


class _EmptyManager:
    """A manager with nothing registered."""

    _instruments: dict = {}
    names: list = []


class TestSimulatedRigsDoNotLock:
    def test_an_all_mock_manager_is_simulated(self):
        from softae.core.run_lock import rig_is_simulated
        from softae.drivers.mock_factory import create_mock_manager

        assert rig_is_simulated(create_mock_manager()) is True

    def test_it_agrees_with_the_arming_interlock_on_what_is_real(self):
        """One definition of "real hardware", not two.

        `assert_hardware_armed` already decides this for motion, and a second private
        notion here would be free to disagree with the one that governs whether the rig
        may move at all. So the lock exemption is exactly "no real motion instruments" —
        including for an empty manager, which the arming interlock also treats as inert.
        """
        from softae.core.hardware_safety import real_motion_instruments
        from softae.core.run_lock import rig_is_simulated
        from softae.drivers.mock_factory import create_mock_manager

        for manager in (create_mock_manager(), _EmptyManager()):
            assert rig_is_simulated(manager) == (not real_motion_instruments(manager))

    def test_an_unreadable_manager_is_assumed_real(self):
        # Skipping the lock is the unsafe direction, so an error must not buy the
        # exemption.
        from softae.core.run_lock import rig_is_simulated

        class _Explodes:
            @property
            def names(self):
                raise RuntimeError("driver layer is broken")

        assert rig_is_simulated(_Explodes()) is False


class TestBasicOwnership:
    def test_acquiring_then_reading_finds_the_owner(self, tmp_path):
        acquire_run_lock(tmp_path, "commissioning blank_short")
        lock = read_run_lock(tmp_path)
        assert lock is not None
        assert lock.pid == os.getpid()
        assert "blank_short" in lock.what

    def test_another_process_is_refused_with_the_owner_named(self, tmp_path):
        # A foreign owner is simulated by another host, which always reads as alive:
        # this process cannot see a PID it cannot reach.
        _write_foreign(tmp_path, what="geometry series geo-1")
        with pytest.raises(RunLockHeld) as exc:
            acquire_run_lock(tmp_path, "something else")
        # The message has to say who, or the operator's only recourse is to delete files.
        assert "geometry series geo-1" in str(exc.value)

    def test_releasing_frees_it(self, tmp_path):
        acquire_run_lock(tmp_path, "x")
        assert release_run_lock(tmp_path) is True
        assert read_run_lock(tmp_path) is None
        acquire_run_lock(tmp_path, "y")          # now available

    def test_releasing_nothing_is_not_an_error(self, tmp_path):
        assert release_run_lock(tmp_path) is False

    def test_the_context_manager_releases_on_exception(self, tmp_path):
        with pytest.raises(ValueError):
            with held_run_lock(tmp_path, "will fail"):
                raise ValueError("boom")
        assert read_run_lock(tmp_path) is None


class TestStalenessIsByLivenessNotClock:
    """A crashed run must not lock the rig until someone deletes a file.

    And no elapsed-time rule can help: a 40-second EIS sweep and a 14-hour anneal are
    both normal, so "old" carries no information about "dead".
    """

    def _write(self, tmp_path, **over):
        data = {"pid": 999_999_999, "what": "crashed run",
                "started_at": "2020-01-01T00:00:00+00:00",
                "host": os.environ.get("COMPUTERNAME") or "", "log_path": ""}
        data.update(over)
        import socket
        data.setdefault("host", socket.gethostname())
        data["host"] = data["host"] or socket.gethostname()
        lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        lock_path(tmp_path).write_text(json.dumps(data), encoding="utf-8")

    def test_a_dead_owner_reads_as_no_lock(self, tmp_path):
        self._write(tmp_path)
        assert read_run_lock(tmp_path) is None

    def test_reading_clears_the_stale_file_so_the_rig_is_usable_again(self, tmp_path):
        self._write(tmp_path)
        read_run_lock(tmp_path)
        assert not lock_path(tmp_path).exists()

    def test_a_stale_lock_does_not_block_a_new_acquire(self, tmp_path):
        self._write(tmp_path)
        lock = acquire_run_lock(tmp_path, "fresh run")
        assert lock.pid == os.getpid()

    def test_a_very_old_but_LIVE_lock_still_holds(self, tmp_path):
        # The point of liveness-over-clock: an overnight anneal is old and legitimate.
        self._write(tmp_path, pid=1, host="some-other-machine",
                    started_at="2020-01-01T00:00:00+00:00")
        assert read_run_lock(tmp_path) is not None
        with pytest.raises(RunLockHeld):
            acquire_run_lock(tmp_path, "impatient second run")

    def test_a_lock_from_another_host_is_never_declared_dead(self, tmp_path):
        # This process cannot see a PID on another machine, and guessing "dead" would
        # hand one rig to two hosts.
        self._write(tmp_path, pid=1, host="some-other-machine")
        lock = read_run_lock(tmp_path)
        assert lock is not None and lock.is_alive

    def test_an_unparseable_lock_is_not_evidence_of_an_owner(self, tmp_path):
        lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        lock_path(tmp_path).write_text("{not json", encoding="utf-8")
        assert read_run_lock(tmp_path) is None
        assert not lock_path(tmp_path).exists()


class TestReleaseIsOwnershipChecked:
    def test_it_refuses_to_release_another_processes_lock(self, tmp_path):
        # A run that crashed and restarted must not delete the lock of the run that
        # replaced it.
        import socket

        lock_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        lock_path(tmp_path).write_text(json.dumps(
            {"pid": os.getpid() + 1, "what": "someone else", "started_at": "",
             "host": socket.gethostname(), "log_path": ""}), encoding="utf-8")
        assert release_run_lock(tmp_path) is False
        assert lock_path(tmp_path).exists()

    def test_break_takes_it_deliberately_and_says_what_it_took(self, tmp_path):
        acquire_run_lock(tmp_path, "commissioning reference_cap")
        broken = break_run_lock(tmp_path)
        assert broken is not None
        assert "reference_cap" in broken.what
        assert read_run_lock(tmp_path) is None

    def test_breaking_nothing_returns_none(self, tmp_path):
        assert break_run_lock(tmp_path) is None


class TestNoTOCTOUWindow:
    def test_an_existing_file_appearing_mid_acquire_is_still_refused(
            self, tmp_path, monkeypatch):
        """Exclusive create, not check-then-write.

        The whole purpose of this file is the case where two things start at once, so a
        window between "is it free?" and "claim it" would be a bug in the one scenario
        it exists for.
        """
        import socket

        import softae.core.run_lock as rl

        real_read = rl.read_run_lock

        def racing_read(project_dir):
            result = real_read(project_dir)
            # Simulate another process winning the race after our check.
            p = rl.lock_path(project_dir)
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(
                    {"pid": os.getpid(), "what": "the other launcher",
                     "started_at": "", "host": socket.gethostname(),
                     "log_path": ""}), encoding="utf-8")
            return result

        monkeypatch.setattr(rl, "read_run_lock", racing_read)
        with pytest.raises(RunLockHeld):
            rl.acquire_run_lock(tmp_path, "loser")


class TestDescribe:
    def test_it_names_the_run_the_pid_and_the_time(self, tmp_path):
        acquire_run_lock(tmp_path, "geometry series geo-7", log_path="C:/logs/x.log")
        text = read_run_lock(tmp_path).describe()
        assert "geo-7" in text
        assert str(os.getpid()) in text
        assert "x.log" in text

    def test_an_unnamed_run_still_describes_cleanly(self):
        assert "unnamed run" in RunLock(pid=5).describe()
