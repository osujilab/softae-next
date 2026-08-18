"""One owner of the rig at a time — a lock that survives across processes.

Every lock in this codebase before now was **in-process**: ``asyncio.Lock`` per
instrument in :mod:`softae.server.manager`, ``QMutex`` in the GUI workers. Those keep
two coroutines or two Qt threads from colliding. They do nothing at all about a second
*process*, and the calibration launcher exists precisely to start one — a headless
commissioning sweep or geometry-series cast that outlives the dialog that launched it.

Without this, an operator pressing **Connect All** twenty minutes into a headless sweep
gets a rig with two owners. On Windows a COM port is usually exclusive so the second
opener often just fails, but the MCP2221 HID bus and VISA paths are less reliably so,
and interleaved commands to a potentiostat while the head is down and the pumps are
armed is not a failure worth discovering empirically.

**Staleness is by process liveness, not by clock.** Same discipline as
``hardware_hash`` for calibrations, and for the same reason: a crashed run must not lock
the rig until someone deletes a file, and no elapsed-time rule can tell a crashed run
from a slow one. A 40-second EIS sweep and a 14-hour anneal are both normal.

.. warning::
   **PID reuse is a real limit.** If the owning process dies and the operating system
   later hands its number to something unrelated, the lock reads as live. Nothing here
   can rule that out, so the lock records *what* it is running and *when* it started and
   :func:`describe` surfaces both — a human can tell "commissioning blank_short, started
   14:02" from a stale number in a way a check cannot. Taking over is therefore an
   explicit, confirmed act (:func:`break_run_lock`), never automatic.
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import structlog

logger = structlog.get_logger(__name__)

#: Lock file name.
LOCK_FILENAME = "rig.lock"

#: Default scope: **one machine, one rig.**
#:
#: An earlier version keyed this on the *project directory*, which is wrong and
#: dangerously so. The rig is a physical object attached to this computer; the project
#: directory is a bookkeeping choice that varies per run — ``softae-commission --mock``
#: already redirects it, and any ``--project`` argument changes it. Two real runs
#: started with different project directories would take two different locks and drive
#: the same COM ports, which is exactly the failure this module exists to prevent, with
#: the added insult of a lock file that says everything is fine.
#:
#: Config path was considered and rejected as a key: a config edit mid-run would change
#: the key and orphan a live lock. A fixed per-user path cannot drift while a run is in
#: flight.
DEFAULT_SCOPE = Path.home() / ".softae"

#: Windows ``GetExitCodeProcess`` sentinel for a process that has not exited.
_STILL_ACTIVE = 259


class RunLockHeld(RuntimeError):
    """Another process owns the rig. Carries the owner so the message can say who."""

    def __init__(self, lock: "RunLock") -> None:
        self.lock = lock
        super().__init__(lock.describe())


def _pid_alive(pid: int) -> bool:
    """Whether *pid* is a live process, on Windows as well as POSIX.

    ``os.kill(pid, 0)`` is the POSIX idiom and is unreliable on Windows, so the NT path
    asks the kernel directly. A process that genuinely exited with code 259 would read
    as alive; that is accepted rather than worked around, because the alternative
    (treating an ambiguous answer as dead) releases a lock that may be real.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                return bool(ok) and code.value == _STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            # An unreadable answer must not release someone else's lock.
            logger.warning("run_lock_pid_check_failed", pid=pid, exc_info=True)
            return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by another user.
        return True
    except OSError:
        return True
    return True


@dataclass(frozen=True)
class RunLock:
    """Who owns the rig, what they are doing, and since when."""

    pid: int = 0
    what: str = ""
    started_at: str = ""
    host: str = ""
    log_path: str = ""

    @property
    def is_alive(self) -> bool:
        """Whether the owning process still exists.

        A lock from **another host** always reads as alive: this process cannot check a
        PID it cannot see, and guessing "dead" would hand the rig to two machines.
        """
        if self.host and self.host != socket.gethostname():
            return True
        return _pid_alive(self.pid)

    def is_mine(self) -> bool:
        return self.pid == os.getpid() and (
            not self.host or self.host == socket.gethostname())

    def describe(self) -> str:
        when = self.started_at or "unknown time"
        where = f" on {self.host}" if self.host and self.host != socket.gethostname() else ""
        tail = f"\n  log: {self.log_path}" if self.log_path else ""
        return (
            f"the rig is held by PID {self.pid}{where} — {self.what or 'unnamed run'}, "
            f"started {when}.{tail}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "what": self.what, "started_at": self.started_at,
                "host": self.host, "log_path": self.log_path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunLock":
        try:
            pid = int(data.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        return cls(
            pid=pid, what=str(data.get("what", "")),
            started_at=str(data.get("started_at", "")),
            host=str(data.get("host", "")),
            log_path=str(data.get("log_path", "")),
        )


def lock_path(scope: str | Path | None = None) -> Path:
    """Where the lock lives. ``None`` means :data:`DEFAULT_SCOPE` — this machine.

    Pass an explicit *scope* only for a test, or for the genuinely unusual case of two
    independent rigs on one computer. Passing a project directory is what the first
    version of this module did and is a bug: see :data:`DEFAULT_SCOPE`.
    """
    return Path(scope if scope is not None else DEFAULT_SCOPE).expanduser() / LOCK_FILENAME


def read_run_lock(scope: str | Path | None = None) -> RunLock | None:
    """The live lock, or ``None``. **Clears a stale one as a side effect.**

    Clearing here rather than in a separate sweep means every caller that asks the
    question also repairs the answer, so a crashed run cannot leave the rig unusable
    until someone thinks to look.
    """
    path = lock_path(scope)
    if not path.exists():
        return None
    try:
        lock = RunLock.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        # An unparseable lock is not evidence of an owner. Remove it, loudly.
        logger.warning("run_lock_unreadable", path=str(path), exc_info=True)
        try:
            path.unlink()
        except OSError:
            pass
        return None

    if lock.is_alive:
        return lock

    logger.info("run_lock_stale_cleared", path=str(path), pid=lock.pid,
                what=lock.what, started_at=lock.started_at,
                msg="owning process is gone — the lock was left by a crashed run")
    try:
        path.unlink()
    except OSError:
        pass
    return None


def acquire_run_lock(
    scope: str | Path | None = None, what: str = "", *, log_path: str = "",
) -> RunLock:
    """Claim the rig. Raises :class:`RunLockHeld` if someone already has it.

    Created with ``open(..., "x")`` — exclusive create — so two launchers racing cannot
    both believe they won. Checking-then-writing would leave exactly that window, and
    the whole point of this file is the case where two things start at once.
    """
    path = lock_path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = read_run_lock(scope)            # also clears a stale one
    if existing is not None:
        # Idempotent for the process that already owns it. A workflow that
        # acquires and then calls a helper that also acquires would otherwise
        # block on itself -- a deadlock with no second party, and the most likely
        # way a new caller meets this module for the first time.
        if existing.is_mine():
            logger.debug("run_lock_reentrant", pid=existing.pid, what=existing.what)
            return existing
        raise RunLockHeld(existing)

    lock = RunLock(
        pid=os.getpid(), what=str(what),
        started_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(), log_path=str(log_path),
    )
    try:
        with open(path, "x", encoding="utf-8") as fh:
            json.dump(lock.to_dict(), fh, indent=2)
    except FileExistsError:
        # Lost the race between the check above and this create.
        current = read_run_lock(scope)
        raise RunLockHeld(current or lock) from None

    logger.info("run_lock_acquired", pid=lock.pid, what=lock.what, path=str(path))
    return lock


def release_run_lock(scope: str | Path | None = None, *, force: bool = False) -> bool:
    """Give the rig back. Returns whether a lock was removed.

    Refuses to remove **another** process's lock unless *force*: releasing on exit is
    the common path, and a run that crashed and restarted must not delete the lock of
    the run that replaced it.
    """
    path = lock_path(scope)
    if not path.exists():
        return False
    try:
        lock = RunLock.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        lock = RunLock()

    if not force and lock.pid and not lock.is_mine():
        logger.warning("run_lock_release_refused", owner_pid=lock.pid,
                       this_pid=os.getpid(),
                       msg="refusing to release a lock this process does not own")
        return False
    try:
        path.unlink()
    except OSError:
        return False
    logger.info("run_lock_released", pid=lock.pid, what=lock.what)
    return True


def break_run_lock(scope: str | Path | None = None) -> RunLock | None:
    """Take the rig from whoever holds it. Returns the lock that was broken.

    Deliberately a separate, explicitly-named function rather than a flag on
    :func:`acquire_run_lock`. PID reuse means a lock can read as live when its owner is
    long gone, so an override has to exist — but it must be something a person chose
    after reading :meth:`RunLock.describe`, not a fallback some code path can take.
    """
    lock = read_run_lock(scope)
    if lock is None:
        return None
    logger.warning("run_lock_broken", pid=lock.pid, what=lock.what,
                   started_at=lock.started_at,
                   msg="operator took the rig from another process")
    release_run_lock(scope, force=True)
    return lock


# ── Liveness, for callers deciding whether to start ──────────────────────────
#
# This is the *liveness* half of a pair, and the pair must not be confused. A run
# lock answers "is something driving the rig **right now**"; an unfinished run row
# (:mod:`softae.core.shutdown`) answers "did something **die**". Each is useless at
# the other's job, in opposite directions:
#
# * the lock cannot be a recovery marker — :func:`read_run_lock` unlinks a stale
#   lock and returns ``None``, so "no lock" and "a crashed run whose lock I just
#   deleted" are the same answer;
# * the row cannot be a liveness check — a live run's row and a crashed run's row
#   are identical, so treating a row as evidence of death marks a **running**
#   campaign ``interrupted`` and parks the rig underneath it.
#
# So every caller that does both asks *this* first. See the ordering comments in
# ``tools/campaign.py`` and ``gui/widgets/unclean_shutdown.py``.

def foreign_run_lock(scope: str | Path | None = None) -> RunLock | None:
    """The rig claim if a **live, other** process holds it, else ``None``.

    Composed from the two predicates that already exist rather than adding a
    third: :func:`read_run_lock` supplies liveness (it returns only locks whose
    owner is still alive, clearing stale ones as it goes) and
    :meth:`RunLock.is_mine` supplies foreignness.

    A lock this process owns is deliberately **not** foreign. A GUI running its
    own sequence, or a campaign re-entering its own claim, is not a second owner
    of anything, and reporting it as one would refuse the ordinary case.
    """
    lock = read_run_lock(scope)
    if lock is None or lock.is_mine():
        return None
    return lock


#: The holder may be **wedged rather than dead**, and the two need different
#: answers. Staleness self-clears — a dead owner's lock is gone the next time
#: anyone reads it — but a hung process stays alive to the OS and holds the rig
#: indefinitely. The override for that already exists and is deliberately manual
#: (:func:`break_run_lock`, surfaced as the Calibration Launcher's "Take the rig?"
#: confirmation), because PID reuse means no automatic check can tell a wedged
#: owner from a live one. Nothing here takes the rig on its own.
WEDGED_HOLDER_ADVICE = (
    "If that process is wedged rather than working — no output, no CPU — take "
    "the rig from it deliberately in the GUI's Calibration Launcher "
    '("Take the rig?"). Do that only once you are sure it is not still driving '
    "the hardware: two processes on one rig is what this check exists to prevent."
)


def busy_rig_message(lock: RunLock, *, action: str) -> str:
    """Why *action* is refused, who holds the rig, and what to do about it.

    Never a bare "busy". The operator's only recourse against an anonymous
    refusal is to start deleting files, so the holder is named — PID, what it is
    running, since when — via :meth:`RunLock.describe`, and every exit from the
    situation is spelled out.
    """
    return (
        f"{action} would drive the same rig, and {lock.describe()}\n"
        "\nWhat to do:\n"
        "  - let it finish, then start this one;\n"
        "  - or stop it at its own terminal (Ctrl-C parks the rig and keeps the "
        "checkpoint, so `softae-campaign resume` continues it);\n"
        f"  - {WEDGED_HOLDER_ADVICE}\n"
        "\nThis refusal is scoped to starting a second automated run. Manual "
        "control at the rig is never refused."
    )


def rig_is_simulated(manager: Any) -> bool:
    """Whether nothing physical is at stake, so the lock may be skipped.

    A simulated run must **not** take the lock: two mock runs collide over nothing, and
    a mock run holding the rig turns a dry run into an outage for a real one. This is
    the single legitimate exemption.

    It delegates to :func:`~softae.core.hardware_safety.real_motion_instruments` rather
    than inspecting driver names here, so "is this real?" has **one** definition shared
    with the arming interlock. A second, private notion of real would be free to
    disagree with the one that governs motion — and the first draft of this function
    did exactly that, calling the mock manager *real* because it registers a genuine
    ``AsyncLiquidHandler``, which `real_motion_instruments` deliberately excludes since
    it owns no port.

    It is a property of the drivers, not of a flag, so a ``--mock`` argument that never
    reached the factory cannot claim the exemption.

    Uses :func:`~softae.core.hardware_safety.probe_motion` rather than
    ``real_motion_instruments`` so that an **unreadable** probe is not mistaken for a
    simulated rig — the two were the same answer until the interlock was fixed, and
    skipping the lock is the unsafe direction.
    """
    try:
        from softae.core.hardware_safety import probe_motion

        real, unreadable = probe_motion(manager)
        return not real and not unreadable
    except Exception:
        logger.warning(
            "run_lock_simulation_check_failed", exc_info=True,
            msg="could not determine whether the rig is real — assuming it is, and "
                "taking the lock")
        return False


@contextmanager
def held_run_lock(
    scope: str | Path | None = None, what: str = "", *, log_path: str = "",
) -> Iterator[RunLock]:
    """Hold the rig for the duration of a block, releasing even on exception.

    The release is unconditional because the alternative — a lock surviving a crash —
    is repaired only by :func:`read_run_lock`'s liveness check, and that repair should
    be the backstop rather than the mechanism.
    """
    before = read_run_lock(scope)
    mine_already = before is not None and before.is_mine()
    lock = acquire_run_lock(scope, what, log_path=log_path)
    try:
        yield lock
    finally:
        # Release only what this block created. A nested `with` must not free the
        # lock its caller is still relying on -- the inner block exiting would
        # otherwise hand the rig away mid-run.
        if not mine_already:
            release_run_lock(scope)
