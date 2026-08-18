"""Make the rig safe when the *process* stops — not only when the loop does.

:mod:`softae.core.safe_park` knows *how* to drive the rig safe. This module
knows *when*, on the headless path, where there is no window to close and no
button to press:

* a **signal** (Ctrl-C, ``SIGTERM``, Ctrl-Break) arrives mid-campaign;
* the campaign **raises** — a crash, a cancellation, an abort;
* the last run **never finished at all**, because the host rebooted under it.

Three rules shape everything here.

**One park per shutdown.** :class:`ParkGuard` claims the park before performing
it, so the signal handler, the campaign's own catch-all and the CLI's teardown
can all ask without three parks racing down one serial line. The claim is taken
under a re-entrant lock and *before* the first driver write, which is what makes
a second Ctrl-C arriving mid-park harmless: the nested handler sees the claim,
declines, and returns.

**Park before disconnect, or do not bother.** ``safe_park`` skips any instrument
that is not ``is_connected`` (``safe_park.py:75-77``), so a park attempted after
teardown parks nothing and says so in ``skipped``. Every caller here runs while
the session is still open.

**A handler is necessary and not sufficient.** On Windows a logoff, a mandated
update restart, or ``TerminateProcess`` runs no Python at all — and
``os.kill(pid, SIGTERM)`` *is* ``TerminateProcess`` there, so ``SIGTERM``
handling is real only on POSIX. What survives that is the unfinished run row, so
:func:`detect_unfinished_runs` and :func:`recover_from_unclean_shutdown` are the
second layer: they run at the *next* launch, when there is time to think. This
is the headless twin of ``gui/widgets/unclean_shutdown.check_unclean_shutdown``
and deliberately shares its query (``DataStore.unfinished_runs``), its alert
text and its order of operations — durable alert first, then mark the rows, then
park — so the two surfaces cannot come to describe the same evidence differently.

Keyed on the **run row**, never on the run lock: ``read_run_lock`` unlinks a
stale lock and returns ``None`` (``run_lock.py:178-209``), so a caller cannot
tell "no lock" from "a crashed run's lock, which I have just deleted". The row is
durable and does not clear itself.

**But the row is not a liveness check, and callers must ask liveness first.**
``DataStore.unfinished_runs`` is project-wide and a *running* campaign's row is
byte-for-byte what a crashed one's looks like. So :func:`detect_unfinished_runs`
asked while another campaign is genuinely running reports that campaign as
crashed, and :func:`recover_from_unclean_shutdown` then marks its row
``interrupted`` and parks the rig underneath it. The two mechanisms are exactly
complementary and each is useless at the other's job:

===================  ==========================================================
:func:`~softae.core.run_lock.foreign_run_lock`  liveness — is someone running *now*?
:func:`detect_unfinished_runs`                  recovery — did something die?
===================  ==========================================================

Both surfaces therefore check the rig lock **before** calling anything here —
``tools/campaign.py`` refuses to start, ``gui/widgets/unclean_shutdown.py`` skips
the check — and both say so at the call site. Deferring costs nothing: the row
outlives every process, so a real crash is still caught at the next launch that
is not racing a live campaign.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import structlog

from softae.core.safe_park import SafeParkResult, safe_park

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

#: What the operator is told about a run that died without unwinding. Written
#: once, here, because the GUI dialog and the CLI must not describe the same
#: physical unknown in two different ways.
UNCLEAN_SHUTDOWN_MESSAGE = (
    "run(s) did not finish cleanly; the previous session ended without "
    "unwinding. The dispenser head holds position without power, so it may have "
    "been left lowered over an electrode — inspect before moving the stage."
)

#: Reason recorded by the recovery park at the next launch.
RECOVERY_PARK_REASON = "recovery after unclean shutdown"


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def shutdown_signals() -> tuple[int, ...]:
    """Signals that mean *this process is being asked to stop*, on this OS.

    ``SIGBREAK`` (Ctrl-Break) exists only on Windows and ``SIGTERM``, while
    present there, is not deliverable to a Python handler — both are resolved by
    name so this module imports and installs on either platform.
    """
    found: list[int] = []
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None and sig not in found:
            found.append(int(sig))
    return tuple(found)


# ── One park per shutdown ────────────────────────────────────────────────────

class ParkGuard:
    """A one-shot, thread-safe, re-entrant-safe park of one manager.

    The first caller to :meth:`park` performs it; every later caller is told it
    already happened and returns ``None``. That is what lets the signal handler,
    :func:`~softae.core.autonomous_wiring.run_autonomous_campaign`'s catch-all
    and the CLI's teardown each ask unconditionally — the guard, not the caller,
    owns the question of whether a park is still owed.

    The claim is taken **before** the first driver write, so a signal delivered
    *during* the park (a second Ctrl-C, which Python may run inside the first
    handler) finds the guard already claimed and adds no second write sequence to
    a serial line that is mid-sequence.

    ``retract_head`` defaults to ``None`` — *do not touch the head* — because
    every caller here is automatic. ``head_retract`` is a conditional flip on a
    belief with no sensor behind it, and a process being killed is exactly when
    that belief is least likely to hold. The result reports the head as
    unverifiable instead, which is the truth.
    """

    def __init__(
        self,
        manager: "InstrumentManager",
        *,
        on_park: Callable[[str, SafeParkResult], None] | None = None,
        retract_head: bool | None = None,
    ) -> None:
        self._manager = manager
        self._on_park = on_park
        self._retract_head = retract_head
        # Re-entrant: a nested signal handler on this same thread must find the
        # lock passable and the flag already set, not deadlock behind itself.
        self._lock = threading.RLock()
        self._claimed = False
        self._finished = False
        self.reason: str | None = None
        self.result: SafeParkResult | None = None

    @property
    def manager(self) -> "InstrumentManager":
        return self._manager

    @property
    def parked(self) -> bool:
        """A park has been claimed (and possibly completed) by someone."""
        return self._claimed

    @property
    def in_progress(self) -> bool:
        """A park is claimed but its driver writes have not finished."""
        return self._claimed and not self._finished

    def park(self, reason: str) -> SafeParkResult | None:
        """Park once. ``None`` when someone else already claimed it.

        Never raises: this runs on shutdown paths where an exception would
        replace whatever was actually wrong with a report about the reporting.
        """
        with self._lock:
            if self._claimed:
                logger.info("shutdown_park_already_claimed",
                            declined=reason, held_by=self.reason)
                return None
            self._claimed = True
            self.reason = reason

        try:
            result = safe_park(self._manager, reason=reason,
                               retract_head=self._retract_head)
        except BaseException as exc:  # noqa: BLE001 - safe_park's contract says
            # it never raises; shutdown is the worst possible place to discover
            # otherwise, so the contract is enforced here rather than trusted.
            logger.error("shutdown_park_raised", reason=reason, error=str(exc))
            result = SafeParkResult(errors=[f"safe_park raised: {exc}"])

        self.result = result
        self._finished = True
        if self._on_park is not None:
            try:
                self._on_park(reason, result)
            except Exception:
                logger.warning("shutdown_park_notify_failed", exc_info=True)
        return result

    def describe(self) -> str:
        """One line an operator can read at the end of a killed run."""
        if not self._claimed:
            return "the rig was NOT parked"
        if self.result is None:
            return f"park in progress ({self.reason})"
        if self.result.ok:
            return f"parked ({self.result.summary()}): {self.reason}"
        return (f"park INCOMPLETE ({self.result.summary()}): "
                + "; ".join(self.result.errors))


_active_lock = threading.Lock()
_active_guard: ParkGuard | None = None


def active_park_guard() -> ParkGuard | None:
    """The guard installed for this process, if any."""
    return _active_guard


def park_on_shutdown(
    manager: "InstrumentManager",
    reason: str,
    *,
    retract_head: bool | None = None,
) -> SafeParkResult | None:
    """Park *manager*, deduplicated against the process's active guard.

    Library code (the campaign wiring) calls this rather than :func:`safe_park`
    directly so that it parks correctly when nobody installed a guard *and*
    does not park twice when someone did.
    """
    guard = active_park_guard()
    if guard is not None and guard.manager is manager:
        return guard.park(reason)
    return ParkGuard(manager, retract_head=retract_head).park(reason)


@contextmanager
def install_signal_park(
    guard: ParkGuard,
    *,
    signals: Sequence[int] | None = None,
) -> Iterator[tuple[int, ...]]:
    """Park on SIGINT/SIGTERM/SIGBREAK for the duration, then restore.

    Yields the signals actually installed — empty when this is not the main
    thread (``signal.signal`` is main-thread only, so a GUI worker or a test
    runner thread gets the guard without the handlers).

    The handler **parks, uninstalls itself, and raises ``KeyboardInterrupt``**.
    Each part is load-bearing:

    * *parks* — for ``SIGTERM`` there is no other Python that will ever run;
    * *uninstalls* — so a further Ctrl-C reaches the default handler. A handler
      that swallowed every signal after the first would leave an operator
      watching a wedged teardown with no way out but the task manager;
    * *raises* — so the exit unwinds through the campaign's own teardown, which
      is what finalizes the run row and keeps the checkpoint, instead of dying
      where it stands.

    A signal arriving *while* the park runs is declined by
    :attr:`ParkGuard.in_progress` and returns, leaving the in-flight park alone.

    The guard is also published as the process's active one, so
    :func:`park_on_shutdown` deep inside the campaign deduplicates against it.
    """
    global _active_guard

    targets = tuple(signals) if signals is not None else shutdown_signals()
    previous: dict[int, object] = {}
    installed: list[int] = []
    restored = threading.Event()

    def _restore() -> None:
        if restored.is_set():
            return
        restored.set()
        for sig in installed:
            try:
                signal.signal(sig, previous[sig])  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 - restoring must not raise on exit
                logger.debug("signal_restore_failed", signal=_signal_name(sig))

    def _handler(signum: int, _frame) -> None:
        name = _signal_name(signum)
        if guard.in_progress:
            logger.warning("shutdown_signal_during_park", signal=name)
            return
        if not guard.parked:
            logger.warning("shutdown_signal_parking", signal=name)
            guard.park(f"{name} received — the process is shutting down")
        _restore()
        raise KeyboardInterrupt(f"{name} received")

    is_main = threading.current_thread() is threading.main_thread()
    if is_main:
        for sig in targets:
            try:
                previous[sig] = signal.signal(sig, _handler)
                installed.append(sig)
            except (ValueError, OSError, RuntimeError, AttributeError) as exc:
                logger.info("signal_park_not_installed", signal=_signal_name(sig),
                            error=str(exc))
    else:
        logger.info("signal_park_not_installed", why="not the main thread")

    with _active_lock:
        outer, _active_guard = _active_guard, guard
    try:
        yield tuple(installed)
    finally:
        _restore()
        with _active_lock:
            _active_guard = outer


# ── Next-launch recovery, keyed on the unfinished run row ────────────────────

@dataclass(frozen=True)
class UnfinishedRuns:
    """Runs whose row was never closed — evidence of a process that was killed.

    Every *soft* exit of a campaign finalizes its row (``_finalize_run`` runs in
    the catch-all and again in the ``finally``), so this marks hard kills only:
    a crash, a power cut, an OS-forced restart. That narrowness is the point —
    it is exactly the population no signal handler can reach.
    """

    runs: tuple[dict, ...]

    def __bool__(self) -> bool:
        return bool(self.runs)

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(str(r.get("run_id")) for r in self.runs)

    def describe(self) -> str:
        ids = self.run_ids
        shown = ", ".join(ids[:3])
        more = f" (+{len(ids) - 3} more)" if len(ids) > 3 else ""
        return (f"{len(ids)} {UNCLEAN_SHUTDOWN_MESSAGE}\n"
                f"   {shown}{more}")


def detect_unfinished_runs(data_store: "DataStore | None") -> UnfinishedRuns | None:
    """Unfinished runs from a previous session, or ``None``.

    Read-only and best-effort — it must never be the reason a campaign cannot
    start. Call it *before* connecting so the operator is told early; do the
    parking afterwards (see :func:`recover_from_unclean_shutdown`).

    .. warning::
       **Check :func:`~softae.core.run_lock.foreign_run_lock` first.** This query
       cannot distinguish a live run from a dead one, so calling it while another
       process holds the rig reports that process's own run as crashed — and the
       recovery that follows parks a campaign that was working. See the module
       docstring.
    """
    if data_store is None:
        return None
    try:
        rows = data_store.unfinished_runs()
    except Exception:
        logger.warning("unfinished_runs_query_failed", exc_info=True)
        return None
    if not rows:
        return None
    unfinished = UnfinishedRuns(tuple(rows))
    logger.warning("unclean_shutdown_detected", count=len(rows),
                   runs=unfinished.run_ids[:3])
    return unfinished


def record_unclean_shutdown(
    unfinished: UnfinishedRuns,
    data_store: "DataStore",
) -> None:
    """Raise the durable alert, then mark the rows ``interrupted``.

    In that order, and for the GUI's reason: the alert is what outlives the
    session, and marking first would leave a window in which the evidence had
    been consumed and nothing recorded it. Marking at all is what stops this
    being reported afresh at every subsequent launch.
    """
    try:
        from softae.core.alerts import WARNING, Alert, raise_alert

        raise_alert(
            Alert(
                kind="unclean_shutdown",
                message=f"{len(unfinished.runs)} {UNCLEAN_SHUTDOWN_MESSAGE}",
                severity=WARNING,
                run_id=unfinished.run_ids[0],
                details={
                    "runs": list(unfinished.run_ids[:10]),
                    # A real physical unknown, not boilerplate: the head is a
                    # motor-driven flipper that holds position unpowered.
                    "head_position_unknown": True,
                    "surface": "headless",
                },
            ),
            data_store=data_store,
        )
    except Exception:
        logger.warning("unclean_shutdown_alert_failed", exc_info=True)

    for run_id in unfinished.run_ids:
        try:
            data_store.finish_run(run_id, "interrupted")
        except Exception:
            logger.warning("mark_interrupted_failed", run_id=run_id)


def recover_from_unclean_shutdown(
    manager: "InstrumentManager",
    data_store: "DataStore",
    *,
    unfinished: UnfinishedRuns | None = None,
    report: Callable[[str], None] | None = None,
) -> SafeParkResult | None:
    """Record and park after a previous session was killed. Never raises.

    **Call this with the instruments connected.** ``safe_park`` skips anything
    that is not ``is_connected``, so a recovery park issued before
    ``connect_all`` records four skips and moves nothing — which would be the
    same defect this priority exists to fix, one launch later.

    Deliberately **not** routed through the shutdown :class:`ParkGuard`: this
    park belongs to the *previous* run, and consuming the guard's single claim
    here would leave the run that is about to start with no park left for its own
    shutdown.
    """
    if unfinished is None:
        unfinished = detect_unfinished_runs(data_store)
    if not unfinished:
        return None

    if report is not None:
        report(unfinished.describe())
    record_unclean_shutdown(unfinished, data_store)

    try:
        result = safe_park(manager, reason=RECOVERY_PARK_REASON)
    except BaseException as exc:  # noqa: BLE001 - see ParkGuard.park
        logger.error("recovery_park_raised", error=str(exc))
        result = SafeParkResult(errors=[f"safe_park raised: {exc}"])
    if report is not None:
        report(f"   recovery park: {result.summary()}"
               + ("" if result.ok else " — " + "; ".join(result.errors)))
    return result
