"""Make a validation run watchable while it runs, from **synchronous** code.

A validation holds the rig for 1.5-3+ hours and published nothing at all. An
operator who opened the desktop GUI at hour two got a banner saying the rig was
busy and no temperature, no RH and no progress -- the run's own console was the
only place any of it existed, and that console is on the machine the run was
started from.

Everything needed to fix that already exists in
:mod:`softae.core.campaign_events`: the ``events.jsonl`` writer, the heartbeat
record, the single-slot ``conditions.json``, and the reader half
(:func:`~softae.core.campaign_events.read_events`,
:func:`~softae.core.campaign_events.liveness`). **This module imports that one
and changes nothing in it.** What it adds is the two pieces that module cannot
supply to *this* caller, plus the vocabulary that describes a validation.

The asyncio problem, and the answer
-----------------------------------
:meth:`~softae.core.campaign_events.CampaignNarrator.start_heartbeat` and
:meth:`~softae.core.campaign_events.ConditionsPublisher.start` both call
``asyncio.ensure_future``. They were written for ``run_autonomous_campaign``,
which owns a running event loop for the whole campaign and dispatches every
blocking instrument call through ``run_in_executor`` -- so its loop is free to
beat straight through an 8-hour anneal.

``cmd_run`` is not that. It uses ``asyncio.run`` for ``connect_all`` and
``disconnect_all`` and nothing else; ``settle_phase``, ``soak_phase``,
``run_cells`` and ``drift_check`` are straight-line synchronous code, and there
is no loop running during any of them -- which is precisely the window that
needs narrating. ``start_heartbeat()`` from there raises or, worse, schedules a
task onto a loop that is torn down at the end of ``asyncio.run(connect_all())``
and never runs again.

So the beat is a **thread**, not a task. It calls the *synchronous*
:meth:`~softae.core.campaign_events.CampaignNarrator.beat`, which is public,
never raises, and appends under
:attr:`~softae.core.campaign_events.CampaignNarrator._lock` -- a
``threading.Lock`` whose own comment says it exists because callbacks reach the
writer from more than one thread. The class was already prepared for this; the
thread is the caller that needed it.

**Why the beat cannot simply be phase boundaries.**
:func:`~softae.core.campaign_events.liveness` counts *any* record, not only a
beat, so the phase records below do keep a watcher fed most of the time. But one
``Extended`` reference sweep is ~120 s, a full settle round is minutes, and a
soak is *hours* of one line every 30 s -- against a staleness rule of three
beats, i.e. 90 s. A watcher would call a perfectly healthy run **stale** for
most of its life. The beat closes exactly that gap and nothing else.

The beat thread touches **no instrument**. It appends a line and rewrites one
small file; it opens no session, takes no serial lock, and cannot delay a sweep.

conditions.json: published from the capture the run already performs
-------------------------------------------------------------------
:class:`~softae.core.campaign_events.ConditionsPublisher` is deliberately **not**
stood up. Two reasons, and the second is the one that decided it.

It is asyncio for the same reason the heartbeat is -- but that alone would only
mean re-implementing it. The reason not to have it at all is that it would be a
**second reader competing for the serial lock the measurement is already using**.
Its own docstring says so: against a headless run it is *net-new Modbus traffic
on the same* ``_serial_lock``. And this harness does not need it, because
:func:`softae.tools.eis_validate.persist` already calls
:func:`~softae.core.conditions_capture.read_environment` after **every single
sweep** -- the same function the publisher wraps. Publishing that capture costs
zero extra instrument transactions, and :meth:`RunNarration.capture` is written
so that one read feeds both the DataStore row and this file rather than two
reads agreeing by convention.

The idle soak is the one window with no sweeps to ride on, and it is also the
window with an idle bus: ``soak_phase`` is already polling both controllers
every 30 s through ``HoldWatch``. A capture on that same cadence adds five
Modbus reads per 30 s to a bus with no sweep in flight, which is the same order
as what the watch already spends there -- and it is the phase an operator at
hour two is actually asking about.

**Visibly stale, never silently stale** -- the publisher's rule, kept.
:meth:`beat` republishes the slot with ``skipped_beats`` incremented and
``completed_at`` *unchanged*, so a 30-minute sweep leaves a file whose mtime
advances and whose numbers openly date themselves.
:class:`~softae.gui.widgets.conditions_source.ConditionsFileSource` measures age
on ``completed_at`` and drops values older than 15 s, so it will correctly render
``--`` through a sweep rather than showing a two-minute-old temperature as
current. That is the intended outcome, not a shortfall: our numbers genuinely
are that old, and the stream's heartbeat -- not this file -- is what says the run
is alive.

What never goes in
------------------
The stream carries **narration and liveness, never scientific record**. Sweep
results, fits, sigma, R1, apexes and arc verdicts are already in the DataStore,
which is the only thing that can say what they mean. A record here is a claim
about what the run was *doing*. Phase names, progress counts and excursion
restarts are narration; a fitted R1 is not, and
``test_the_stream_carries_no_scientific_value`` is what keeps it that way.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

from softae.core.campaign_events import (
    CONDITIONS_TMP_SUFFIX,
    DEFAULT_HEARTBEAT_S,
    EVENTS_FILENAME,
    conditions_path,
    open_narrator,
)
from softae.core.conditions_capture import ENV_KEYS

logger = structlog.get_logger(__name__)

#: How long :meth:`RunNarration.close` waits for the beat thread before giving
#: up on it. The thread only appends a line and rewrites a small file, so a join
#: that does not complete promptly means the disk is wedged -- and a validation
#: run that has just parked must not be held open by its own log.
JOIN_TIMEOUT_S = 5.0

#: The phase spine, in order. These are the ``old``/``new`` values on the
#: ``state`` records, and they are the answer to "how far along is it?".
PHASE_STARTING = "starting"
PHASE_APPROACH = "approach"
PHASE_SETTLE = "settle"
PHASE_SOAK = "soak"
PHASE_CELLS = "cells"
PHASE_DRIFT = "drift"
PHASE_REPORT = "report"
PHASE_PARK = "park"
PHASE_FINISHED = "finished"


def _stamp() -> str:
    """The ISO-8601 UTC stamp ``conditions.json``'s readers parse.

    Written here rather than imported because ``campaign_events._stamp`` is
    private to that module, and a second file reaching into it would make a
    private helper part of this tool's contract -- the same reasoning
    ``gui/widgets/conditions_source.py`` gives for keeping its own parser.
    """
    return datetime.now(tz=timezone.utc).isoformat()


def _null_environment() -> dict[str, Any]:
    """The five keys, all unread -- the shape ``read_environment`` guarantees."""
    return {key: None for key in ENV_KEYS}


class RunNarration:
    """One run's ``events.jsonl`` and ``conditions.json``, driven synchronously.

    A **null object when it cannot open**: every method stays callable and does
    nothing, so no call site needs an ``if``. A failure to narrate can never fail
    a validation run -- that is the contract
    :mod:`~softae.core.campaign_events` keeps toward a campaign, and this keeps
    the same one toward the harness.

    Parameters
    ----------
    run_dir
        This run's directory. Both sidecars are written inside it, and it is what
        the rig claim's ``log_path`` advertises.
    narrator
        The underlying :class:`~softae.core.campaign_events.CampaignNarrator`, or
        ``None`` for an inert narration. Built by :func:`open_narration`.
    heartbeat_s
        Beat cadence. Left at
        :data:`~softae.core.campaign_events.DEFAULT_HEARTBEAT_S` deliberately:
        :func:`~softae.core.campaign_events.liveness` applies its three-beat
        staleness rule against that same constant, so a run that beat on a
        different clock would be judged by a rule it does not obey. ``0``
        disables the thread.
    now
        Monotonic clock, injected so a test can time a capture without waiting.
    wait
        The beat loop's wait, defaulting to the stop event's. Injected so the
        loop body can be driven deterministically rather than by sleeping.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        narrator: Any = None,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
        now: Callable[[], float] = time.monotonic,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.heartbeat_s = float(heartbeat_s)
        self._narrator = narrator
        self._now = now
        self._stop = threading.Event()
        self._wait = wait if wait is not None else self._stop.wait
        self._thread: threading.Thread | None = None
        self._degraded = False
        #: The phase last announced, so :meth:`state` can fill ``old`` itself.
        #: Kept here rather than threaded through ``cmd_run`` as a local because a
        #: transition is a fact about the run, and a caller that has to remember
        #: where it came from is a caller that will eventually pass the wrong
        #: ``old`` on the one exit path nobody exercised.
        self._phase = PHASE_STARTING

        # Guards the conditions slot only. The event stream has its own lock,
        # inside `CampaignNarrator`, and reaching for that one from here would
        # be a second discipline over the same file.
        self._slot = threading.Lock()
        self._env: dict[str, Any] = _null_environment()
        self._started_at: str | None = None
        self._completed_at: str | None = None
        self._read_ms: int | None = None
        self._skipped_beats = 0

    # ── Identity ────────────────────────────────────────────────────────

    @property
    def live(self) -> bool:
        """Whether a stream was actually opened."""
        return self._narrator is not None

    @property
    def log_path(self) -> str:
        """What the rig claim should advertise -- and ``""`` when there is nothing.

        :func:`~softae.core.rig_session.claim_rig_session` leaves this field empty
        by default because a directory that holds some *other* run's stream,
        offered as the live holder's, is a lie. This run writes its own stream
        into its own directory before the claim is taken, which answers that
        objection -- but only while the stream exists. If it could not be opened
        the field goes back to empty rather than naming a directory a watcher
        would find nothing in.
        """
        return str(self.run_dir) if self.live else ""

    @property
    def events_path(self) -> Path:
        return self.run_dir / EVENTS_FILENAME

    @property
    def conditions_path(self) -> Path:
        return conditions_path(self.run_dir)

    # ── Narration ───────────────────────────────────────────────────────

    def record(self, event_type: str, **payload: Any) -> None:
        """Append one narration record. Never raises, and never a measurement."""
        if self._narrator is None:
            return
        try:
            self._narrator.record(event_type, payload)
        except Exception:                                 # pragma: no cover
            self._warn("eis_validate_narrate_failed")

    @property
    def phase(self) -> str:
        """The phase last announced through :meth:`state`."""
        return self._phase

    def state(self, new: str, **payload: Any) -> None:
        """Enter *new*. Emitted in ``run_autonomous_campaign``'s ``state`` shape.

        ``old`` comes from :attr:`phase` rather than from the caller, so the
        chain of transitions is continuous by construction on every exit path --
        including the ones reached from inside a ``finally``.
        """
        old, self._phase = self._phase, new
        self.record("state", old=old, new=new, **payload)

    def progress(self, phase: str, done: int, total: int, **payload: Any) -> None:
        """How far through *phase* this run is. Counts only -- never a result."""
        self.record("progress", phase=phase, done=int(done), total=int(total),
                    **payload)

    def beat(self) -> None:
        """One heartbeat, plus a republish of the conditions slot. Never raises.

        The republish carries the **last completed** read with its original
        stamps and ``skipped_beats`` incremented, so a reader can tell "the
        publisher is alive and the numbers are two minutes old" from "the
        publisher died", which are different problems with different answers.
        """
        if self._narrator is not None:
            try:
                self._narrator.beat()
            except Exception:                             # pragma: no cover
                self._warn("eis_validate_beat_failed")
        with self._slot:
            self._skipped_beats += 1
        self._write_conditions()

    # ── Conditions ──────────────────────────────────────────────────────

    def capture(self, manager: Any) -> dict[str, Any]:
        """Read the rig once, publish it, and hand the reading back.

        **One read, two consumers.** The caller records the returned dict into
        ``conditions`` rows; this file gets the same values. Returning it rather
        than reading twice is what makes "publishing costs no extra instrument
        traffic" a property of the code rather than a promise about it.

        Never raises: :func:`~softae.core.conditions_capture.read_environment`
        already answers ``None`` for anything it cannot read, and a failure worse
        than that publishes nulls rather than a stale value wearing a fresh
        stamp.
        """
        from softae.core.conditions_capture import read_environment

        started_at = _stamp()
        began = self._now()
        try:
            env = read_environment(manager)
        except Exception:                                 # pragma: no cover
            env = _null_environment()
            self._warn("eis_validate_conditions_read_failed")
        with self._slot:
            self._env = {key: env.get(key) for key in ENV_KEYS}
            self._started_at = started_at
            self._completed_at = _stamp()
            self._read_ms = int(round(max(0.0, self._now() - began) * 1000))
            self._skipped_beats = 0
        self._write_conditions()
        return dict(env)

    def payload(self) -> dict[str, Any]:
        """The slot as it is written -- key-for-key what ``ConditionsPublisher``
        writes, because :class:`~softae.gui.widgets.conditions_source.ConditionsFileSource`
        reads both and must not learn a second shape."""
        with self._slot:
            return {
                "started_at": self._started_at,
                "completed_at": self._completed_at,
                "read_ms": self._read_ms,
                "env": dict(self._env),
                "skipped_beats": self._skipped_beats,
            }

    # ── Lifetime ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin beating on a daemon thread. Never raises."""
        if self._narrator is None or self.heartbeat_s <= 0 or self._thread is not None:
            return
        try:
            self._thread = threading.Thread(
                target=self._beat_loop, name="eis-validate-heartbeat", daemon=True)
            self._thread.start()
        except Exception:                                 # pragma: no cover
            self._thread = None
            self._warn("eis_validate_heartbeat_start_failed")

    def close(self) -> None:
        """Stop beating and close the stream. Never raises. Idempotent."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(JOIN_TIMEOUT_S)
        if self._narrator is not None:
            self._narrator.close()

    def __enter__(self) -> RunNarration:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ── Internal ────────────────────────────────────────────────────────

    def _beat_loop(self) -> None:
        """Beat until asked to stop. ``Event.wait`` returns ``True`` when set,
        so the stop is prompt rather than one cadence late."""
        while not self._wait(self.heartbeat_s):
            self.beat()

    def _write_conditions(self) -> None:
        """Replace the single slot, atomically. Never raises, never appends.

        Written next to the target and renamed onto it -- same directory, so the
        rename is same-volume and therefore atomic on NTFS and POSIX alike. A
        reader polling this path sees one whole payload or the previous one,
        never a prefix of either.
        """
        if self._narrator is None:
            return
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            target = self.conditions_path
            tmp = target.with_name(target.name + CONDITIONS_TMP_SUFFIX)
            tmp.write_text(json.dumps(self.payload(), indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            self._warn("eis_validate_conditions_write_failed")

    def _warn(self, event: str) -> None:
        """First failure is a warning; the rest are debug.

        A narration stream that has started failing usually keeps failing, and
        one warning per beat over a three-hour run would bury the log the
        operator actually reads under the failure of the log they do not.
        """
        if self._degraded:
            logger.debug(event, run_dir=str(self.run_dir))
            return
        self._degraded = True
        logger.warning(event, run_dir=str(self.run_dir), exc_info=True)


def open_narration(run_dir: str | Path, **kwargs: Any) -> RunNarration:
    """Open a run's narration. **Always returns an object**, never ``None``.

    :func:`~softae.core.campaign_events.open_narrator` answers ``None`` when it
    cannot open, and its caller checks. This returns an inert
    :class:`RunNarration` instead, because the alternative is a null check at
    every one of a dozen call sites inside a run block three sessions are
    writing -- and the one that gets forgotten is an ``AttributeError`` raised
    out of a harness that actuates a heater.

    ``heartbeat_s=0`` is passed **down** to the narrator deliberately: it is that
    class's own switch for "do not start an asyncio task", so the loop-bound path
    is disabled at the source rather than merely not called. The synchronous
    :meth:`~softae.core.campaign_events.CampaignNarrator.beat` is unaffected, and
    it is the only thing this module drives.
    """
    return RunNarration(run_dir, narrator=open_narrator(run_dir, heartbeat_s=0),
                        **kwargs)


__all__ = [
    "JOIN_TIMEOUT_S", "PHASE_APPROACH", "PHASE_CELLS", "PHASE_DRIFT",
    "PHASE_FINISHED", "PHASE_PARK", "PHASE_REPORT", "PHASE_SETTLE",
    "PHASE_SOAK", "PHASE_STARTING", "RunNarration", "open_narration",
]
