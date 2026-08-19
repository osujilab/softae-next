"""The campaign's run-directory sidecars: narration out, control in.

``events.jsonl`` (stage 3, D7) is the durable channel a watcher reads;
``control.json`` (stage 4, D1) is the one small file a watcher writes. Both live
beside the run, both are best-effort, and neither can refuse anyone anything.

Stages 3 and 4 of ``docs/SubAgent docs/campaign_attach_architecture.md``.

The two directions share this module because they share one discipline — the
run-directory siting, the never-raise contract, the asyncio task on the campaign's
own loop — and because keeping them together is what makes the acknowledgement
cheap: a control request is answered on the *same* stream everything else is
narrated to, rather than on a second channel with its own failure modes.

**Nothing here decides anything.** The watcher reads a request, hands it to a
handler, and records what the handler said. What Pause and Abort *mean* belongs
to :class:`~softae.core.autonomous_loop.AutonomousLoop`.

Why it exists
-------------
``run_autonomous_campaign``'s ``emit()`` is purely in-memory dispatch, and the
wiring module says so in its own comments (*"The event stream dies with the
process"*). The scientific record is already durable three times over — DOE rows,
``measurements``, ``settle.json``, ``alerts``, ``campaign_checkpoints`` — so what
dies is the *narration*: which iteration, which mode was resolved, which step was
recovered, why it parked. A watcher that arrives at iteration 40 can rebuild the
convergence curve from ``doe_parameters`` and cannot rebuild the log.

The second gap is **liveness**. ``campaign_checkpoints.updated_at`` advances only
when an iteration *completes*, so inside a five-hour anneal a wedged process and a
healthy slow one are indistinguishable. The heartbeat is the answer, and it is why
this module owns an asyncio task rather than only a file handle.

What it deliberately does not carry
-----------------------------------
Measurements, spectra, DOE rows, settle verdicts as *record*. Every scientific
fact in the stream is already in a table or a sidecar; this file is narration and
liveness. A record here is a claim about *what the campaign was doing*, never
about what it found. If a future change finds itself persisting a measurement
through this writer, that is the bug.

Records also omit ``run_id`` and the campaign name. Both are redundant: the file
lives at ``runs/<run_id>/events.jsonl``, the run lock's ``what`` carries
``campaign:<name>:<run_id>``, and the first record in the file is ``run_started``,
which names both. Repeating them on every line would be several megabytes of
saying the same thing.

Reuse
-----
The writer is :class:`~softae.workflows.experiment_logger.ExperimentLogger` —
the append-only, flush-per-record JSONL writer this codebase already has. A second
JSONL writer in the same process with its own flush discipline is exactly the
duplication worth avoiding. What this module adds around it is the part
``ExperimentLogger`` has no business knowing: a fixed sidecar name beside the run,
the best-effort contract, the heartbeat task, and the size bound.

Discipline
----------
**Best-effort, never raises into the campaign.** A failure to narrate must never
fail a trial — the same contract ``settle.json`` keeps
(``autonomous_wiring``'s ``settle_sidecar_failed``) and the same one
``conditions_capture`` keeps for environment reads. The first failure is a
warning; the rest are debug, because a storm of them must not bury the log that
the operator actually reads.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)

#: Filename beside the run, fixed rather than timestamped. A reader discovers the
#: run directory from the rig lock's ``log_path`` and must be able to open the
#: stream without globbing — one campaign owns one run directory, so there is
#: nothing here for a timestamp to disambiguate.
EVENTS_FILENAME = "events.jsonl"

#: Where a rotated generation goes. Exactly one is kept; see :meth:`_rotate`.
PREVIOUS_FILENAME = "events.1.jsonl"

#: Seconds between heartbeats.
#:
#: The argument, since the obvious cadences are wrong in both directions. The
#: case a watcher cares about is precisely the one a between-steps heartbeat
#: cannot serve: a single step can be an 8-hour anneal, so a beat that ticks only
#: at step boundaries is silent for exactly as long as the operator most wants to
#: know. This heartbeat is therefore an asyncio task on its own clock, and the
#: cadence is chosen against two costs.
#:
#: *Too slow* and staleness cannot be called: a watcher needs several missed
#: beats before declaring a process wedged (one missed beat is a busy event loop,
#: not a corpse), so a 5-minute cadence means a 15-minute verdict — long enough
#: that an operator on a remote desktop reaches for Task Manager instead.
#:
#: *Too fast* and the file stops being small. At 30 s a beat is ~120 bytes, so
#: ~2880 beats/day ≈ 345 kB/day; narration over the same day is a few hundred
#: records. A week-long campaign lands around 2.5 MB — negligible, tailable, and
#: replayable from byte 0 without a reader strategy. At 5 s it would be ~2 MB/day
#: and 15 MB/week for no added answer, since nothing a human does with this file
#: resolves faster than half a minute.
#:
#: 30 s with a **three-beat (90 s) staleness rule** is the recommendation. It is
#: cheap enough to leave on for a multi-day unattended run and prompt enough that
#: "is it alive?" is answered before the question becomes "should I drive in?".
DEFAULT_HEARTBEAT_S = 30.0

#: Rotate at 32 MB, keeping one previous generation — so the stream costs at most
#: 64 MB on disk, permanently.
#:
#: This is a backstop, not a routine event. At the cadence above the arithmetic
#: says ~350 kB/day, so an ordinary campaign never reaches it: 32 MB is roughly
#: three months of continuous running. What it bounds is the pathological case —
#: a step-recovery storm emitting thousands of ``step_recovered`` records, or a
#: campaign left running far longer than anyone planned. Leaving that unbounded on
#: the machine that also holds the DataStore is the failure this cap exists for.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024


class CampaignNarrator:
    """Persist the campaign's event vocabulary, and prove the process is alive.

    Not a new vocabulary. :meth:`record` takes exactly what
    ``run_autonomous_campaign``'s ``emit()`` already produces — ``run_started``,
    ``suggestion``, ``result``, ``settle_verdict``, ``park``, ``safe_park``,
    ``state``, ``step_recovered``, ``step_skipped``, ``run_finished`` and the
    rest — and writes it through unchanged under its own ``type``. A reader can
    feed a replayed line straight into the same handler that consumes the live
    dispatch.

    Parameters
    ----------
    run_dir
        The run's directory; the stream is written inside it. Created if absent,
        following the ``settle.json`` convention.
    heartbeat_s
        Beat cadence in seconds. ``0`` disables the heartbeat entirely.
    max_bytes
        Rotation threshold. ``0`` disables the bound.
    now, sleep
        Injection seams for tests, so the long-step case can be driven on a fake
        clock rather than by waiting. ``now`` returns wall-clock seconds; ``sleep``
        is awaited between beats.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
        max_bytes: int = DEFAULT_MAX_BYTES,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / EVENTS_FILENAME
        self.heartbeat_s = float(heartbeat_s)
        self._max_bytes = int(max_bytes)
        self._now = now
        self._sleep = sleep

        # `emit` is called from the event-loop thread today, but callbacks reach
        # it from the executor's pool often enough that a lock is cheaper than
        # the interleaved half-lines it would otherwise take to notice.
        self._lock = threading.Lock()
        self._writer: Any | None = None
        self._bytes = 0
        self._rotate_at = self._max_bytes
        self._seq = 0
        self._degraded = False
        self._task: asyncio.Task[None] | None = None

        # Liveness state, so a beat can say what the campaign was last doing.
        self._started_at = self._now()
        self._phase = "starting"
        self._phase_at = self._started_at
        self._iteration: int | None = None

        self._open()

    # ── Public API ──────────────────────────────────────────────────────

    def record(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        """Append one narration record. Never raises."""
        body = dict(payload or {})
        self._phase = event_type
        self._phase_at = self._now()
        iteration = body.get("iteration")
        if isinstance(iteration, int):
            self._iteration = iteration
        self._append({"type": event_type, **body})

    def beat(self) -> None:
        """Append one heartbeat. Never raises.

        Carries everything a watcher needs to compute staleness *and* to say
        what it is waiting on: the stamp, the iteration, the phase, and how long
        that phase has run. A beat whose ``phase_age_s`` is 20000 during an
        anneal is a healthy campaign; the same beat missing for 90 s is not.
        """
        self._append({
            "type": "heartbeat",
            "iteration": self._iteration,
            "phase": self._phase,
            "phase_age_s": round(max(0.0, self._now() - self._phase_at), 1),
            "uptime_s": round(max(0.0, self._now() - self._started_at), 1),
        })

    def start_heartbeat(self) -> None:
        """Begin beating on the running event loop.

        An asyncio task rather than a thread because it can be: sync instrument
        methods are dispatched through ``run_in_executor`` onto the shared I/O
        pool (``server/base_instrument.py``), so the event loop stays free for
        the whole of an 8-hour anneal. That is the fact that makes a beat inside
        a long step possible at all, and it is what the injected-clock test pins.
        """
        if self.heartbeat_s <= 0 or self._task is not None:
            return
        try:
            self._task = asyncio.ensure_future(self._heartbeat_loop())
        except Exception:
            self._warn("campaign_heartbeat_start_failed")

    async def aclose(self) -> None:
        """Stop beating and close the file. Never raises."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:
                # Including the CancelledError we just asked for. Nothing this
                # task can raise is worth propagating into a campaign teardown.
                pass
        self.close()

    def close(self) -> None:
        """Close the file handle. Never raises. Idempotent."""
        with self._lock:
            writer, self._writer = self._writer, None
            if writer is None:
                return
            try:
                writer.close()
            except Exception:
                self._warn("campaign_events_close_failed")

    # ── Internal ────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        while True:
            await self._sleep(self.heartbeat_s)
            self.beat()

    def _open(self) -> None:
        """Open the stream for append. A failure here degrades, it does not raise.

        Append rather than truncate: a campaign resumed into the same run
        directory continues one history rather than destroying the one that
        explains why the first attempt stopped.
        """
        # Deferred: importing it pulls in `softae.workflows.__init__`, which
        # imports `workflow_executor`, which imports back into `softae.core`.
        # Nothing today closes that loop, but `core` reaching up into `workflows`
        # at module scope is how it would.
        from softae.workflows.experiment_logger import ExperimentLogger

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._bytes = self.path.stat().st_size if self.path.exists() else 0
            self._rotate_at = self._max_bytes
            self._writer = ExperimentLogger(
                self.run_dir, "campaign", filename=EVENTS_FILENAME)
        except Exception:
            self._writer = None
            self._warn("campaign_events_open_failed")

    def _append(self, record: dict[str, Any]) -> None:
        with self._lock:
            if self._writer is None:
                return
            try:
                record = {"ts": _stamp(), "seq": self._seq, **record}
                self._writer.log_record(record)
                self._seq += 1
                # `stat()` rather than counting the string we wrote: the file is
                # opened in text mode, so on Windows every ``\n`` costs two bytes
                # and a length-based count would under-report the cap by ~1%.
                # One extra syscall per record is nothing at this cadence.
                self._bytes = self.path.stat().st_size
                if 0 < self._rotate_at <= self._bytes:
                    self._rotate()
            except Exception:
                self._warn("campaign_events_write_failed")

    def _rotate(self) -> None:
        """Move the stream aside, keeping exactly one previous generation.

        Best-effort like everything else here, and for a Windows-specific reason
        worth naming: a reader tailing the file holds it open without
        ``FILE_SHARE_DELETE``, so the rename can fail with a sharing violation
        through no fault of ours. When it does we keep writing to the file we
        have and try again a megabyte later — a watcher that is holding the file
        open is a watcher that will let go.
        """
        # Deferred: importing it pulls in `softae.workflows.__init__`, which
        # imports `workflow_executor`, which imports back into `softae.core`.
        # Nothing today closes that loop, but `core` reaching up into `workflows`
        # at module scope is how it would.
        from softae.workflows.experiment_logger import ExperimentLogger

        try:
            self._writer.close()
            os.replace(self.path, self.run_dir / PREVIOUS_FILENAME)
            rotated = True
        except Exception:
            self._warn("campaign_events_rotate_failed")
            rotated = False

        # Either way the handle is closed and a new one is needed; the only
        # difference is whether it lands on a fresh file or the one we failed to
        # move, and how soon we try again.
        try:
            self._writer = ExperimentLogger(
                self.run_dir, "campaign", filename=EVENTS_FILENAME)
        except Exception:
            self._writer = None
            self._warn("campaign_events_reopen_failed")
            return

        self._bytes = self.path.stat().st_size
        if rotated:
            self._rotate_at = self._max_bytes
            self._writer.log_record({
                "ts": _stamp(), "seq": self._seq, "type": "stream_rotated",
                "previous": PREVIOUS_FILENAME,
            })
            self._seq += 1
        else:
            self._rotate_at = self._bytes + max(1 << 20, self._max_bytes // 16)

    def _warn(self, event: str) -> None:
        """First failure is a warning; the rest are debug.

        A narration stream that has started failing usually keeps failing — a
        full disk, a revoked permission — and one warning per record would bury
        the log an operator actually reads under the failure of the log they do
        not.
        """
        if self._degraded:
            logger.debug(event, path=str(self.path))
            return
        self._degraded = True
        logger.warning(event, path=str(self.path), exc_info=True)


def _stamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def open_narrator(run_dir: str | Path, **kwargs: Any) -> CampaignNarrator | None:
    """Build a narrator, or ``None`` if even constructing one fails.

    The caller is a campaign that must start whether or not it can be watched, so
    the constructor's own best-effort contract is backed by one more layer here:
    ``None`` means "run unnarrated", never "do not run".

    The cadence default is resolved *here*, at call time, rather than being baked
    into the constructor's signature at import time — so it stays one knob with
    one value rather than a module constant and a frozen copy of it.
    """
    kwargs.setdefault("heartbeat_s", DEFAULT_HEARTBEAT_S)
    kwargs.setdefault("max_bytes", DEFAULT_MAX_BYTES)
    try:
        return CampaignNarrator(run_dir, **kwargs)
    except Exception:
        logger.warning("campaign_narrator_unavailable", run_dir=str(run_dir),
                       exc_info=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# control.json — the one file a watcher writes (stage 4, D1)
# ═══════════════════════════════════════════════════════════════════════════

#: Beside ``events.jsonl``, for the same reason: a reader finds the run
#: directory from the rig lock's ``log_path`` and must be able to name both
#: files without globbing.
CONTROL_FILENAME = "control.json"

#: Written next to the target and renamed onto it. Same directory, so the
#: rename is same-volume and therefore atomic on both NTFS and POSIX.
CONTROL_TMP_SUFFIX = ".tmp"

#: Seconds between polls of ``control.json``.
#:
#: **Polled, not watched.** A filesystem-notification API would be the obvious
#: alternative and it is the wrong one here: it needs a dependency this tree
#: does not carry, ``ReadDirectoryChangesW`` is unreliable on the network and
#: removable volumes a project directory can live on, and a missed notification
#: is a silently un-actioned Abort — the one failure this channel exists to
#: prevent. A poll cannot miss an edit; it can only be late by one interval, and
#: it degrades to "late" rather than to "never".
#:
#: One second, and the cost is one ``stat()`` per second. The file is only
#: *parsed* when its ``(mtime, size)`` changes, so a campaign nobody is
#: controlling — the overwhelmingly common case — pays a stat and nothing else.
#: The precedent in the tree is ``_approver``'s 0.02 s poll of ``loop.state``;
#: this channel is driven by a human pressing a button rather than by a state
#: machine, and nothing a human does resolves in 20 ms. Against Abort's latency
#: budget it is invisible: the mechanism it drives lands within one *anneal*
#: poll, which is measured in tens of seconds.
DEFAULT_CONTROL_POLL_S = 1.0

#: The complete vocabulary. Three scopes exist (E-Stop, Abort, Pause) and
#: E-Stop is rig-scale and not a campaign control, so exactly two of them are
#: here — plus the inverse of the resumable one.
#:
#: Anything else is not a request this channel did not understand; it is a
#: request this channel refuses to guess at. Board-exchange answers, approval
#: answers and alert acknowledgements are all argued and declined in D2.
CONTROL_ACTIONS = ("pause", "resume", "abort")

#: Outcome recorded when the file could not be read as a request at all.
CONTROL_UNREADABLE = "unreadable"
#: Outcome recorded for a request at or below the last one acted on.
CONTROL_STALE = "ignored_stale"
#: Outcome recorded for whatever was already in the file when the campaign
#: started. See :class:`ControlWatcher` on why it is never obeyed.
CONTROL_PRE_EXISTING = "ignored_pre_existing"


@dataclass(frozen=True)
class ControlRequest:
    """One operator request, as it appears on disk."""

    seq: int
    action: str
    reason: str = ""
    requested_at: str = ""
    requested_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "action": self.action, "reason": self.reason,
            "requested_at": self.requested_at, "requested_by": self.requested_by,
        }


def control_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / CONTROL_FILENAME


def read_control_request(run_dir: str | Path) -> ControlRequest | None:
    """Parse ``control.json``, or ``None`` if it is not a request.

    ``None`` covers *every* way the file can fail to be one: absent, empty,
    truncated mid-write, not a JSON object, no ``action``, an ``action`` outside
    :data:`CONTROL_ACTIONS`, a ``seq`` that is not an integer.

    **The failure direction is deliberate and is the whole point.** A spurious
    Abort costs a board, an anneal and a night; a missed one costs a poll
    interval, because the operator is standing there and presses it again. So
    every ambiguity resolves to *not a request*, never to *halt*.
    """
    try:
        raw = control_path(run_dir).read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        # A half-written file, most likely. The writer renames onto the target
        # so this should be unreachable for our own writer — but a reader that
        # trusts its writer's discipline is a reader that obeys a torn file the
        # day something else writes one.
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if not isinstance(action, str) or action not in CONTROL_ACTIONS:
        return None
    try:
        seq = int(data.get("seq"))
    except (TypeError, ValueError):
        return None
    return ControlRequest(
        seq=seq,
        action=action,
        reason=str(data.get("reason") or ""),
        requested_at=str(data.get("requested_at") or ""),
        requested_by=str(data.get("requested_by") or ""),
    )


def write_control_request(
    run_dir: str | Path,
    action: str,
    *,
    reason: str = "",
    requested_by: str = "",
) -> ControlRequest:
    """Place a request for the campaign to pick up. Atomic; raises on failure.

    Raises rather than degrading, unlike everything else in this module: the
    writer is the operator's *only* feedback that their button did something, so
    a silent failure here is the one failure they cannot see. The campaign side
    is best-effort; the request side is not.

    ``seq`` is one past whatever is already on disk, so a request can never be
    confused with the one before it — including across a resume into the same
    run directory, which is the case the guard was written for.
    """
    run_dir = Path(run_dir)
    previous = read_control_request(run_dir)
    request = ControlRequest(
        seq=(previous.seq + 1) if previous else 1,
        action=action,
        reason=reason,
        requested_at=_stamp(),
        requested_by=requested_by,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    target = control_path(run_dir)
    tmp = target.with_name(target.name + CONTROL_TMP_SUFFIX)
    tmp.write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")
    # Same directory, so same volume, so atomic. A reader polling this path sees
    # the old request or the new one and never a prefix of either.
    os.replace(tmp, target)
    logger.warning("campaign_control_written", path=str(target),
                   action=action, seq=request.seq)
    return request


class ControlWatcher:
    """Poll ``control.json`` and dispatch what it finds. Never raises.

    Parameters
    ----------
    run_dir
        The run's directory; ``control.json`` is read from inside it.
    handlers
        ``{action: callable(ControlRequest) -> outcome_str}``. The watcher does
        not know what an action means — it delivers and records.
    on_ack
        Called with one flat dict per request the watcher formed an opinion
        about, *including* the ones it declined. **A request that is silently
        ignored is worse than no control**, so the refusals are acknowledged as
        loudly as the acceptances.
    poll_s, sleep
        Cadence and its injection seam, so the tests drive this on a fake clock.

    Idempotence and races
    ---------------------
    ``seq`` is monotone and the watcher records the highest it has acted on, so
    the same request read twice is acted on once. A doubled Pause therefore
    resolves either by the seq guard (same file) or by the loop's own
    ``already_paused`` (a second, higher-seq request) — deterministic both ways.

    **Abort beats Pause, in every ordering.** Abort is terminal and Pause is
    not, so there is no sequence in which a Pause meaningfully undoes an Abort,
    while the reverse is well-defined and is already how the executor behaves:
    ``abort()`` accepts ``PAUSED``, and the pause loop is checked *before* the
    abort check precisely so a paused executor is released by the state change
    that must stop it. The loop mirrors that — ``abort()`` sets the resume event
    — and a Pause arriving after an Abort is acknowledged as ``ended``.

    The channel is a **slot, not a queue**. Two requests written between polls
    leave only the second on disk, and the second is the operator's most recent
    intent, which is the one to obey.

    A pre-existing request
    ----------------------
    Whatever is in the file at construction time is recorded as
    :data:`CONTROL_PRE_EXISTING` and **never obeyed**. A campaign resumed into
    the same run directory would otherwise read yesterday's Abort and park
    itself on the first poll, which is the spurious halt this whole design
    orders itself around avoiding.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        handlers: dict[str, Callable[[ControlRequest], str]],
        on_ack: Callable[[dict[str, Any]], Any] | None = None,
        poll_s: float = DEFAULT_CONTROL_POLL_S,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = control_path(self.run_dir)
        self.poll_s = float(poll_s)
        self._handlers = dict(handlers)
        self._on_ack = on_ack
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._acted_seq = 0
        self._signature: tuple[int, int] | None = None
        self._degraded = False

        existing = read_control_request(self.run_dir)
        self._signature = self._stat()
        if existing is not None:
            self._acted_seq = existing.seq
            self._ack(existing, CONTROL_PRE_EXISTING)

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin polling on the running event loop. Never raises."""
        if self.poll_s <= 0 or self._task is not None:
            return
        try:
            self._task = asyncio.ensure_future(self._watch_loop())
        except Exception:
            self._warn("campaign_control_start_failed")

    async def aclose(self) -> None:
        """Stop polling. Never raises."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except BaseException:
            # Including the CancelledError we just asked for.
            pass

    def poll_once(self) -> ControlRequest | None:
        """Read, validate, dispatch. Returns what was acted on, or ``None``.

        Synchronous and self-contained so the whole decision surface — stale
        seq, torn file, unknown action, handler outcome — is testable without an
        event loop.
        """
        try:
            signature = self._stat()
            if signature is None:
                # No file. Forget the last signature so a control.json that is
                # deleted and re-written with the same size and timestamp is
                # still noticed.
                self._signature = None
                return None
            if signature == self._signature:
                return None
            self._signature = signature

            request = read_control_request(self.run_dir)
            if request is None:
                # Recorded, not merely dropped: an operator whose file was
                # rejected must be able to find out why it did nothing. Only
                # once per edit, because the signature guard above means a file
                # that stays broken is not re-reported every second.
                logger.warning("campaign_control_unreadable", path=str(self.path))
                self._ack(None, CONTROL_UNREADABLE)
                return None

            if request.seq <= self._acted_seq:
                self._ack(request, CONTROL_STALE)
                return None

            self._acted_seq = request.seq
            handler = self._handlers.get(request.action)
            if handler is None:
                self._ack(request, CONTROL_UNREADABLE)
                return None
            try:
                outcome = str(handler(request))
            except Exception:
                logger.warning("campaign_control_handler_failed",
                               action=request.action, exc_info=True)
                self._ack(request, "handler_failed")
                return None
            self._ack(request, outcome)
            return request
        except Exception:
            self._warn("campaign_control_poll_failed")
            return None

    # ── Internal ────────────────────────────────────────────────────────

    async def _watch_loop(self) -> None:
        while True:
            await self._sleep(self.poll_s)
            self.poll_once()

    def _stat(self) -> tuple[int, int] | None:
        try:
            info = self.path.stat()
        except Exception:
            return None
        return (info.st_mtime_ns, info.st_size)

    def _ack(self, request: ControlRequest | None, outcome: str) -> None:
        """Record receipt and outcome. Never raises.

        On stage 3's stream rather than a channel of its own, so the answer to
        "did my Abort arrive?" sits in the same file, in the same order, as what
        the campaign was doing when it did.
        """
        if self._on_ack is None:
            return
        ack: dict[str, Any] = {"outcome": outcome, "path": str(self.path)}
        if request is not None:
            ack.update(seq=request.seq, action=request.action,
                       reason=request.reason, requested_by=request.requested_by)
        try:
            self._on_ack(ack)
        except Exception:
            self._warn("campaign_control_ack_failed")

    def _warn(self, event: str) -> None:
        """First failure is a warning; the rest are debug — as above."""
        if self._degraded:
            logger.debug(event, path=str(self.path))
            return
        self._degraded = True
        logger.warning(event, path=str(self.path), exc_info=True)


def open_control_watcher(
    run_dir: str | Path,
    *,
    handlers: dict[str, Callable[[ControlRequest], str]],
    **kwargs: Any,
) -> ControlWatcher | None:
    """Build a watcher, or ``None`` if even constructing one fails.

    The mirror of :func:`open_narrator`, and for the mirror reason: a campaign
    that cannot be controlled must still run. ``None`` means *uncontrollable
    from outside*, never *do not start* — the operator still has the terminal,
    the signal handler and the rig itself.
    """
    kwargs.setdefault("poll_s", DEFAULT_CONTROL_POLL_S)
    try:
        return ControlWatcher(run_dir, handlers=handlers, **kwargs)
    except Exception:
        logger.warning("campaign_control_unavailable", run_dir=str(run_dir),
                       exc_info=True)
        return None
