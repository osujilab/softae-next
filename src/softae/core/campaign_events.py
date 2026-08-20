"""The campaign's run-directory sidecars: narration out, control in.

``events.jsonl`` (stage 3, D7) is the durable channel a watcher reads;
``control.json`` (stage 4, D1) is the one small file a watcher writes;
``conditions.json`` (stage 5, S5.F) is the single slot the campaign republishes
so a watcher holding no instrument sessions can still see the rig. All three live
beside the run, all three are best-effort, and none can refuse anyone anything.

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
from typing import Any, Awaitable, Callable, Sequence

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
# Reading events.jsonl — cursor, rotation, liveness (stage 5, S5.B)
# ═══════════════════════════════════════════════════════════════════════════
#
# Stage 3 shipped the writer and no reader, so every consumer that wanted the
# stream — the attached GUI's sidebar, the E-Stop ladder's ack wait, the Pause
# and Abort buttons — would have written its own tailer. This is that tailer,
# written once, headless, with no Qt anywhere near it: everything below is a
# pure function of what is on disk plus a cursor, so the whole surface is
# testable without a window and reusable from a script.
#
# The one discipline that is not negotiable is stated at :meth:`_rotate`:
# **the handle is opened, read to EOF and closed inside every call.** A tailer
# that keeps the file open holds it without ``FILE_SHARE_DELETE``, ``os.replace``
# then fails with a sharing violation, ``_rotate`` gives up and keeps writing to
# the un-rotated file — and the 32 MB cap is silently off for as long as the
# watcher is attached. The cap exists because this process shares a disk with the
# DataStore. A reader is not allowed to disable it.

#: The three liveness bands, named once here so no widget invents a fourth or
#: disagrees about where the boundaries fall.
LIVENESS_LIVE = "live"
LIVENESS_QUIET = "quiet"
LIVENESS_STALE = "stale"


@dataclass(frozen=True)
class EventCursor:
    """Where a reader got to: a generation, and a count of records inside it.

    ``lines_read`` counts **complete lines consumed**, not bytes, and the reason
    is two lines of this file rather than a preference. A byte offset stops
    meaning anything the moment :meth:`CampaignNarrator._rotate` moves the stream
    aside; and the stream is written in *text* mode, so on Windows every ``\\n``
    costs two bytes on disk and one character on read — which is exactly why
    :meth:`CampaignNarrator._append` stats the file rather than counting the
    string it wrote.

    A line is counted whether or not it yielded a record. A blank line, or a
    complete line that is not JSON, advances the cursor and returns nothing —
    otherwise one corrupt record would stall the reader on it forever.

    ``generation`` counts the rotations **this reader has followed**, from 0. It
    is deliberately not the writer's absolute generation number: the writer keeps
    exactly one previous generation, so a reader attaching after two rotations
    cannot know it was the second and has nothing to do differently if it did.
    What the number is for is noticing a discontinuity — if it advances, the
    reader crossed a boundary rather than simply read further.
    """

    generation: int = 0
    lines_read: int = 0


def events_path(run_dir: str | Path) -> Path:
    """The live stream, beside the run. Named, never globbed — see the module top."""
    return Path(run_dir) / EVENTS_FILENAME


def previous_events_path(run_dir: str | Path) -> Path:
    """The one retained earlier generation. Exists only after a rotation."""
    return Path(run_dir) / PREVIOUS_FILENAME


def read_events(
    run_dir: str | Path,
    *,
    cursor: EventCursor | None = None,
) -> tuple[list[dict[str, Any]], EventCursor]:
    """Read everything written since ``cursor``, and say where that leaves us.

    ``cursor=None`` is a replay: the full history the run directory still holds,
    which after a rotation means ``events.1.jsonl`` followed by ``events.jsonl``.
    Pass the returned cursor back on the next poll and only what is new comes
    back.

    Never raises. A missing run directory, a stream that does not exist yet, a
    permission failure — all of them are "nothing new", because a watcher that
    dies when the file it watches is briefly unavailable is worse than one that
    is briefly behind.

    The four properties that make this correct, each forced by something the
    writer already does:

    **The handle does not outlive the call.** See the section comment above: a
    held handle turns the size cap off.

    **A truncated final line is "not yet written", not an error.**
    ``ExperimentLogger._write`` writes then flushes, so a reader can arrive
    between the two. Only text up to the last ``\\n`` is treated as lines, the
    partial tail is left uncounted, and the next poll returns it whole.

    **A rotation is followed rather than absorbed.** The writer announces one by
    making ``stream_rotated`` the first record of the new file, so a current
    stream that opens with that record is a generation the cursor may predate. If
    it does, the tail of ``events.1.jsonl`` is delivered first and the new
    generation from 0 — no gap, and no record twice.

    **A generation the cursor cannot be inside is re-read, not trusted.** If the
    position is past the end of the current file, the file underneath the reader
    was replaced; everything it now holds is returned and the generation
    advances, so a consumer can see the discontinuity rather than silently lose
    the difference.

    Known limit, stated rather than hidden: with two integers a *second* rotation
    is detected by the position falling off the end of the new generation. A
    reader positioned within the first few lines of a 32 MB generation when it
    rotates would not notice. At the shipped cap that is a ~100,000-line
    generation and a reader that has read almost none of it, which is not a state
    a poller reaches.
    """
    run_dir = Path(run_dir)
    lines = _complete_lines(events_path(run_dir))
    opening = _first_record(lines)
    current_is_rotated = (
        opening is not None and opening.get("type") == "stream_rotated")

    generation = 0 if cursor is None else max(0, int(cursor.generation))
    position = 0 if cursor is None else max(0, int(cursor.lines_read))

    if position > len(lines):
        crossed = True          # the file we were reading is not this file
    elif current_is_rotated and generation == 0:
        crossed = True          # the rotation happened after our cursor was cut
    elif not current_is_rotated and generation > 0:
        crossed = True          # our generation vanished without announcing it
    else:
        crossed = False

    if not crossed:
        return _records(lines[position:]), EventCursor(generation, len(lines))

    events: list[dict[str, Any]] = []
    if current_is_rotated:
        # `position` indexes the generation now sitting in events.1.jsonl, so
        # this is its tail — empty when the cursor had already consumed it.
        events.extend(_records(_complete_lines(previous_events_path(run_dir))[position:]))
    events.extend(_records(lines))
    return events, EventCursor(generation + 1, len(lines))


def last_heartbeat(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The newest ``heartbeat`` record, or ``None`` if the stream carries none.

    :meth:`CampaignNarrator.beat` puts ``phase`` and ``phase_age_s`` on every
    beat precisely so a watcher can say *what* the campaign is waiting on and not
    merely that it is waiting. Exposed here so the sidebar line and the E-Stop
    dialog read the same record rather than each scanning for it.
    """
    for record in reversed(list(events)):
        if record.get("type") == "heartbeat":
            return record
    return None


def liveness(
    events: Sequence[dict[str, Any]],
    *,
    now: Any,
    heartbeat_s: float = DEFAULT_HEARTBEAT_S,
) -> str:
    """``"live"`` / ``"quiet"`` / ``"stale"`` from the records, not from mtime.

    The three-beat rule argued at :data:`DEFAULT_HEARTBEAT_S`, implemented once:
    under one beat is :data:`LIVENESS_LIVE`, one to three beats is
    :data:`LIVENESS_QUIET`, three beats or more (90 s at the shipped cadence) is
    :data:`LIVENESS_STALE`. The boundaries close downward — exactly one beat is
    already ``quiet``, exactly three is already ``stale`` — because a watcher
    that rounds in the other direction reports a wedged process as healthy for
    one more interval.

    From mtime it would be simpler and wrong: a rotation rewrites mtime without
    the campaign having done anything, and a stream on a network volume can carry
    a timestamp from the other machine's clock. The records carry their own
    stamps, written by the process whose liveness is the question.

    **Any record counts, not only a beat.** A ``suggestion`` written two seconds
    ago is proof the process is alive whatever the heartbeat task is doing, and
    the cadence still bounds the silence, because a live campaign beats every
    ``heartbeat_s`` even when nothing else happens. Use :func:`last_heartbeat`
    when the question is *what* it is doing rather than *whether* it is there.

    ``events`` is the stream as the caller has accumulated it, not one poll's
    delta — a tailer that passes only what the last poll returned would report
    ``stale`` the first time nothing new arrived. Keeping the newest record and
    passing ``[record]`` is enough. An empty sequence is ``stale``: never having
    seen the campaign is not evidence that it lives.

    ``now`` accepts a :class:`~datetime.datetime` or epoch seconds. A campaign
    that disabled its heartbeat has disabled liveness rather than redefined it,
    so a non-positive ``heartbeat_s`` measures against the module default.
    """
    beat = float(heartbeat_s) if heartbeat_s and float(heartbeat_s) > 0 \
        else DEFAULT_HEARTBEAT_S
    stamp = _newest_stamp(events)
    if stamp is None:
        return LIVENESS_STALE
    age = _epoch(now) - stamp
    if age < beat:
        return LIVENESS_LIVE
    if age < 3 * beat:
        return LIVENESS_QUIET
    return LIVENESS_STALE


# ── Reader internals ────────────────────────────────────────────────────────

def _complete_lines(path: Path) -> list[str]:
    """Every whole line in the file — opened, read to EOF and closed right here.

    The close is the contract, not an implementation detail; see the section
    comment. Splitting on ``"\\n"`` and dropping the last fragment is what makes
    a half-flushed final line invisible until it is finished: ``"a\\nb\\n"`` gives
    two lines, ``"a\\nb"`` gives one and leaves ``b`` for the next poll.

    ``splitlines`` is deliberately not used — it also breaks on ``\\r``, ``\\x0b``
    and the Unicode separators, any of which inside a JSON string value would
    manufacture two torn lines out of one good record.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return text.split("\n")[:-1]


def _records(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Parse what parses. A line that does not is dropped, never raised."""
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _first_record(lines: Sequence[str]) -> dict[str, Any] | None:
    parsed = _records(lines[:1])
    return parsed[0] if parsed else None


def _newest_stamp(events: Sequence[dict[str, Any]]) -> float | None:
    for record in reversed(list(events)):
        stamp = _parse_stamp(record.get("ts"))
        if stamp is not None:
            return stamp
    return None


def _parse_stamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _epoch(moment)


def _epoch(moment: Any) -> float:
    """Seconds since the epoch, from a datetime or from a number.

    A naive datetime is read as UTC, matching :func:`_stamp`, which is the only
    thing that writes the timestamps this is compared against.
    """
    if isinstance(moment, datetime):
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    return float(moment)


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


def ack_answers_request(ack: dict[str, Any], request: Any) -> bool:
    """Whether *ack* is the campaign's answer to *request*.

    Two surfaces now wait on an ack — the Pause/Abort bar
    (:class:`softae.gui.widgets.campaign_control.CampaignControlRequester`) and
    the E-Stop ladder's rung 2 — and the rule has one subtlety, so it lives here
    once rather than in each of them. A watcher that matched acks its own way
    would resolve a press against somebody else's answer, which is the failure
    mode this whole channel exists to remove.

    **The subtlety.** An ack normally quotes the ``seq`` of the request it
    answers, and that is the only thing worth matching on. The exception is
    :meth:`ControlWatcher._ack` called with no request at all — the
    :data:`CONTROL_UNREADABLE` case, where the file could not be parsed and so
    *has* no seq to quote. That ack is matched by outcome instead, which is sound
    only because the caller snapshots its cursor **before** it writes: there is
    one ``control.json`` per run directory, ours is the newest thing written to
    it, and an ``unreadable`` recorded after our write is about our write.
    Dropping it for carrying no seq would silently swallow exactly the answer the
    operator most needs — that the campaign never read what they asked for.
    """
    seq = ack.get("seq")
    if seq is None:
        return ack.get("outcome") == CONTROL_UNREADABLE
    return seq == getattr(request, "seq", None)


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


# ═══════════════════════════════════════════════════════════════════════════
# conditions.json — what the campaign knows about the rig (stage 5, S5.F)
# ═══════════════════════════════════════════════════════════════════════════
#
# An attached GUI opens no instrument sessions, so it cannot read a temperature
# the campaign owns: a read is a serial transaction on a bus another process is
# using. The campaign therefore publishes, and this is the publisher. Without it
# the Monitoring tab — the panel an operator opens at 2 a.m. to see whether the
# rig is still at setpoint — is blank for the whole of an unattended run.
#
# Three properties carry the whole design, and each is load-bearing for a reason
# outside monitoring:
#
#   * **Cadence is a ceiling, not a guarantee.** One clock, one in-flight read.
#     A beat that finds a read still running counts itself skipped and returns.
#     Never a queue: one contended read can outlast six beats, and a backlog of
#     six pending instrument reads fired at once is a monitoring knob taking the
#     serial lock away from the campaign that owns it.
#   * **The read never runs on the event loop.** `read_environment` is
#     straight-line synchronous code over up to five blocking driver calls, and
#     `AsyncTempController._with_retry` holds `_serial_lock` for a deadline
#     measured in tens of seconds. Awaited directly, one bad read would stall the
#     loop that also runs the heartbeat *and* the ~1 s `control.json` poll — so a
#     comfort knob could delay an operator's Abort. It goes to a worker thread.
#   * **Visibly stale rather than silently stale.** The file always carries the
#     last *completed* read plus `started_at` / `completed_at` / `read_ms` /
#     `skipped_beats`, and is rewritten on a skipped beat too. A frozen file
#     would otherwise be ambiguous between "the publisher died" and "the read is
#     stuck", which are different problems with different answers.

#: Beside ``events.jsonl`` and ``control.json``, discovered the same way.
CONDITIONS_FILENAME = "conditions.json"

#: Written next to the target and renamed onto it — same directory, so the
#: rename is same-volume and therefore atomic on both NTFS and POSIX. Identical
#: discipline to :func:`write_control_request`, and identically load-bearing:
#: a reader polling this path sees one whole payload or the previous one, never
#: a prefix of either.
CONDITIONS_TMP_SUFFIX = ".tmp"

#: Seconds between conditions beats. ``0`` disables the publisher entirely.
#:
#: **A single slot, never appended.** The events stream deliberately carries no
#: measurements, and a conditions file that grew would be a measurement log by
#: another name — an unversioned second copy of what ``conditions`` rows already
#: hold, written by the process least able to say what it means.
#:
#: On the traffic, honestly: against an *attended* run this is a reduction (the
#: GUI's own poller runs at 2 s), but an attached GUI polls nothing, so against
#: a *headless* run this is net-new Modbus traffic on the same ``_serial_lock``
#: the anneal hold polls. ``0`` is a supported and reasonable value.
DEFAULT_CONDITIONS_POLL_S = 5.0


def conditions_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / CONDITIONS_FILENAME


class ConditionsPublisher:
    """Publish the rig conditions the campaign can see. Never raises.

    Parameters
    ----------
    run_dir
        The run's directory; ``conditions.json`` is written inside it.
    manager
        The instrument manager the campaign owns. Read through
        :func:`~softae.core.conditions_capture.read_environment` unless *read*
        is supplied.
    poll_s
        Beat cadence, a **ceiling** (see below). ``0`` publishes nothing.
    read
        Injection seam: a **synchronous** zero-argument callable returning an
        :class:`~softae.core.conditions_capture.Environment`. Always dispatched
        with :func:`asyncio.to_thread`, never called on the loop.
    now, sleep
        The clock and its sleep, injected for tests.

    Its own task, its own clock
    ---------------------------
    Deliberately not folded into the narrator's heartbeat. The 30 s beat cadence
    is load-bearing for the three-beat "wedged" rule a watcher applies to a
    headless campaign, so sharing one clock would let a monitoring-comfort knob
    silently redefine what *wedged* means: set the conditions cadence to 5 s to
    get a livelier Monitoring tab and the staleness verdict quietly becomes 15 s,
    which a busy event loop reaches without anything being wrong. Two knobs, two
    clocks, two tasks — and neither can move the other.

    Cadence is a ceiling
    --------------------
    One task drives the clock and at most one read is ever outstanding. A beat
    that fires while a read is in flight increments :attr:`skipped_beats`,
    republishes the last completed value with its older stamps, and returns. The
    beats that did not happen are therefore *counted and visible* rather than
    queued: a 33 s read at a 5 s cadence leaves ``skipped_beats == 6`` and fires
    exactly one read, not seven.

    Teardown
    --------
    :meth:`aclose` cancels the clock and detaches from any in-flight read
    without waiting for it — a thread blocked inside a driver retry cannot be
    cancelled, and a campaign teardown must not wait on one. (The interpreter's
    own executor shutdown still joins that thread at loop close; the driver's
    retry deadline bounds it.)
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        manager: Any = None,
        poll_s: float = DEFAULT_CONDITIONS_POLL_S,
        read: Callable[[], Any] | None = None,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.path = conditions_path(self.run_dir)
        self.poll_s = float(poll_s)
        self.skipped_beats = 0
        self._manager = manager
        self._read = read if read is not None else self._read_manager
        self._now = now
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._pending: asyncio.Future[Any] | None = None
        self._in_flight = False
        self._degraded = False

        self._started_at: str | None = None
        self._completed_at: str | None = None
        self._read_ms: int | None = None
        self._read_began = 0.0
        self._env: dict[str, Any] = _null_environment()

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin publishing on the running event loop. Never raises."""
        if self.poll_s <= 0 or self._task is not None:
            return
        try:
            self._task = asyncio.ensure_future(self._publish_loop())
        except Exception:
            self._warn("campaign_conditions_start_failed")

    async def aclose(self) -> None:
        """Stop publishing and let go of any in-flight read. Never raises."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except BaseException:
                # Including the CancelledError we just asked for. Nothing this
                # task can raise is worth propagating into a campaign teardown.
                pass
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.cancel()
            try:
                await pending
            except BaseException:
                pass

    def payload(self) -> dict[str, Any]:
        """The record as it is written — last completed read plus its stamps."""
        return {
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "read_ms": self._read_ms,
            "env": dict(self._env),
            "skipped_beats": self.skipped_beats,
        }

    # ── Internal ────────────────────────────────────────────────────────

    async def _publish_loop(self) -> None:
        # Sleeps first, like the heartbeat and the control watcher: at start-up
        # the campaign is connecting and configuring instruments, and the first
        # thing a monitoring read should not do is join that queue for the
        # serial lock. One cadence of delay costs an attaching operator nothing.
        while True:
            await self._sleep(self.poll_s)
            self._beat()

    def _beat(self) -> None:
        """One beat: start a read, or count this beat as skipped. Never blocks."""
        if self._in_flight:
            self.skipped_beats += 1
            self._write()
            return
        self._in_flight = True
        self._started_at = _stamp()
        self._read_began = self._now()
        try:
            self._pending = asyncio.ensure_future(self._read_and_publish())
        except Exception:
            self._in_flight = False
            self._warn("campaign_conditions_dispatch_failed")

    async def _read_and_publish(self) -> None:
        """Await one threaded read, then publish it. Never raises."""
        env: Any = None
        try:
            env = await asyncio.to_thread(self._read)
        except asyncio.CancelledError:
            self._in_flight = False
            raise
        except Exception:
            # `read_environment` returns nulls rather than raising, so reaching
            # here means something worse — but it means the same thing to a
            # reader, and it must not take the publisher down with it.
            self._warn("campaign_conditions_read_failed")
        finally:
            self._in_flight = False
        self._completed_at = _stamp()
        self._read_ms = int(round(max(0.0, self._now() - self._read_began) * 1000))
        # A failed read publishes nulls rather than the previous value: an old
        # number carrying a fresh stamp is the one lie this file must not tell.
        self._env = dict(env) if isinstance(env, dict) else _null_environment()
        self._write()

    def _read_manager(self) -> Any:
        # Deferred: `conditions_capture` is a leaf, but importing it at module
        # scope would put an instrument-facing import inside the sidecar module
        # every reader of the stream also imports.
        from softae.core.conditions_capture import read_environment

        return read_environment(self._manager)

    def _write(self) -> None:
        """Replace the single slot. Never raises, never appends."""
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + CONDITIONS_TMP_SUFFIX)
            tmp.write_text(json.dumps(self.payload(), indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            self._warn("campaign_conditions_write_failed")

    def _warn(self, event: str) -> None:
        """First failure is a warning; the rest are debug — as above."""
        if self._degraded:
            logger.debug(event, path=str(self.path))
            return
        self._degraded = True
        logger.warning(event, path=str(self.path), exc_info=True)


def _null_environment() -> dict[str, Any]:
    """The five keys, all unread — the shape ``read_environment`` guarantees.

    Named here rather than imported so a reader of ``conditions.json`` always
    gets the same keys even when the capture module never ran.
    """
    return {
        "stage_temp_sp_C": None,
        "chamber_air_C": None,
        "stage_temp_pv_C": None,
        "rh_sp_pct": None,
        "rh_pv_pct": None,
    }


def open_conditions_publisher(
    run_dir: str | Path,
    *,
    manager: Any = None,
    **kwargs: Any,
) -> ConditionsPublisher | None:
    """Build a publisher, or ``None`` if even constructing one fails.

    The mirror of :func:`open_narrator` and :func:`open_control_watcher`, for
    the mirror reason: a campaign that cannot be *watched* must still run.
    ``None`` means "unobserved", never "do not start".
    """
    kwargs.setdefault("poll_s", DEFAULT_CONDITIONS_POLL_S)
    try:
        return ConditionsPublisher(run_dir, manager=manager, **kwargs)
    except Exception:
        logger.warning("campaign_conditions_unavailable", run_dir=str(run_dir),
                       exc_info=True)
        return None
