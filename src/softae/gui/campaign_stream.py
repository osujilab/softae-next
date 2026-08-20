"""What an attached window knows about the run: the event stream, accumulated.

An attached window holds no sessions and no in-process runner, so every question
it can answer about the campaign — which iteration, which phase, is it alive, did
it park — is answered by ``events.jsonl`` and nothing else. This turns that stream
into the four or five short strings the window actually displays, and keeps no Qt
in the process so the rules are testable without a window.

Two things it is deliberately not
---------------------------------
**Not a second vocabulary.** The three liveness bands and "which record is the
newest heartbeat" belong to :mod:`softae.core.campaign_events`; this calls
:func:`~softae.core.campaign_events.liveness` and
:func:`~softae.core.campaign_events.last_heartbeat` rather than re-deriving them
from timestamps, because a watcher that rounds staleness its own way reports a
wedged campaign as healthy for one more interval than the CLI does.

**Not an interpreter of results.** ``suggestion`` and ``result`` payloads carry
the science, and the convergence and scatter buffers already know that
vocabulary. All this takes from them is the iteration number.

The cadence, and why it is not faster
-------------------------------------
:func:`~softae.core.campaign_events.read_events` uses no byte offset, on purpose:
an offset is meaningless across a rotation, and holding the handle open between
polls makes ``os.replace`` fail on Windows and silently disables the 32 MB cap. So
**every poll reads the whole live file** — ~2.5 MB after a week at the shipped
heartbeat cadence, 32 MB at the cap. The interval is therefore a real cost and is
chosen against what can actually change:

* the heartbeat is 30 s (:data:`~softae.core.campaign_events.DEFAULT_HEARTBEAT_S`)
  and the conditions sidecar republishes at 5 s
  (:data:`~softae.core.campaign_events.DEFAULT_CONDITIONS_POLL_S`), so polling
  faster than 5 s re-reads a megabyte to redisplay the same string;
* the tick this replaces for the park indicator was the purge timer's, at 30 s,
  so 5 s is six times *more* responsive than what an owner window has today;
* at 5 s the whole-file cost is 2.5× lower than the instrument poller's 2 s and
  the reads are page-cache warm.

The number lives on :data:`softae.gui.main_window._CAMPAIGN_POLL_MS`, which owns
the timer; this class has no clock of its own.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from softae.core.campaign_events import (
    LIVENESS_LIVE,
    LIVENESS_QUIET,
    LIVENESS_STALE,
    EventCursor,
    last_heartbeat,
    liveness,
    read_events,
)

logger = structlog.get_logger(__name__)

#: How each liveness band reads on a status line. The middle one matters most:
#: "quiet" must not look like a fault, because one missed beat is a busy event
#: loop, and it must not look like health either.
_LIVENESS_WORDS = {
    LIVENESS_LIVE: "live",
    LIVENESS_QUIET: "quiet",
    LIVENESS_STALE: "NO HEARTBEAT",
}


class CampaignStreamView:
    """Poll a run directory; keep the little that a status line needs.

    Parameters
    ----------
    run_dir
        The campaign's run directory, from the rig lock's ``log_path``.
    campaign
        ``(name, run_id)`` from
        :func:`~softae.gui.widgets.rig_owner.campaign_identity`, for labelling.
    reader, now
        Injected for tests. *reader* has
        :func:`~softae.core.campaign_events.read_events`' signature.
    """

    def __init__(
        self,
        run_dir: str,
        *,
        campaign: tuple[str, str] | None = None,
        reader: Callable[..., Any] = read_events,
        now: Callable[[], Any] | None = None,
    ) -> None:
        self.run_dir = str(run_dir)
        self.campaign = campaign
        self._reader = reader
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._cursor: EventCursor | None = None

        self._seen = False
        self._latest: dict[str, Any] | None = None
        self._heartbeat: dict[str, Any] | None = None
        self._state: str | None = None
        self._iteration: int | None = None
        self._park_reason: str | None = None
        self._finished: str | None = None

    # ── Polling ──────────────────────────────────────────────────────────────

    def poll(self) -> None:
        """Read whatever is new and fold it in. Never raises.

        A run directory that has vanished, a stream not written yet, a
        permission failure — all of them are "nothing new". The last known state
        stays on screen, which is the honest answer: the campaign was doing that
        when we last heard, and the liveness word is what says we have not heard
        since.
        """
        try:
            events, self._cursor = self._reader(self.run_dir, cursor=self._cursor)
        except Exception:
            logger.warning("campaign_stream_poll_failed", run_dir=self.run_dir)
            return
        # "Which record is the newest heartbeat" is the stream module's rule, not
        # this one's — asked of each batch, so a poll that carried no beat leaves
        # the previous one standing rather than blanking the phase.
        self._heartbeat = last_heartbeat(events) or self._heartbeat
        for record in events:
            self._absorb(record)

    def _absorb(self, record: dict[str, Any]) -> None:
        self._seen = True
        if record.get("ts"):
            # Kept for liveness, which measures the newest record of any kind:
            # a `result` written two seconds ago proves the process is alive
            # whatever the heartbeat task is doing.
            self._latest = record
        kind = record.get("type")
        if kind == "state":
            self._state = record.get("new") or self._state
        elif kind == "park":
            self._park_reason = str(record.get("reason") or "the campaign parked")
        elif kind == "run_finished":
            self._finished = str(record.get("status") or "finished")
        iteration = record.get("iteration")
        if isinstance(iteration, int):
            self._iteration = iteration

    # ── What the window shows ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return (self.campaign or ("the campaign", ""))[0] or "the campaign"

    @property
    def phase(self) -> str | None:
        """The campaign's last named phase, off its newest heartbeat."""
        beat = self._heartbeat
        return None if beat is None else (beat.get("phase") or None)

    @property
    def phase_age_s(self) -> Any:
        beat = self._heartbeat
        return None if beat is None else beat.get("phase_age_s")

    @property
    def park_reason(self) -> str | None:
        """Why the campaign parked, or ``None``.

        Latched, like every other park in this application: a park means a human
        may have to reach into the rig, and nothing in the stream retracts one.
        """
        return self._park_reason

    @property
    def finished(self) -> str | None:
        """The ``run_finished`` status, or ``None`` while the run is going."""
        return self._finished

    def liveness(self) -> str:
        """``live`` / ``quiet`` / ``stale``, by the stream's own three-beat rule."""
        return liveness([self._latest] if self._latest else [], now=self._now())

    def auto_status(self) -> str:
        """The campaign's own progress — for the sidebar's Autonomous slot.

        A campaign *is* an autonomous run, so this slot is the one place the
        remote run belongs. The string is composed the same way the in-process
        signal's is: a short phrase, no markup, idle words avoided so the label
        does not grey itself out while a run is going.
        """
        if not self._seen:
            return f"{self.name} — no records read yet"
        if self._finished:
            return f"{self.name} finished ({self._finished})"
        bits = [self.name]
        if self._state:
            bits.append(str(self._state))
        elif self.phase:
            bits.append(str(self.phase))
        if self._iteration is not None:
            bits.append(f"iter {self._iteration}")
        bits.append(_LIVENESS_WORDS.get(self.liveness(), self.liveness()))
        return " · ".join(bits)

    def ht_status(self) -> str:
        """Why the HT slot is idle — for the sidebar's HT Experiment slot.

        An attached window cannot run an HT experiment: every workflow goes
        through ``WorkflowExecutor``, which refuses while another process holds
        the rig lock. A bare "Idle" is true and useless — it is the same word an
        operator sees on a free rig, at the moment the difference matters most —
        so this says *why*, and follows the stream: when the run finishes, the
        sentence becomes the one that names the way back in.
        """
        if self._finished:
            return (
                f"idle — {self.name} has finished; Init → Connect All takes the rig"
            )
        return f"idle — {self.name} holds the rig"
