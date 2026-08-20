"""The E-Stop escalation ladder — how an attached window reaches a rig it cannot drive.

An attached window holds no instrument sessions, so its E-Stop cannot park
anything: :func:`~softae.core.safe_park.safe_park` against a manager with nothing
connected files every subsystem under ``skipped``, raises nothing, and reports
``ok is True`` having sent not one byte to the rig. The stop has to reach the
*campaign*, which does hold the sessions. That is a request, it travels by file,
and it can fail in ways a red button must not hide — so it is a **ladder**, with
each rung's budget stated and each rung's answer shown.

The four rungs
--------------
======  =============================================================  ==========
1       ``abort`` written to ``control.json``                          automatic
2       wait for a ``control_ack`` carrying our ``seq``                T1 = 15 s
3       wait for ``park`` **and** ``safe_park`` records                T2 = 120 s
4       break the lock, terminate the recorded PID, connect, park      **never**
======  =============================================================  ==========

**The timeouts bound how long this waits before *offering* the next rung. They
never perform one.** That is the single most important property in this module.
Auto-advancing would convert a slow park — a long EIS sweep in flight, a
contended serial bus, a heater taking its full Modbus retry window — into a
killed process mid-dispense, which is precisely the un-parked death the whole
control channel exists to avoid. So every timeout lands in an ``offered_*``
state, and only :meth:`EstopLadder.advance` and :meth:`EstopLadder.take_over`
move out of one; neither is reachable from :meth:`EstopLadder.poll`.

The budgets, and their arithmetic
---------------------------------
**T1 = 15 s** is the 1 s control poll
(:data:`~softae.core.campaign_events.DEFAULT_CONTROL_POLL_S`) plus slack. No ack
by then does not mean "slow"; it means the watcher is not running — wrong run
directory, wedged loop, or a dead process behind a live-looking lock.

**T2 = 120 s** is 1 s of control poll, up to a 60 s humidity hold poll
(``DEFAULT_RH_POLL_S``), and ``safe_park`` itself — four subsystems at up to 10 s
of Modbus retry each — which sums to about 101 s. 120 s is *"it should be here by
now"*, not *"it has failed"*, and the difference is why rung 4 is a question and
not a consequence.

Why the cross-host limit is declared before the press
-----------------------------------------------------
:attr:`softae.core.run_lock.RunLock.is_alive` reports a lock from another host as
**always alive**, deliberately — guessing "dead" would hand the rig to two
machines. So against a cross-host holder there is no PID this process may check,
no PID it may kill, and the sessions are on a machine it cannot reach. Rungs 3
and 4 are unreachable, and :attr:`EstopLadder.reachable_rungs` says so at
construction. A red button whose guarantee silently varies by launch location is
worse than one that admits its limit.

Rung 4 is a CLAUDE.md class-4 act
---------------------------------
It terminates a process. :func:`terminate_pid` takes an **integer**, never a
name: on this machine the operator's GUI, the VS Code extension hosts and pytest
all run the same interpreter out of the same virtualenv, so no image-name or
path filter can separate them, and an image-name kill has already taken the
operator's live GUI down twice. PID reuse remains an unmitigated limit
(``run_lock`` says so), which is why the caller shows ``started_at`` and ``what``
as plausibility evidence and requires the operator to type the number.

And rung 4 **acquires before it parks**: ``break_run_lock`` → terminate → claim →
``connect_all`` → ``safe_park``. A park issued before the connect would command
nothing and report ``ok`` for it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from softae.core.campaign_events import (
    EventCursor,
    ack_answers_request,
    read_events,
    write_control_request,
)
from softae.core.safe_park import SafeParkResult, safe_park_async

logger = structlog.get_logger(__name__)

#: Rung 2's budget: the campaign's 1 s control poll plus slack. See the module
#: docstring — an expiry here is evidence about the *watcher*, not about speed.
T1_ACK_S = 15.0

#: Rung 3's budget: control poll + a 60 s hold poll + ``safe_park``'s own Modbus
#: retry window ≈ 101 s, rounded up. "It should be here by now", not "it failed".
T2_PARK_S = 120.0

#: The reason stamped on the abort request, and on the park rung 4 performs.
ESTOP_REASON = "operator E-Stop"

#: Rung 1 has been written; waiting for the campaign to acknowledge it.
STATE_AWAITING_ACK = "awaiting_ack"
#: Acknowledged, and waiting for the park the abort should produce.
STATE_AWAITING_PARK = "awaiting_park"
#: T1 expired with no ack. Rung 3 is *offered*, not entered.
STATE_OFFERED_WAIT = "offered_wait"
#: T2 expired with no park. Rung 4 is *offered*, not performed.
STATE_OFFERED_TAKEOVER = "offered_takeover"
#: The campaign parked. Terminal, and the outcome this ladder wants.
STATE_PARKED = "parked"
#: Acknowledged, and this machine can go no further — the cross-host terminal.
STATE_ACKED_ONLY = "acked_only"
#: No rung remains that this process may reach. The cross-host dead end.
STATE_EXHAUSTED = "exhausted"
#: Rung 4 ran. Terminal.
STATE_TAKEN_OVER = "taken_over"
#: Nothing pressed yet.
STATE_IDLE = "idle"

#: What the operator is told in each state, and what the next act would be.
#: Written here rather than in the dialog so the sentences are testable and so a
#: second surface cannot invent a softer one.
STATE_NOTES: dict[str, str] = {
    STATE_IDLE: "Nothing has been requested yet.",
    STATE_AWAITING_ACK: (
        "Abort written. Waiting for the campaign to acknowledge it — it polls "
        "once a second."
    ),
    STATE_AWAITING_PARK: (
        "The campaign acknowledged the abort. Waiting for it to park the rig: a "
        "step already running finishes first, so this is normally where the time "
        "goes."
    ),
    STATE_OFFERED_WAIT: (
        "NO ACKNOWLEDGEMENT. The campaign has not read the request, which means "
        "the watcher is not running — a wrong run directory, a wedged loop, or a "
        "dead process behind a live-looking lock. Waiting longer is still "
        "reasonable; taking the rig is the other answer."
    ),
    STATE_OFFERED_TAKEOVER: (
        "NO PARK RECORD. The campaign was asked to stop and has not reported "
        "parking. Take the rig only if it is wedged rather than working — check "
        "the elapsed time and the last event line first."
    ),
    STATE_PARKED: "The campaign parked the rig. Its own sessions issued the stop.",
    STATE_ACKED_ONLY: (
        "The campaign acknowledged the abort. Nothing further can be done from "
        "this machine — the sessions are on another host."
    ),
    STATE_EXHAUSTED: (
        "No acknowledgement, and no rung left: the campaign is on another host, "
        "so this process can neither check its PID nor reach its instruments. "
        "Stop it at that machine."
    ),
    STATE_TAKEN_OVER: "The rig was taken from the campaign and parked from here.",
}

#: Shown instead of :data:`STATE_NOTES`' idle line when something holds the rig
#: but publishes no control channel — a bench sequence, an HT workflow. Rungs 1–3
#: do not exist for it, so the ladder opens at its last rung rather than
#: pretending to have asked.
NOTE_NO_CHANNEL = (
    "Whatever holds the rig is not a campaign, so it reads no control file and "
    "there is nothing to request: rungs 1-3 do not exist here. The only act left "
    "is taking the rig, which is manual and confirmed — it is not something this "
    "window will do on its own."
)

#: …and when that holder is also on another machine, there is no act left at all.
NOTE_NOTHING_REACHABLE = (
    "The rig is held by a non-campaign process on another machine. This window "
    "can neither ask it to stop nor reach its instruments. Stop it at that "
    "machine."
)

#: The states a timeout can put the ladder into. Nothing leaves one of these
#: except an operator act — asserted by the tests, and the reason they are named.
OFFERED_STATES = (STATE_OFFERED_WAIT, STATE_OFFERED_TAKEOVER)

#: Terminal states: the ladder stops polling the clock in them.
TERMINAL_STATES = (
    STATE_PARKED, STATE_ACKED_ONLY, STATE_EXHAUSTED, STATE_TAKEN_OVER)


def reachable_rungs(*, run_dir: str | None, cross_host: bool) -> tuple[int, ...]:
    """Which rungs a press can ever enter, from the two facts that decide it.

    A module function rather than only a property, because the **button** must
    answer this before it is pressed and the **ladder** must answer it while it
    is climbing. Two copies of the rule is how a red button comes to promise a
    rung the machinery behind it cannot take.

    ================================  ==============  ===========================
    Holder                            Rungs           Why
    ================================  ==============  ===========================
    a campaign, this host             ``(1,2,3,4)``   the full ladder
    a campaign, another host          ``(1,2)``       its lock reads as alive by
                                                      design, so there is no PID
                                                      here to check or stop, and
                                                      the sessions are on a
                                                      machine we cannot reach
    not a campaign, this host         ``(4,)``        it reads no control file,
                                                      so there is nothing to ask
                                                      — taking the rig is the
                                                      only act left, and it is
                                                      manual and confirmed
    not a campaign, another host      ``()``          neither reachable
    ================================  ==============  ===========================
    """
    requestable = run_dir is not None
    if cross_host:
        return (1, 2) if requestable else ()
    return (1, 2, 3, 4) if requestable else (4,)


def terminate_pid(pid: int) -> bool:
    """Terminate exactly one process, **by number**. Whether it was signalled.

    The signature is the safety property: this takes an ``int``, so there is no
    argument that could name ``python.exe``. ``taskkill /IM``, ``pkill``,
    ``killall`` and ``Get-Process <name> | Stop-Process`` are forbidden here
    without exception — on this machine the operator's GUI, the VS Code extension
    hosts and pytest share one interpreter and one virtualenv, and only the
    command line separates them.

    Two refusals, both unconditional:

    * a PID at or below zero is not a process, and on POSIX a non-positive
      argument to ``kill`` is a *broadcast* — the one way this function could
      become the thing it exists to prevent;
    * this process's own PID, because the window asking is not the one to stop.
    """
    pid = int(pid)
    if pid <= 0:
        logger.warning("estop_terminate_refused", pid=pid,
                       msg="not a process id — nothing was signalled")
        return False
    if pid == os.getpid():
        logger.warning("estop_terminate_refused", pid=pid,
                       msg="that is this window; refusing to terminate ourselves")
        return False

    logger.warning("estop_terminate_pid", pid=pid,
                   msg="operator took the rig — terminating the recorded PID only")
    if os.name == "nt":
        import ctypes

        PROCESS_TERMINATE = 0x0001
        kernel32 = ctypes.windll.kernel32       # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if not handle:
            logger.warning("estop_terminate_unreachable", pid=pid,
                           msg="could not open the process — already gone, or "
                               "owned by another user")
            return False
        try:
            return bool(kernel32.TerminateProcess(handle, 1))
        finally:
            kernel32.CloseHandle(handle)

    import signal

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        logger.warning("estop_terminate_failed", pid=pid, exc_info=True)
        return False
    return True


@dataclass(frozen=True)
class TakeoverResult:
    """What rung 4 actually did — or why it did nothing.

    ``refused`` is the important field: it is set whenever the gate declined, and
    a gate that declines silently is indistinguishable from one that fired.
    """

    refused: str = ""
    broke_lock: bool = False
    terminated: bool = False
    connected: bool = False
    claimed: bool = False
    park: SafeParkResult | None = None
    #: Every step attempted, in order. The ordering property rung 4 depends on —
    #: connect *before* park — is asserted against this.
    steps: tuple[str, ...] = field(default_factory=tuple)

    @property
    def performed(self) -> bool:
        return not self.refused

    def headline(self) -> tuple[str, bool]:
        """The park's verdict, or the refusal. ``(text, severe)``.

        Delegated to :meth:`~softae.core.safe_park.SafeParkResult.headline` so
        the takeover's report and every other park's are chosen by one rule — in
        particular so a takeover whose connect failed is headed *"NOTHING WAS
        COMMANDED"* rather than reported as a stop.
        """
        if self.refused:
            return f"The rig was NOT taken: {self.refused}", True
        if self.park is None:
            return "The rig was taken, but no park was attempted.", True
        return self.park.headline()

    def describe(self) -> str:
        """What each step did, then the park's own account of itself."""
        if self.refused:
            return self.refused
        lines = [
            f"rig lock broken: {'yes' if self.broke_lock else 'no'}",
            f"campaign process terminated: {'yes' if self.terminated else 'no'}",
            f"rig claimed by this window: {'yes' if self.claimed else 'no'}",
            f"instrument sessions opened: {'yes' if self.connected else 'NO'}",
        ]
        if self.park is not None:
            lines.append("")
            lines.append(self.park.describe())
        return "\n".join(lines)


class EstopLadder:
    """One press of an attached window's E-Stop, and everything it can escalate to.

    Headless on purpose — no Qt anywhere in here — so the rule that matters most
    ("no timer path reaches the kill") is testable on an injected clock without a
    window, and so a script could reuse the same guarantee.

    Parameters
    ----------
    run_dir
        The campaign's run directory, from the rig lock's ``log_path``. ``None``
        when something holds the rig but publishes no control channel: rungs 1–3
        are then unreachable and :attr:`reachable_rungs` says so.
    lock
        The :class:`~softae.core.run_lock.RunLock` behind the attach decision.
        Rung 4's PID comes from here and from nowhere else.
    cross_host
        Whether the holder is on another machine. Forced by the caller from
        ``lock.host`` versus ``socket.gethostname()`` — the same comparison
        ``RunLock.is_alive`` makes — rather than re-derived here, so the button's
        label and this ladder cannot disagree about what a press can reach.
    manager
        This process's instrument manager, for rung 4 only. Rungs 1–3 never
        touch it.
    writer, reader, clock, breaker, terminator, claimer, connector, parker
        Every collaborator with a consequence is injected, which is what lets a
        test drive the whole ladder past both timeouts and then assert that
        *breaker* and *terminator* were never called.
    """

    def __init__(
        self,
        run_dir: str | None,
        *,
        lock: Any = None,
        cross_host: bool = False,
        campaign: tuple[str, str] | None = None,
        manager: Any = None,
        writer: Callable[..., Any] = write_control_request,
        reader: Callable[..., Any] = read_events,
        clock: Callable[[], float] = time.monotonic,
        breaker: Callable[[], Any] | None = None,
        terminator: Callable[[int], bool] = terminate_pid,
        claimer: Callable[[Any], Any] | None = None,
        connector: Callable[[], Any] | None = None,
        parker: Callable[..., Any] = safe_park_async,
        requested_by: str | None = None,
    ) -> None:
        self.run_dir = str(run_dir) if run_dir else None
        self.lock = lock
        self.cross_host = bool(cross_host)
        self.campaign = campaign
        self._manager = manager
        self._writer = writer
        self._reader = reader
        self._clock = clock
        self._breaker = breaker
        self._terminator = terminator
        self._claimer = claimer
        self._connector = connector
        self._parker = parker
        self._requested_by = requested_by or f"softae GUI E-Stop (pid {os.getpid()})"

        self._state = STATE_IDLE
        self._cursor: EventCursor | None = None
        self._request: Any = None
        self._rung_started_at: float | None = None
        self._ack: dict[str, Any] | None = None
        self._saw_park = False
        self._saw_safe_park = False
        self._park_record: dict[str, Any] | None = None
        self._last_record: dict[str, Any] | None = None

    # ── What a press can reach, known before it happens ──────────────────────

    @property
    def reachable_rungs(self) -> tuple[int, ...]:
        """Which rungs this ladder may ever enter. Fixed at construction.

        The rule is :func:`reachable_rungs`, shared with the button so the label
        the operator read and the ladder they are now watching cannot disagree.
        """
        return reachable_rungs(run_dir=self.run_dir, cross_host=self.cross_host)

    @property
    def state(self) -> str:
        return self._state

    @property
    def note(self) -> str:
        """The operator-facing sentence for the current state."""
        if self._state == STATE_IDLE and self.run_dir is None:
            return NOTE_NO_CHANNEL if self.reachable_rungs else NOTE_NOTHING_REACHABLE
        return STATE_NOTES.get(self._state, self._state)

    @property
    def request(self) -> Any:
        """The abort request rung 1 wrote, or ``None``."""
        return self._request

    @property
    def ack(self) -> dict[str, Any] | None:
        """The ``control_ack`` that answered rung 1, or ``None``."""
        return self._ack

    @property
    def park_record(self) -> dict[str, Any] | None:
        """The campaign's ``safe_park`` record, once it lands."""
        return self._park_record

    @property
    def elapsed_s(self) -> float:
        """How long the current rung has been waiting.

        Shown throughout, because *elapsed time beside the newest event line* is
        what lets an operator tell "still working" from "wedged" — the same
        discipline CLAUDE.md §5 requires of a human deciding whether to kill a
        process, offered to the person who has to decide it here.
        """
        if self._rung_started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._rung_started_at)

    @property
    def budget_s(self) -> float | None:
        """The current rung's budget, or ``None`` where waiting is not timed."""
        if self._state == STATE_AWAITING_ACK:
            return T1_ACK_S
        if self._state == STATE_AWAITING_PARK:
            return T2_PARK_S
        return None

    @property
    def last_event_line(self) -> str:
        """The newest record of ``events.jsonl``, on one line.

        Deliberately the *raw* record rather than a gloss: this is shown next to
        the elapsed clock so a wedged campaign can be told from a busy one, and
        an interpretation layer is exactly what could hide the difference.
        """
        record = self._last_record
        if record is None:
            return "(no events read yet)"
        rest = {k: v for k, v in record.items() if k not in ("ts", "type")}
        try:
            detail = json.dumps(rest, default=str)
        except Exception:              # pragma: no cover - defensive
            detail = str(rest)
        if len(detail) > 200:
            detail = detail[:197] + "…"
        return f"{record.get('ts', '?')}  {record.get('type', '?')}  {detail}"

    @property
    def may_advance(self) -> bool:
        """Whether the operator may enter rung 3 from here."""
        return self._state == STATE_OFFERED_WAIT and 3 in self.reachable_rungs

    @property
    def may_take_over(self) -> bool:
        """Whether the operator may perform rung 4 from here.

        ``True`` in exactly two places: after rung 3's budget expired, and from
        the outset when there is no control channel at all (nothing to request,
        so nothing to escalate *through*). Never as a consequence of a poll.
        """
        if 4 not in self.reachable_rungs:
            return False
        if self._state == STATE_OFFERED_TAKEOVER:
            return True
        return self._state == STATE_IDLE and self.run_dir is None

    @property
    def takeover_pid(self) -> int:
        """The one PID rung 4 may terminate. ``0`` when there is none."""
        try:
            return int(getattr(self.lock, "pid", 0) or 0)
        except (TypeError, ValueError):
            return 0

    # ── Rung 1 ───────────────────────────────────────────────────────────────

    def start(self) -> Any:
        """Write the abort and begin waiting for the ack. Raises if the write fails.

        Raising rather than degrading is
        :func:`~softae.core.campaign_events.write_control_request`'s deliberate
        contract and it is right here too: the write is the operator's only
        evidence that a red button did anything, so a failure is the one failure
        they must see.

        The cursor is snapshotted **before** the write, so an ack already on disk
        — the previous operator's, or an answer to the CLI — can never be
        mistaken for the answer to this press. The same snapshot is why rung 3
        cannot be satisfied by history: this reads the run's whole past and keeps
        **only the newest line for display**, deliberately not the ``park`` and
        ``safe_park`` flags. A campaign that parked once already, recovered and
        carried on would otherwise answer rung 3 the instant it was entered, with
        a record written hours before the operator's press.
        """
        if self.run_dir is None:
            raise RuntimeError(
                "There is no campaign to ask: whatever holds the rig published "
                "no run directory, so there is nowhere to write an abort.")
        events, self._cursor = self._reader(self.run_dir)
        for record in events:
            if record.get("ts"):
                self._last_record = record
        self._request = self._writer(
            self.run_dir, "abort", reason=ESTOP_REASON,
            requested_by=self._requested_by)
        logger.warning("estop_abort_requested", run_dir=self.run_dir,
                       seq=getattr(self._request, "seq", None))
        self._enter(STATE_AWAITING_ACK)
        return self._request

    # ── Rungs 2 and 3: waiting, and the offers a budget produces ─────────────

    def poll(self) -> str:
        """Fold in whatever is new, apply the current rung's budget, return the state.

        **This method cannot reach rung 4, and cannot enter rung 3 either.** Both
        transitions out of an ``offered_*`` state are operator acts
        (:meth:`advance`, :meth:`take_over`); all a budget expiry does here is
        change which offer is on screen. Never raises: a run directory that
        vanished mid-wait leaves the last known state standing, which is honest —
        the campaign was doing that when we last heard, and the elapsed clock is
        what says we have not heard since.
        """
        if self._state in TERMINAL_STATES or self._state == STATE_IDLE:
            return self._state
        if self.run_dir is not None:
            try:
                events, self._cursor = self._reader(self.run_dir, cursor=self._cursor)
            except Exception:
                logger.warning("estop_ladder_poll_failed", run_dir=self.run_dir)
                events = []
            self._absorb(events)

        if self._saw_park and self._saw_safe_park:
            # The campaign parked — whichever rung we were on, and whichever
            # offer was on screen. **An offer withdraws itself the moment the
            # thing it exists for happens.** A takeover offer left standing after
            # a late park is an invitation to kill a process that has already
            # done what it was asked, which is the same mistake as auto-advancing
            # with the operator's hand on the button instead of a timer's.
            return self._enter(STATE_PARKED)

        if self._state == STATE_AWAITING_ACK:
            if self._ack is not None:
                self._enter(STATE_AWAITING_PARK if 3 in self.reachable_rungs
                            else STATE_ACKED_ONLY)
            elif self.elapsed_s >= T1_ACK_S:
                self._enter(STATE_OFFERED_WAIT if 3 in self.reachable_rungs
                            else STATE_EXHAUSTED)
        elif self._state == STATE_AWAITING_PARK:
            if self.elapsed_s >= T2_PARK_S:
                self._enter(STATE_OFFERED_TAKEOVER)
        elif self._state == STATE_OFFERED_WAIT and self._ack is not None:
            # The ack arrived late, while the offer was on screen. That answers
            # the question the offer was asking, so the offer withdraws itself.
            self._enter(STATE_AWAITING_PARK)
        return self._state

    def advance(self) -> str:
        """Operator act: take the offered rung 3 and keep waiting for the park."""
        if not self.may_advance:
            return self._state
        self._enter(STATE_AWAITING_PARK)
        return self._state

    def _absorb(self, events: list[dict[str, Any]]) -> None:
        for record in events:
            if record.get("ts"):
                self._last_record = record
            kind = record.get("type")
            if kind == "control_ack" and self._ack is None and self._request is not None:
                if ack_answers_request(record, self._request):
                    self._ack = record
                    logger.warning("estop_abort_acknowledged",
                                   outcome=record.get("outcome"))
            elif kind == "park":
                self._saw_park = True
            elif kind == "safe_park":
                self._saw_safe_park = True
                self._park_record = record

    def _enter(self, state: str) -> str:
        if state != self._state:
            logger.info("estop_ladder_state", was=self._state, now=state)
            self._state = state
            self._rung_started_at = self._clock()
        return self._state

    # ── Rung 4: never automatic, and never by image name ─────────────────────

    async def take_over(self, *, confirmed: bool) -> TakeoverResult:
        """Break the lock, terminate the recorded PID, connect, park. In that order.

        Three gates, and each refuses rather than proceeding:

        * the ladder must be *offering* rung 4 (:attr:`may_take_over`), which no
          poll can bring about;
        * *confirmed* must be ``True`` — the caller's evidence that a human read
          ``lock.describe()`` and typed the PID back;
        * the lock must carry a PID this process is allowed to signal, which
          :func:`terminate_pid` checks again on its own account.

        **The connect precedes the park and that ordering is the point.** A park
        against a manager with nothing connected files every subsystem under
        ``skipped``, raises nothing, and reports ``ok`` — so parking first would
        produce a reassuring result about a rig this process had not yet spoken
        to. The claim sits between them because ``rig_session`` rules that
        whoever opens the ports is who holds the lock; it is best-effort, because
        bookkeeping must never be what stops a stop.
        """
        if not self.may_take_over:
            return TakeoverResult(
                refused="the ladder is not offering it — nothing was touched.")
        if not confirmed:
            return TakeoverResult(refused="the operator did not confirm it.")

        pid = self.takeover_pid
        steps: list[str] = []
        logger.warning(
            "estop_takeover_begin", pid=pid,
            what=getattr(self.lock, "what", ""),
            started_at=getattr(self.lock, "started_at", ""),
            msg="operator took the rig from the campaign after the E-Stop ladder "
                "reached its last rung")

        broke = self._call(self._breaker, steps, "break_lock") is not None
        terminated = bool(self._call(
            self._terminator, steps, "terminate", pid) if pid else False)
        claimed = self._call(self._claimer, steps, "claim", self._manager) is not None

        connected = False
        try:
            steps.append("connect")
            await self._connect()
            connected = True
        except Exception:
            # Not fatal: some ports may have opened, and a park across a
            # partially connected rig is worth strictly more than no park.
            # `SafeParkResult` reports exactly how far it got.
            logger.warning("estop_takeover_connect_failed", exc_info=True)

        steps.append("park")
        park = await self._parker(
            self._manager, reason=f"{ESTOP_REASON} — takeover", retract_head=None)

        self._enter(STATE_TAKEN_OVER)
        return TakeoverResult(
            broke_lock=broke, terminated=terminated, connected=connected,
            claimed=claimed, park=park, steps=tuple(steps))

    async def _connect(self) -> None:
        if self._connector is not None:
            result = self._connector()
        else:
            result = self._manager.connect_all()
        if hasattr(result, "__await__"):
            await result

    def _call(self, fn: Any, steps: list[str], name: str, *args: Any) -> Any:
        """Run one best-effort takeover step, recording that it was attempted.

        ``None`` on absence or failure. Every step here is repair, not the stop
        itself: the park at the end is the safety act, and it must not be
        skippable because a lock file would not unlink.
        """
        if fn is None:
            return None
        steps.append(name)
        try:
            return fn(*args)
        except Exception:
            logger.warning("estop_takeover_step_failed", step=name, exc_info=True)
            return None


def default_ladder_collaborators() -> dict[str, Any]:
    """The production wiring for rung 4, resolved late.

    Imported inside the function rather than at module scope so this module stays
    importable — and testable — without dragging in the lock and session machinery
    a test has no use for.
    """
    from softae.core.rig_session import claim_rig_session
    from softae.core.run_lock import break_run_lock

    return {"breaker": break_run_lock, "claimer": claim_rig_session}
