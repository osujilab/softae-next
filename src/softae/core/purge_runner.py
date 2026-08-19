"""Performing a due purge, and the idle-rest state it happens in (P8).

`core.purge` decides *when* and *how much*. This performs it — and it is the
**only mechanism in the system that moves hardware with nobody asking it to**,
which shapes every decision here.

Shipped inert. ``[purge] actuate`` defaults to **false**: the schedule runs, the
consumption is billed to the runway projection, and every due purge is logged as
a dry run. Nothing moves until an operator turns it on at the bench. That is the
same posture the measurement quality gate ships in, for a stronger reason — this
one has a physical effect rather than a data one.

**Idle rest.** The rig's resting state between runs is *stage at the flush
position, head lowered*, tip immersed. That protects the dispenser tip, and it is
where a purge can happen with no stage motion at all.

⚠️ **Idle rest is the opposite of a safe park and must never be confused with
it.** `safe_park` retracts the head because a fault occurred and a human may
reach in; idle rest lowers it because nothing is wrong. Entering idle rest while
a park reason is outstanding would take a rig that was deliberately made safe and
quietly put its head back down — so that transition is gated, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

# `RigPose` and `classify_pose` were defined here and now live in
# `core/rig_pose.py`, because a *third* consumer appeared — Pause's "next safe
# interruption" — and a control path must not import the actuation path to ask
# where the rig is. Re-exported so this module's own code, its callers and
# `tests/test_purge_runner.py` see no change: it was a move, not a fork, and one
# definition of quiescent is the whole point.
from softae.core.rig_pose import (  # noqa: F401  (re-exported)
    FLUSH_TOLERANCE_MM,
    QUIESCENT_POSES,
    RigPose,
    _flush_position,
    classify_pose,
    safe_to_interrupt,
)

logger = structlog.get_logger(__name__)

#: Escalate to a durable alert once a purge has been owed this long. Deferral is
#: safe and expected — a long campaign holds the rig for hours — but a deferral
#: that never resolves is functionally the same as purging being switched off,
#: and at info level the two look identical in the log. Four intervals at the
#: default 15 min cadence.
DEFAULT_DEFER_ALERT_S = 3600.0


class IdleRestState:
    """Whether the rig is *known* to be resting at the flush station.

    Tracked explicitly and never inferred from hardware, because the one signal
    that looks like it would work does not: **the head is down at idle rest and
    also down mid-cast**, so head position alone cannot tell the two apart. A
    purge fired on that confusion dispenses into an electrode well.

    Defaults to *not* at rest, so a runner given no state refuses idle purges
    until something has actually brought the rig to rest. The unsafe direction
    requires an explicit act; the safe direction is the default.
    """

    def __init__(self, at_rest: bool = False) -> None:
        self._at_rest = bool(at_rest)

    @property
    def at_rest(self) -> bool:
        return self._at_rest

    def mark_entered(self) -> None:
        self._at_rest = True

    def mark_left(self) -> None:
        self._at_rest = False


@dataclass
class PurgeOutcome:
    """What a purge attempt did, or why it did nothing."""

    performed: bool = False
    dry_run: bool = False
    volumes_uL: dict[int, float] = field(default_factory=dict)
    skipped_reason: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def total_uL(self) -> float:
        return sum(self.volumes_uL.values())

    def summary(self) -> str:
        if self.skipped_reason:
            return f"purge skipped: {self.skipped_reason}"
        if self.dry_run:
            return (
                f"purge DUE ({self.total_uL:.0f} µL) — not actuating "
                f"([purge] actuate is off)"
            )
        if self.performed:
            state = "ok" if not self.errors else f"with errors {self.errors}"
            return f"purged {self.total_uL:.0f} µL {state}"
        return "no purge due"


class PurgeRunner:
    """Performs due purges, subject to every precondition being satisfied.

    Preconditions are checked *here* rather than trusted from the caller, because
    the caller may be a background timer with no idea what the rig is doing.
    """

    #: This runner can actually dispense; see :class:`NullPurgeRunner` for why
    #: callers ask rather than testing for ``None``.
    performs_purges = True

    def __init__(
        self,
        manager: Any,
        scheduler: Any,
        *,
        waste_ledger: Any = None,
        park_reason: Any = None,
        idle_rest: "IdleRestState | None" = None,
        activity: Any = None,
        flush_xy: "tuple[float, float] | None" = None,
        flush_rate_uL_min: float = 200.0,
        data_store: Any = None,
        defer_alert_after_s: "float | None" = DEFAULT_DEFER_ALERT_S,
    ) -> None:
        self._manager = manager
        self._scheduler = scheduler
        self._waste = waste_ledger
        #: Callable returning a park reason (truthy = parked). A parked rig is
        #: parked for a reason; resuming unprompted actuation would undo it.
        self._park_reason = park_reason
        #: Tracks the idle-rest *intent*; the authority on whether a purge may
        #: happen is now :func:`classify_pose`, read from the hardware.
        self.idle_rest = idle_rest if idle_rest is not None else IdleRestState()
        #: :class:`~softae.core.rig_activity.RigActivity`, or ``None`` to skip
        #: the ownership check (only sensible when the caller *is* the owner).
        self._activity = activity
        #: Resolved once rather than per tick; ``None`` defers to the calibration.
        self._flush_xy = tuple(flush_xy) if flush_xy is not None else None
        self._flush_rate = float(flush_rate_uL_min)
        self._data_store = data_store
        #: Escalate once a purge has been owed this long; ``None`` disables it.
        self._defer_alert_after_s = defer_alert_after_s
        self._defer_alerted = False

        from softae.core.rig_activity import PURGE_INSTRUMENTS

        #: What a purge occupies — the syringe, plus the stage in case it has to
        #: travel to the basin first.
        self._purge_instruments = PURGE_INSTRUMENTS

    def _pose(self) -> "RigPose":
        return classify_pose(self._manager, flush_xy=self._flush_xy)

    def _note_deferred(self, due: Any, *, reason: str) -> None:
        """Escalate a line that has been deferred for too long.

        Deferral is safe; *silent* deferral is not. A rig busy with a long
        campaign refuses every tick for hours, and at info level that reads
        identically to a healthy 30-second skip — so the one failure mode that
        actually looks like success gets its own durable alert.

        Raised once per crossing, not once per tick, or an overnight run would
        bury the alert table under thousands of identical rows.
        """
        threshold = self._defer_alert_after_s
        if threshold is None or due.overdue_s < threshold:
            self._defer_alerted = False
            return
        if self._defer_alerted:
            return
        self._defer_alerted = True

        from softae.core.alerts import WARNING, Alert, raise_alert

        hours = due.overdue_s / 3600.0
        raise_alert(
            Alert(
                kind="purge",
                severity=WARNING,
                message=(
                    f"Anti-clog purge has been deferred for {hours:.1f} h "
                    f"({reason}). Lines carrying particulate stock may be "
                    f"clogging. Purging resumes automatically once the rig is "
                    f"free — this is a warning, not a stopped purge."
                ),
                details={"overdue_s": round(due.overdue_s, 1), "reason": reason,
                         "volumes_uL": dict(due.volumes_uL)},
            ),
            data_store=self._data_store,
        )

    # ── Preconditions ────────────────────────────────────────────────────

    def _blocking_reason(self, *, allow_positioning: bool,
                         owns_rig: bool) -> str | None:
        """Why a purge must not happen now, or ``None`` if it may."""
        if self._park_reason is not None:
            try:
                reason = self._park_reason()
            except Exception:
                reason = "park state unknown"
            if reason:
                return f"rig is parked ({reason})"

        # Ownership, not head position. Once purging can position the rig
        # itself, "the head is up" no longer implies "nothing else is running" —
        # a workflow mid-anneal has a raised head and would be very surprised to
        # find the stage somewhere else.
        #
        # `owns_rig` is an in-run caller saying "I *am* the claim holder" — the
        # anneal poll hook, or the executor's concurrent purge window. Without
        # it the purge would refuse against the very run that invoked it, and
        # nothing could ever purge mid-campaign.
        if not owns_rig and self._activity is not None:
            # Instrument-scoped: a step occupying only the potentiostat does not
            # stop the syringe purging. A whole-rig claim still blocks everything.
            blocker = self._activity.conflicts(self._purge_instruments)
            if blocker is not None:
                return f"rig is in use ({blocker})"

        return None

    def _pose_blocking(self, *, allow_positioning: bool) -> str | None:
        """Why the rig's current pose forbids a purge, or ``None``."""
        pose = self._pose()
        if pose is RigPose.AT_FLUSH:
            return None
        if pose is RigPose.HEAD_UP:
            if allow_positioning:
                return None
            return "head is raised and this caller does not permit repositioning"
        if pose is RigPose.HEAD_DOWN_ELSEWHERE:
            # Casting into a well, or dwelling on the wick. Dispensing here
            # contaminates the sample; moving here drags the tip.
            return "head is lowered away from the flush basin (casting or wicking)"
        return "rig pose could not be read"

    # ── Actuation ────────────────────────────────────────────────────────

    def maybe_purge(
        self,
        *,
        context: str = "idle",
        require_idle_rest: bool = True,
        allow_positioning: bool = True,
        owns_rig: bool = False,
        end_at_idle_rest: bool = True,
    ) -> PurgeOutcome:
        """Purge if one is due and every precondition holds.

        Three rig poses, three different answers (see :class:`RigPose`):

        * **AT_FLUSH** — head already down in the basin. Purge in place. This
          covers idle rest, a precondition flush, and an anneal parked there.
        * **HEAD_UP** — travel to the basin and lower, then purge, provided
          ``allow_positioning``. A raised head is no longer a refusal.
        * **HEAD_DOWN_ELSEWHERE / UNKNOWN** — casting into a well, dwelling on
          the wick, or unreadable. Never purge, never move.

        ``allow_positioning=False`` is for a caller that owns the hardware and
        wants a purge only if it costs no motion. ``owns_rig=True`` says the
        caller *is* the current claim holder — an in-run purge window — so the
        ownership check must not refuse it against its own run.

        ``end_at_idle_rest`` decides what pose to leave behind, and **an in-run
        caller must pass False**. Idle rest leaves the head *down* in the basin,
        but both `precondition_flush` and `single_drop_simul` open with a bare
        `move_to` and no retract — so a purge that lowered the head mid-run
        would make the very next step trip the stage head guard. With False the
        purge restores the pose it found.
        """
        due = self._scheduler.due()
        if due is None:
            return PurgeOutcome()

        blocking = self._blocking_reason(allow_positioning=allow_positioning,
                                         owns_rig=owns_rig)
        if blocking is None:
            blocking = self._pose_blocking(allow_positioning=allow_positioning)
        if blocking is not None:
            # Do NOT mark as purged — the line is still stagnating, and pretending
            # otherwise would hide a genuinely overdue line behind a reset timer.
            # Every refusal is therefore a DEFERRAL: `due()` is derived from the
            # per-pump timers rather than being a queued event, so the next tick
            # recomputes the same purge as still owed, with overdue_s larger.
            logger.info("purge_skipped", context=context, reason=blocking,
                        overdue_s=round(due.overdue_s, 1))
            self._note_deferred(due, reason=blocking)
            return PurgeOutcome(skipped_reason=blocking, volumes_uL=due.volumes_uL)

        settings = self._scheduler.settings
        if not settings.actuate:
            # Dry run: report, and reset the timer so the log shows the intended
            # cadence rather than one continuous "overdue" state.
            logger.info(
                "purge_dry_run", context=context, volumes_uL=due.volumes_uL,
                total_uL=due.total_uL, overdue_s=round(due.overdue_s, 1),
                msg="[purge] actuate is off — nothing dispensed",
            )
            self._scheduler.note_purged()
            return PurgeOutcome(dry_run=True, volumes_uL=due.volumes_uL)

        return self._perform(due, context=context,
                             end_at_idle_rest=end_at_idle_rest)

    def _perform(self, due: Any, *, context: str,
                 end_at_idle_rest: bool = True) -> PurgeOutcome:
        """Dispense the purge. Best-effort per pump; never raises."""
        errors: list[str] = []
        moved: dict[int, float] = {}

        # Establish the pose if it is not already right. Only reachable from
        # HEAD_UP — the pose check refused every other case that needs motion.
        pose_before = self._pose()
        repositioned = pose_before is not RigPose.AT_FLUSH
        if repositioned:
            result = enter_idle_rest(
                self._manager, park_reason=self._park_reason,
                state=self.idle_rest, flush_xy=self._flush_xy,
            )
            if not result.entered:
                # Could not get to the basin: skip without resetting the timer,
                # so the line comes due again rather than being quietly forgotten.
                return PurgeOutcome(
                    skipped_reason=f"could not reach the flush basin: {result.reason}"
                )

        try:
            syringe = self._manager.get("syringe")
        except Exception as exc:
            return PurgeOutcome(skipped_reason=f"syringe unavailable: {exc}")

        for pump_id, volume in sorted(due.volumes_uL.items()):
            if volume <= 0:
                continue
            try:
                # Goes through the normal choke point, so the stock ledger
                # debits it and a depleted reservoir refuses it exactly as it
                # would for a trial dispense.
                syringe.single_pump(
                    res_vol=1000, ID=int(pump_id),
                    rate=self._flush_rate, dispense_vol=float(volume),
                )
                moved[int(pump_id)] = float(volume)
            except Exception as exc:
                errors.append(f"pump {pump_id}: {exc}")
                logger.warning("purge_pump_failed", pump_id=pump_id, error=str(exc))

        # Purged volume goes to the flush station, i.e. to waste. Recorded even
        # on partial failure — fluid that moved is in the container regardless.
        if moved and self._waste is not None:
            try:
                self._waste.add(sum(moved.values()))
            except Exception:
                logger.warning("purge_waste_record_failed", exc_info=True)

        # Only reset timers for lines that actually moved: a pump that failed is
        # still stagnating and should come due again immediately.
        if moved:
            self._scheduler.note_purged(list(moved))

        # Put the head back where it was found, unless the caller wants the rig
        # left at idle rest (the background timer, between runs).
        #
        # This is load-bearing mid-run: idle rest leaves the head DOWN, and both
        # `precondition_flush` and `single_drop_simul` open with a bare
        # `move_to` and no retract — so a purge that lowered the head would make
        # the very next step trip the stage head guard and fail the channel.
        if repositioned and not end_at_idle_rest:
            if not leave_idle_rest(self._manager, state=self.idle_rest):
                errors.append("could not retract the head after purging")

        logger.info("purge_performed", context=context, volumes_uL=moved,
                    total_uL=sum(moved.values()), errors=errors,
                    repositioned=repositioned, left_at_rest=end_at_idle_rest)
        return PurgeOutcome(performed=bool(moved), volumes_uL=moved, errors=errors)


class NullPurgeRunner:
    """The purge harness, absent — a runner that does nothing and says so.

    Purging is **optional**. A rig with no ``[purge]`` schedule attached is
    correctly configured, not broken, so its absence should not push a ``None``
    branch out to every caller. This absorbs the call instead.

    ⚠️ **It absorbs an optional side effect only. It never answers a question
    about the rig.** Pose, park state, idle rest and "is a purge owed" are all
    facts about hardware, and a null object that invented them would be the
    exact failure the *undeclared is unknown, never empty* convention exists to
    prevent — a fabricated "no" is indistinguishable from a measured one. Every
    attribute other than the two defined below therefore raises
    :class:`NotImplementedError` rather than returning a plausible default.
    """

    #: ``False`` here, ``True`` on :class:`PurgeRunner`. A caller that also has
    #: to stand up *machinery* around a purge asks this first: the executor's
    #: concurrent purge window costs a thread and an asyncio task per co-runnable
    #: step, and spinning that up to call a no-op would be a real behaviour
    #: change dressed up as one. This is a static fact about the class, not a
    #: claim about the rig.
    performs_purges = False

    def __init__(self, *, reason: str = "no purge scheduler is attached") -> None:
        self._reason = str(reason)
        # Logged once at construction rather than per call: an overnight campaign
        # offers hundreds of purge windows, and a line each would bury the log in
        # reports that nothing happened.
        logger.info("purge_runner_absent", reason=self._reason)

    def maybe_purge(self, **_kwargs: Any) -> PurgeOutcome:
        """Do nothing, and report that nothing was done. Never raises.

        The outcome describes **this call** — no purge was performed, here is
        why — and makes no claim about whether one was owed. ``performed`` and
        ``dry_run`` are both false, so a caller that only surfaces outcomes
        worth showing (the GUI status bar) stays silent.
        """
        return PurgeOutcome(
            skipped_reason=f"purging is not configured ({self._reason})"
        )

    def __getattr__(self, name: str) -> Any:
        # Dunders fall through as AttributeError so ordinary duck typing
        # (`hasattr(x, "__await__")`, copy, repr) keeps working; anything else
        # is a caller reaching for rig state this object does not have and must
        # not invent.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"NullPurgeRunner has no {name!r}: purging is not configured "
            f"({self._reason}), and a null object may absorb a purge but never "
            f"answer a question about the rig."
        )


# ── Idle rest ────────────────────────────────────────────────────────────────

@dataclass
class IdleRestResult:
    entered: bool
    reason: str | None = None


def enter_idle_rest(
    manager: Any,
    *,
    park_reason: Any = None,
    state: "IdleRestState | None" = None,
    flush_xy: "tuple[float, float] | None" = None,
) -> IdleRestResult:
    """Bring the rig to its resting state: at the flush station, head down.

    Works from **any** starting pose, which is what makes it usable as the
    system-wide "run finished" convention: it retracts first if the head is
    down, travels, then lowers. Retracting first is not optional — the stage
    head guard refuses a move while the head is lowered, and rightly so.

    Refuses while a park reason is outstanding. A parked rig was deliberately
    made safe — putting its head back down would erase the one visible sign that
    something went wrong, and would do so silently overnight.
    """
    if park_reason is not None:
        try:
            reason = park_reason()
        except Exception:
            reason = "park state unknown"
        if reason:
            logger.info("idle_rest_refused", reason=reason)
            return IdleRestResult(False, f"rig is parked ({reason})")

    # Retract before travelling, from whatever pose the last run left behind.
    try:
        syringe = manager.get("syringe")
        is_up = getattr(syringe, "is_head_up", None)
        if callable(is_up) and not is_up():
            syringe.head_retract()
    except Exception as exc:
        logger.warning("idle_rest_retract_failed", error=str(exc))
        return IdleRestResult(False, f"could not retract before moving: {exc}")

    try:
        if flush_xy is None:
            flush_xy = _flush_position()

        stage = manager.get("stage")
        stage.move_to(float(flush_xy[0]), float(flush_xy[1]))
    except Exception as exc:
        logger.warning("idle_rest_move_failed", error=str(exc))
        return IdleRestResult(False, f"could not reach the flush station: {exc}")

    try:
        manager.get("syringe").head_descend()
    except Exception as exc:
        logger.warning("idle_rest_head_failed", error=str(exc))
        return IdleRestResult(False, f"could not lower the head: {exc}")

    # Only now is the rig genuinely at rest — after both the move and the head
    # succeeded. Marking earlier would authorise a purge at a position the stage
    # never reached.
    if state is not None:
        state.mark_entered()
    logger.info("idle_rest_entered", flush_xy=list(flush_xy))
    return IdleRestResult(True)


def leave_idle_rest(manager: Any, *, state: "IdleRestState | None" = None) -> bool:
    """Retract the head before anything else moves. ``True`` on success.

    The flag is cleared *first*: from the moment something wants the rig out of
    rest it is no longer a safe purge target, and a failed retract must not leave
    a stale "at rest" belief behind for a background timer to act on.
    """
    if state is not None:
        state.mark_left()
    try:
        manager.get("syringe").head_retract()
        logger.info("idle_rest_left")
        return True
    except Exception as exc:
        logger.warning("idle_rest_leave_failed", error=str(exc))
        return False
