"""Who owns the rig right now (P8).

The anti-clog purge is the only mechanism in the system that moves hardware with
nobody asking it to. Once it is allowed to *position* the rig — travel to the
flush basin and lower the head — head position stops being a usable proxy for
"is it safe to act", because the purge itself changes head position. The real
question has to be answerable directly: **does something else own the hardware?**

This is a claim registry, not a lock. Nothing blocks on it; the purge asks and
defers. That is deliberate — a background timer that could *wait* on the rig is
a background timer that can deadlock a run, and a purge is never urgent enough
to be worth that. The interval is a floor, so deferring simply means the purge
happens at the next boundary.

Claims are re-entrant by owner and released even when a run raises, because a
registry that leaks a claim silently disables purging for the rest of the
session — a failure that looks exactly like "the harness is off".

**Claims are instrument-scoped.** A whole-rig claim (``instruments=None``) is
the conservative default and conflicts with everything — that is what a campaign
takes, and it is why the background timer correctly defers for the whole run.
But a step that occupies only the potentiostat does not prevent the syringe from
purging, and expressing that requires knowing *which* instruments a claim covers
rather than just that something is running. Without the scoping, every in-run
purge opportunity has to be special-cased the way the anneal one was.

**A claim can be suspended.** A run held at a quiescent boundary — paused by the
operator, or by the executor's own consecutive-failure hold — is not driving
anything, so it hands the instruments back. The obvious way to express that is
to release the claim and re-acquire on resume, and it is wrong: a released claim
makes a paused rig indistinguishable from an idle one, so the purge sees a free
rig and purges *without asking*, at the exact moment a human is most likely to be
standing at it. Suspension is therefore a third state rather than an absence —
:meth:`conflicts` skips a suspended owner (manual control is re-enabled, which is
the point) while :meth:`suspended_conflict` still reports it (so the purge can
ask). Claim depth is preserved across a suspend/unsuspend round trip, which is
what makes a pause nested inside a pause harmless where a second ``release``
would have dropped a real claim.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterable

import structlog

logger = structlog.get_logger(__name__)

#: Instruments a purge needs: the syringe to dispense, and the stage in case it
#: has to travel to the flush basin first.
PURGE_INSTRUMENTS = frozenset({"syringe", "stage"})


class RigActivity:
    """Tracks which activities hold which instruments."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        #: owner -> stack of claimed instrument sets (``None`` = whole rig).
        self._claims: dict[str, list[frozenset | None]] = {}
        #: owner -> why it is currently holding without driving. Membership is
        #: the state; the value is only ever displayed.
        self._suspended: dict[str, str] = {}

    def acquire(self, owner: str,
                instruments: "Iterable[str] | None" = None) -> None:
        """Claim *instruments* for *owner*; ``None`` claims the whole rig.

        Re-entrant: nested claims stack, and each release pops one.
        """
        scope = None if instruments is None else frozenset(instruments)
        with self._lock:
            self._claims.setdefault(owner, []).append(scope)
            logger.debug("rig_claim_acquired", owner=owner,
                         instruments=sorted(scope) if scope else "whole-rig",
                         depth=len(self._claims[owner]))

    def release(self, owner: str) -> None:
        """Drop one claim for *owner*. Releasing an unheld claim is not an error.

        Tolerant on purpose: the alternative is an exception on a cleanup path,
        which would mask whatever actually went wrong in the run.
        """
        with self._lock:
            stack = self._claims.get(owner)
            if not stack:
                return
            stack.pop()
            if not stack:
                self._claims.pop(owner, None)
                # Dropped with the last claim, never carried: a run that ends
                # while paused would otherwise leave a suspension behind for
                # good, and the purge would keep asking about a run that
                # finished last Tuesday.
                self._suspended.pop(owner, None)
            logger.debug("rig_claim_released", owner=owner)

    def suspend(self, owner: str, *, reason: str = "") -> None:
        """Mark *owner* as holding its claim without driving anything.

        Idempotent, and a no-op for an owner with no claim — for the same
        reason :meth:`release` tolerates one. Idempotence is what makes the
        executor's self-pause safe *inside* an operator pause: suspending twice
        costs nothing, whereas releasing twice would drop a live claim.
        """
        with self._lock:
            if owner not in self._claims:
                return
            self._suspended[owner] = reason
            logger.debug("rig_claim_suspended", owner=owner, reason=reason)

    def unsuspend(self, owner: str) -> None:
        """Mark *owner* as driving again. Unknown owners are ignored.

        Membership, not a counter: an unsuspend from an inner hold clears an
        outer pause too. That is the safe direction — the owner goes back to
        conflicting, so the rig is guarded again rather than silently released.
        """
        with self._lock:
            if self._suspended.pop(owner, None) is not None:
                logger.debug("rig_claim_unsuspended", owner=owner)

    @contextmanager
    def claimed(self, owner: str, instruments: "Iterable[str] | None" = None):
        """Hold a claim for the duration of the block, releasing on any exit."""
        self.acquire(owner, instruments)
        try:
            yield self
        finally:
            self.release(owner)

    @property
    def busy(self) -> bool:
        """Whether *anything* holds the rig."""
        with self._lock:
            return bool(self._claims)

    def conflicts(self, instruments: Iterable[str]) -> str | None:
        """The owner blocking use of *instruments*, or ``None`` if free.

        A whole-rig claim conflicts with everything; scoped claims conflict only
        on overlap. Erring toward conflict is the safe direction — a purge that
        is wrongly deferred costs a delayed purge, one wrongly allowed collides
        with a live measurement.

        **Suspended owners are skipped**, because a suspended owner is by
        definition not driving. This is the whole of the pause ruling: there is
        no second predicate for a caller to keep in sync, and the existing purge
        caller gets the answer for free. Use :meth:`suspended_conflict` to see
        who was skipped.
        """
        return self._owner_blocking(instruments, suspended=False)

    def suspended_conflict(self, instruments: Iterable[str]) -> str | None:
        """The owner who *would* block *instruments* but is suspended.

        The difference between a paused rig and an idle one — which is to say,
        whether a human is probably standing at it. The purge asks rather than
        assumes when this returns an owner.
        """
        return self._owner_blocking(instruments, suspended=True)

    def _owner_blocking(self, instruments: Iterable[str], *,
                        suspended: bool) -> str | None:
        want = frozenset(instruments)
        with self._lock:
            for owner, stack in self._claims.items():
                if (owner in self._suspended) is not suspended:
                    continue
                for scope in stack:
                    if scope is None or (scope & want):
                        return owner
        return None

    def owners(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._claims))

    def describe(self) -> str:
        with self._lock:
            return ", ".join(
                f"{owner} ({self._suspended[owner] or 'paused'})"
                if owner in self._suspended else owner
                for owner in sorted(self._claims)
            ) or "idle"


#: Attributes on a workflow-like object that hold :class:`WorkflowStep` lists.
#: ``Workflow`` carries the last three; ``steps`` is accepted so a flattened
#: step list works too. Unioning the *templates* rather than calling
#: ``resolve_steps()`` is deliberate: it is wider (a workflow with
#: ``iterations=0`` still names what its loop would drive) and it allocates
#: nothing, and wider is the direction :meth:`RigActivity.conflicts` asks for.
_STEP_LIST_ATTRS = ("steps", "setup", "loop_steps", "teardown")


def workflow_instruments(workflow) -> frozenset[str] | None:
    """Which instruments *workflow* will drive, or ``None`` for the whole rig.

    ``None`` — the conservative whole-rig scope — is returned when the union is
    empty or anything at all goes wrong reading it. A scope derivation that
    fails must **widen, never narrow**: the cost of a claim that is too broad is
    a manual control refused during a run the operator can pause, and the cost
    of one that is too narrow is a jog into a moving stage.
    """
    try:
        names = {
            str(getattr(step, "instrument", "")).strip()
            for attr in _STEP_LIST_ATTRS
            for step in (getattr(workflow, attr, None) or ())
        }
    except Exception:
        logger.warning("workflow_instruments_unreadable", exc_info=True)
        return None
    names.discard("")
    return frozenset(names) or None
