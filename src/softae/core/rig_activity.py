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
            logger.debug("rig_claim_released", owner=owner)

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
        """
        want = frozenset(instruments)
        with self._lock:
            for owner, stack in self._claims.items():
                for scope in stack:
                    if scope is None or (scope & want):
                        return owner
        return None

    def owners(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._claims))

    def describe(self) -> str:
        owners = self.owners()
        return ", ".join(owners) if owners else "idle"
