"""Where the rig is, and whether it is safe to stop here.

One definition of *quiescent*, shared by everything that needs one.

Why this is its own module
--------------------------
:func:`classify_pose` was written for the anti-clog purge harness and lived in
``core/purge_runner.py`` — **the only module in the system that moves hardware
with nobody asking it to**. It has three consumers now rather than two:

===========================  ==========================================
consumer                     what it asks
===========================  ==========================================
``PurgeRunner``              may I dispense here, and may I travel first?
``MainWindow`` idle rest     may the rig go to rest?
``AutonomousLoop.pause``     may the campaign *hold* here?
===========================  ==========================================

The third is a **control** path, and a control path taking an import dependency
on the actuation path is backwards: reaching ``classify_pose`` through
``purge_runner`` drags in ``PurgeRunner``, the purge scheduler, the waste ledger
and ``IdleRestState`` — none of which a pause has any business knowing about.
Nothing about the classification is purge-specific: it reads ``is_head_up()`` and
the stage position, and both facts are about the rig rather than about purging.

``purge_runner`` re-exports every name that moved, so its own code, its callers
and ``tests/test_purge_runner.py`` are untouched. **This is a move, not a fork.**
A second notion of "safe to stop here" is exactly the failure mode worth
preventing: it is how a Pause comes to hold at a pose a purge would have refused.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: How close to the calibrated flush coordinates counts as "at the flush basin",
#: in mm. Generous relative to stage jitter (~0.001 mm of mock noise, and real
#: encoder repeatability far below this) and far tighter than the distance to
#: the nearest electrode, so it can never mistake a well for the basin.
FLUSH_TOLERANCE_MM = 2.0


class RigPose(Enum):
    """Where the rig is, as it bears on whether a purge may happen here.

    Head position alone is not enough — **the head is down at the flush basin
    and down over an electrode mid-cast**, and those two demand opposite
    responses. Classification therefore reads the stage position too.
    """

    #: Head down at the flush basin — purge in place, no motion needed. This is
    #: idle rest, and also a precondition flush or an anneal hold parked there.
    AT_FLUSH = "at_flush"
    #: Head raised, wherever. Safe to travel to the flush basin and lower.
    HEAD_UP = "head_up"
    #: Head down somewhere other than the flush basin — casting into a well or
    #: dwelling on the wick. NEVER purge, and never move: the head guard would
    #: refuse the travel anyway, and dispensing here contaminates the sample.
    HEAD_DOWN_ELSEWHERE = "head_down_elsewhere"
    #: State could not be read. Treated exactly like HEAD_DOWN_ELSEWHERE.
    UNKNOWN = "unknown"


#: The poses at which a run may be *held*. Deliberately the complement of the
#: two the purge runner refuses, so the two answers cannot drift apart.
#:
#: ``HEAD_UP`` qualifies for holding even though it does not qualify for purging
#: in place: a purge at ``HEAD_UP`` has to *travel* first, and a hold moves
#: nothing at all. The refused pair is the same in both cases —
#: ``HEAD_DOWN_ELSEWHERE`` is a tip sitting in a drop, and ``UNKNOWN`` is folded
#: into it because "I could not tell" and "it is unsafe" must lead to the same
#: action.
QUIESCENT_POSES = frozenset({RigPose.AT_FLUSH, RigPose.HEAD_UP})


def classify_pose(
    manager: Any, *, flush_xy: "tuple[float, float] | None" = None
) -> RigPose:
    """Read the rig's current pose from the hardware, not from a belief flag.

    Unreadable state resolves to :attr:`RigPose.UNKNOWN`, which is refused —
    "I could not tell" and "it is unsafe" must lead to the same action.
    """
    try:
        syringe = manager.get("syringe")
    except Exception:
        return RigPose.UNKNOWN

    is_up = getattr(syringe, "is_head_up", None)
    if not callable(is_up):
        return RigPose.UNKNOWN
    try:
        if is_up():
            return RigPose.HEAD_UP
    except Exception:
        return RigPose.UNKNOWN

    # Head is down — the only question left is whether it is down somewhere safe.
    try:
        if flush_xy is None:
            flush_xy = _flush_position()
        stage = manager.get("stage")
        x, y = stage.live_position()
    except Exception:
        return RigPose.UNKNOWN

    at_flush = (
        abs(float(x) - float(flush_xy[0])) <= FLUSH_TOLERANCE_MM
        and abs(float(y) - float(flush_xy[1])) <= FLUSH_TOLERANCE_MM
    )
    return RigPose.AT_FLUSH if at_flush else RigPose.HEAD_DOWN_ELSEWHERE


def safe_to_interrupt(
    manager: Any, *, flush_xy: "tuple[float, float] | None" = None
) -> bool:
    """May a run be **held** where the rig currently stands?

    This is the predicate behind Pause's *"next safe interruption"*, and it is
    :func:`classify_pose` with no second opinion layered on top. A ``False`` here
    is never a refusal of anything — it defers the hold to a later boundary, and
    the loop's own between-cycles gate is the backstop that always qualifies. So
    an unreadable rig delays a Pause by at most one trial; it can never strand
    one, and it can never refuse an operator anything.
    """
    return classify_pose(manager, flush_xy=flush_xy) in QUIESCENT_POSES


def _flush_position() -> tuple[float, float]:
    """Calibrated flush-basin coordinates, with the engine's own fallback."""
    from softae.core.deposition_steps import deposition_positions

    positions = deposition_positions()
    return tuple(getattr(positions, "flush", None) or (0.0, 0.0))
