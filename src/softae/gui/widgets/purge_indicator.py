"""What the purge schedule looks like right now — decided here, painted elsewhere.

The anti-clog purge used to be visible for **eight seconds every fifteen
minutes**: ``main_window._on_purge_tick`` pushed ``PurgeOutcome.summary()`` into
a status-bar message and let it expire. A deferral was rendered not at all. That
is the invisibility this module exists to end — an activity nobody can see is an
activity nobody can arbitrate.

**A decision, not a widget.** Everything worth asserting lives in
:func:`purge_indicator`, a pure function of the retained outcome plus a clock;
:class:`~softae.gui.widgets.purge_badge.PurgeBadge` only paints what it is
handed. That split is what makes the overdue state testable at all — it is
reached by *time passing while blocked*, which no widget test could arrange.

Three states the operator actually distinguishes, and the middle one is where
badges usually go wrong:

``purged``
    Fluid moved. Neutral-positive, no attention.
``dry_run``
    A purge came due and ``[purge] actuate`` is off, so nothing was dispensed.
    **Rendered neutral.** This is the shipped default and therefore the *normal*
    case; a permanent amber for the normal case is how an indicator becomes
    wallpaper, and an operator who has learned to ignore it will also ignore the
    overdue one.
``overdue``
    A purge is owed *right now* and something is preventing it. This is the
    attention state, and it **dominates regardless of** ``actuate``: a line
    stagnating is a fact about the fluid, not about whether the harness is armed.

**How "overdue" is actually reached — the part that is easy to get wrong.**
:meth:`~softae.core.purge_runner.PurgeRunner.maybe_purge` calls
``scheduler.note_purged()`` inside its ``not settings.actuate`` branch, so a dry
run **resets the per-pump timers** exactly as a real purge does. A blocked purge
returns *before* that call, deliberately, so its timers keep running. The
consequence, and it is the whole design of this badge:

    ``overdue_s`` measures **time spent blocked or declined**, never time since
    the interval elapsed.

On a free, idle rig under the shipped default the badge therefore alternates
between ``dry_run`` and ``scheduled`` and never lights. An implementer who
assumes overdue accumulates from the interval builds a badge that never fires
*and* cannot tell why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: How close to due counts as ``near``. Four ticks of the 30 s purge timer: near
#: enough to mean "about to happen", far enough that the line is not repainted
#: every tick on its way there.
NEAR_WINDOW_S = 120.0


@dataclass(frozen=True)
class PurgeIndicator:
    """One rendering decision about the purge schedule.

    ``state`` is the only field anything branches on; ``headline`` and ``detail``
    are the words, and ``attention`` is the request for the operator's eye.
    """

    #: ``"not_ours" | "unconfigured" | "overdue" | "near" | "dry_run" |
    #: "purged" | "scheduled"``, in precedence order.
    state: str
    #: One short line for a ~130 px sidebar column.
    headline: str
    #: The tooltip — room for the reason, the volumes and the actuate caveat.
    detail: str
    #: Seconds a purge has been owed *and prevented*. ``0.0`` unless ``overdue``.
    overdue_s: float = 0.0
    #: Pulse, and keep pulsing until purged or acknowledged.
    attention: bool = False


def purge_indicator(
    scheduler: Any,
    *,
    last_outcome: Any = None,
    last_at: float | None = None,
    now: float = 0.0,
    acknowledged_at: float | None = None,
    attached_holder: str | None = None,
) -> PurgeIndicator:
    """Decide how the purge schedule should read.

    Parameters
    ----------
    scheduler
        The live :class:`~softae.core.purge.PurgeScheduler`, or ``None`` when no
        purge harness is attached. Read, never mutated.
    last_outcome, last_at
        The last :class:`~softae.core.purge_runner.PurgeOutcome` worth retaining
        and the clock reading it was retained at. ``MainWindow`` keeps only
        outcomes that *said* something (performed, dry run, or skipped), so a
        "purged 3 min ago" line is not wiped by the next no-op tick.
    now, acknowledged_at
        Same clock as *last_at* and as the scheduler's — ``time.monotonic`` in
        production, an injected counter in tests.
    attached_holder
        Short name of the process holding the rig when this window is attached.
        Truthy means this window's scheduler describes *its own* launch, not the
        campaign's dispensing, so no timing at all may be reported.
    """
    if attached_holder:
        return PurgeIndicator(
            "not_ours",
            f"Purge: owned by {attached_holder}",
            # No minutes, no volumes. `_purge_scheduler` is constructed in both
            # modes but an attached window's per-pump timers started at *its*
            # launch and know nothing about the campaign's dispensing, so any
            # number here would be fabricated.
            f"{attached_holder} owns the rig. This window opened no instrument "
            "sessions, so it has no purge schedule of its own and reports none.",
        )

    if scheduler is None:
        return PurgeIndicator(
            "unconfigured",
            "Purge: not configured",
            "No anti-clog purge schedule is attached. A rig without one is "
            "correctly configured, not broken.",
        )

    try:
        settings = scheduler.settings
        due = scheduler.due()
        next_in = scheduler.next_due_in_s()
    except Exception:
        return PurgeIndicator(
            "unconfigured",
            "Purge: unavailable",
            "The purge scheduler could not be read.",
        )

    if not settings.enabled:
        return PurgeIndicator("unconfigured", "Purge: off", settings.describe())

    if due is not None:
        return _overdue(due, settings, last_outcome, now, acknowledged_at)

    if next_in is not None and next_in <= NEAR_WINDOW_S:
        return PurgeIndicator(
            "near",
            f"Purge: due in {_duration(next_in)}",
            f"{_volumes(settings.per_purge_uL())} is due in "
            f"{_duration(next_in)}.{_actuate_note(settings)}",
        )

    if getattr(last_outcome, "dry_run", False):
        return PurgeIndicator(
            "dry_run",
            f"Purge: would have purged {_total(last_outcome):.0f} µL",
            f"A purge came due {_ago(last_at, now)} and "
            f"{_volumes(last_outcome.volumes_uL)} would have moved. "
            "[purge] actuate is off — nothing was dispensed. "
            f"Next purge in {_duration(next_in)}.",
        )

    if getattr(last_outcome, "performed", False):
        return PurgeIndicator(
            "purged",
            f"Purge: purged {_total(last_outcome):.0f} µL {_ago(last_at, now)}",
            f"{_volumes(last_outcome.volumes_uL)} dispensed to the flush basin "
            f"{_ago(last_at, now)}. Next purge in {_duration(next_in)}.",
        )

    return PurgeIndicator(
        "scheduled",
        f"Purge: next in {_duration(next_in)}",
        f"{_volumes(settings.per_purge_uL())} every "
        f"{_duration(settings.interval_s)}.{_actuate_note(settings)}",
    )


def _overdue(due: Any, settings: Any, last_outcome: Any, now: float,
             acknowledged_at: float | None) -> PurgeIndicator:
    """The one state that asks for the operator's eye."""
    overdue_s = float(due.overdue_s)
    reason = getattr(last_outcome, "skipped_reason", None)
    detail = (
        f"{_volumes(due.volumes_uL)} has been owed for {_duration(overdue_s)} "
        "and could not be dispensed."
    )
    if reason:
        detail += f" Deferring: {reason}."
    if not settings.actuate:
        # Stated, not coloured. The lines stagnate whether or not the harness
        # is armed, which is why this state is amber either way.
        detail += (
            " [purge] actuate is off, so nothing would be dispensed — but the "
            "lines are stagnating regardless."
        )
    detail += " Click to acknowledge."
    return PurgeIndicator(
        "overdue",
        f"Purge: OVERDUE by {_duration(overdue_s)}",
        detail,
        overdue_s=overdue_s,
        attention=not _acknowledged(overdue_s, now, acknowledged_at,
                                    settings.interval_s),
    )


def _acknowledged(overdue_s: float, now: float, acknowledged_at: float | None,
                  interval_s: float) -> bool:
    """Is the operator's "yes, I know" still standing?

    A click silences the pulse, but not forever and not across episodes:

    * **Superseded by a further interval.** Attention resumes once ``overdue_s``
      has grown past the acknowledged value by a full ``interval_s``. Otherwise
      one click silences an unbounded problem.
    * **Not carried into a later episode.** ``overdue_s`` grows one-for-one with
      the clock while a purge stays blocked, so an acknowledgement made *inside*
      this episode is younger than the episode itself. One older than the
      episode belongs to a previous one — the timers were reset in between by a
      purge or a dispense — and is therefore stale.
    """
    if acknowledged_at is None:
        return False
    age = now - acknowledged_at
    if not 0.0 <= age <= overdue_s:
        return False
    return age < interval_s


# ── Words ────────────────────────────────────────────────────────────────────

def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.0f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.0f} min"
    return f"{seconds / 3600.0:.1f} h"


def _ago(last_at: float | None, now: float) -> str:
    return "just now" if last_at is None else f"{_duration(now - last_at)} ago"


def _total(outcome: Any) -> float:
    return float(getattr(outcome, "total_uL", 0.0) or 0.0)


def _volumes(volumes_uL: dict) -> str:
    total = sum(volumes_uL.values())
    per_pump = ", ".join(f"pump {p} {v:.0f} µL"
                         for p, v in sorted(volumes_uL.items()))
    return f"{total:.0f} µL ({per_pump})" if per_pump else "A purge"


def _actuate_note(settings: Any) -> str:
    return "" if settings.actuate else " [purge] actuate is off — a due purge is logged, not dispensed."
