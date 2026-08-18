"""Who owns the rig, said in one voice — and never used to refuse anything.

The cross-process rig lock (:mod:`softae.core.run_lock`) is the only authority on
which process is driving the hardware. The GUI now renders that fact in two
places — the instrument table on the Init tab and the banner on Manual Control —
and both must say it in the *same* words. Two renderings of one file is how an
operator ends up comparing "OCCUPIED" against "busy" and having to work out
whether they disagree.

**Nothing here gates, refuses, or takes over.** That is deliberate and it is the
whole design:

    The operator standing at the rig is the final authority, and they frequently
    reach for manual control *because* something has gone wrong. Refusing them at
    that moment is the failure, not the protection.

So this module answers "who holds the rig, and since when" and stops. Stopping a
campaign is a *designated* control with its own scope — E-Stop is rig-scale,
Abort is campaign-scale and terminal, Pause is campaign-scale and resumable — and
none of those belong in an interlock that fires on a jog button.

What the banner may safely *say* is bounded by what can actually be commanded
across a process boundary, which today is nothing: see :func:`campaign_identity`.
"""

from __future__ import annotations

from typing import Any

#: The Init tab's word for "a different process holds the rig". Shared so the two
#: renderings cannot drift into two vocabularies.
OCCUPIED = "OCCUPIED"

#: ``what`` prefix a campaign stamps on the lock (see ``run_lock`` D8). The rest
#: is ``<campaign name>:<run id>``.
CAMPAIGN_PREFIX = "campaign:"


def owner_line(rig_lock: Any) -> str:
    """One-line owner summary — ``describe()`` is multi-line and too tall for a cell.

    Names the PID, the run and the start time rather than just "busy", because PID
    reuse means the lock can read as live when its owner is long gone (see
    :mod:`softae.core.run_lock`). A person can tell "commissioning blank_short,
    started 14:02" from a stale number; a check cannot.
    """
    what = rig_lock.what or "unnamed run"
    when = rig_lock.started_at or "unknown time"
    return f"held by PID {rig_lock.pid} — {what}, started {when}"


def foreign_rig_lock() -> Any:
    """The live rig lock **if another process holds it**, else ``None``.

    A lock this process owns is not foreign: a GUI running its own HT sequence or
    in-process campaign is not a second owner of anything, and reporting it as one
    would make the banner cry wolf on the ordinary case.

    Never raises. This decorates a view; a lock file that cannot be read must not
    take a tab down with it.
    """
    try:
        from softae.core.run_lock import read_run_lock

        lock = read_run_lock()
    except Exception:
        return None
    if lock is None or lock.is_mine():
        return None
    return lock


def campaign_identity(rig_lock: Any) -> tuple[str, str] | None:
    """``(campaign name, run id)`` if *rig_lock* is a campaign's, else ``None``.

    The lock's ``what`` is the discovery channel: a campaign stamps
    ``campaign:<name>:<run_id>`` on it, so reading ownership and identifying the
    run are one file read rather than a registry. A workflow-scoped lock (the
    executor's ``workflow '<name>'``) has no campaign identity and correctly
    returns ``None`` — an HT sequence is not a campaign and must not be labelled
    as one.
    """
    what = getattr(rig_lock, "what", "") or ""
    if not what.startswith(CAMPAIGN_PREFIX):
        return None
    rest = what[len(CAMPAIGN_PREFIX):]
    name, _, run_id = rest.partition(":")
    return name, run_id
