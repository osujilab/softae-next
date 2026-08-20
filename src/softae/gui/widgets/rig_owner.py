"""Who owns the rig, said in one voice — and never used to refuse *on the lock*.

The cross-process rig lock (:mod:`softae.core.run_lock`) is the only authority on
which process is driving the hardware. The GUI now renders that fact in two
places — the instrument table on the Init tab and the banner on Manual Control —
and both must say it in the *same* words. Two renderings of one file is how an
operator ends up comparing "OCCUPIED" against "busy" and having to work out
whether they disagree.

**Nothing here refuses on the lock.** That is deliberate and it is still the
whole design:

    The operator standing at the rig is the final authority, and they frequently
    reach for manual control *because* something has gone wrong. Refusing them at
    that moment is the failure, not the protection.

So this module answers "who holds the rig, and since when" and stops. Stopping a
campaign is a *designated* control with its own scope — E-Stop is rig-scale,
Abort is campaign-scale and terminal, Pause is campaign-scale and resumable — and
none of those belong in an interlock that fires on a jog button.

Amendment — attach mode
-----------------------
This paragraph used to read *"Nothing here gates, refuses, or takes over."* It
was written when every window owned the sessions it drove, and under that
assumption "refuse" and "refuse on the lock" were one sentence. They stopped
being one sentence when a window could be launched **attached** to a campaign in
another process (:mod:`softae.gui.launch_mode`): such a window opened no
instrument session, so a jog from it reaches nothing whatever this module says.
The operator ruled that it must then fail *legibly*, naming the run — so this
module now also supplies the words for that (:func:`attached_owner_line`,
:func:`attached_refusal_line`), and the Manual Control tab refuses with them.

Three things keep that inside the ruling rather than around it:

* **The predicate is the launch decision, not the lock.** It is fixed before a
  port is opened and never re-derived at press time; a lock read inside a jog
  handler is the forbidden lockout no matter what it is called.
* **It removes no capability**, because an attached window has none to remove.
  It replaces a driver exception naming nothing with a sentence naming the run.
* **It is leavable in one operator act** — Init tab → Connect All — after which
  the window owns sessions and refuses nothing.

A window that owns its own sessions is *unchanged*: it is told who else is on the
rig, and then it actuates, foreign lock or not.

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

#: The word for "*this window* opened none of the sessions it can see". Distinct
#: from :data:`OCCUPIED`, which is a fact about the rig: the rig can be occupied
#: while this window still owns its own instruments, and that case is not this
#: one. Shared by the toolbar notice and the Manual Control banner.
ATTACHED = "ATTACHED"


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
    """:func:`softae.core.run_lock.foreign_run_lock`, wrapped so a view survives it.

    **The predicate is not reimplemented here.** ``run_lock`` owns "is a *live,
    other* process holding the rig", and it has to: a headless campaign must be
    able to ask the same question without importing ``softae.gui``, so a second
    copy of the rule could only ever drift from the one the CLI enforces.

    What this adds — and the only reason the wrapper exists — is that it **never
    raises**. This decorates a view, and a lock file on a dead network share must
    not take a tab down with it; the CLI deliberately does *not* swallow that,
    because a run that cannot determine who owns the rig must refuse to start
    rather than start blind.
    """
    try:
        from softae.core.run_lock import foreign_run_lock

        return foreign_run_lock()
    except Exception:
        return None


def owner_status_line(
    rig_lock: Any,
    *,
    phase: str | None = None,
    phase_age_s: Any = None,
) -> str:
    """The monitoring sidebar's one line about who has the rig. ``""`` = hide it.

    Three states, not two, and the third is the one a two-state line gets wrong:

    ==========================  ================================================
    no foreign lock             ``""`` — the line is hidden; a permanent "rig
                                free" label is noise the operator learns to skip
    a campaign holds it         ``<name> · <phase> · <age>s — See Monitoring
                                tab``: there is a run to follow, and somewhere
                                to follow it
    something else holds it     ``OCCUPIED — <owner_line>``: a bench sequence or
                                an HT workflow publishes no stream and no
                                control channel, so there is nothing to attach
                                to and nothing to pause. Saying so is the whole
                                point — the alternative renders it as a campaign
                                and offers stops that reach nothing
    ==========================  ================================================

    *phase* and *phase_age_s* come straight off the campaign's last ``heartbeat``
    record (:func:`~softae.core.campaign_events.last_heartbeat`). Before the
    first beat arrives the line says so rather than showing a zero age, because
    "0s" and "we have not heard anything yet" are opposite facts.
    """
    if rig_lock is None:
        return ""
    identity = campaign_identity(rig_lock)
    if identity is None:
        return f"{OCCUPIED} — {owner_line(rig_lock)}"
    name = identity[0] or "campaign"
    if not phase:
        return f"{name} · waiting for the first heartbeat — See Monitoring tab"
    return f"{name} · {phase} · {_age(phase_age_s)} — See Monitoring tab"


def _age(phase_age_s: Any) -> str:
    """``"12s"``, or ``"age unknown"`` for anything that is not a number."""
    try:
        return f"{int(round(float(phase_age_s)))}s"
    except (TypeError, ValueError):
        return "age unknown"


def attached_owner_line(
    campaign: tuple[str, str] | None,
    rig_lock: Any = None,
) -> str:
    """``ATTACHED — campaign 'x' (run y) owns the rig``, holder named if given.

    The headline every attach-mode surface starts from, so the toolbar notice and
    the Manual Control banner cannot end up saying "ATTACHED" and "attached to a
    campaign" side by side. *campaign* is
    :attr:`~softae.gui.launch_mode.LaunchMode.campaign`; ``None`` is the honest
    degenerate case — something holds the rig and publishes no identity — and is
    rendered as "another process" rather than invented into a campaign.
    """
    if campaign:
        name, run_id = campaign
        who = f"campaign '{name}' (run {run_id})" if run_id else f"campaign '{name}'"
    else:
        who = "another process"
    line = f"{ATTACHED} — {who} owns the rig"
    return line if rig_lock is None else f"{line}: {owner_line(rig_lock)}"


def attached_refusal_line(
    campaign: tuple[str, str] | None,
    rig_lock: Any = None,
) -> str:
    """Why a manual command is not sent from an attached window, naming the run.

    Deliberately not :func:`~softae.core.run_lock.busy_rig_message`: that one
    closes with *"Manual control at the rig is never refused"*, which is exactly
    the claim attach mode qualifies, so quoting it here would ship a message that
    contradicts itself. What is reused instead is its *shape* — name the holder,
    then spell out every exit — and the routing sentence the Manual Control
    banner already uses in owner mode.
    """
    return (
        f"{attached_owner_line(campaign, rig_lock)}. This window opened no "
        "instrument sessions, so manual commands are not sent from here — they "
        "would reach nothing. To act on the rig: pause or abort the campaign "
        "from the process that owns it, or use the E-Stop on the main toolbar "
        "to park the whole rig. Once that run has finished, Connect All on the "
        "Init tab takes ownership and this window actuates again."
    )


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
