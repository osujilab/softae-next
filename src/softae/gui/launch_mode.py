"""Owner or attached — decided **once**, at launch, before a port is opened.

The desktop GUI has always assumed it owns the rig. It arms the hardware
interlock, asks the operator to eyeball the dispenser head, and schedules
``connect_all()`` — unconditionally, at start-up. Every one of those is wrong
while a headless campaign is driving the same instruments: the connect opens
ports the campaign owns (the exact collision
:meth:`softae.gui.tabs.tab_init.InitCalibrationTab._refuse_if_rig_held` exists to
prevent, reached by the one path that never consults it), the arming licenses a
process that holds no sessions, and the head prompt asks a question whose answer
is stale the moment a running campaign flips the head.

**The invariant this serves:** a process may command only the instrument sessions
it opened. Everything else it may read, narrate, or request — never drive.

Three properties, and each is load-bearing:

*It is decided at launch, not at the moment of danger.* The alternative — read the
lock again on the way out and branch — puts a filesystem read that
:func:`softae.core.run_lock.read_run_lock` can fail on inside the decision "do I
make the rig safe?". Ownership changes only by an operator act (Init tab →
Connect All, which is guarded by the same predicate), never by re-derivation.

*It is a pure function.* ``gui/app.py`` constructs a ``QApplication`` and so can
carry no tests of its own; extracting the decision is what makes the rule
testable at all. Nothing here reads Qt, opens a session, or has a side effect.

*Unknown means occupied.* :func:`softae.gui.widgets.unclean_shutdown.check_unclean_shutdown`
already treats an unreadable lock as "someone might be running", two lines above
the head prompt this module now gates, and says why: *"Deferring costs a launch;
guessing wrong costs the run."* This is that same rule, one level up. It is also
why the reader here is the **raising** :func:`~softae.core.run_lock.foreign_run_lock`
rather than :func:`~softae.gui.widgets.rig_owner.foreign_rig_lock`: the
never-raises wrapper answers ``None`` for *both* "the rig is free" and "I could
not tell", and collapsing those two into owner mode would open ports on exactly
the evidence we just failed to obtain. The swallow belongs here, where it can
resolve to the conservative answer, not in the reader.

The predicate itself is **not** reimplemented. ``run_lock`` owns "is a live,
*other* process holding the rig", and stays the only implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from softae.core.run_lock import foreign_run_lock
from softae.gui.widgets.rig_owner import campaign_identity, owner_line


@dataclass(frozen=True)
class LaunchMode:
    """What this window is allowed to do with the rig, for its whole lifetime.

    ``attached`` is the only field anything gates on; the rest is what the
    operator is told and what an attached view needs to render.
    """

    #: ``False`` → this process may open, arm and command. ``True`` → it may
    #: read, narrate and request, and nothing else.
    attached: bool

    #: ``(campaign name, run id)`` when there is a run to attach *to*, else
    #: ``None``. ``None`` in attached mode means occupied-but-not-attachable:
    #: something holds the rig, but it publishes no stream and offers no
    #: control channel, so there is nothing to tail, pause or abort.
    campaign: tuple[str, str] | None

    #: The campaign's run directory (``lock.log_path``) — where ``events.jsonl``
    #: and ``control.json`` live. ``None`` whenever ``campaign`` is ``None``.
    run_dir: str | None

    #: The :class:`~softae.core.run_lock.RunLock` behind the decision, for
    #: rendering. ``None`` in owner mode, and also when the lock could not be
    #: read — in which case ``attached`` is still ``True``.
    holder: Any | None

    #: One operator-facing sentence saying which mode this is and why.
    reason: str

    @property
    def owner(self) -> bool:
        """Whether this process may open instrument sessions and command them."""
        return not self.attached

    @property
    def attachable(self) -> bool:
        """Whether there is a campaign to tail and control, not merely an owner."""
        return self.campaign is not None


def decide_launch_mode(
    *,
    lock_reader: Callable[[], Any] = foreign_run_lock,
    identify: Callable[[Any], tuple[str, str] | None] = campaign_identity,
) -> LaunchMode:
    """Owner or attached, from the rig lock alone. Never raises.

    The branches mirror :func:`softae.tools.campaign._running_campaign_run_dir`
    deliberately — "not a campaign" and "a campaign that published no run
    directory" are its two refusal branches, and the GUI must not invent a third
    account of what "nothing to control" means.

    Both collaborators are injected so the decision can be tested without a lock
    file, and both are wrapped: a reader that raises, *or an identifier that
    does*, resolves to attached mode rather than to owner mode.
    """
    def describe(lock: Any) -> str:
        try:
            return owner_line(lock)
        except Exception:
            return "held by another process"

    try:
        lock = lock_reader()
    except Exception as exc:
        return LaunchMode(
            attached=True, campaign=None, run_dir=None, holder=None,
            reason=(
                "The rig lock could not be read, so this session assumes another "
                f"process is driving the rig and will not open, arm or command "
                f"anything ({exc.__class__.__name__})."
            ),
        )

    if lock is None:
        return LaunchMode(
            attached=False, campaign=None, run_dir=None, holder=None,
            reason="No other process holds the rig — this session owns the instruments.",
        )

    try:
        identity = identify(lock)
    except Exception:
        identity = None

    if identity is None:
        return LaunchMode(
            attached=True, campaign=None, run_dir=None, holder=lock,
            reason=(
                f"The rig is {describe(lock)} — not a campaign, so there is "
                "nothing to attach to. This session will not open, arm or "
                "command anything."
            ),
        )

    run_dir = str(getattr(lock, "log_path", "") or "").strip()
    name, run_id = identity
    if not run_dir:
        return LaunchMode(
            attached=True, campaign=None, run_dir=None, holder=lock,
            reason=(
                f"Campaign '{name}' holds the rig but published no run directory, "
                "so there is no stream to follow and no way to pause or abort it "
                "from here. This session will not open, arm or command anything."
            ),
        )

    return LaunchMode(
        attached=True, campaign=(name, run_id), run_dir=run_dir, holder=lock,
        reason=(
            f"Campaign '{name}' (run {run_id}) holds the rig. This session is "
            "attached: it follows the run and can request a stop, but opens no "
            "instrument session of its own."
        ),
    )
