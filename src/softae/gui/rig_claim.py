"""One adapter between a runner tab and the window's rig claim.

Three tabs drive hardware on daemon threads — HT (``tab_experiment``), Sandbox
(``tab_sandbox``) and Arrhenius (``tab_arrhenius``) — and each needs the same
two things of its host: take :meth:`softae.gui.main_window.MainWindow.rig_run`
when there is a window offering one, and degrade to a no-op when there is not.
The second half matters as much as the first: most of the suite constructs these
tabs with no parent, and a tab that only works inside the shell is a tab that
cannot be tested.

**Why the claim matters at all.** The claim is what makes the background
anti-clog purge *defer*. Until this module existed, ``rig_run`` had exactly one
caller — the HT tab — so an Arrhenius sweep and a Sandbox run claimed nothing,
and the purge timer was free to travel the stage to the flush basin and fire the
syringe in the middle of either. Latent only because ``[purge] actuate`` ships
``false``.

**A function, not a mixin, and deliberately so.** The obvious home is
``DaemonRunnerMixin``, which all three tabs already inherit. But ``BOTabBase``
inherits it too, and ``tests/test_autonomous_run_mixin.py`` pins that the Live-BO
host has **no** ``_rig_run``: its in-process execution was removed along with the
wrapper, and an inherited adapter would silently hand it back. A free function is
reachable by the three that want it and invisible to the one that must not have
it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable


class RigRunClaim:
    """A run's handle on the claim it is currently holding.

    Yielded by :meth:`softae.gui.main_window.MainWindow.rig_run` so a run can
    say *"I am held right now"* without learning the owner string's spelling or
    reaching into the window's registry. That indirection is the point: the
    owner string is composed on the run thread (``f"ht:{wf.name}"``) and the
    suspension is toggled from the executor's asyncio thread, and a string that
    has to match exactly across that boundary is a string that will eventually
    not match. A mismatched owner does not raise — it creates a second registry
    entry that never drains.

    :meth:`set_held` deliberately has the exact signature of
    ``WorkflowExecutor.on_pause_hold``, so wiring the two together is an
    assignment rather than a lambda that could capture the wrong thing.
    """

    __slots__ = ("_activity", "_owner", "_reason")

    def __init__(self, activity, owner: str, *, reason: str = "paused") -> None:
        #: ``None`` makes every method a no-op — see :data:`NULL_RIG_CLAIM`.
        self._activity = activity
        self._owner = owner
        self._reason = reason

    def set_held(self, held: bool) -> None:
        """Suspend the claim while the run is held; restore it when it drives.

        A *suspended* owner is the one that **permits** manual control, so this
        is the operator's pause ruling in one line. Both directions are
        tolerant of an owner with no claim (see ``RigActivity.suspend`` /
        ``unsuspend``), which is what makes a callback that outlives its ``with``
        block harmless rather than a phantom suspension.
        """
        if self._activity is None:
            return
        if held:
            self._activity.suspend(self._owner, reason=self._reason)
        else:
            self._activity.unsuspend(self._owner)


#: What a windowless host's claim looks like: a handle that answers ``set_held``
#: and does nothing with it. Shared rather than constructed per call because it
#: holds no state — and because a *class* with the method is what keeps the
#: no-window path from being the one path that raises ``AttributeError``.
#: ``nullcontext(None)`` was the previous shape and would do exactly that as
#: soon as any caller wrote ``as claim``.
NULL_RIG_CLAIM = RigRunClaim(None, "")


def rig_run(host, owner: str, *,
            instruments: "Iterable[str] | None" = None,
            manage_rest: bool = True):
    """Claim the rig for a run on *host*'s window. A context manager.

    ``instruments`` is the claim's scope. ``None`` is the conservative whole-rig
    claim that conflicts with everything, and it is what a scope derivation
    returns when it fails — widening on doubt, per
    :meth:`softae.core.rig_activity.RigActivity.conflicts`.

    ``manage_rest`` decides whether the window also takes the rig out of, and
    back to, idle rest around the block. A run that drives no fluidics wants the
    claim and nothing else: retracting the tip out of the flush basin for the
    hours an Arrhenius sweep lasts is worse for the line than leaving it there,
    and travelling the stage home afterwards would be motion this run never asked
    for.

    Yields a :class:`RigRunClaim` so the run can suspend its **own** claim while
    it is held at a pause (``with rig_run(...) as claim``). Callers that ignore
    the yielded value are unaffected.

    Returns a null context when the host has no window offering ``rig_run`` — a
    tab used outside the GUI shell, or in a test, stays fully usable. It yields
    :data:`NULL_RIG_CLAIM` rather than ``None`` for that reason: most of the
    suite constructs these tabs with no parent, so the no-window path is the
    *common* path in tests, and it has to answer the same calls as the real one.
    """
    window = host.window() if callable(getattr(host, "window", None)) else None
    factory = getattr(window, "rig_run", None)
    if not callable(factory):
        return nullcontext(NULL_RIG_CLAIM)
    return factory(owner, instruments=instruments, manage_rest=manage_rest)
