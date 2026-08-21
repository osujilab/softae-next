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

    Returns a null context when the host has no window offering ``rig_run`` — a
    tab used outside the GUI shell, or in a test, stays fully usable.
    """
    window = host.window() if callable(getattr(host, "window", None)) else None
    factory = getattr(window, "rig_run", None)
    if not callable(factory):
        return nullcontext()
    return factory(owner, instruments=instruments, manage_rest=manage_rest)
