"""Which campaign a stop request would reach — one answer for every surface.

Two surfaces can now ask a running campaign to pause or abort: ``softae-campaign
control`` in a terminal, and the Pause/Abort buttons in the tab that surfaces the
run. Both need the same two things before they can write anything:

* **where** the campaign's ``control.json`` goes, and
* **why there is nowhere to write it**, when that is the answer.

Those live here rather than in either surface, because the failure this prevents
is not a crash. If the CLI and the GUI each carry their own account of "no
campaign to control", they can disagree — the terminal refusing while the button
is live, or the reverse — and an operator holding a mouse in one hand and a
prompt in the other has no way to tell which one is right about a rig that is
mid-anneal. One implementation, three refusal branches, quoted identically by
both.

**Discovery is the rig lock and nothing else.** The campaign already publishes
``what = "campaign:<name>:<run_id>"`` and ``log_path = <run directory>`` when it
claims the rig, so a controller reads one file it did not have to invent. A
second registry that could disagree with the lock is how a rig ends up with two
owners and two stories about it.

**Nothing here swallows.** :func:`softae.core.run_lock.read_run_lock` can fail on
a filesystem this process does not control, and the conservative answer to that
differs by caller: the CLI has always let it surface rather than start blind,
while a GUI timer must not take a tab down. That choice belongs at the call site,
which can resolve it to the conservative answer for *its* surface — the same
argument :mod:`softae.gui.launch_mode` makes for reading the lock with the
raising reader and swallowing one level up.

The latency sentences live here for the same reason as the refusals: they are the
operator-facing statement of what a Pause actually guarantees, and a tooltip that
paraphrases them into a promise the code cannot keep is worse than no tooltip.
The CLI prints them and the buttons show them, from one string each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from softae.core.run_lock import read_run_lock

#: What Abort costs and when it lands. Printed by ``softae-campaign control
#: abort`` and shown on the Abort button — said plainly, because the latency is
#: not uniform and promising that it is would be the wrong kind of reassurance.
ABORT_LATENCY_NOTE = (
    "Abort is immediate during a temperature hold and takes effect at the next "
    "step boundary otherwise. The rig is parked and the checkpoint is kept."
)

#: What Pause does **not** do. The second sentence is the load-bearing one: a
#: Pause that dropped the setpoint would destroy the anneal it exists to
#: preserve, so the operator is told what is left running.
PAUSE_LATENCY_NOTE = (
    "Pause stops the run issuing new steps and then holds; a step already "
    "running finishes first. Setpoints, lamp and head are left exactly as they "
    "are."
)

#: The inverse. Nothing is re-driven and nothing is re-initialised, which is what
#: makes "resume" mean *continue* rather than *restart*
#: (:meth:`softae.core.autonomous_loop.AutonomousLoop.resume`).
RESUME_LATENCY_NOTE = (
    "Resume lifts a pause: the run continues from the step it was holding. The "
    "iteration counter and the checkpoint are untouched by the round trip."
)

#: Keyed by :data:`softae.core.campaign_events.CONTROL_ACTIONS`, so a surface
#: that grows a button gets the sentence rather than writing one.
CONTROL_LATENCY_NOTES: dict[str, str] = {
    "abort": ABORT_LATENCY_NOTE,
    "pause": PAUSE_LATENCY_NOTE,
    "resume": RESUME_LATENCY_NOTE,
}


@dataclass(frozen=True)
class CampaignTarget:
    """Where a control request goes, or why it has nowhere to go."""

    #: The campaign's run directory (``lock.log_path``) — where ``control.json``
    #: and ``events.jsonl`` live. ``None`` on every refusal branch.
    run_dir: str | None

    #: On success, the lock's ``what`` (``campaign:<name>:<run_id>``). On a
    #: refusal, one lower-case clause naming what was found instead, written to
    #: read after "No campaign to control: ".
    detail: str

    @property
    def controllable(self) -> bool:
        """Whether there is a campaign to pause or abort."""
        return self.run_dir is not None

    @property
    def refusal(self) -> str:
        """The refusal as an operator reads it. Empty when there is a target."""
        return "" if self.controllable else f"No campaign to control: {self.detail}"


def find_running_campaign(
    *,
    lock_reader: Callable[[], Any] | None = None,
) -> CampaignTarget:
    """The live campaign's run directory, or why we cannot say.

    Three refusal branches, and each is a different fact about the rig rather
    than three spellings of "busy":

    ============================  ==================================================
    Nothing holds the rig         there is no run to control, and starting one is
                                  the operator's next move, not stopping one
    Something else holds it       a bench sequence or an HT workflow publishes no
                                  control channel — the answer is *not* "abort it"
    A campaign with no run dir    it holds the rig but named no directory, so there
                                  is nowhere to put the request
    ============================  ==================================================

    *lock_reader* is injected for tests only — production callers pass nothing,
    because a controller that reads a different lock than the one the campaign
    wrote is the disagreement this module exists to prevent. It is resolved
    here rather than as a default *value* so that a test which substitutes
    :func:`~softae.core.run_lock.read_run_lock` on this module reaches every
    surface at once, which is what makes "the CLI and the GUI share one
    implementation" assertable rather than merely intended.
    """
    lock = (lock_reader or read_run_lock)()
    if lock is None:
        return CampaignTarget(None, "no process holds the rig.")
    if not lock.what.startswith("campaign:"):
        return CampaignTarget(
            None,
            f"the rig is held by something that is not a campaign — {lock.describe()}",
        )
    if not lock.log_path:
        return CampaignTarget(
            None,
            f"the campaign did not publish a run directory — {lock.describe()}",
        )
    return CampaignTarget(lock.log_path, lock.what)
