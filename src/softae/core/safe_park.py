"""Drive the rig to a physically safe state — the one canonical stop sequence.

Used by two callers that must never diverge (principle P-3, no second path to the
hardware):

* the GUI **Emergency Stop** button, and
* an **unattended campaign** that has decided to park (bounded retries exhausted,
  a hard fault class, a gate timeout, or reservoir depletion).

Before this existed the sequence lived only inside a Qt worker, so a headless run
had no way to make the rig safe — it simply stopped issuing commands and left the
head down, the heater at setpoint, and the lamp on.

**Scope.** This makes *hardware* safe. It deliberately does **not** abort the
executor or the loop, or record anything: cancelling work and reporting are the
caller's concerns, and conflating them would make the sequence untestable and
unreusable. Every step is best-effort and independent — one failing instrument
must never prevent the others from being made safe — so this function does not
raise. Inspect :class:`SafeParkResult` to see what actually happened.

**What "safe" is worth here.** Nothing on this rig reads back. Every axis below
is *commanded*; none is *verified*; the dispenser head is *unverifiable* and
always will be until it grows a sensor. :class:`SafeParkResult` keeps those three
apart so no surface can round them up into "stopped / safe", and the head is not
moved at all unless a human has just said which way it is pointing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import structlog

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

#: Setpoint the temperature controller is driven to when parking (°C).
DEFAULT_SAFE_TEMP_C = 10.0

#: Pumps halted by :func:`safe_park` (via ``halt_pump``, not via a dispense).
DEFAULT_PUMP_IDS: tuple[int, ...] = (0, 1, 2)

#: The head axis, named once. It appears in every result because it is
#: **permanently** unverifiable: the pneumatic head is a two-state flipper with
#: no position feedback, so no park in any configuration can ever report where
#: it is. Adding a sensor is the only thing that retires this line.
HEAD_UNVERIFIABLE = "dispenser head position: no sensor exists — never confirmed"

#: A park suspends anti-clog purging for as long as it is outstanding
#: (``PurgeRunner._blocking_reason`` refuses while a park reason is set, and that
#: check runs *before* the pose check, so it holds in every pose).
#:
#: That is correct — resuming unprompted actuation would undo the park — but it
#: is a **cost that grows with park duration**, and it is not the head policy that
#: causes it: it was equally true when the park retracted the head, and a raised
#: head is the *more* exposed pose, not the less. So the note is unconditional
#: and says what an operator has to go and look at.
PURGE_SUSPENDED_NOTE = (
    "anti-clog purging is refused for as long as this park stands. A short park "
    "is harmless; a long one leaves any line carrying particulate stock stagnant "
    "with the tip wherever the park found it. Check those lines before resuming, "
    "and clear the park to let purging restart."
)

#: The three operator headlines a park can land under. They live here, beside
#: :meth:`SafeParkResult.describe`, because they are claims about the same
#: result — and a headline each surface derives for itself is how two dialogs
#: come to say different things about one park. :meth:`SafeParkResult.headline`
#: is the only thing that chooses between them.
HEADLINE_COMMANDED = "Stop commands were issued."
HEADLINE_PARTIAL = "PARTIAL STOP — something refused to go safe."
HEADLINE_NOTHING = (
    "NOTHING WAS COMMANDED — no instrument was connected to this process."
)


@dataclass
class SafeParkResult:
    """Outcome of a park attempt, in the only three grades this rig supports.

    The distinction is the point. Before it, ``ok`` meant *"no exception was
    raised"* and every write landed in one undifferentiated ``actions`` list, so
    the E-Stop could report "All instruments stopped / safe" about a heater it
    had merely written a setpoint to and a head it had neither moved nor sensed.

    ==================  =========================================================
    ``commanded``       The write was issued and did not raise. Nothing read back.
    ``verified``        A read-back confirmed the new state. **Empty today** — no
                        axis is verified yet; the temperature PV is readable and
                        is the first candidate (deliberately out of scope: it
                        needs its own "how close counts as parked" threshold).
    ``unverifiable``    No sensor exists. The head, permanently.
    ==================  =========================================================

    ``errors`` and ``skipped`` are unchanged: a subsystem that refused, and one
    that was absent or disconnected. A partial park is reported, never hidden.
    """

    commanded: list[str] = field(default_factory=list)      # issued, not confirmed
    verified: list[str] = field(default_factory=list)       # read back
    unverifiable: list[str] = field(default_factory=list)   # no sensor
    errors: list[str] = field(default_factory=list)         # "subsystem: message"
    skipped: list[str] = field(default_factory=list)        # absent/disconnected
    #: Known consequences of having parked — not failures, not claims about
    #: hardware. A park that silently suspends the anti-clog harness is a park
    #: whose real cost is invisible until a line is dead, and "invisible" is the
    #: property this class exists to remove.
    notes: list[str] = field(default_factory=list)

    @property
    def actions(self) -> list[str]:
        """Everything the park *did* — commanded plus verified.

        Kept because callers outside this module log it (the campaign's park
        event, the headless shutdown report). It is deliberately read-only: a
        caller appending to it was appending to a list of claims, and which grade
        of claim was exactly the thing being lost.
        """
        return [*self.commanded, *self.verified]

    @property
    def ok(self) -> bool:
        """No commanded write raised.

        **Not** a claim that the rig is safe, and never read as one in operator
        text — see :meth:`describe`. With ``verified`` empty, ``ok`` is a
        statement about exceptions, not about hardware.
        """
        return not self.errors

    @property
    def commanded_anything(self) -> bool:
        """Whether this park reached the hardware **at all**.

        The question ``ok`` is routinely mistaken for. A park against a manager
        whose instruments are absent or disconnected files every one under
        ``skipped``, raises nothing, and so reports ``ok is True`` having sent
        not one byte to the rig. That is not a bug in ``ok`` — it is exactly
        what ``ok`` says — but a surface that heads such a result *"Stop
        commands were issued."* is telling the operator the rig was stopped by
        a process that never spoke to it.

        ``verified`` is included even though it is empty on this rig today: if
        an axis ever graduates to read-back, a park that verified something
        certainly commanded something.
        """
        return bool(self.commanded or self.verified)

    def headline(self) -> tuple[str, bool]:
        """The operator's one-sentence verdict, and whether it is a *warning*.

        Three grades, and the order between the first two is deliberate:

        * **errors present** → :data:`HEADLINE_PARTIAL`. Something was reached
          and refused, so instruments *were* connected — which is precisely
          what :data:`HEADLINE_NOTHING` denies. Reporting "no instrument was
          connected" about a rig that answered and said no would be a second
          false statement replacing the first.
        * **nothing commanded, nothing raised** → :data:`HEADLINE_NOTHING`.
        * otherwise → :data:`HEADLINE_COMMANDED`.

        Returns ``(text, severe)``. ``severe`` is a plain bool rather than a Qt
        icon on purpose: this module is imported by the headless campaign and
        must not grow a GUI dependency. Each dialog maps the bool to its own
        icon; none of them re-derives the three-way choice.
        """
        if self.errors:
            return HEADLINE_PARTIAL, True
        if not self.commanded_anything:
            return HEADLINE_NOTHING, True
        return HEADLINE_COMMANDED, False

    def summary(self) -> str:
        parts = [f"{len(self.commanded)} commanded"]
        if self.verified:
            parts.append(f"{len(self.verified)} verified")
        if self.unverifiable:
            parts.append(f"{len(self.unverifiable)} unverifiable")
        if self.errors:
            parts.append(f"{len(self.errors)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)

    def describe(self) -> str:
        """The operator's paragraph — what was commanded versus what was checked.

        Written for the E-Stop dialog, which used to say *"All instruments
        stopped / safe."* This says instead what the system is actually entitled
        to assert.
        """
        blocks: list[tuple[str, list[str]]] = [
            ("Commanded (sent — nothing on this rig reads back to confirm it)",
             self.commanded),
            ("Verified (read back and confirmed)", self.verified),
            ("NOT verifiable (no sensor — look at the hardware)", self.unverifiable),
            ("Failed", self.errors),
            ("Skipped (absent or disconnected)", self.skipped),
            ("Consequences of the park", self.notes),
        ]
        out: list[str] = []
        for title, items in blocks:
            if not items:
                continue
            out.append(title + ":")
            out.extend(f"  • {item}" for item in items)
        if not self.verified:
            out.append("Nothing was verified. This rig has no read-back on any "
                       "parked axis.")
        return "\n".join(out)


def _instrument(manager: "InstrumentManager", name: str, result: SafeParkResult):
    """Return a *connected* instrument, or ``None`` (recording why)."""
    try:
        inst = manager.get(name)
    except Exception:
        result.skipped.append(f"{name}: not registered")
        return None
    if not getattr(inst, "is_connected", False):
        result.skipped.append(f"{name}: not connected")
        return None
    return inst


def _park_head(syr, retract_head: bool | None, result: SafeParkResult) -> None:
    """Apply the head policy. See :func:`safe_park` for why ``None`` is the default.

    Exactly one of the three branches issues motion, and only on an explicit
    instruction from a human.
    """
    if retract_head is None:
        result.unverifiable.append(
            HEAD_UNVERIFIABLE + "; this park did not move it (no operator answer)")
        logger.warning("safe_park_head_untouched",
                       msg="automatic park: head left exactly where it was")
        return

    if not retract_head:
        # Recorded as commanded, not skipped: a deliberate choice by the operator
        # is something the park *did*, and the log is the only durable trace of
        # why the head was found down next session.
        result.commanded.append("head left lowered (operator instruction)")
        result.unverifiable.append(HEAD_UNVERIFIABLE)
        return

    try:
        syr.head_retract()
        # "Commanded", never "retracted". head_retract() is a *conditional flip*
        # on a belief with no feedback: if the belief already says "up" it issues
        # nothing at all, and the old wording claimed a retraction that never
        # happened.
        result.commanded.append("head retract commanded (operator instruction)")
        result.unverifiable.append(HEAD_UNVERIFIABLE)
    except Exception as exc:
        result.errors.append(f"syringe head: {exc}")


def _halt_pumps(syr, pump_ids: Sequence[int], result: SafeParkResult) -> None:
    """Stop every pump via the driver's dedicated halt — never via a dispense."""
    halt = getattr(syr, "halt_pump", None)
    if not callable(halt):
        # Deliberately an error and deliberately *not* a fallback to
        # ``single_pump``: falling back would silently restore the refusal this
        # whole path exists to remove, and would do so exactly when a reservoir
        # is empty.
        result.errors.append(
            "pumps: driver exposes no halt_pump() — nothing stopped the pumps")
        return

    halted: list[int] = []
    for pump_id in pump_ids:
        try:
            halt(int(pump_id))
            halted.append(int(pump_id))
        except Exception as exc:
            result.errors.append(f"pump {pump_id} stop: {exc}")
    if halted:
        result.commanded.append(f"pumps {halted} halted")


def safe_park(
    manager: "InstrumentManager",
    *,
    reason: str = "",
    pump_ids: Sequence[int] = DEFAULT_PUMP_IDS,
    safe_temp_C: float = DEFAULT_SAFE_TEMP_C,
    retract_head: bool | None = None,
) -> SafeParkResult:
    """Drive the rig to a safe state. Never raises.

    Order is deliberate: the head first (when a human has asked for it) so it is
    clear of the board before anything else is touched, then halt fluid motion,
    then remove the thermal and optical load.

    Parameters
    ----------
    reason:
        Recorded in the log line — this is often the only durable trace of *why*
        an unattended campaign stopped.
    retract_head:
        Three-valued, and the default **reverses** what this function used to do.

        ``None`` (default) — **do not touch the head.** Every automatic caller
        gets this: an unattended fault park, a campaign park, an unclean-shutdown
        recovery. The result reports the head as unverifiable and no motion is
        issued.

        ``True`` — retract, because a human said so (the E-Stop's post-stop
        offer, an operator-driven exit).

        ``False`` — leave it lowered, because a human said so (*Safe Exit*).

    Why the default reversed
    ------------------------
    The previous default argued that every automatic caller should retract
    *because nobody is present to decide*. That reasoning is sound about the
    **decision** and wrong about the **capability**. ``head_retract`` is not an
    action; it is a conditional flip on a belief with no feedback
    (``AsyncSyringe.is_head_up``: *"the software's belief, not a sensed value"*).
    So it has two failure modes and both were live:

    * belief says up, head is actually down → **nothing happens**, and the park
      reported ``"head retracted"`` anyway;
    * belief says down, head is actually up → **the flip drives the head down**,
      onto the board, as the emergency response to a fault.

    Leaving a head down costs a sample. Driving it down costs hardware. And the
    belief goes stale *because* something physical went wrong or a human reached
    in — the same population of events that triggers a park. Acting on a belief is
    least defensible exactly when the belief is least likely to hold, so absent an
    operator the correct response to an unknown is to add no motion to it.

    An **age-based freshness heuristic was considered and rejected.** Belief
    desyncs from physical events, not from elapsed time, so age is uncorrelated
    with truth: a five-second-old belief invalidated by a crash is worse than a
    five-hour-old one nobody disturbed. It would have looked principled and
    decided by coin flip.

    The operator remains the sensor, via ``set_head_state`` (which issues no
    motion) — see the E-Stop's post-stop prompt.

    Interaction with the anti-clog harness (P8)
    -------------------------------------------
    A park suspends purging entirely: ``PurgeRunner._blocking_reason`` refuses
    while any park reason is outstanding, ahead of the pose check, so no pose
    re-enables it. Two consequences, and only the second is new:

    * **The suspension is not caused by this change.** It follows from the park
      latch and was equally true when the park retracted. It is reported in
      ``notes`` (:data:`PURGE_SUSPENDED_NOTE`) so a long park's cost to a
      particulate line is visible rather than discovered.
    * **Leaving the head alone is, if anything, better for the tip.** A retract
      moved the tip *out* of whatever it was in — including out of the flush
      basin, which is exactly where idle rest deliberately parks it to keep it
      wet. Under the new default a rig parked from idle rest keeps its tip
      immersed; a rig parked mid-cast keeps a lowered tip in a drop. Neither is
      made worse, and the first is made better.

    What does **not** change: ``classify_pose`` reads ``is_head_up()``, the same
    belief, and its ``UNKNOWN`` state comes from an unreadable or absent reader —
    never from a park declining to move anything. This change does not widen the
    purge gate's refusal surface.
    """
    result = SafeParkResult()
    logger.warning("safe_park_start", reason=reason or "unspecified",
                   retract_head=retract_head)

    # 1. The head — motion only on an explicit human instruction.
    syr = _instrument(manager, "syringe", result)
    if syr is not None:
        _park_head(syr, retract_head, result)
        # 2. Halt fluid motion.
        _halt_pumps(syr, pump_ids, result)
        # 3. The cost of having parked, said out loud rather than discovered.
        result.notes.append(PURGE_SUSPENDED_NOTE)

    # 4. Remove the thermal load.
    tc = _instrument(manager, "temp_controller", result)
    if tc is not None:
        try:
            tc.write_sp(safe_temp_C, print_flag=0)
            # Commanded, not verified. The PV *is* readable, so this is the one
            # axis that could graduate — filed, not fixed: it needs its own
            # thresholds (how close to setpoint counts as parked, how long to
            # wait), and an E-Stop must not block on a heater cooling down.
            result.commanded.append(f"temperature setpoint → {safe_temp_C} °C")
        except Exception as exc:
            result.errors.append(f"temperature: {exc}")

    # 5. Lamp off.
    lamp = _instrument(manager, "lamp", result)
    if lamp is not None:
        try:
            lamp.off()
            result.commanded.append("lamp off")
        except Exception as exc:
            result.errors.append(f"lamp: {exc}")

    log = logger.error if result.errors else logger.warning
    log(
        "safe_park_done",
        reason=reason or "unspecified",
        ok=result.ok,
        commanded=result.commanded,
        verified=result.verified,
        unverifiable=result.unverifiable,
        errors=result.errors,
        skipped=result.skipped,
        notes=result.notes,
    )
    return result


async def safe_park_async(
    manager: "InstrumentManager",
    *,
    reason: str = "",
    pump_ids: Sequence[int] = DEFAULT_PUMP_IDS,
    safe_temp_C: float = DEFAULT_SAFE_TEMP_C,
    retract_head: bool | None = None,
) -> SafeParkResult:
    """:func:`safe_park` off the event loop, for async callers (the campaign loop).

    The driver calls are blocking serial I/O; running them inline would stall the
    loop for seconds while it is trying to shut down cleanly.
    """
    return await asyncio.to_thread(
        safe_park, manager,
        reason=reason, pump_ids=tuple(pump_ids), safe_temp_C=safe_temp_C,
        retract_head=retract_head,
    )
