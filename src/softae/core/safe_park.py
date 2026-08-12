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

#: Pumps commanded to a near-zero dispense to halt any in-flight motion.
DEFAULT_PUMP_IDS: tuple[int, ...] = (0, 1, 2)


@dataclass
class SafeParkResult:
    """Outcome of a park attempt.

    ``ok`` is ``True`` only when every attempted action succeeded.  A partial
    park is still valuable — and is reported, not hidden — because the operator
    needs to know which subsystem refused to go safe.
    """

    actions: list[str] = field(default_factory=list)   # what succeeded
    errors: list[str] = field(default_factory=list)    # "subsystem: message"
    skipped: list[str] = field(default_factory=list)   # absent/disconnected

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = [f"{len(self.actions)} ok"]
        if self.errors:
            parts.append(f"{len(self.errors)} failed")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)


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


def safe_park(
    manager: "InstrumentManager",
    *,
    reason: str = "",
    pump_ids: Sequence[int] = DEFAULT_PUMP_IDS,
    safe_temp_C: float = DEFAULT_SAFE_TEMP_C,
    retract_head: bool = True,
) -> SafeParkResult:
    """Drive the rig to a safe state. Never raises.

    Order is deliberate: **retract the head first** so it is clear of the board
    before anything else is touched, then halt fluid motion, then remove the
    thermal and optical load.

    Parameters
    ----------
    reason:
        Recorded in the log line — this is often the only durable trace of *why*
        an unattended campaign stopped.
    retract_head:
        Whether to raise the head. **Defaults to true, and every automatic caller
        leaves it that way** — an unattended park, a fault park and a window close
        all raise it, because nobody is present to decide otherwise.

        The one caller that passes ``False`` is the operator pressing *Safe Exit*
        after being asked. Retracting is not universally the safe act: a head left
        deliberately lowered is holding a position (an anneal hold in the flush
        basin, a mid-cast pause, an in-drop mix), and raising it drags the tip
        clear of the drop it is sitting in. That is a judgement only the person at
        the rig can make, so it is asked rather than assumed — and never assumed on
        their behalf when they are not there.
    """
    result = SafeParkResult()
    logger.warning("safe_park_start", reason=reason or "unspecified",
                   retract_head=retract_head)

    # 1. Head up and clear of the board.
    syr = _instrument(manager, "syringe", result)
    if syr is not None:
        if retract_head:
            try:
                syr.head_retract()
                result.actions.append("head retracted")
            except Exception as exc:
                result.errors.append(f"syringe head: {exc}")
        else:
            # Recorded as an action, not a skip: a deliberate choice by the operator
            # is something the park *did*, and the log is the only durable trace of
            # why the head was found down next session.
            result.actions.append("head left lowered (operator choice)")

        # 2. Halt fluid motion (near-zero dispense stops an in-flight command).
        halted: list[int] = []
        for pump_id in pump_ids:
            try:
                syr.single_pump(1000, int(pump_id), 0.1, 0.001)
                halted.append(int(pump_id))
            except Exception as exc:
                result.errors.append(f"pump {pump_id} stop: {exc}")
        if halted:
            result.actions.append(f"pumps {halted} halted")

    # 3. Remove the thermal load.
    tc = _instrument(manager, "temp_controller", result)
    if tc is not None:
        try:
            tc.write_sp(safe_temp_C, print_flag=0)
            result.actions.append(f"temperature setpoint → {safe_temp_C} °C")
        except Exception as exc:
            result.errors.append(f"temperature: {exc}")

    # 4. Lamp off.
    lamp = _instrument(manager, "lamp", result)
    if lamp is not None:
        try:
            lamp.off()
            result.actions.append("lamp off")
        except Exception as exc:
            result.errors.append(f"lamp: {exc}")

    log = logger.error if result.errors else logger.warning
    log(
        "safe_park_done",
        reason=reason or "unspecified",
        ok=result.ok,
        actions=result.actions,
        errors=result.errors,
        skipped=result.skipped,
    )
    return result


async def safe_park_async(
    manager: "InstrumentManager",
    *,
    reason: str = "",
    pump_ids: Sequence[int] = DEFAULT_PUMP_IDS,
    safe_temp_C: float = DEFAULT_SAFE_TEMP_C,
    retract_head: bool = True,
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
