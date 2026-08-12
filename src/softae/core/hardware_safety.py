"""Hardware-arming interlock — a hard gate against accidental real motion.

Motion on real hardware (stage moves, dispenser head, piezo) must never be
triggerable by a stray, scripted, or agent-issued command.  This module is the
single choke point: before any code drives a manager that holds **real** motion
instruments, it must pass :func:`assert_hardware_armed`, which refuses unless
the operator has *deliberately* armed the system.

Arming is intentionally out-of-band and cannot be satisfied by simply running a
command with the wrong flags:

* set the environment variable ``SOFTAE_ALLOW_HARDWARE=1`` (a conscious,
  session-scoped act on the rig), **or**
* call :func:`arm_hardware` from a trusted, human-driven context (e.g. a GUI
  control that confirms intent).

Mock managers never trip the interlock, so simulation and tests are unaffected.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Environment variable that arms real hardware for the session.
ARM_ENV_VAR = "SOFTAE_ALLOW_HARDWARE"
_ARMED_VALUES = frozenset({"1", "true", "yes", "on"})

#: Instruments capable of physical motion — the ones the interlock protects.
MOTION_INSTRUMENTS = ("stage", "syringe", "piezo")

# Process-level arm, for deliberate in-process arming by trusted callers.
_process_armed = False


class HardwareNotArmedError(RuntimeError):
    """Raised when real motion is requested but hardware is not armed."""


def arm_hardware(enabled: bool = True) -> None:
    """Arm (or disarm) real hardware motion for this process.

    Use only from a trusted, human-driven context after confirming intent.
    Headless/agent paths should rely on the operator setting
    :data:`ARM_ENV_VAR` instead, so they cannot self-arm.
    """
    global _process_armed
    _process_armed = bool(enabled)
    logger.info("hardware_arm_changed", armed=_process_armed)


def hardware_is_armed() -> bool:
    """True if hardware has been deliberately armed (env var or process flag)."""
    if _process_armed:
        return True
    return os.environ.get(ARM_ENV_VAR, "").strip().lower() in _ARMED_VALUES


def probe_motion(manager: Any) -> tuple[list[str], list[str]]:
    """``(real, unreadable)`` motion instruments.

    The second list is the whole point, and its absence was a hole in the interlock.
    The previous implementation caught every exception and ``continue``\\ d, so a
    driver layer that raised produced an **empty list — indistinguishable from "all
    mocks"**. :func:`assert_hardware_armed` then evaluated ``if real and not armed``,
    found ``real`` falsy, raised nothing, and let the stage move unarmed. The same
    empty list made :mod:`softae.gui.app` decline to arm, so the two compounded into
    *nothing armed, nothing blocked, motion proceeds*.

    Enumeration is attempted **first**: a manager whose ``names`` cannot be read yields
    every motion instrument as unreadable rather than none as real, because "I could
    not look" and "I looked and found nothing" must not be the same answer.

    Detection is by driver class: mock drivers are named ``Mock*``; everything else
    owning a motion instrument name is treated as real hardware. The ``liquid_handler``
    coordinator is intentionally excluded (it owns no port — its risk is its
    stage/syringe sub-instruments, which are checked directly).
    """
    try:
        present = set(manager.names)
    except Exception:
        logger.warning(
            "motion_probe_enumeration_failed", exc_info=True,
            msg="cannot list instruments — treating every motion instrument as "
                "unreadable rather than as absent",
        )
        return [], list(MOTION_INSTRUMENTS)

    real: list[str] = []
    unreadable: list[str] = []
    for name in MOTION_INSTRUMENTS:
        if name not in present:
            continue
        try:
            inst = manager.get(name)
        except Exception:
            logger.warning("motion_probe_failed", instrument=name, exc_info=True)
            unreadable.append(name)
            continue
        if not type(inst).__name__.startswith("Mock"):
            real.append(name)
    return real, unreadable


def real_motion_instruments(manager: Any) -> list[str]:
    """Names of registered **real** (non-mock) motion instruments.

    Unchanged contract, so every existing caller is unaffected. Callers that must
    *fail closed* need :func:`probe_motion` instead — this function cannot express
    "unreadable", which is exactly how the interlock came to be defeatable.
    """
    return probe_motion(manager)[0]


def assert_hardware_armed(manager: Any, *, action: str = "move hardware") -> list[str]:
    """Raise :class:`HardwareNotArmedError` if real motion is unarmed.

    Returns the list of real motion instruments detected (empty for a mock
    manager, in which case the call is always a no-op).

    **Fails closed on an unreadable probe.** If the driver layer raises, this cannot
    tell a real stage from a mock one, and the safe reading of "I don't know" is "assume
    it can move". Arming still overrides — an armed operator has already declared
    intent — so the refusal only ever blocks the case where nobody said it was allowed
    *and* nothing could confirm it was safe.
    """
    real, unreadable = probe_motion(manager)

    if unreadable and not hardware_is_armed():
        raise HardwareNotArmedError(
            f"SAFETY INTERLOCK: refusing to {action}. Could not determine whether "
            f"{unreadable} are real hardware or mocks — the driver probe raised. "
            f"An unreadable probe is treated as real motion, because the alternative "
            f"is moving a stage nobody confirmed was simulated. Fix the driver, or "
            f"set {ARM_ENV_VAR}=1 to proceed deliberately."
        )
    if unreadable:
        logger.warning(
            "hardware_probe_unreadable_but_armed", action=action,
            instruments=unreadable,
            msg="proceeding on the operator's arming, not on a successful check",
        )

    if real and not hardware_is_armed():
        raise HardwareNotArmedError(
            f"SAFETY INTERLOCK: refusing to {action}. Real motion instruments "
            f"{real} are present but hardware is not armed. This prevents "
            f"accidental/scripted/agent-driven motion. To deliberately allow it, "
            f"set {ARM_ENV_VAR}=1 in your shell before running (or arm from a "
            f"trusted GUI control)."
        )
    if real:
        logger.warning("hardware_armed_action", action=action, instruments=real)
    return real


def attach_head_guard(manager: Any, *, stage: str = "stage",
                      syringe: str = "syringe") -> bool:
    """Let the stage see head state so it can refuse to move while lowered.

    The stage driver has no reference to the syringe, so the head state is
    injected as an attribute — the same pattern as the reservoir ledger and the
    purge scheduler, and for the same reason: one entry point for every host, so
    the interlock cannot be live in the GUI and inert in the headless CLI.

    Returns ``True`` if the guard was attached. A rig missing either instrument
    is not an error; it simply has nothing to guard.
    """
    try:
        stage_obj = manager.get(stage)
        syringe_obj = manager.get(syringe)
    except Exception:
        logger.info("head_guard_not_attached", reason="instrument missing")
        return False
    if stage_obj is None or syringe_obj is None:
        return False

    stage_obj.head_source = syringe_obj
    logger.info("head_guard_attached", stage=stage, syringe=syringe)
    return True
