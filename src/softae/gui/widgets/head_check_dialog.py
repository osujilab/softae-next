"""Dispenser-head position verification dialog and start-gate helper.

The pneumatic dispenser head has no position feedback, and an operator can flip
it manually at any time without the software knowing.  These helpers ask the
operator to *register* the current physical position so the software belief
(``syringe.is_head_up()``) matches reality before any automated sequence relies
on it.  Nothing here senses the head — it only records what the operator reports
(and, at a start-gate, optionally retracts for safety).

Two entry points:

``ask_head_state`` / ``register_head_state``
    The launch-time flow: ask, then record the answer.  No motion.

``verify_head_before_run``
    The start-gate flow used before an HT experiment or autonomous campaign.
    Dismiss → abort (returns ``False``); *Lowered* → register **then** issue a
    safety retract before the caller moves the stage.  Registering first matters
    because ``head_retract`` is conditional on the belief.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class HeadState(Enum):
    """Operator's reported dispenser-head position."""

    RAISED = "raised"        # retracted, safe travel position (is_up=True)
    LOWERED = "lowered"      # descended, dispensing position (is_up=False)
    CANCELLED = "cancelled"  # dialog dismissed without an answer


def ask_head_state(parent: QWidget | None, *, context: str = "") -> HeadState:
    """Modal prompt: is the dispenser head currently Raised or Lowered?

    Returns :attr:`HeadState.CANCELLED` if the operator dismisses the dialog.
    Issues no motion and touches no driver — pure question.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Confirm Dispenser Head Position")
    box.setIcon(QMessageBox.Icon.Question)
    prompt = "Confirm the current physical position of the dispenser head"
    if context:
        prompt += f" before {context}"
    box.setText(prompt + ".")
    box.setInformativeText(
        "The system cannot sense the head position, and it may have been "
        "flipped manually.\n\n"
        "• Raised — head retracted (safe travel position)\n"
        "• Lowered — head descended (dispensing position)"
    )
    raised_btn = box.addButton("Raised", QMessageBox.ButtonRole.YesRole)
    lowered_btn = box.addButton("Lowered", QMessageBox.ButtonRole.NoRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(raised_btn)
    box.exec()

    clicked = box.clickedButton()
    if clicked is raised_btn:
        return HeadState.RAISED
    if clicked is lowered_btn:
        return HeadState.LOWERED
    return HeadState.CANCELLED


def register_head_state(manager: "InstrumentManager", state: HeadState) -> bool:
    """Write a reported head state into the syringe driver (no motion).

    Returns ``True`` if a state was recorded, ``False`` for
    :attr:`HeadState.CANCELLED` or when the syringe is unavailable.
    """
    if state is HeadState.CANCELLED:
        return False
    try:
        syr = manager.get("syringe")
    except Exception:
        return False
    syr.set_head_state(state is HeadState.RAISED)
    return True


def _safety_retract(parent: QWidget | None, manager: "InstrumentManager") -> bool:
    """Issue a blocking safety retract; return ``True`` on success.

    Runs synchronously behind a wait cursor — this is a deliberate pre-run
    gate, so a brief pause is acceptable.  A failure is surfaced and reported
    so the caller can abort rather than move the stage with the head down.
    """
    try:
        syr = manager.get("syringe")
    except Exception as exc:
        QMessageBox.critical(
            parent, "Head Retract Failed",
            f"Could not access the syringe to retract the head: {exc}",
        )
        return False
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        syr.head_retract()
    except Exception as exc:
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(
            parent, "Head Retract Failed",
            f"The safety retract did not complete: {exc}\n\n"
            "The run was not started.",
        )
        return False
    QApplication.restoreOverrideCursor()
    return True


def verify_head_before_run(
    parent: QWidget | None,
    manager: "InstrumentManager",
    *,
    context: str = "starting the run",
) -> bool:
    """Start-gate head verification. Returns ``False`` if the run must not start.

    Policy: dismiss → abort (``False``); *Raised* → register and proceed;
    *Lowered* → register **then** safety-retract before the caller moves the
    stage.  If the retract fails, aborts (``False``).
    """
    state = ask_head_state(parent, context=context)
    if state is HeadState.CANCELLED:
        return False
    register_head_state(manager, state)
    if state is HeadState.LOWERED:
        return _safety_retract(parent, manager)
    return True
